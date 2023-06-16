import os
import time
import argparse
import datetime
import importlib
import numpy as np

import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from libs.dataset import PairedImageDataset
from libs.utils import CSVLogger, make_dir, save_config
from libs.model_initializtion import kaiming_normal_init


def main(args):
    # create directory to store checkpoints and experiment results
    if args.init_epoch != 0:
        # if continue training from a checkpoint, check whether the relevant experiment directory exists
        assert os.path.exists(args.exp_dir), "Path specified in exp_dir not found!"
    else:
        # create new directory with
        args.exp_dir = make_dir(args.exp_dir, allow_repeat=True)
        print("Files will be exported to: {}".format(args.exp_dir))
        # save configurations to file
        save_config(args, args.exp_dir)

    # check and create relevant directory for storing relevant experiment results
    checkpoints_dir = make_dir(os.path.join(args.exp_dir, 'checkpoints'))
    log_dir = make_dir(os.path.join(args.exp_dir, 'train_log'))

    # set the random seed
    torch.manual_seed(1)

    # specify the device used for computing: GPU ('cuda') or CPU ('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ####################################################################################################################

    # define models
    generator = importlib.import_module('models.{}'.format(args.generator)).Generator()
    if args.discriminator is not None:
        discriminator = importlib.import_module('models.{}'.format(args.discriminator)).Discriminator()
        models = {'G': generator, 'D': discriminator}
    else:
        models = {'AE': generator}

    if args.init_epoch != 0:
        # load checkpoints
        for m in models:
            checkpoints_path =  os.path.join(checkpoints_dir, '{}_{}.pth'.format(m, args.init_epoch))
            models[m].load_state_dict(torch.load(checkpoints_path))
            # deploy models to device
            models[m].to(device)
    else:
        # initialize weights
        for m in models:
            models[m].apply(kaiming_normal_init)
            # deploy models to device
            models[m].to(device)

    ####################################################################################################################

    # define image pre-processing methods
    tra = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # specify directories of training set and testing set respectively
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    assert os.path.exists(train_dir) and os.path.exists(test_dir), "Invalid value for the argument \"--data_split\""

    # create dataset objects
    train_dataset = PairedImageDataset(train_dir, img_splice=args.img_splice, transforms=tra, crop_size=args.crop_size,
                                       random_flip=args.random_flip)
    #test_dataset = PairedImageDataset(test_dir, img_splice=args.img_splice, transforms=tra, crop_size=args.crop_size,
    #                                  random_flip=args.random_flip)

    # create data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    #test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    ####################################################################################################################

    # define loss functions
    if args.discriminator is not None:
        criteria = {'criterion_GAN': torch.nn.MSELoss() if args.lsgan else torch.nn.BCEWithLogitsLoss(),
                    'criterion_pixelwise': torch.nn.L1Loss()
                    }
    else:
        criteria = {'criterion_AE': torch.nn.MSELoss()}
    criteria = {c: criteria[c].to(device) for c in criteria}

    # define optimizers for each model
    optimizers = {'optimizer_{}'.format(m):
                      torch.optim.Adam(models[m].parameters(), lr=args.learn_rate, betas=(0.5, 0.999)) for m in models
                  }

    ####################################################################################################################

    # training models
    train(models, device, train_data_loader, criteria, optimizers, checkpoints_dir, log_dir, args)

    print("All Finished!")


def train_gan(input, ground_truth, device, G, D, criterion_GAN, criterion_pixelwise, optimizer_G, optimizer_D):
    # calculate output of image discriminator (PatchGAN)
    patch = (1, input.size(2) // 2 ** 4, input.size(3) // 2 ** 4)

    # adversarial ground truths
    valid = Variable(torch.Tensor(np.ones((input.size(0), *patch))).to(device), requires_grad=False)
    fake = Variable(torch.Tensor(np.zeros((input.size(0), *patch))).to(device), requires_grad=False)

    ##### train generator #####

    optimizer_G.zero_grad()

    # feed forward and obtain generated results
    output = G(input)

    # feed forward and obtain discrimination results
    pred_fake = D(output, input)

    # calculate the GAN loss
    loss_G = criterion_GAN(pred_fake, valid)
    # calculate the pixel-wise loss between translated image and real image
    loss_pixel = criterion_pixelwise(output, ground_truth)
    # calculaten the total loss of the generator
    loss_total = loss_G + 100 * loss_pixel

    # update network parameters
    loss_total.backward()
    optimizer_G.step()

    ##### train discriminator #####

    optimizer_D.zero_grad()

    # calculate the real loss
    pred_real = D(ground_truth, input)
    loss_real = criterion_GAN(pred_real, valid)
    # calculate fake loss
    pred_fake = D(output.detach(), input)
    loss_fake = criterion_GAN(pred_fake, fake)
    # calculate the total loss of the discriminator
    loss_D = 0.5 * (loss_real + loss_fake)

    # update network parameters
    loss_D.backward()
    optimizer_D.step()

    return {'D loss': loss_D, 'G loss': loss_G, 'pixel loss': loss_pixel, 'total loss': loss_total}


def train_autoencoder(input, ground_truth, AE, criterion_AE, optimizer_AE):

    optimizer_AE.zero_grad()

    # feed forward and obtain generated results
    output = AE(input)

    # calculate the pixel-wise loss
    loss = criterion_AE(output, ground_truth)

    # update network parameters
    loss.backward()
    optimizer_AE.step()

    return {'loss': loss}



def train(models, device, data_loader, criteria, optimizers, checkpoints_dir, log_dir, args):
    prev_time = time.time()

    # train each epoch
    for epoch in range(args.init_epoch + 1, args.num_epochs + 1):

        # create logger to record training loss data of each iteration
        columns = ['batch', 'D loss', 'G loss', 'pixel loss', 'total loss'] if args.discriminator is not None else ['batch', 'loss']
        logger = CSVLogger(os.path.join(log_dir, 'epoch_{}.csv'.format(epoch)), columns_names=columns, append=False)

        # train each iteration
        for i, batch in enumerate(data_loader):
            # assign variable
            input = Variable(batch["in"]).to(device)   # input image
            ground_truth = Variable(batch["gt"]).to(device)   # ground truth image

            # train model
            if args.discriminator is not None:
                losses = train_gan(input, ground_truth, device, **models, **criteria, **optimizers)
            else:
                losses = train_autoencoder(input, ground_truth, **models, **criteria, **optimizers)

            # record training log data
            logger.add_row_from_dict(dict({'batch': i}, **losses))

            # estimate time left
            batches_done = (epoch-1) * len(data_loader) + i
            batches_left = args.num_epochs * len(data_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # print log
            loss_message = ' '.join(['{}: {:.4f}'.format(k, v) for k, v in losses.items()])
            print("\r[Epoch {}/{}] [Batch {}/{}] [{}] Time: {}".format(
                epoch, args.num_epochs, i, len(data_loader), loss_message, time_left
            ), end='', flush=True)

        # save model checkpoints
        if args.checkpoint != -1 and epoch % args.checkpoint == 0:
            for m in models:
                torch.save(models[m].state_dict(), os.path.join(checkpoints_dir,'{}_{}.pth'.format(m, epoch)))

    # save trained model
    for m in models:
        torch.save(models[m].state_dict(), os.path.join(args.exp_dir, '{}_epoch={}.pth'.format(m, args.num_epochs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Derain (training)')
    parser.add_argument('--exp_dir',        type=str,
                        help='specify the path to the directory to save the experiment results and logging data')
    parser.add_argument('--data_dir',       type=str,
                        help='specify the path to the directory of the dataset')
    parser.add_argument('--generator',      type=str,
                        help='specify the module name of the generator network')
    parser.add_argument('--discriminator',  type=str,       default=None,
                        help='specify the module name of the discriminator network,'
                             'if not specified, the model will be trained as a denoising Autoencoder')
    parser.add_argument('--lsgan',          action="store_true",                    default=False,
                        help='specify whether to use LSGAN loss or not when training as a GAN model')
    parser.add_argument('--img_splice',     type=str,       choices=['xy', 'yx'],   default=None,
                        help='specify how the inputs and ground truths in the dataset are spliced into one image:'
                             '\'xy\' stands for the inputs at left-hand-side and the ground truth as right-hand-side,'
                             'default None, where the inputs and ground truths are stored as different images')
    parser.add_argument('--crop_size',      type=int,       nargs='*',              default=None,
                        help='specify the size for the randomly cropping of the input images during training,'
                             'default None for not cropping')
    parser.add_argument('--random_flip',    action="store_true",                    default=False,
                        help='specify whether to randomly flip the training images horizontally as data augmentation')
    parser.add_argument('--num_epochs',     type=int,       default=200,
                        help='define the number of epochs for the training process')
    parser.add_argument('--init_epoch',     type=int,       default=0,
                        help='specify the initial epoch to start from (specify the corresponding checkpoint to load)')
    parser.add_argument("--checkpoint",     type=int,       default=-1,
                        help="interval between model checkpoints (by defalut, no checkpoint will be saved)")
    parser.add_argument('--learn_rate',     type=float,     default=0.0002,
                        help='define the learning rate')
    parser.add_argument('--batch_size',     type=int,       default=1,
                        help='define the batch size')
    parser.add_argument('--num_workers',    type=int,       default=0,
                        help='define the number of workers for the data loaders')
    args = parser.parse_args()

    main(args)
