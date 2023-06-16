import os
import cv2
import argparse
import importlib
from skimage.measure import compare_psnr, compare_ssim

import torch
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image

from libs.utils import CSVLogger, make_dir
from libs.dataset import PairedImageDataset


def main(args):
    assert os.path.exists(args.model_file), "Model file not found in the specified path: {}".format(args.model_file)

    # create directory to store the evaluation results
    args.exp_dir = make_dir(args.exp_dir, allow_repeat=True)

    # set the random seed
    torch.manual_seed(1)

    # specify the device used for computing: GPU ('cuda') or CPU ('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model
    model = importlib.import_module('models.{}'.format(args.generator)).Generator()

    # load model
    model.load_state_dict(torch.load(args.model_file))

    # deploy model to device
    model.to(device)

    # define image pre-processing methods
    tra = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # create dataset objects
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    assert os.path.exists(train_dir) and os.path.exists(test_dir), "Invalid value for the argument \"--data_split\""
    train_dataset = PairedImageDataset(train_dir, img_splice=args.img_splice, transforms=tra, crop_size=args.crop_size)
    test_dataset = PairedImageDataset(test_dir, img_splice=args.img_splice, transforms=tra, crop_size=args.crop_size)

    # evaluate on the training set
    print("Evaluating on the training set...")
    train_log_file = os.path.join(args.exp_dir, 'training_set_evaluation.csv')
    train_exp_img_dir = make_dir(os.path.join(args.exp_dir, 'training_set_results')) if args.exp_img else None
    infer(model, device, train_dataset, train_log_file, exp_img_dir=train_exp_img_dir)

    # evaluate on the testing set
    print("Evaluating on the testing set...")
    test_log_file = os.path.join(args.exp_dir, 'testing_set_evaluation.csv')
    test_exp_img_dir = make_dir(os.path.join(args.exp_dir, 'testing_set_results')) if args.exp_img else None
    infer(model, device, test_dataset, test_log_file, exp_img_dir=test_exp_img_dir)

    print("All Finished!")


# inference on the specified dataset
def infer(model, device, dataset, log_file, exp_img_dir=None):
    # create a logger to record evaluation results of each data instance
    logger = CSVLogger(log_file, columns_names=['index', 'image file', 'PSNR', 'SSIM'], append=False)

    for i in range(len(dataset)):

        print("\r[{} / {}] ".format(i, len(dataset)), end='', flush=True)

        # retrieve instance of images data from the dataset
        img_pair = dataset[i]

        # retrieve the name of the input image files
        img_file = os.path.basename(dataset.in_list[i])

        # get tensors of inputs and ground truths from the dataset
        input = Variable(img_pair["in"]).to(device)
        ground_truth = Variable(img_pair["gt"]).to(device)

        # inference
        with torch.no_grad():
            model.eval()
            output = model(torch.unsqueeze(input, 0))[0]

        # calculate the PSNR and SSIM between the generated results and their corresponding ground truths
        psnr, ssim = evaluate(output, ground_truth)

        # record evaluation results of the instance
        logger.add_row_from_list(list(map(str, [i, img_file, psnr, ssim])))

        # save the generated results as image files in the directory if specified
        if exp_img_dir is not None:
            exp_image = torch.cat((input.data, output.data, ground_truth.data), -2)
            save_image(exp_image, os.path.join(exp_img_dir, '{}.jpg'.format(i)), nrow=1, normalize=True)

    print("Ok!")


# evaluate the generated output
def evaluate(output, ground_truth):
    image_out = output.data.permute(1, 2, 0).cpu().numpy()
    image_gt = ground_truth.data.permute(1, 2, 0).cpu().numpy()
    image_out = cv2.normalize(src=image_out, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image_gt = cv2.normalize(src=image_gt, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    psnr = compare_psnr(image_out, image_gt)
    ssim = compare_ssim(image_out, image_gt, multichannel=True)

    print('PSNR: {:.4f} | SSIM: {:.4f}'.format(psnr, ssim))

    return psnr, ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Derain (evaluation)')
    parser.add_argument('--exp_dir',        type=str,
                        help='specify the path to the directory to save the generated images and evaluation results')
    parser.add_argument('--data_dir',       type=str,
                        help='specify the path to the directory of the dataset')
    parser.add_argument('--generator',      type=str,
                        help='specify the module name of the generator network')
    parser.add_argument('--model_file',     type=str,
                        help='specify the path to where the trained generator model is saved')
    parser.add_argument('--img_splice',     type=str,       choices=['xy', 'yx'],   default=None,
                        help='specify how the inputs and ground truths in the dataset are spliced into one image:'
                             '\'xy\' stands for the inputs at left-hand-side and the ground truth as right-hand-side,'
                             'default None, where the inputs and ground truths are stored as different images')
    parser.add_argument('--crop_size',      type=int,       nargs='*',              default=None,
                        help='specify the size for the randomly cropping of the input images during training,'
                             'default None for not cropping')
    parser.add_argument('--exp_img',        action="store_true",                    default=False,
                        help='specify whether to export the generated images to files')
    args = parser.parse_args()

    main(args)
