import os
import random
from PIL import Image
from torch.utils.data import Dataset


class PairedImageDataset(Dataset):
    def __init__(self, data_dir, img_splice=None, transforms=None, crop_size=None, random_flip=False):
        in_list_path = os.path.join(data_dir, 'in_list.txt')
        assert os.path.exists(in_list_path), "File \"in_list.txt\" not found!"
        self.in_list = open(in_list_path).read().splitlines()

        if img_splice is None:
            gt_list_path = os.path.join(data_dir, 'gt_list.txt')
            assert os.path.exists(gt_list_path), "File \"gt_list.txt\" not found!"
            self.gt_list = open(gt_list_path).read().splitlines()
            assert len(self.in_list) == len(self.gt_list), "Length of the two file lists are inconsistent!"
        else:
            assert img_splice in ['xy', 'yx'], "Invalid value for the argument \"--img_splice\"!"
            self.img_splice = img_splice

        if crop_size is not None:
            assert isinstance(crop_size, list) and (len(crop_size) == 1 or len(crop_size) == 2), \
                "Invalid value for the argument \"--crop_size\""
            self.crop_size = crop_size if len(crop_size) == 2 else crop_size * 2
        else:
            self.crop_size = None

        self.random_flip = random_flip

        self.transforms = transforms

    def __getitem__(self, index):
        if hasattr(self, 'gt_list'):
            img_in = Image.open(self.in_list[index])
            img_gt = Image.open(self.gt_list[index])
        else:
            img = Image.open(self.in_list[index])
            w, h = img.size
            assert w % 2 == 0
            img_left = img.crop((0, 0, w / 2, h))
            img_right = img.crop((w / 2, 0, w, h))
            if self.img_splice == 'xy':
                img_in, img_gt = img_left, img_right
            else:
                assert self.img_splice == 'yx'
                img_in, img_gt = img_right, img_left

        assert img_in.size == img_gt.size

        if self.crop_size is not None:
            # apply random crop to images
            img_in, img_gt = self._random_crop([img_in, img_gt])

        if self.random_flip:
            # apply random crop to images
            img_in, img_gt = self._random_flip([img_in, img_gt])

        if self.transforms is not None:
            # apply transforms to images
            img_in = self.transforms(img_in)
            img_gt = self.transforms(img_gt)

        return {"in": img_in, "gt": img_gt}

    def __len__(self):
        return len(self.in_list)

    # randomly crop the images to specific size
    def _random_crop(self, images):
        w, h = images[0].size
        try:
            left = random.randrange(w - self.crop_size[0])
        except ValueError:
            assert (w - self.crop_size[0]) <= 0
            left = (w - self.crop_size[0]) // 2
        try:
            top = random.randrange(h - self.crop_size[1])
        except ValueError:
            assert (h - self.crop_size[1]) <= 0
            top = (h - self.crop_size[1]) // 2
        right = left + self.crop_size[0]
        bottom = top + self.crop_size[1]
        return (img.crop((left, top, right, bottom)) for img in images)

    # randomly flip the images horizontally
    def _random_flip(self, images):
        if random.choice([True, False]):
            return (img.transpose(Image.FLIP_LEFT_RIGHT) for img in images)
        else:
            return tuple(images)