import torch
from torch.utils.data import Dataset


class PairedDistortedDataGenerator(Dataset):
    def __init__(self, synthesizer, distort_len=256, distort_range=256, img_size=(256, 256), dataset_len=100000, transforms=None):
        assert synthesizer.distort_len == distort_len
        assert synthesizer.img_size == img_size
        self.synthesizer = synthesizer
        self.dataset_len = dataset_len
        self.distort_len = distort_len
        self.distort_range = distort_range
        '''
        if isinstance(img_size, tuple) && len(img_size) == 2:
            self.img_size = img_size
        else:
            assert isinstance(img_size, int)
            self.img_size = tuple([img_size] * 2)
        '''
        self.transforms = transforms

    def __getitem__(self, index):
        assert index < self.dataset_len, "Index {} exceeds the size of the dataset {}.".format(index, self.dataset_len)
        torch.manual_seed(index)
        img_clean = torch.randint(256, self.img_size)
        distort_vector = torch.randint(self.distort_range, (self.distort_len))

        img_distorted = self.synthesizer(img_clean.clone, distort_vector)

        if self.transforms is not None:
            # apply transforms to images
            img_distorted = self.transforms(img_distorted)
            img_clean = self.transforms(img_clean)

        return {"in": img_distorted, "gt": img_clean}

    def __len__(self):
        return self.dataset_len