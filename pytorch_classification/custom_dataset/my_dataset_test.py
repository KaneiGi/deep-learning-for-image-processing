from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSetTest(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img = Image.open(self.images_path[index])
        if img.mode != "RGB":
            raise ValueError('image:{} is not RGB mode'.format(self.images_path[index]))

        label = self.images_class[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels





if __name__ == '__main__':
    a = list(range(2))
    b = list(range(2))
    a, b = tuple(zip(a, b))
    print(a, b)
    c = tuple(range(10))
    a, b, *_ = c
    print(a, b)
