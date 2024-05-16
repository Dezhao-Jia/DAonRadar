import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        """
        :param root_dir(str): 包含图像的跟目录路径 - 必填
        :param transform: 数据增强函数 - 可选
        """
        self.root_dirs = root_dirs
        self.transform = transform
        self.img_files = []
        self.labels = []

        for label, root_dir in enumerate(self.root_dirs):
            img_files = os.listdir(root_dir)
            self.img_files.extend(img_files)
            self.labels.extend([label] * len(img_files))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dirs[self.labels[index]], self.img_files[index])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        # 获取标签
        label = self.labels[index]
        sample = {'image': image, 'label': label}
        return sample


class LoadData:
    def __init__(self):
        self.train_dirs = ["Data/S01/1", "Data/S02/1"]
        self.test_dirs = ["Data/S01/2", "Data/S02/2"]

    def get_datasets(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_datesets = MyDataset(self.train_dirs, transform)
        test_datasets = MyDataset(self.test_dirs, transform)

        return [train_datesets, test_datasets]
