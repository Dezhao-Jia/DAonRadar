import os
import re
import torch
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, image_dir, labels_dir, transform=None):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        image = Image.open(self.image_dir[idx])
        image = self.transform(image)

        label = self.labels_dir[idx] - 1
        sample = {'image': image, 'label': label}
        return sample


def splitData(root, src='normal', tar='fast', train_num=300, test_num=30):  # ./Data
    random.seed(123)
    src_train_path, src_train_label = [], []
    tar_train_path, tar_train_label = [], []
    tar_test_path, tar_test_label = [], []

    pattern = re.compile(r'S\d{2}')
    per_subject = [pid for pid in os.listdir(root) if os.path.isdir(os.path.join(root, pid)) and pattern.match(pid)]
    print('per subject Info :', per_subject)
    for subject in per_subject:
        src_path = os.path.join(root, subject, src)  # ./Data/S1/normal
        tag_path = os.path.join(root, subject, tar)  # ./Data/S1/fast
        print("src_path :", src_path, '\t', "tar_path :", tag_path)
        # samples_per_subject_per_domain ==350
        temp_p, temp_l = [], []
        for filename in os.listdir(src_path):
            temp_p.append(os.path.join(src_path, filename))
            temp_l.append(int(subject[-1]))
        src_train_path.extend(random.sample(temp_p, train_num))
        src_train_label.extend(random.sample(temp_l, train_num))

        tmp_p, tmp_l = [], []
        for filename in os.listdir(tag_path):
            tmp_p.append(os.path.join(tag_path, filename))
            tmp_l.append(int(subject[-1]))

        tar_train_path.extend(random.sample(tmp_p, train_num))
        tar_train_label.extend(random.sample(tmp_l, train_num))
        filter = [x for x in tmp_p if x not in tar_train_path]
        tar_test_path.extend(random.sample(filter, test_num))
        tar_test_label.extend(random.sample(tmp_l, test_num))

    return [src_train_path, src_train_label], [tar_train_path, tar_train_label], [tar_test_path, tar_test_label]


def get_Loader(root_path, source_path='normal', target_path='fast', batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    src_train, tag_train, tag_test = splitData(root_path, source_path, target_path)
    src_train_dataset = MyDataset(src_train[0], src_train[-1])
    tag_train_dataset = MyDataset(tag_train[0], tag_train[-1])
    tag_test_dataset = MyDataset(tag_test[0], tag_test[-1])

    src_train_loader = DataLoader(src_train_dataset, batch_size=batch_size, shuffle=True)
    tag_train_loader = DataLoader(tag_train_dataset, batch_size=batch_size, shuffle=True)
    tag_test_loader = DataLoader(tag_test_dataset, batch_size=batch_size, shuffle=False)

    return [src_train_loader, tag_train_loader, tag_test_loader]
