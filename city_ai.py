import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(self.root_dir, "leftImg8bit", self.split)
        self.label_dir = os.path.join(self.root_dir, "gtFine", self.split)

        self.images = []
        self.labels = []

        for city in os.listdir(self.image_dir):
            city_image_dir = os.path.join(self.image_dir, city)
            city_label_dir = os.path.join(self.label_dir, city)

            for file_name in os.listdir(city_image_dir):
                image_path = os.path.join(city_image_dir, file_name)
                label_path = os.path.join(city_label_dir, file_name.replace("_leftImg8bit.png", "_gtFine_labelIds.png"))

                self.images.append(image_path)
                self.labels.append(label_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

# 定义数据预处理管道
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        w, h = image.size
        x1 = random.randint(0, w - self.size[0])
        y1 = random.randint(0, h - self.size[1])
        x2 = x1 + self.size[0]
        y2 = y1 + self.size[1]
        image = transforms.functional.crop(image, y1, x1, self.size[1], self.size[0])
        label = transforms.functional.crop(label, y1, x1, self.size[1], self.size[0])
        return image, label

transform_train = transforms.Compose([
    RandomCrop((256, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = CityscapesDataset(root_dir="/path/to/cityscapes", split="train", transform=transform_train)
val_dataset = CityscapesDataset(root_dir="/path/to/cityscapes", split="val", transform=transform_val)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
