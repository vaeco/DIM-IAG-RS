import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from PIL import Image

import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])


def is_image_file(filename: str):
    return any(filename.lower().endswith(extension) for extension in [".png", ".jpg", ".bmp", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = transform(img)
    return img

class DIMDataset(data.Dataset):
    def __init__(self, data_dir):
        super(DIMDataset, self).__init__()
        self.data_dir = data_dir

        input_folder = self.data_dir + '/input'
        target_folder = self.data_dir + '/target'
        self.input_filenames = [join(input_folder, x) for x in listdir(input_folder) if is_image_file(x)]
        self.target_filenames = [join(target_folder, x) for x in listdir(target_folder) if is_image_file(x)]

        self.num = len(self.input_filenames)

    def __getitem__(self, index):
        im1 = load_img(self.input_filenames[index])
        target_index = self.target_filenames.index(self.input_filenames[index].replace('input', 'target'))
        im2 = load_img(self.target_filenames[target_index])

        return im1, im2

    def __len__(self):
        return self.num


if __name__ == '__main__':
    data = DIMDataset(r'D:\datasets\ReF_DIM')
    print(data.__len__())