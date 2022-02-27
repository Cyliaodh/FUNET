import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import PIL
from data_management.utils import file_reader


class DatasetCINEDE(Dataset):
    def __init__(self, folder_path, transform=None, img_size=None):
        super(DatasetCINEDE, self).__init__()

        if img_size is None:
            img_size = [256, 256]

        self.img_size = img_size
        self.folder_path = folder_path
        self.img_size = img_size
        self.transform = transform

        self.cine_images = []
        self.de_images = []
        self.cine_masks = []
        self.de_masks = []

        for i, folder in enumerate(sorted(os.listdir(folder_path))):

            cine_img_path = os.path.join(folder_path, folder + '/patient{:03d}.nii.gz'.format(i + 1))
            cine_gt_path = os.path.join(folder_path, folder + '/patient{:03d}_gt.nii.gz'.format(i + 1))
            de_img_path = os.path.join(folder_path, folder + '/Case_{:03d}.nii.gz'.format(i + 1))
            de_gt_path = os.path.join(folder_path, folder + '/Case_{:03d}_gt.nii.gz'.format(i + 1))

            _, cine_img = file_reader(cine_img_path)
            _, cine_gt = file_reader(cine_gt_path)
            _, de_img = file_reader(de_img_path)
            _, de_gt = file_reader(de_gt_path)

            for slice in range(cine_img.shape[2]):
                self.cine_images.append(self.resize(cine_img[:, :, slice]))
                self.cine_masks.append(self.resize(cine_gt[:, :, slice]))
                self.de_images.append(self.resize(de_img[:, :, slice]))
                self.de_masks.append(self.resize(de_gt[:, :, slice]))

    def __len__(self):
        return len(self.de_images)

    def __getitem__(self, index):

        img1 = self.cine_images[index]
        img2 = self.de_images[index]

        img1 = torch.from_numpy(np.vstack(img1).astype(np.float))
        img2 = torch.from_numpy(np.vstack(img2).astype(np.float))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        mask1 = self.cine_masks[index]
        mask2 = self.de_masks[index]

        mask1 = torch.from_numpy(np.vstack(mask1))
        mask2 = torch.from_numpy(np.vstack(mask2))

        img1, mask1, img2, mask2 = img1[None, :], mask1, img2[None, :], mask2

        return img1.float(), mask1.type(torch.LongTensor), img2.float(), mask2.type(torch.LongTensor)

    def resize(self, image, interp_option=PIL.Image.NEAREST):
        image_resized = np.array(Image.fromarray(image).resize(self.img_size, resample=interp_option))
        return image_resized
