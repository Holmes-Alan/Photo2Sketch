import torch.utils.data as data
import torch
import numpy as np
import cv2 as cv
from os import listdir
from os.path import *
from PIL import Image, ImageOps, ImageFile
import random
from glob import glob
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def is_flow_file(filename):
    return any(filename.endswith(extension) for extension in [".flo"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def rescale_img(img_in, w, h):
    # (w, h) = img_in.size
    new_size_in = tuple([w, h])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img.crop((0, 0, ih, iw))
    #y, cb, cr = img.split()
    return img


def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))

    # info_patch = {
    #    'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    # color_factor = 1.2
    # contrast_factor = 1.2
    # bright_factor = 1.1
    # sharp_factor = 1.1
    # img_tar = ImageEnhance.Color(img_tar).enhance(color_factor)
    # img_tar = ImageEnhance.Contrast(img_tar).enhance(contrast_factor)
    # img_tar = ImageEnhance.Brightness(img_tar).enhance(bright_factor)
    # img_tar = ImageEnhance.Sharpness(img_tar).enhance(sharp_factor)
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, info_aug


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h - self.th) // 2:(self.h + self.th) // 2, (self.w - self.tw) // 2:(self.w + self.tw) // 2, :]


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, ref_dir, fineSize=256):
        super(DatasetFromFolder, self).__init__()

        self.data_dir = data_dir
        self.ref_dir = ref_dir

        # self.transform = transforms.Compose([
        #     transforms.Resize((288, 288)),
        #     transforms.RandomCrop(fineSize),
        #     transforms.RandomHorizontalFlip(),
        #     # transforms.RandomVerticalFlip(),
        #     transforms.ToTensor()])

        self.style_transform = transforms.Compose([
            transforms.Resize((fineSize, fineSize)),
            # transforms.RandomCrop(fineSize),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor()])

        img_dir = data_dir + '/train2017'
        edge_dir = data_dir + '/edge'
        self.input_filenames = sorted(glob(join(img_dir, '*.jpg')))
        self.edge_filenames = sorted(glob(join(edge_dir, '*.jpg'))) # .png for old edge and .jpg for HED
        self.ref_filenames = sorted(glob(join(ref_dir, '*/*.JPEG')))
        self.ref_len = len(self.ref_filenames)
        self.input_len = len(self.input_filenames)

    def transform(self, image, mask1, mask2):
        # Resize
        resize = transforms.Resize(size=(288, 288))
        image = resize(image)
        mask1 = resize(mask1)
        mask2 = resize(mask2)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(256, 256))
        image = TF.crop(image, i, j, h, w)
        mask1 = TF.crop(mask1, i, j, h, w)
        mask2 = TF.crop(mask2, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask1 = TF.hflip(mask1)
            mask2 = TF.hflip(mask2)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask1 = TF.vflip(mask1)
            mask2 = TF.vflip(mask2)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask1 = TF.to_tensor(mask1)
        mask2 = TF.to_tensor(mask2)

        return image, mask1, mask2

    def __getitem__(self, index):
        img = cv.imread(self.input_filenames[index])
        YCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
        img, _, _ = cv.split(YCrCb)
        edge = 255 - cv.Canny(img, 100, 200)
        edge = Image.fromarray(edge)
        input = load_img(self.input_filenames[index])
        edge_gt = load_img(self.edge_filenames[index])
        rand_no = torch.randint(0, self.ref_len, (1,)).item()
        ref = load_img(self.ref_filenames[rand_no])


        # input = self.transform(input)
        input, edge, edge_gt = self.transform(input, edge, edge_gt)
        ref = self.style_transform(ref)

        return input, edge, edge_gt, ref

    def __len__(self):
        return self.input_len


class DatasetFromFolder_test(data.Dataset):
    def __init__(self, data_dir, fineSize=256):
        super(DatasetFromFolder_test, self).__init__()

        self.data_dir = data_dir

        img_dir = data_dir + '/train2017'
        edge_dir = data_dir + '/edge'
        self.input_filenames = sorted(glob(join(img_dir, '*.jpg')))
        self.edge_filenames = sorted(glob(join(edge_dir, '*.png')))
        self.input_len = len(self.input_filenames)

        self.data_transform = transforms.Compose([
            transforms.Resize((fineSize, fineSize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = cv.imread(self.input_filenames[index])
        YCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
        img, _, _ = cv.split(YCrCb)
        edge = 255 - cv.Canny(img, 100, 200)
        edge = Image.fromarray(edge)
        input = load_img(self.input_filenames[index])
        edge_gt = load_img(self.edge_filenames[index])


        # input = self.transform(input)
        input = self.data_transform(input)
        edge = self.data_transform(edge)
        edge_gt = self.data_transform(edge_gt)

        return input, edge, edge_gt

    def __len__(self):
        return self.input_len


