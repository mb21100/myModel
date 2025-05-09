import torch
import numpy as np
import cv2


def random_crop(d_img, config):
    b, c, h, w = d_img.shape
    top = np.random.randint(0, h - config.crop_size)
    left = np.random.randint(0, w - config.crop_size)
    d_img_org = crop_image(top, left, config.crop_size, img=d_img)
    return d_img_org


def crop_image(top, left, patch_size, img=None):
    tmp_img = img[:, :, top:top + patch_size, left:left + patch_size]
    return tmp_img


def five_point_crop(idx, d_img, config):
    new_h = config.crop_size
    new_w = config.crop_size
    b, c, h, w = d_img.shape
    if idx == 0:
        top = 0
        left = 0
    elif idx == 1:
        top = 0
        left = w - new_w
    elif idx == 2:
        top = h - new_h
        left = 0
    elif idx == 3:
        top = h - new_h
        left = w - new_w
    elif idx == 4:
        center_h = h // 2
        center_w = w // 2
        top = center_h - new_h // 2
        left = center_w - new_w // 2
    d_img_org = crop_image(top, left, config.crop_size, img=d_img)

    return d_img_org


def split_dataset_kadid10k(txt_file_name, split_seed=20):
    np.random.seed(split_seed)
    object_data = []
    with open(txt_file_name, 'r') as listFile:
        for line in listFile:
            dis, score = line.split()
            dis = dis[:-1]
            if dis[1:3] not in object_data:
                object_data.append(dis[1:3])
    
    np.random.shuffle(object_data)
    np.random.seed(20)

    l = len(object_data)
    train_name = object_data[:int(l * 0.8)]
    val_name = object_data[int(l * 0.8):]
    return train_name, val_name


def split_dataset_tid2013(txt_file_name, split_seed=20):
    np.random.seed(split_seed)
    object_data = []
    with open(txt_file_name, 'r') as listFile:
        for line in listFile:
            score, dis = line.split()
            if dis[1:3] not in object_data:
                object_data.append(dis[1:3])
    
    np.random.shuffle(object_data)
    np.random.seed(20)

    l = len(object_data)
    train_name = object_data[:int(l * 0.8)]
    val_name = object_data[int(l * 0.8):]
    return train_name, val_name


def split_dataset_live(txt_file_name, split_seed=20):
    np.random.seed(split_seed)
    object_data = []
    with open(txt_file_name, 'r') as listFile:
        for line in listFile:
            i1, i2, ref, dis, score, h, w = line.split()
            if ref[8:] not in object_data:
                object_data.append(ref[8:])
    
    np.random.shuffle(object_data)
    np.random.seed(20)

    l = len(object_data)
    train_name = object_data[:int(l * 0.8)]
    val_name = object_data[int(l * 0.8):]
    return train_name, val_name


def split_dataset_csiq(txt_file_name, split_seed=20):
    np.random.seed(split_seed)
    object_data = []
    with open(txt_file_name, 'r') as listFile:
        for line in listFile:
            dis, score= line.split()
            dis_name, dis_type, idx_img, _ = dis.split(".")
            if dis_name not in object_data:
                object_data.append(dis_name)
    
    np.random.shuffle(object_data)
    np.random.seed(20)

    l = len(object_data)
    train_name = object_data[:int(l * 0.8)]
    val_name = object_data[int(l * 0.8):]
    return train_name, val_name


class RandCrop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        
    def __call__(self, sample):
        # r_img : C x H x W (numpy)
        d_img = sample['d_img_org']
        score = sample['score']

        c, h, w = d_img.shape
        new_h = self.patch_size
        new_w = self.patch_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        ret_d_img = d_img[:, top: top + new_h, left: left + new_w]

        sample = {
            'd_img_org': ret_d_img,
            'score': score
        }
        return sample


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        d_img = sample['d_img_org']
        score = sample['score']
        d_img = (d_img - self.mean) / self.var
        sample = {'d_img_org': d_img, 'score': score}
        return sample


class RandHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']
        prob_lr = np.random.random()
        # np.fliplr needs HxWxC
        if prob_lr > 0.5:
            d_img = np.fliplr(d_img).copy()
        
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        return sample

class RandFlip(object):
    def __init__(self):
        pass
    
    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']
        prob_lr = np.random.random()
        prob_ud = np.random.random()
        # np.flipud needs HxWxC
        if prob_ud > 0.5 :
            d_img = np.flipud(d_img).copy()
        
        if prob_lr > 0.5:
            d_img = np.fliplr(d_img).copy()
        
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        return sample

class RandFlipRotate(object):
    def __init__(self, max_angle=5, prob_rotate=0.5):
        self.max_angle = max_angle      # 최대 회전 각도 (°)
        self.prob_rotate = prob_rotate  # 회전 적용 확률 (0~1)
    
    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']
        
        # 수직/수평 flip
        if np.random.random() > 0.5:
            d_img = np.flipud(d_img).copy()
        if np.random.random() > 0.5:
            d_img = np.fliplr(d_img).copy()
        
        # 확률에 따라 회전 적용
        if np.random.random() < self.prob_rotate:
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            h, w = d_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            d_img = cv2.warpAffine(d_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        return sample



class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']
        d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        score = torch.from_numpy(score).type(torch.FloatTensor)
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        return sample