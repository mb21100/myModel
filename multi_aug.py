from PIL import Image
from torchvision import transforms
import numpy as np
import torch

class MultiGeometricAug(object):
    def __init__(self, num_aug=5, crop_size=224):
        self.num_aug = num_aug
        self.crop_size = crop_size
        # 여기서는 기하학적 변환만 사용 (랜덤 크롭, 수평 반전, 회전 등)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
        ])
        # ToTensor는 나중에 적용
        self.to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        # sample['d_img_org']는 numpy array 형식이라고 가정 (C, H, W) 형식
        # 이를 PIL 이미지로 변환하기 위해 (H, W, C)로 transpose하고, [0,1] 범위로 복원
        d_img = sample['d_img_org']
        # (C, H, W) → (H, W, C), 값 범위를 0-255로 변환
        d_img = np.transpose(d_img, (1, 2, 0))
        d_img = (d_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(d_img)

        aug_list = []
        for _ in range(self.num_aug):
            # 각 반복마다 독립적으로 변환 적용
            aug_img = self.transform(pil_img)
            # 변환된 이미지를 Tensor로 변환
            aug_tensor = self.to_tensor(aug_img)
            aug_list.append(aug_tensor)
        sample['d_img_aug'] = aug_list
        return sample
