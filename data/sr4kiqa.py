import os
import torch
import numpy as np
import cv2 

class SR4KIQA(torch.utils.data.Dataset):
    def __init__(self,dis_path,txt_folder_name,txt_file_name): #def __init__(self,dis_path,txt_folder_name,transform,txt_file_name):
        super(SR4KIQA, self).__init__()
        self.dis_path = dis_path #/user/home/mb21100/data/SR4KIQA
        self.txt_folder_name = txt_folder_name #sr4kiqa.txt
        #self.transform = transform
        self.txt_file_name = txt_file_name #/user/home/mb21100/data/SR4KIQA/MOS.csv

        self.row2folder = load_row2folder(txt_folder_name) #"01":"Animal1" "02":"Animal2" ... "24":"TextSimpOut2"

        dis_files_data, score_data = [], []
        with open(txt_file_name, 'r') as listFile:
            for line in listFile:
                dis,x,score = line.split(',')
                dis = dis.strip()
                score = float(score.strip())
                dis_files_data.append(dis)
                score_data.append(x)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        #score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    # min-max (range 0-1 MOS score)
    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])


    def __getitem__(self, idx):
            d_img_name = self.data_dict['d_img_list'][idx]

            splitted = d_img_name.split('_')

            if splitted[0].startswith("DPSRGAN") or splitted[0].startswith("DPSR"):
                row_number = splitted[2]
                subfolder = self.row2folder[row_number]
            else:
                subfolder = splitted[0]
            

            d_img_path = os.path.join(self.dis_path, subfolder, d_img_name)
            d_img = cv2.imread(d_img_path, cv2.IMREAD_COLOR)
            d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
            d_img = np.array(d_img).astype('float32') / 255.0 # 이미지 픽셀 값을 0-1 범위로 정규화
            # (H, W, C) → (C, H, W)
            d_img = np.transpose(d_img, (2, 0, 1))
            
            score = self.data_dict['score_list'][idx]
            sample = {
                'd_img_org': d_img,
                'score': score,
                'd_name': d_img_name,
                'subfolder':subfolder
            }
            return sample

def load_row2folder(mapping_file_path):
    """
    mapping_file_path: 폴더 이름들이 저장된 txt 파일의 경로
    반환: {"01": "Animal1", "02": "Animal2", ...} 형태의 딕셔너리
    """
    row2folder = {}
    with open(mapping_file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    # 1부터 시작하는 번호를 2자리 문자열로 만들기 (예: 1 -> "01")
    for idx, folder_name in enumerate(lines, start=1):
        key = f"{idx:02d}"
        row2folder[key] = folder_name
    return row2folder


