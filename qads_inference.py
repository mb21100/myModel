import os
import torch
import numpy as np
import pandas as pd
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from utils.inference_process import ToTensor, Normalize, five_point_crop, sort_file,random_crop
from data.qads import QADS
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, kendalltau
from models.maniqa import MANIQA


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def remove_module_prefix(state_dict, prefix="module.module."):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eval_epoch(config, net, test_loader):
    with torch.no_grad():
        net.eval()
        name_list = []
        pred_list = []
        
        with open(config.valid_path + '/output.txt', 'w') as f:
            for data in tqdm(test_loader):
                pred = 0
                for i in range(config.num_avg_val):
                    x_d = data['d_img_org'].cuda()
                    x_d = random_crop(x_d, config)
                    pred += net(x_d)

                pred /= config.num_avg_val
                d_name = data['d_name']
                pred = pred.cpu().numpy()
                name_list.extend(d_name)
                pred_list.extend(pred)
            for i in range(len(name_list)):
                f.write(name_list[i] + ',' + str(pred_list[i]) + '\n')
            print(len(name_list))
        f.close()


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path
        "db_name": "QADS",  # 이번엔 PIPAL22 데이터셋
        "data_path": "/home/mb21100/data/QADS/super-resolved_images",  # 데이터 경로 (예시)
        "txt_file_name": "/home/mb21100/data/QADS/mos_with_names2.txt",
        
        
        # optimization
        "batch_size": 10,
        "num_avg_val": 5,
        "crop_size": 224, #512

        # device
        "num_workers": 8,

        # load & save checkpoint
        "valid": "./output/eval",
        "valid_path": "./output/valid/qads_inference_eval",
        #"model_path": "./output/models/model_maniqa_pipal/epoch3" # epoch 에서 가장 좋은 성능을 만든 epoch 번호로 수정 ##### 여기 수정하고 돌리기기
        #"model_path": "./output/models/model_maniqa_pipal/model_maniqa_pipal_epoch43.pth"
        # "model_path": "./output2/models/model_maniqa_pipal/model_maniqa_pipal_epoch35.pth" # hidden_dim = 768 / SRCC: 0.8799 PLCC: 0.8753
        #"model_path": "./output4/models/model_maniqa_pipal/model_maniqa_pipal_epoch45.pth" #SRCC: 0.8633 PLCC: 0.8556
        #"model_path": "./output5/models/model_maniqa_pipal/model_maniqa_pipal_epoch13.pth"  # SRCC: 0.8669 PLCC: 0.8607
        #"model_path": "./output6/models/model_maniqa_pipal2/model_maniqa_pipal_epoch45.pth" #SRCC: 0.8642 PLCC: 0.8618
        #"model_path": "./output7/models/model_maniqa_pipal/model_maniqa_pipal_epoch40.pth" # SRCC: 0.8336 PLCC: 0.8292
        #"model_path": "./output7/models/model_maniqa_pipal0/model_maniqa_pipal_epoch5.pth" #SRCC: 0.8693 PLCC: 0.8657
        #"model_path": "./output7/models/model_maniqa_pipal1/model_maniqa_pipal_epoch5.pth" #SRCC: 0.8749 PLCC: 0.8693
        #"model_path": "./output7/models/model_maniqa_pipal/model_maniqa_pipal_epoch5.pth" #SRCC: 0.8736 PLCC: 0.8685
        "model_path": "./output8/models/model_maniqa_pipal/model_maniqa_pipal_epoch3.pth" # SRCC: 0.8682 PLCC: 0.8627
        # 이 모델은 PIPAL21 로 훈련한 모델에다가, SR4KIQA 로 fine-tuning 한 후에 사용.
    })

    if not os.path.exists(config.valid):
        os.makedirs(config.valid)

    if not os.path.exists(config.valid_path):
        os.makedirs(config.valid_path)

    
    # data load
    test_dataset = QADS(
        dis_path=config.data_path,
        txt_file_name = config.txt_file_name, # 나중에 srcc,krcc,plcc 구할 때 사용하자자
        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )

    #MANIQA 모델 생성 및 GPU에 올리기, DataParallel로 멀티-GPU 지원
    net = MANIQA(

        num_outputs=1,

        img_size=224,

        drop=0.3,

        hidden_dim=768,


    )

    state_dict = torch.load(config.model_path, map_location=torch.device("cuda"))
    state_dict = remove_module_prefix(state_dict, prefix="module.")
    net.load_state_dict(state_dict)
    net = net.cuda()

    # net = torch.load(config.model_path)
    # net = net.cuda()

    losses, scores = [], []
    eval_epoch(config, net, test_loader)
    sort_file(config.valid_path + '/output.txt')

    df_gt = pd.read_csv("/home/mb21100/data/QADS/mos_with_names2.txt",
                    names=['filename','gt'],
                    sep=',', header=None)
    # 혹시 파일명 끝에 공백이 있을 수 있으니 .strip()
    df_gt['filename'] = df_gt['filename'].str.strip()

    # 2) 모델 예측값 불러오기
    df_pred = pd.read_csv("./output/valid/qads_inference_eval/output.txt",
                        names=['filename','pred'],
                        sep=',', header=None)
    df_pred['filename'] = df_pred['filename'].str.strip()

    # 3) filename을 기준으로 merge (inner join)
    df_merged = pd.merge(df_gt, df_pred, on='filename', how='inner')

    # 4) SRCC, PLCC, KRCC 계산
    srcc, _ = spearmanr(df_merged['pred'], df_merged['gt'])
    plcc, _ = pearsonr(df_merged['pred'], df_merged['gt'])
    krcc, _ = kendalltau(df_merged['pred'], df_merged['gt'])

    print(f"SRCC: {srcc:.4f}")
    print(f"PLCC: {plcc:.4f}")
    print(f"KRCC: {krcc:.4f}")
        
        