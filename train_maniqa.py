import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from models.maniqa import MANIQA
from config import Config
from utils.process import RandCrop, ToTensor, RandHorizontalFlip, Normalize, five_point_crop
from utils.inference_process import random_crop
from scipy.stats import spearmanr, pearsonr
from data.pipal21 import PIPAL21
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm


def remove_module_prefix(state_dict, prefix="module.module."):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict





# 랜덤 시드 고정정
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 로그 폴더/파일 생성 및설정정
def set_logging(config):
    if not os.path.exists(config.log_path): 
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train() # 모델을 학습 모드로 전환환
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    
    for data in tqdm(train_loader):
        x_d = data['d_img_org'].cuda() #입력 이미지를 GPU 로 전송송
        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()  # 정답(score)을 GPU로 전송
    
        pred_d = net(x_d) # 모델 추론

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels) # 손실 계산
        losses.append(loss.item())

        loss.backward() # 역전파
        optimizer.step() # 가중치 업데이트
        scheduler.step() # learning rate 조정

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    print('Train Epoch: {} / Loss: {:.4f} / SRCC: {:.4f} / PLCC: {:.4f}'.format(epoch + 1, ret_loss, rho_s, rho_p))
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p


def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval() # 모델을 평가 모드로 전환
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        # 여러 개의 랜덤 크롭/전처리를 통해 평균 예측값 산출
        for data in tqdm(test_loader):
            pred = 0
            for i in range(config.num_avg_val):
                x_d = data['d_img_org'].cuda()
                labels = data['score']
                labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                x_d = random_crop(x_d, config)
                pred += net(x_d)

            pred /= config.num_avg_val # 평균내기
            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient

        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
        return np.mean(losses), rho_s, rho_p

 # 메인 실행 부분
if __name__ == '__main__':
    # cpu 관련 thread 수 설정
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    model = MANIQA(embed_dim=72, num_outputs=1, drop=0.1, img_size=224, num_tab=2, scale=0.8)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected shape: (2, 1)




    setup_seed(20)

    # Config 객체를 사용해 설정값 로드 (데이터 경로, 학습 파라미터, 모델 하이퍼파라미터 등)
    # config file
    config = Config({
        # dataset path
        "db_name": "PIPAL",
        "train_dis_path": "/home/mb21100/data/PIPAL/Train_Distort/",
        "val_dis_path": "/home/mb21100/data/PIPAL/Val_Distort/",
        "train_txt_file_name": "./data/pipal21_train.txt",
        "val_txt_file_name": "./data/pipal21_val.txt",

        # optimization
        "batch_size": 8,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 100,
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_avg_val": 1,
        "crop_size": 224,
        "num_workers": 8,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.13,
        
        # load & save checkpoint
        "model_name": "model_maniqa_pipal",
        "output_path": "./output2",
        "snap_path": "./output2/models/",               # directory for saving checkpoint
        "log_path": "./output2/log/maniqa/",
        "log_file": ".txt",
        "tensorboard_path": "./output2/tensorboard/"
    })

    # 경로가 없으면 새로 생성
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)

    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)
    
    if not os.path.exists(config.tensorboard_path):
        os.mkdir(config.tensorboard_path)

    config.snap_path += config.model_name
    config.log_file = config.model_name + config.log_file
    config.tensorboard_path += config.model_name

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)

    # 데이터셋 로드: PIPAL21을 사용해 학습 및 검증 데이터셋 구성
    # data load
    train_dataset = PIPAL21(
        dis_path=config.train_dis_path,
        txt_file_name=config.train_txt_file_name,
        transform=transforms.Compose(
            [
                RandCrop(config.crop_size),
                Normalize(0.5, 0.5),
                RandHorizontalFlip(),
                ToTensor()
            ]
        ),
    )
    val_dataset = PIPAL21(
        dis_path=config.val_dis_path,
        txt_file_name=config.val_txt_file_name,
        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
    )

    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )
    # 모델 재초기화 (pre-trained model 로드)
    #net = torch.load(config.model_path)
    net = MANIQA(

        num_outputs=1,

        img_size=224,

        drop=0.3,

        hidden_dim=512,

        fusion_type='concat'

    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net = nn.DataParallel(net)

    # loss function
    # 손실 함수, 옵티마이저, 스케줄러 설정
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    # make directory for saving weights
    # 체크포인트 저장 디렉토리 생성
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    # train & validation
    # Training & validation 루프
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0

    # 300 번 epoch 반복
    for epoch in range(0, config.n_epoch):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p = train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader)

        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)

        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))
            loss, rho_s, rho_p = eval_epoch(config, epoch, net, criterion, val_loader)
            logging.info('Eval done...')

        # 체크포인트 저장 (매 epoch마다)
        checkpoint_path = os.path.join(
            config["snap_path"],
            f"{config['model_name']}_epoch{epoch+1}.pth"
        )
        torch.save(net.state_dict(), checkpoint_path)


        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))