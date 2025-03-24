import os
import torch
import numpy as np
import random
import json
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from config import Config
from utils.inference_process import ToTensor, Normalize, five_point_crop, sort_file,random_crop
from data.sr4kiqa import SR4KIQA
from tqdm import tqdm
from sklearn.model_selection import KFold  # 추가 for cross-validation
from scipy.stats import spearmanr, pearsonr, kendalltau
import torch.nn  # nn.DataParallel 사용 위해
from utils.process import RandCrop, ToTensor, RandHorizontalFlip, Normalize, five_point_crop
from models.maniqa import MANIQA
#from multi_aug import MultiGeometricAug #### implement augmentation
import torch.nn as nn

def remove_module_prefix(state_dict, prefix="module.module."):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict




# TransformWrapper 클래스는 변함 없이 사용
class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        """
        dataset: 원래의 데이터셋 (예: SR4KIQA)
        transform: sample에 적용할 transform (예: Normalize, ToTensor 등)
        """
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform:
            transformed_sample = self.transform(sample)  # sample 전체를 전달
            # transform 과정에서 일부 추가 정보가 누락될 수 있으므로 원래 sample의 d_name, score, subfolder를 보존
            for key in ["d_name", "score", "subfolder"]:
                if key not in transformed_sample:
                    transformed_sample[key] = sample[key]
            sample = transformed_sample
        return sample

    def __len__(self):
        return len(self.dataset)



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_epoch(config,epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # 에포크 동안 예측값과 라벨 저장 (평가용)
    pred_epoch = []
    labels_epoch = []
    
    for data in tqdm(train_loader):
        x_d = data['d_img_org'].cuda()  # 원본 이미지 (배치 전체)
        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
        
        # 하나의 이미지에 대해 config.num_avg_val 만큼 랜덤 크롭을 추출하여 각 crop마다 업데이트
        for i in range(config.num_avg_val):
            x_d_crop = random_crop(x_d, config)  # 랜덤 크롭 추출
            pred = net(x_d_crop)  # crop에 대한 예측 (B, output)
            
            optimizer.zero_grad()
            loss = criterion(torch.squeeze(pred), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            
            # 각 crop의 예측값과 라벨을 저장 (평가 시 여러 crop의 결과를 확인할 수 있음)
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    # 에포크별 상관계수 계산
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    ret_loss = np.mean(losses)
    #logging.info('train epoch:{} / loss:{:.4f} / SRCC:{:.4f} / PLCC:{:.4f}'.format(epoch + 1, ret_loss, rho_s, rho_p))
    print('Train Epoch: {} / Loss: {:.4f} / SRCC: {:.4f} / PLCC: {:.4f}'.format(epoch + 1, ret_loss, rho_s, rho_p))
    return ret_loss, rho_s, rho_p
    
def eval_epoch(config, iteration, fold, epoch, net, criterion, test_loader, log_f):
    with torch.no_grad():
        losses = []
        net.eval()  # 평가 모드 전환
        pred_epoch = []
        labels_epoch = []
        file_name_list = []  # 각 샘플의 파일 이름 저장

        for data in tqdm(test_loader):
            pred = 0
            d_names = data.get('d_name', None)
            for i in range(config.num_avg_val):
                x_d = data['d_img_org'].cuda()
                labels = data['score']
                labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                x_d = random_crop(x_d, config)
                pred += net(x_d)
            pred /= config.num_avg_val
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

            if d_names is not None:
                file_name_list.extend(d_names)
        
        # iteration, fold, epoch 정보를 포함한 예측 결과 저장 파일 생성
        pred_log_file = os.path.join(config.log_path, f"iter_{iteration}_fold_{fold}_predictions_epoch_{epoch+1}.txt")
        with open(pred_log_file, "w") as fout:
            for fname, pred_val, true_val in zip(file_name_list, pred_epoch, labels_epoch):
                fout.write(f"{fname}, {pred_val}, {true_val}\n")
        
        log_f.write(f"Iteration {iteration}, Fold {fold}, Epoch {epoch+1}: Predictions saved to {pred_log_file}\n")

        # 상관계수 계산
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_k, _ = kendalltau(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        
        print(f"Iteration {iteration}, Fold {fold}, Epoch {epoch+1} ===== Loss: {np.mean(losses):.4f} ===== SRCC: {rho_s:.4f} ===== PLCC: {rho_p:.4f}")
        return np.mean(losses), rho_s, rho_p, rho_k

# def eval_epoch(config, epoch, net, criterion, test_loader, log_f):
#     with torch.no_grad():
#         losses = []
#         net.eval()  # 평가 모드 전환
#         pred_epoch = []
#         labels_epoch = []
#         file_name_list = []  # 각 샘플의 파일 이름을 저장

#         for data in tqdm(test_loader):
#             pred = 0
#             # 배치 내에서 파일 이름 가져오기 (d_name 키가 존재한다면)
#             d_names = data.get('d_name', None)
#             for i in range(config.num_avg_val):
#                 x_d = data['d_img_org'].cuda()
#                 labels = data['score']
#                 labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
#                 x_d = random_crop(x_d,config)
#                 pred += net(x_d)
#             pred /= config.num_avg_val
#             loss = criterion(torch.squeeze(pred), labels)
#             losses.append(loss.item())

#             pred_batch_numpy = pred.data.cpu().numpy()
#             labels_batch_numpy = labels.data.cpu().numpy()
#             pred_epoch = np.append(pred_epoch, pred_batch_numpy)
#             labels_epoch = np.append(labels_epoch, labels_batch_numpy)

#             if d_names is not None:
#                 file_name_list.extend(d_names)
        
#         # 예측 결과를 파일에 저장
#         pred_log_file = os.path.join(config.log_path, f"predictions_epoch_{epoch+1}.txt")
#         with open(pred_log_file, "w") as fout:
#             for fname, pred_val in zip(file_name_list, pred_epoch):
#                 fout.write(f"{fname}, {pred_val}\n")
        
#         # log_f에도 기록
#         log_f.write(f"Epoch {epoch+1}: Predictions saved to {pred_log_file}\n")

#         # 상관계수 계산
#         rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
#         rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
#         rho_k, _ = kendalltau(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        
#         print(f"Epoch:{epoch+1} ===== loss:{np.mean(losses):.4f} ===== SRCC:{rho_s:.4f} ===== PLCC:{rho_p:.4f}")
#         return np.mean(losses), rho_s, rho_p, rho_k


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config 설정
    config = Config({
        # dataset 관련
        "db_name": "SR4KIQA",
        "dis_path": "/home/mb21100/data/SR4KIQA",  # 이 폴더 아래에 24개의 subfolder와 MOS.csv가 있음.
        "txt_folder_name": "./data/sr4kiqa.txt",
        "txt_file_name": "/home/mb21100/data/SR4KIQA/MOS.csv",

        # optimization
        "batch_size": 4, # for gpu // how many images data loader will bring at one time
        "learning_rate": 1e-6, #2e-5
        "num_avg_val": 45, #25
        "crop_size": 224,
        "n_iteration": 1,
        "weight_decay":  1e-4, #1e-4
        "val_freq": 1,
        "T_max": 50, #100
        "eta_min": 1e-7,
        "num_workers": 8,

        # model 관련
        "patch_size": 8, # in one cropped image(224x224), make the patch.
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.1,
        "n_epoch_fold": 25, #5

        # load & save checkpoint, 로그, 텐서보드 등
        #"model_path": "./epoch3",  # in 5 epochs, epoch3 has the best performance on plcc and srcc.
        "model_name": "model_maniqa_sr4kiqa2",
        "output_path": "./output/sr4kiqa2/",
        "snap_path": "./output/models/sr4kiqa2",       # checkpoint 저장 폴더
        "log_path": "./output/log/sr4kiqa2/",
        "log_file": "sr4kiqa_log.txt",
        "tensorboard_path": "./output/tensorboard/sr4kiqa2"
        
    })
    os.makedirs(config["snap_path"], exist_ok=True)
    os.makedirs(config["log_path"], exist_ok=True)
    os.makedirs(config["tensorboard_path"], exist_ok=True)
    os.makedirs(config["output_path"], exist_ok=True)

    # 로그 파일 오픈 (append 모드)
    log_file_path = os.path.join(config["log_path"], config["log_file"])
    
    log_f = open(log_file_path, "a")

    all_results = {}
    
    # data load (SR4KIQA 데이터셋 로드; SR4KIQA 클래스가 __getitem__에서 subfolder 정보를 포함한다고 가정)
    sr4kiqa_dataset = SR4KIQA(
        dis_path=config.dis_path,
        txt_folder_name=config.txt_folder_name,
        txt_file_name=config.txt_file_name,
    )

    # unique_subfolders 파일 읽기 (24개의 subfolder 이름)
    with open(config.txt_folder_name, "r") as f:
        unique_subfolders = [line.strip() for line in f if line.strip()]
    print(unique_subfolders)

    # 각 폴더에 해당하는 샘플의 인덱스 매핑 생성
    folder_to_indices = {folder: [] for folder in unique_subfolders}
    for idx in range(len(sr4kiqa_dataset)):
        sample = sr4kiqa_dataset[idx]
        folder = sample['subfolder']
        folder_to_indices[folder].append(idx)

    ##############################################################################################################
    # 5번 iteration에 대해 5-fold cross validation 진행
    for iteration in range(config.n_iteration):
        print(f"Iteration: {iteration}")
        iteration_results = {}

        # KFold로 폴더 단위로 분할 (unique_subfolders 기준)
        kf = KFold(n_splits=5, shuffle=True, random_state=20)
        for fold, (train_folder_idx, valid_folder_idx) in enumerate(kf.split(unique_subfolders)):
            fold_info = {}  # fold별 정보 저장 딕셔너리

            train_folders = [unique_subfolders[i] for i in train_folder_idx]
            valid_folders = [unique_subfolders[i] for i in valid_folder_idx]
            fold_info["train_folders"] = train_folders
            fold_info["valid_folders"] = valid_folders

            print(f"Fold {fold} - Train Folders: {train_folders}, Valid Folders: {valid_folders}")
            log_f.write(f"Fold {fold} - Train Folders: {train_folders}, Valid Folders: {valid_folders}\n")


            # 모델 재초기화 (pre-trained model 로드)
            #net = torch.load(config.model_path)
            net = MANIQA(

                num_outputs=1,

                img_size=224,

                drop=0.1,

                hidden_dim=512,

                fusion_type='concat'

            )
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # net = net.to(device)

            state_dict = torch.load("./output/models/model_maniqa_pipal_ver1/model_maniqa_pipal_epoch43.pth", map_location=torch.device("cuda"))
            state_dict = remove_module_prefix(state_dict, prefix="module.")
            net.load_state_dict(state_dict)
            net = net.cuda()
            # # DataParallel 여부 확인 후, 실제 모델에 접근
            # if isinstance(net, torch.nn.DataParallel):
            #     base_model = net.module  # net.module이 실제 모델(MANIQA)
            # else:
            #     base_model = net

            # # vit 객체가 있는지 확인 (MANIQA에서는 base_model.vit로 접근)
            # if hasattr(base_model, 'vit'):
            #     vit_model = base_model.vit

            #     # 1. Patch Embedding 레이어 동결
            #     if hasattr(vit_model, 'patch_embed'):
            #         for param in vit_model.patch_embed.parameters():
            #             param.requires_grad = False
            #         print("Patch Embedding 레이어 동결 완료.")

            #     # 2. 첫 num_freeze_blocks개의 Transformer 블록 동결
            #     num_freeze_blocks = 4  # 필요에 따라 조절
            #     if hasattr(vit_model, 'blocks'):
            #         for block in vit_model.blocks[:num_freeze_blocks]:
            #             for param in block.parameters():
            #                 param.requires_grad = False
            #         print(f"ViT의 처음 {num_freeze_blocks}개 Block 동결 완료.")
            #     else:
            #         print("vit.blocks를 찾지 못했습니다.")

            # else:
            #     print("MANIQA 모델에서 vit를 찾지 못했습니다. 동결할 레이어가 없습니다.")

            # 이후 옵티마이저 생성
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, net.parameters()),  # requires_grad=True인 파라미터만 업데이트
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            criterion = torch.nn.MSELoss()

            # 스케줄러 생성
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)
            # 해당 폴더에 속하는 샘플 인덱스 모으기
            train_indices = []
            valid_indices = []
            for folder in train_folders:
                train_indices.extend(folder_to_indices[folder])
            for folder in valid_folders:
                valid_indices.extend(folder_to_indices[folder])

            # Subset을 이용해 DataLoader 생성
            train_subset = Subset(sr4kiqa_dataset, train_indices)
            valid_subset = Subset(sr4kiqa_dataset, valid_indices)

            train_dataset_with_transform = TransformWrapper(
                train_subset,
                transform=transforms.Compose([
                    #MultiGeometricAug(num_aug=5, crop_size=config.crop_size), ## implement it
                    Normalize(0.5, 0.5),
                    RandHorizontalFlip(),
                    ToTensor()
                ])
            )
            valid_dataset_with_transform = TransformWrapper(
                valid_subset,
                transform=transforms.Compose([
                    Normalize(0.5, 0.5),
                    ToTensor()
                ])
            )

            train_loader = DataLoader(
                dataset=train_dataset_with_transform,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                drop_last=True,
                shuffle=True
            )
            eval_loader = DataLoader(
                dataset=valid_dataset_with_transform,
                batch_size=config.batch_size,
                num_workers=1,
                drop_last=True,
                shuffle=False
            )

            fold_results = []  # 각 epoch 결과를 저장할 리스트

            # 학습/검증 루프: 각 fold에서 config.n_epoch_fold 만큼 반복
            pre_plcc =0
            pre_srcc=0
            for epoch in range(config.n_epoch_fold):
                # (a) 훈련
                train_loss = train_epoch(config,epoch, net, criterion, optimizer, scheduler, train_loader)
                #print("finish training")
                # (b) 검증
                val_loss, srcc, plcc, krcc = eval_epoch(config, iteration, fold, epoch, net, criterion, eval_loader, log_f)
                #print("finish evaluating")

                # log_msg = f"Iteration {iteration}, Fold {fold}, Epoch {epoch+1}/{config['n_epoch_fold']} - " \
                #           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, " \
                #           f"SRCC: {srcc:.4f}, PLCC: {plcc:.4f}, KRCC: {krcc:.4f}\n"

                #print(log_msg)
                #log_f.write(log_msg)

                epoch_result = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "srcc": srcc,
                    "plcc": plcc,
                    "krcc": krcc
                }
                fold_results.append(epoch_result)

                # 체크포인트 저장 (매 epoch마다)
                checkpoint_path = os.path.join(
                    config["snap_path"],
                    f"{config['model_name']}_iter{iteration}_fold{fold}_epoch{epoch+1}.pth"
                )
                torch.save(net.state_dict(), checkpoint_path)

                if (plcc > pre_plcc or srcc > pre_srcc):
                    pre_plcc = plcc
                    state_dict = torch.load(checkpoint_path, map_location=torch.device('cuda'))
                    net.load_state_dict(state_dict)
                    best_checkpoint_path = os.path.join(
                                config["snap_path"],
                                f"{config['model_name']}_iter{iteration}_fold{fold}_best.pth"
                            )

            # fold 정보에 epoch 결과 추가
            fold_info["results"] = fold_results
            iteration_results[f"fold_{fold}"] = fold_info

        all_results[f"iteration_{iteration}"] = iteration_results

        # iteration 결과 저장 (JSON 파일)
        if not os.path.exists(config["output_path"]):
            os.makedirs(config["output_path"], exist_ok=True)
        iteration_file = os.path.join(config["output_path"], f"iteration_{iteration}_results.json")
        with open(iteration_file, "w") as f:
            json.dump(iteration_results, f, indent=4)

        print(f"[Iteration {iteration}] Cross Validation Done.\n")

    # 모든 결과 저장 (전체 결과 JSON)

    all_results_file = os.path.join(config["output_path"], "all_results.json")
    with open(all_results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    log_f.close()
