import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import torch.nn as nn

def add_gaussian_noise(data, mean=0.0, std=0.1):
    """
    EEG 데이터에 가우시안 노이즈 추가
    :param data: EEG 데이터. 크기는 (배치 크기, 채널 수, EEG 채널 수, 시간 스텝 수)
    :param mean: 가우시안 노이즈의 평균
    :param std: 가우시안 노이즈의 표준편차
    :return: 노이즈가 추가된 EEG 데이터
    """
    noise = torch.randn(data.size()) * std + mean
    return data + noise


def random_masking(X, mask_size):
    """
    X의 너비(시간) 차원에서 랜덤하게 마스킹합니다.
    mask_size: 마스킹할 너비(시간)의 크기
    """
    _, _, _, width = X.shape
    mask = torch.ones_like(X)

    for _ in range(mask_size):
        batch_indices = torch.randint(0, X.shape[0], (1,))
        channel_indices = torch.randint(0, X.shape[1], (1,))
        height_indices = torch.randint(0, X.shape[2], (1,))
        time_indices = torch.randint(0, width, (1,))
        mask[batch_indices, channel_indices, height_indices, time_indices] = 0
    return X * mask  # 원본 데이터에 마스크 적용


class CustomDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.load_data()
        self.torch_form()

        if self.args.masking:
            self.masking = random_masking(self.X, mask_size = 100)
            self.X = torch.cat([self.X, self.masking])
            self.y = torch.cat([self.y, self.y])


        if self.args.gaussian:
            self.gaussian_noise1 = add_gaussian_noise(self.X,mean=0, std=0.10)
            self.gaussian_noise2 = add_gaussian_noise(self.X, mean=0, std=0.05)


        if self.args.masking:
            self.X = torch.cat([self.X, self.masking])
            self.Y = torch.cat([self.y, self.y])

        if self.args.gaussian:
            self.X = torch.cat([self.X, self.gaussian_noise1, self.gaussian_noise2], dim=0)
            try:
                self.y = torch.cat([self.Y, self.y, self.y], dim=0)
            except AttributeError:
                self.y = torch.cat([self.y, self.y, self.y], dim=0)


        # 데이터의 길이만큼 인덱스를 생성하고 섞음
        indices = torch.randperm(self.X.size(0))

        # 생성한 인덱스를 사용해 self.X와 self.y의 순서를 재배열하여 쌍을 유지
        self.X = self.X[indices]
        self.y = self.y[indices]

    def load_data(self):
        s = self.args.train_subject[0]
        if self.args.phase == 'train':
            self.X = np.load(f"./data/S{s:02}_train_X.npy")
            self.y = np.load(f"./data/S{s:02}_train_y.npy")
        else:
            self.X = np.load(f"./data/S{s:02}_test_X.npy")
            self.y = np.load(f"./answer/S{s:02}_y_test.npy")
        if len(self.X.shape) <= 3:
            self.X = np.expand_dims(self.X, axis=1)

    def torch_form(self):
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]
        sample = [X,y]
        return sample


def data_loader(args):
    print("[Load data]")
    # Load train data
    args.phase = "train"

    trainset = CustomDataset(args)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Load val data
    args.phase = "val"
    valset = CustomDataset(args)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Print
    print(f"train_set size: {train_loader.dataset.X.shape}")
    print(f"val_set size: {val_loader.dataset.X.shape}")
    print("")
    return train_loader, val_loader
