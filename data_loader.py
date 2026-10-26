import os
from os.path import splitext
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import tempfile
from pathlib import Path

# label对照表
label_MAPPING = {
    'anger': 0,
    'contentment': 1,
    'disgust': 2,
    'fear': 3,
    'guilt': 4,
    'neutral': 5,
    'sadness': 6,
    'surprise': 7,
}

# 特征长度
FEATURES_LENGTH = {
    'audeep': 4096,
    'deepspectrum': 1920,
    'opensmile': 6374,
    'xbow': 2000,
}


def lazy_read(filepath, filename, force_reload=False):
    cache = Path(tempfile.gettempdir()) / f'{filename}.npz'
    # print(cache)
    if cache.is_file() and not force_reload:
        data = np.load(cache)['data']
    else:
        data = np.load(filepath)['data']
        np.savez_compressed(cache, data=data)
    return data


def load_feature(feature_type, feature_path):
    """加载已有特征集"""
    sep = ';' if feature_type == 'opensmile' else ','  # 指定分隔符,
    feature_files = os.path.join(feature_path, feature_type, "features.csv")
    feat_df = pd.read_csv(feature_files, sep=sep, quotechar="'")  # 加载特征集
    return feat_df


class TrainLoader(Dataset):

    def __init__(self, feat_df, train_path, wav_path, feature_type, **kwargs):
        super(TrainLoader, self).__init__()
        self.data_list = []  # 用来记录原始特征
        self.data_list_feat = []  # 用来记录指定特征
        self.data_label = []  # 记录标签
        self.feature_type = feature_type
        # 加载label
        label_df = pd.read_csv(train_path, sep=',')

        # 加载指定特征
        if feature_type:
            name = 'index'
            self.train_df = feat_df[feat_df[name].str.contains('train')].reset_index(drop=True)  # 获取训练特征集
        # if feature_type:
        #     name = 'file_name'
        #     self.train_df = feat_df[feat_df[name].str.contains('train')].reset_index(drop=True)  # 获取训练特征集

        for index in range(len(label_df)):
            label = label_MAPPING[label_df.loc[index, 'label']]
            # 记录标签
            self.data_label.append(label)
            # 记录指定特征索引
            if feature_type:
                self.data_list_feat.append(
                    self.train_df[self.train_df[name] == label_df.loc[index, 'filename']].index.item())
            # 记录原始语音文件
            file_name = splitext(label_df.loc[index, 'filename'])[0]
            self.data_list.append(file_name)

    def __getitem__(self, index):
        # 指定特征
        if self.feature_type:
            feature = self.train_df.iloc[self.data_list_feat[index], 1:].astype(float).values
        else:
            feature = [0.0]
        # data = torch.FloatTensor(feature)
        layers = [16]
        # layers = [i for i in range(17,25)]

        all_data = []
        for layer in layers:
            filepath = os.path.join(f"E:/work/USERNAME/DEMOS/features/diff/it/wav2vec/finetune1/wav2vec_{layer}/",
                                    str(self.data_list[index]) + f"_{layer}.npz")
            temp_data = np.load(filepath)['data']
            all_data.append(temp_data)
            # filepath = os.path.join(f"E:/work/USERNAME/DEMOS/features/diff/it/wav2vec/finetune0/wav2vec_{layer}/",
            #                         str(self.data_list[index]) + f"_{layer}.npz")
            # temp_data = np.load(filepath)['data']
            # all_data.append(temp_data)
        data = torch.FloatTensor(all_data)
        return data, self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class Loader(Dataset):

    def __init__(self, feat_df, target_path, wav_path, feature_type, **kwargs):
        super(Loader, self).__init__()
        self.data_list = []  # 用来记录原始特征
        self.data_list_feat = []  # 用来记录指定特征
        self.data_label = []  # 记录标签
        self.feature_type = feature_type
        # 加载label
        label_df = pd.read_csv(target_path, sep=',')

        # 加载指定特征
        if feature_type:
            name = 'index'
            self.train_df = feat_df[feat_df[name].str.contains('train')].reset_index(drop=True)  # 获取训练特征集

        for index in range(len(label_df)):
            label = label_MAPPING[label_df.loc[index, 'label']]
            # 记录标签
            self.data_label.append(label)
            # 记录指定特征索引
            if feature_type:
                self.data_list_feat.append(
                    self.train_df[self.train_df[name] == label_df.loc[index, 'filename']].index.item())
            # 记录原始语音文件
            file_name = splitext(label_df.loc[index, 'filename'])[0]
            self.data_list.append(file_name)

    def __getitem__(self, index):
        # 指定特征
        if self.feature_type:
            feature = self.train_df.iloc[self.data_list_feat[index], 1:].astype(float).values
        else:
            feature = [0.0]
        # data = torch.FloatTensor(feature)
        layers = [16]
        # layers = [i for i in range(17,25)]

        all_data = []
        for layer in layers:
            filepath = os.path.join(f"D:/work/USERNAME/DEMOS/features/same/it/wav2vec/finetune1/wav2vec_{layer}/",
                                    str(self.data_list[index]) + f"_{layer}.npz")
            temp_data = np.load(filepath)['data']
            all_data.append(temp_data)
            # filepath = os.path.join(f"D:/work/USERNAME/DEMOS/features/same/it/wav2vec/finetune0/wav2vec_{layer}/",
            #                         str(self.data_list[index]) + f"_{layer}.npz")
            # temp_data = np.load(filepath)['data']
            # all_data.append(temp_data)
        data = torch.FloatTensor(all_data)
        return data

    def __len__(self):
        return len(self.data_list)


class DevelLoader(Dataset):

    def __init__(self, feat_df, devel_path, wav_path, feature_type, **kwargs):
        super(DevelLoader, self).__init__()
        self.data_list = []  # 用来记录原始文件
        self.data_list_feat = []  # 用来记录指定特征
        self.data_label = []  # 记录标签
        self.feature_type = feature_type
        # 加载label
        label_df = pd.read_csv(devel_path, sep=',')

        # 加载指定特征
        if feature_type:
            name = 'index'
            self.devel_df = feat_df[feat_df[name].str.contains('devel')].reset_index(drop=True)

        for index in range(len(label_df)):
            label = label_MAPPING[label_df.loc[index, 'label']]
            # 记录标签
            self.data_label.append(label)
            # 记录指定特征索引
            if feature_type:
                self.data_list_feat.append(
                    self.devel_df[self.devel_df[name] == label_df.loc[index, 'filename']].index.item())
            file_name = splitext(label_df.loc[index, 'filename'])[0]
            self.data_list.append(file_name)

    def __getitem__(self, index):
        # 指定特征
        if self.feature_type:
            feature = self.devel_df.iloc[self.data_list_feat[index], 1:].astype(float).values
        else:
            feature = [0.0]
        # data = torch.FloatTensor(feature)
        layers = [16]
        # layers = [i for i in range(17,25)]

        all_data = []
        for layer in layers:
            filepath = os.path.join(f"E:/work/USERNAME/DEMOS/features/diff/it/wav2vec/finetune1/wav2vec_{layer}/",
                                    str(self.data_list[index]) + f"_{layer}.npz")
            temp_data = np.load(filepath)['data']
            all_data.append(temp_data)
            # filepath = os.path.join(f"E:/work/USERNAME/DEMOS/features/diff/it/wav2vec/finetune0/wav2vec_{layer}/",
            #                         str(self.data_list[index]) + f"_{layer}.npz")
            # temp_data = np.load(filepath)['data']
            # all_data.append(temp_data)
        data = torch.FloatTensor(all_data)

        return data, self.data_label[index]

    def __len__(self):
        return len(self.data_list)
