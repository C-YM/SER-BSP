import os
from os import makedirs
from os.path import splitext

import numpy as np
import pandas as pd
import torch
from models.CNN_ATT_LSTM_SINGLE import CNN_ATT_LSTM_SINGLE


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = CNN_ATT_LSTM_SINGLE().to(device)
pretrained_dict = torch.load('model_best_0.000010_1.pkl')
model_dict = model.state_dict()
# 关键在于下面这句，从model_dict中读取key、value时，用if筛选掉不需要的网络层
pretrained_dict = {key: value for key, value in pretrained_dict.items() if
                   (key in model_dict and 'Prediction' not in key)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

OUTPUT_DIR = "E:/work/USERNAME/DEMOS/features/diff/it/wav2vec/cnn_ex"
makedirs(OUTPUT_DIR, exist_ok=True)
devel_path = 'E:/work/USERNAME/DEMOS/lab/diff/train.csv'
label_df = pd.read_csv(devel_path, sep=',')
with torch.no_grad():
    for index in range(len(label_df)):
        layers = [i for i in range(1, 25)]
        all_data = []
        for layer in layers:
            filepath = os.path.join(f"E:/work/USERNAME/DEMOS/features/diff/it/wav2vec/wav2vec_{layer}/",
                                    str(splitext(label_df.loc[index, 'filename'])[0]) + f"_{layer}.npz")
            temp_data = np.load(filepath)['data']
            all_data.append(temp_data)
        data = torch.FloatTensor(all_data)
        data = data.unsqueeze(0).to(device)
        _, feature = model(data)
        embedding = feature.cpu().squeeze(0).numpy()
        np.savez_compressed(f"{OUTPUT_DIR}/" + splitext(label_df.loc[index, 'filename'])[0], data=embedding)
