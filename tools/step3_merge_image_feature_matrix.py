import json
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm


extract_type = 'image'
data_type = 'retrieval_100'
# data_type = 'train_random_20k'

dataset_path = './data/features'
h5_filename = os.path.join(dataset_path, f'{extract_type}_features_{data_type}.h5')
import glob
anno_root = './data/analyze'
split_path = os.path.join(anno_root, f'{data_type}.json')

with open(split_path, 'r', encoding='utf-8') as f:
    annotations = json.load(f)


dataline = annotations
# 1. 获取样本总数 N
N = len(annotations)
print(f"Total lines: {N}")

# 如果数据为空，直接退出
if N == 0:
    print(f"No data found in {split_path}, exit.")
    exit(0)

# 2. 先用一个样本探测隐藏维度 dim
dim = 512

image_feature_root = './data/temp/image_features'
# 3. 在 HDF5 文件里创建一个 (N, dim) 的 dataset
with h5py.File(h5_filename, 'w') as datafile:
    dset = datafile.create_dataset(
        'features',
        shape=(N, dim),
        dtype='float32',
        chunks=(1, dim)  # 按行写入时更高效
    )

    i = 0
    for item in tqdm(dataline):
        img_fea_paths = []
        for image_id in item['image_ids']:
            if '_' in image_id:
                image_id = image_id.split('_')[0]
            img_fea_paths.append(os.path.join(image_feature_root, image_id + '.h5'))

        img_features = []
        for img_fea_path in img_fea_paths:
            with h5py.File(img_fea_path, 'r') as f:
                feature = np.array(f['feature'])
                img_features.append(feature)
        if len(img_features) == 1:
            img_feature = img_features[0]  # 直接使用单个特征
        elif len(img_features) > 1:
            img_feature = np.mean(img_features, axis=0, keepdims=True)  # 计算均值，并保持 (1, 512) 形状
        else:
            raise ValueError(f"No valid features found for item {item}")

        dset[i, :] = img_feature
        i += 1

print(f"Features saved to {h5_filename}")
