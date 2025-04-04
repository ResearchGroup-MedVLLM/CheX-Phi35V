import json
import os
import h5py
import torch
from tqdm import tqdm
from open_clip.tokenizer import HFTokenizer
from open_clip.model import CustomTextCLIP

cast_dtype = 'fp16'
config_path = 'open_clip_config.json'
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

pretrained_cfg = config['preprocess_cfg']
model_cfg = config['model_cfg']

model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
model.load_state_dict(torch.load('BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.pth'), strict=False)

tokenizer = HFTokenizer("./tokenizer")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

extract_type = 'question'
# extract_type = 'explanation'
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
sample_text = dataline[0][extract_type]
with torch.no_grad():
    sample_inputs = tokenizer(sample_text).to(device)
    sample_feature, _ = model.encode_text(sample_inputs)  # shape = (batch=1, dim)
    dim = sample_feature.shape[-1]
print(f"Detected feature dim: {dim}")

# 3. 在 HDF5 文件里创建一个 (N, dim) 的 dataset
with h5py.File(h5_filename, 'w') as datafile:
    dset = datafile.create_dataset(
        'features',
        shape=(N, dim),
        dtype='float32',
        chunks=(1, dim)  # 按行写入时更高效
    )

    # 4. 遍历每一个文本，逐条写入
    with torch.no_grad():
        i = 0
        for item in tqdm(dataline):
            question = item[extract_type]
            text_input = tokenizer(question).to(device)
            text_feature, _ = model.encode_text(text_input)
            # text_feature shape = (1, dim)
            # 写入到 dset 的第 i 行
            dset[i, :] = text_feature[0].detach().cpu().numpy()
            i += 1

print(f"Features saved to {h5_filename}")
