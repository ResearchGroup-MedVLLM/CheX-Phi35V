import json
import glob
import os.path
import h5py
import torch
from PIL import Image
from tqdm import tqdm
from open_clip.tokenizer import HFTokenizer
from open_clip.model import CustomTextCLIP
from open_clip.transform import image_transform

cast_dtype = 'fp16'
config_path = 'open_clip_config.json'
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

pretrained_cfg = config['preprocess_cfg']
model_cfg = config['model_cfg']

model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)

model.load_state_dict(torch.load('BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.pth'))
tokenizer = HFTokenizer("./tokenizer")


preprocess_val = image_transform(
    model.visual.image_size,
    is_train=False,
    mean=getattr(model.visual, 'image_mean', None),
    std=getattr(model.visual, 'image_std', None),
)

dataset_path = './data/images/MIMIC_CXR_JPG/'
feature_path = './data/temp/image_features/'
test_imgs = glob.glob(os.path.join(dataset_path,  '*.*'))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()


for img_path in tqdm(test_imgs):
    img = preprocess_val(Image.open(img_path))
    image = img.unsqueeze(dim=0).to(device)
    image_features, image_features_all = model.encode_image(image)

    image_features = image_features.detach().cpu().numpy()
    image_features_all = image_features_all.detach().cpu().numpy()

    filename = img_path.replace(dataset_path, feature_path)
    path, name = os.path.split(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    fea_name = name.split('.')[0]+'.h5'
    fea_path = os.path.join(path, fea_name)

    with h5py.File(fea_path, 'w') as f:
        f.create_dataset('feature', data=image_features)
        # f.create_dataset('feature_noproj', data=image_features_all)

    print(fea_path, image_features.shape, image_features_all.shape)
