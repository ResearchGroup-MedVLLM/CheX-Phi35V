import os
import json
import torch
import re
import gc

from torch.utils.data import Dataset
from datasets import Dataset, Image, Sequence, IterableDataset
from PIL import Image as PILImage
import os
import json
import random


def load_mimicext_dpo_dataset(annotation_file="", vis_root="", transform=None):
    """
    Load a dataset from an annotation file and associated image root, and return a `datasets.Dataset`.

    Parameters:
        annotation_file (str): Path to the annotation file containing image IDs and captions.
        vis_root (str): Root directory where images are stored.
        transform (callable, optional): Optional transform to be applied on a PIL image.

    Returns:
        datasets.Dataset: A Dataset object compatible with Hugging Face's Datasets library.
    """
    # Load annotations
    with open(annotation_file, 'r') as file:
        annotations = json.load(file)
        print(len(annotations))
        
    print('Loading Annotations...')

    # Prepare data storage for Dataset.from_dict
    data_dict = {
        # "image_paths": [],
        "images": [],
        "question": [],
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "image_ids": []
    }

    for ann in annotations:
        question = ann["question"]
        chosen = ann["chosen"]
        rejected = ann["rejected"]

        task_name = ann["task_name"]
        if '_region_' in task_name:
            image_paths = [os.path.join(vis_root.replace('MIMIC_CXR_JPG', 'MIMIC_CXR_JPG_region'),
                                        f'{img_id}.png') for img_id in ann["image_ids"]]
        else:
            image_paths = [os.path.join(vis_root, f'{img_id}.png') for img_id in ann["image_ids"]]
        prompt = '\n'.join([f"<|image_{iid}|>" for iid in range(1, len(image_paths) + 1)])
        prompt += question

        # if transform:
        #     images = [transform(image) for image in images]

        # Add to data dictionary
        # data_dict["image_paths"].append(image_paths)
        data_dict["images"].append(image_paths)
        data_dict["question"].append(question)
        data_dict["prompt"].append(prompt)
        data_dict["chosen"].append(chosen)
        data_dict["rejected"].append(rejected)
        data_dict["image_ids"].append(ann["image_ids"])
    # print('Convert to a Hugging Face Dataset ...')

    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.cast_column("images", Sequence(Image()))
    return dataset


