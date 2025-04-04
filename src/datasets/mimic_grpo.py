import os
import json
import torch
import re

from trl.trainer.dpo_trainer import DataCollatorForPreference
# from trl import PreferenceCollator
from torch.utils.data import Dataset


from datasets import Dataset, Image, Sequence
from PIL import Image as PILImage
import os
import json


def load_mimicext_grpo_dataset(annotation_file="", vis_root="", transform=None):
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
        # annotations = json.load(file)[:500]  # Limit to 256 entries if needed
        # try 10k entries
        annotations = json.load(file)[:10]
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
            image_paths = [os.path.join(vis_root.replace('images_512', 'images_with_region'),
                                        f'{img_id}.png') for img_id in ann["image_ids"]]
        else:
            image_paths = [os.path.join(vis_root, f'{img_id}.png') for img_id in ann["image_ids"]]
        # image_paths = image_paths[:1]
        # images = [PILImage.open(image_path).convert('RGB') for image_path in image_paths]
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


class MiMiCEXTDPODataset(Dataset):
    def __init__(self, annotation_file="", vis_root="", name="", transform=None):
        """
        Initialize the dataset.

        Parameters:
            annotation_file (str): Path to the annotation file containing image IDs and captions.
            vis_root (str): Root directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a PIL image.
        """
        with open(annotation_file, 'r') as file:
            self.annotation = json.load(file)[:256]
        print('Loading Annotations...')
        self.vis_root = vis_root
        self.name = name
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.annotation)

    def __getitem__(self, index):
        """
        Retrieve a sample from the dataset at the specified index.
        """
        ann = self.annotation[index]
        question = ann["question"]
        chosen = ann["chosen"]
        rejected = ann["rejected"]

        task_name = ann["task_name"]
        if '_region_' in task_name:
            image_paths = [os.path.join(self.vis_root.replace('images_512', 'images_with_region'),
                                        f'{img_id}.png') for img_id in ann["image_ids"]]
        else:
            image_paths = [os.path.join(self.vis_root, f'{img_id}.png') for img_id in ann["image_ids"]]

        images = [PILImage.open(image_path).convert('RGB') for image_path in image_paths]
        prompt = '\n'.join([f"<|image_{iid}|>" for iid in range(1, len(image_paths) + 1)])
        prompt += question

        if self.transform:
            images = [self.transform(image) for image in images]

        return {
            "image_paths": image_paths,
            "images": images,
            "question": question,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "image_ids": ann["image_ids"]
        }


class MiMiCEXTDPODataCollator(DataCollatorForPreference):
    def __init__(self, processor, pad_token_id=32000):
        super().__init__(pad_token_id=pad_token_id)
        self.processor = processor
        self.pad_token_id = pad_token_id

    def torch_call(self, examples):
        print(examples)
        assert len(examples) == 1, 'Phi-3-V only supports batch_size == 1'
        example = examples[0]

        prompt = example["prompt"]
        images = example["images"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        prompt_message = {
            'role': 'user',
            'content': prompt,
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )
        # 处理 prompts 和 images，生成 prompt_input_ids 和 attention_mask
        prompt_outputs = self.processor(prompt, images, return_tensors='pt')
        prompt_input_ids = prompt_outputs["input_ids"]
        prompt_attention_mask = prompt_outputs["attention_mask"]

        # 对 chosen 和 rejected 分别进行 tokenizer 编码
        chosen_outputs = self.processor.tokenizer(chosen, add_special_tokens=False, return_tensors="pt")
        rejected_outputs = self.processor.tokenizer(rejected, add_special_tokens=False, return_tensors="pt")

        # 获取 chosen 和 rejected 的 input_ids 和 attention_mask
        chosen_input_ids = chosen_outputs["input_ids"]
        chosen_attention_mask = chosen_outputs["attention_mask"]
        rejected_input_ids = rejected_outputs["input_ids"]
        rejected_attention_mask = rejected_outputs["attention_mask"]

        # 构建 DPOTrainer 所需的返回格式
        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }



