import os
import json
import torch
import re
import random
from torch.utils.data import Dataset
from PIL import Image

class MiMiCEXTVQADataset(Dataset):
    def __init__(self, annotation_file='', vis_root='', name="", transform=None):
        """
        Initialize the dataset.

        Parameters:
            annotation_file (str): Path to the annotation file containing image IDs and captions.
            vis_root (str): Root directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a PIL image.
        """
        with open(annotation_file, 'r') as file:
            self.annotation = json.load(file)

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
        explanation = ann["explanation"]
        answer = ann["answer"]

        if self.name == "diff":
            img_file, img_file1 = ann["image_ids"][0], ann["image_ids"][1]
            image_path = os.path.join(self.vis_root, img_file + '.png')
            image_path1 = os.path.join(self.vis_root, img_file1 + '.png')
            image = Image.open(image_path).convert('RGB')
            image1 = Image.open(image_path1).convert('RGB')
            prompt = f'<|image_1|>\n<|image_2|>\n{question}'
        else:
            img_file, img_file1 = ann["image_id"], None
            image_path = os.path.join(self.vis_root, img_file + '.png')
            image = Image.open(image_path).convert('RGB')
            image1, image_path1 = None, None
            prompt = f'<|image_1|>\n{question}'

        if self.transform:
            image = self.transform(image)
            if self.name == "diff":
                image1 = self.transform(image1)
            
        return {
            "image_path": image_path,
            "image_path1": image_path1,
            "image": image,
            "image1": image1,
            "question": question,
            "prompt": prompt,
            "explanation": explanation,
            "answer": answer,
            "image_id": img_file,
            "image_id1": img_file1
        }

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = '. '.join(tokens) + '.'
        return report


class MiMiCEXTVQADatasetEval(Dataset):
    def __init__(self, annotation_file='', vis_root='', vis_root2="", transform=None):
        """
        Initialize the dataset.

        Parameters:
            annotation_file (str): Path to the annotation file containing image IDs and captions.
            vis_root (str): Root directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a PIL image.
        """
        with open(annotation_file, 'r') as file:
            self.annotation = json.load(file)

        self.annotation = self.annotation
        self.vis_root = vis_root
        self.vis_root2 = vis_root2
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
        task_name = ann["task_name"]
        image_ids = ann["image_ids"]

        if len(image_ids) > 1:
            img_file, img_file1 = image_ids[0], image_ids[1]
            image_path = os.path.join(self.vis_root, img_file + '.png')
            image_path1 = os.path.join(self.vis_root, img_file1 + '.png')
            image = Image.open(image_path).convert('RGB')
            image1 = Image.open(image_path1).convert('RGB')
            prompt = f'<|image_1|>\n<|image_2|>\n{question}'
        else:
            img_file, img_file1 = image_ids[0], None
            try:
                image_path = os.path.join(self.vis_root, img_file + '.png')
                image = Image.open(image_path).convert('RGB')
            except:
                image_path = os.path.join(self.vis_root2, img_file + '.png')
                image = Image.open(image_path).convert('RGB')
            
            image1, image_path1 = None, None
            prompt = f'<|image_1|>\n{question}'

        if self.transform:
            image = self.transform(image)
            if len(image_ids) > 1:
                image1 = self.transform(image1)

        return {
            "image_path": image_path,
            "image_path1": image_path1,
            "image": image,
            "image1": image1,
            "question": question,
            "prompt": prompt,
            "task_name":task_name,
            "image_id": img_file,
            "image_id1": img_file1
        }


class MiMiCEXTVQADatasetTest(Dataset):
    def __init__(self, annotation_file='', vis_root='', vis_root2='', name="", transform=None):
        """
        Initialize the dataset.

        Parameters:
            annotation_file (str): Path to the annotation file containing image IDs and captions.
            vis_root (str): Primary root directory where images are stored.
            vis_root2 (str): Secondary root directory used as备用 when image not found in vis_root.
            transform (callable, optional): Optional transform to be applied on a PIL image.
        """
        with open(annotation_file, 'r') as file:
            self.annotation = json.load(file)

        self.vis_root = vis_root
        self.vis_root2 = vis_root2
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
        explanation = ann["explanation"]
        answer = ann["answer"]

        # 如果包含 "image_ids" 键，表示两张图片
        if "image_ids" in ann:
            img_file, img_file1 = ann["image_ids"][0], ann["image_ids"][1]
            image_path = os.path.join(self.vis_root, img_file + '.png')
            image_path1 = os.path.join(self.vis_root, img_file1 + '.png')
            image = Image.open(image_path).convert('RGB')
            image1 = Image.open(image_path1).convert('RGB')
            prompt = f'<|image_1|>\n<|image_2|>\n{question}'

        # 只有 "image_id" 键，表示单张图片
        elif "image_id" in ann:
            img_file = ann["image_id"]
            img_file1 = None
            try:
                image_path = os.path.join(self.vis_root, img_file + '.png')
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                image_path = os.path.join(self.vis_root2, img_file + '.png')
                image = Image.open(image_path).convert('RGB')

            image1, image_path1 = None, None
            prompt = f'<|image_1|>\n{question}'
        else:
            raise KeyError("Annotation does not contain 'image_id' or 'image_ids'.")

        if self.transform:
            image = self.transform(image)
            if image1 is not None:
                image1 = self.transform(image1)

        return {
            "image_path": image_path,
            "image_path1": image_path1,
            "image": image,
            "image1": image1,
            "question": question,
            "prompt": prompt,
            "explanation": explanation,
            "answer": answer,
            "image_id": img_file,
            "image_id1": img_file1
        }




class MiMiCEXTVQADataCollator:
    def __init__(self, processor):
        self.processor = processor
        # self.max_length = max_length

    def __call__(self, examples):
        assert len(examples) == 1, 'Phi-3-V only supports batch_size == 1'
        example = examples[0]

        if example['image1'] is None:
            image = [example['image']]
        else:
            image = [example['image'], example['image1']]
        
        question = example['question']
        answer = example['explanation']
        prompt = example['prompt']

        prompt_message = {
            'role': 'user',
            'content': prompt,
        }

        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )
        answer = f'{answer}<|end|>\n<|endoftext|>'

        # mask questions for labels
        batch = self.processor(prompt, image, return_tensors='pt')
        prompt_input_ids = batch['input_ids']
        # Do not add bos token to answer
        answer_input_ids = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
        ignore_index = -100
        labels = torch.cat(
            [
                torch.tensor([ignore_index] * len(prompt_input_ids[0])).unsqueeze(0),
                answer_input_ids,
            ],
            dim=1,
        )

        batch['input_ids'] = input_ids
        del batch['attention_mask']
        batch['labels'] = labels

        return batch


# without images
class MiMiCEXTVQADatasetTest_withoutImages(Dataset):
    def __init__(self, annotation_file='', vis_root='', vis_root2='', name="", transform=None):
        """
        Initialize the dataset.

        Parameters:
            annotation_file (str): Path to the annotation file containing image IDs and captions.
            vis_root (str): Primary root directory where images are stored.
            vis_root2 (str): Secondary root directory used as备用 when image not found in vis_root.
            transform (callable, optional): Optional transform to be applied on a PIL image.
        """
        with open(annotation_file, 'r') as file:
            self.annotation = json.load(file)

        self.vis_root = vis_root
        self.vis_root2 = vis_root2
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
        explanation = ann["explanation"]
        answer = ann["answer"]
        prompt = f'{question}'
        # 如果包含 "image_ids" 键，表示两张图片
        # if "image_ids" in ann:
        #     img_file, img_file1 = ann["image_ids"][0], ann["image_ids"][1]
        #     image_path = os.path.join(self.vis_root, img_file + '.png')
        #     image_path1 = os.path.join(self.vis_root, img_file1 + '.png')
        #     image = Image.open(image_path).convert('RGB')
        #     image1 = Image.open(image_path1).convert('RGB')
        #     # prompt = f'<|image_1|>\n<|image_2|>\n{question}'
        #     prompt = f'{question}'

        # # 只有 "image_id" 键，表示单张图片
        # elif "image_id" in ann:
        #     img_file = ann["image_id"]
        #     img_file1 = None
        #     try:
        #         image_path = os.path.join(self.vis_root, img_file + '.png')
        #         image = Image.open(image_path).convert('RGB')
        #     except Exception as e:
        #         image_path = os.path.join(self.vis_root2, img_file + '.png')
        #         image = Image.open(image_path).convert('RGB')

        #     image1, image_path1 = None, None
        #     prompt = f'{question}'
        # else:
        #     raise KeyError("Annotation does not contain 'image_id' or 'image_ids'.")

        # if self.transform:
        #     image = self.transform(image)
        #     if image1 is not None:
        #         image1 = self.transform(image1)

        return {
            "image_path": None,
            "image_path1": None,
            "image": None,
            "image1": None,
            "question": question,
            "prompt": prompt,
            "explanation": explanation,
            "answer": answer,
            "image_id": None,
            "image_id1": None
        }


# without question
class MiMiCEXTVQADatasetTest_withoutQuestion(Dataset):
    def __init__(self, annotation_file='', vis_root='', vis_root2='', name="", transform=None):
        """
        Initialize the dataset.

        Parameters:
            annotation_file (str): Path to the annotation file containing image IDs and captions.
            vis_root (str): Primary root directory where images are stored.
            vis_root2 (str): Secondary root directory used as备用 when image not found in vis_root.
            transform (callable, optional): Optional transform to be applied on a PIL image.
        """
        with open(annotation_file, 'r') as file:
            self.annotation = json.load(file)

        self.vis_root = vis_root
        self.vis_root2 = vis_root2
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
        explanation = ann["explanation"]
        answer = ann["answer"]

        # 如果包含 "image_ids" 键，表示两张图片
        if "image_ids" in ann:
            img_file, img_file1 = ann["image_ids"][0], ann["image_ids"][1]
            image_path = os.path.join(self.vis_root, img_file + '.png')
            image_path1 = os.path.join(self.vis_root, img_file1 + '.png')
            image = Image.open(image_path).convert('RGB')
            image1 = Image.open(image_path1).convert('RGB')
            # prompt = f'<|image_1|>\n<|image_2|>\n{question}'
            prompt = f'<|image_1|>\n<|image_2|>'

        # 只有 "image_id" 键，表示单张图片
        elif "image_id" in ann:
            img_file = ann["image_id"]
            img_file1 = None
            try:
                image_path = os.path.join(self.vis_root, img_file + '.png')
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                image_path = os.path.join(self.vis_root2, img_file + '.png')
                image = Image.open(image_path).convert('RGB')

            image1, image_path1 = None, None
            prompt = f'<|image_1|>'
        else:
            raise KeyError("Annotation does not contain 'image_id' or 'image_ids'.")

        if self.transform:
            image = self.transform(image)
            if image1 is not None:
                image1 = self.transform(image1)

        return {
            "image_path": image_path,
            "image_path1": image_path1,
            "image": image,
            "image1": image1,
            "question": question,
            "prompt": prompt,
            "explanation": explanation,
            "answer": answer,
            "image_id": img_file,
            "image_id1": img_file1
        }