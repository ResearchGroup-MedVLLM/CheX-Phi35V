import os
import re
import json
import torch
import random
from torch.utils.data import Dataset
from PIL import Image

class IUXRAYDataset(Dataset):
    def __init__(self, annotation_file='', vis_root='', mode='train', transform=None):
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
        self.annotation = self.annotation[mode]
        self.img_ids = {ann['id']: idx for idx, ann in enumerate(self.annotation)}
        self.transform = transform
        # self.questions = ["generate a report", "describe xxx in detail"]
        # self.questions = ['Describe the given chest x-ray image in detail.',
        #                   'Take a look at this chest x-ray and describe the findings and impression.',
        #                   'Could you provide a detailed description of the given x-ray image?',
        #                   'Describe the given chest x-ray image as detailed as possible.',
        #                   "What are the key findings in this chest x-ray image?",
        #                   "What is the most prominent feature visible in this chest x-ray image, and how is it indicative of the patient's health?",
        #                   "What are the finding and overall impression provided by this chest x-ray image?",
        #                   "List any concerns or abnormal findings in this chest x-ray image.",
        #                   "Is the overall impression provided by this chest x-ray image normal or abnormal? Answer based on the observed findings.",
        #                   "Based on the findings in this chest x-ray image, what is the overall impression?",
        #                   "Summarize the overall impression of this chest x-ray image.",
        #                   "Identify any abnormalities present in the given chest x-ray image."
        #                   ]
        # 空问题
        # self.question = [""]

        # # Xray-GPT alignment
        # self.questions = ["Describe the given chest x-ray image in detail.",
        #                   "Take a look at this chest x-ray and describe the findings and impression.",
        #                   "Could you provide a detailed description of the given x-ray image?",
        #                   "Describe the given chest x-ray image as detailed as possible.",
        #                   "What are the key findings in this chest x-ray image?",
        #                   "Could you highlight any abnormalities or concerns in this chest x-ray image?",
        #                   "What specific features of the lungs and heart are visible in this chest x-ray image?",
        #                   "What is the most prominent feature visible in this chest x-ray image, and how is it indicative of the patient's health?",
        #                   "What are the finding and overall impression provided by this chest x-ray image?",
        #                   "Is the overall impression provided by this chest x-ray image normal or abnormal? Answer based on the observed findings.",
        #                   "Are there any indications of infection or inflammation in this chest x-ray image, and if so, what is the likely cause?",
        #                   "Are there any visible indications of enlargement or abnormalities in the patient's lymph nodes in this chest x-ray image?",
        #                   "Based on the findings in this chest x-ray image, what is the overall impression?",
        #                   "Is there any potential complications or risks are associated with the observed abnormalities in this chest x-ray image? or the x-ray is normal?"
        #                   ]

        # # Xray-GLM  
        self.questions = [
            'What diagnosis can be made from this chest X-ray image?',
            'What is in the background of this image?',
            'Describe this image in detail.',
            'Take a look at this image and describe what you notice.',
            'Please provide a detailed description of the image.',
            'Can you describe the content of this image for me?'
        ]
        #   huatuoGPT  Multi-Image Question Set + chest X-ray
        # self.questions = [
        #     'Describe these image in detail.',
        #     'What details stand out in these images?',
        #     'Analyze the images in a comprehensive and detailed manner',
        #     'Take a look at these image and describe what you notice.',
        #     'Please provide a detailed description of these pictures.',
        #     'Could you describe the contents of these images for me?',
        #     'What can be diagnosed from these two chest X-ray images?'
        # ]

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
        image_path = ann["image_path"]

        images = []
        for img_name in ann["image_path"]:
            images.append(Image.open(os.path.join(self.vis_root, img_name)).convert('RGB'))

        if self.transform:
            images = [self.transform(img) for img in images]

        question = random.choice(self.questions)
        # question = self.question[0]
        answer = self.clean_report_mimic_cxr(ann['report'])

        return {
            "image_path": image_path,
            "images": images,
            "question": question,
            "answer": answer,
            "image_id": self.img_ids[ann["id"]]
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
    

class IUXRAYDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        assert len(examples) == 1, 'Phi-3-V only supports batch_size == 1'
        example = examples[0]

        images = example['images']
        question = example['question']
        answer = example['answer']
        content_prompt = ''
        for i in range(len(images)):
            content_prompt += f'<|image_{i+1}|>\n'
        
        prompt_message = {
            'role': 'user',
            'content': f'{content_prompt}{question}',
        }

        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )

        answer = f'{answer}<|end|>\n<|endoftext|>'

        # mask questions for labels
        batch = self.processor(prompt, images, return_tensors='pt')
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


class IUXRAYDataCollatorLoad:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        assert len(examples) == 1, 'Phi-3-V only supports batch_size == 1'
        example = examples[0]

        images = example['images']
        question = example['question']
        answer = example['answer']
        content_prompt = ''
        for i in range(len(images)):
            content_prompt += f'<|image_{i+1}|>\n'
        
        prompt_message = {
            'role': 'user',
            'content': f'{content_prompt}{question}',
        }

        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )

        answer = f'{answer}<|end|>\n<|endoftext|>'

        # mask questions for labels
        batch = self.processor(prompt, images, return_tensors='pt')
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

        batch['inputs_embeds'] = inputs_embeds
        del batch['attention_mask']
        batch['labels'] = labels

        return batch