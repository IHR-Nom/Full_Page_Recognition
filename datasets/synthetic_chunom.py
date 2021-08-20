import glob
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list


class SyntheticChuNomImage(Dataset):
    def __init__(self, root, max_img_w, max_img_h, max_length, limit, transform, repeat=3,
                 mode='training'):
        super().__init__()

        self.root = root
        self.max_img_w = max_img_w
        self.max_img_h = max_img_h
        self.transform = transform
        self.image_folder_dir = os.path.join(root, 'lines')
        self.ground_truth_file = os.path.join(root, 'ascii', 'lines.txt')
        self.line_image_list = []
        self.image_list = []
        for line_image_file in glob.glob(os.path.join(self.image_folder_dir, '*/*', '*.png')):
            self.line_image_list.append(line_image_file)
        for _ in range(repeat):
            random.shuffle(self.line_image_list)
            index = 0
            while index in range(len(self.line_image_list)):
                group_size = random.randint(0, 10)
                sub_image_list = []
                for grp_index in range(group_size):
                    if (index + grp_index) == len(self.line_image_list):
                        break
                    sub_image_list.append(self.line_image_list[index + grp_index])
                self.image_list.append(sub_image_list)
                index += group_size

        train_set_size = round(len(self.image_list) * 0.8)

        if mode == 'validation':
            self.image_list = self.image_list[train_set_size:]
        if mode == 'training':
            self.image_list = self.image_list[: train_set_size]

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True)
        self.max_length = max_length + 1

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image, image_ground_truth = combine_line_image_and_get_ground_truth(self.max_img_w, self.max_img_h,
                                                                            self.image_list[idx],
                                                                            self.ground_truth_file)
        image.save('synthetic_iam.png')

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0), self.max_img_w, self.max_img_h)

        caption_encoded = self.tokenizer.encode_plus(
            image_ground_truth, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True,
            return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
                1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


def combine_line_image_and_get_ground_truth(max_w, max_h, sub_line_image_list, ground_truth_file):
    if len(sub_line_image_list) == 0:
        return Image.new('L', (max_w, max_h), 255), ""

    combined_image = Image.new('RGB', (max_w, max_h), (255, 255, 255))
    combined_image_ground_truth = ""
    indent_top = random.randint(int(max_h * 0.05), int(max_h * 0.1))
    temp_height = indent_top
    indent = random.randint(int(max_w * 0.05), int(max_w * 0.1))
    ref_w, ref_h = int(max_w - indent * 2), int((max_h - indent_top) / 10)
    for line_image_name in sub_line_image_list:
        with Image.open(line_image_name) as image:
            width, height = image.size
            ratio = min(ref_w / width, ref_h / height)
            resized_image = image.resize((int(width * ratio), int(height * ratio)), Image.ANTIALIAS)
            # Combine line image
            width, height = resized_image.size
            combined_image.paste(resized_image, (indent, temp_height))
            temp_height += height

            # Get ground truth of line image
            with open(ground_truth_file, 'r') as file:
                for line in file:
                    if (not line.startswith('#')) and (
                            line.startswith(os.path.basename(os.path.splitext(line_image_name)[0]))):
                        combined_image_ground_truth += line.split(" ")[8].replace('|', " ")
                        break
    combined_image = combined_image.convert('L')

    return combined_image, combined_image_ground_truth


def build_dataset(config, transforms, mode='training'):
    data = SyntheticChuNomImage(config.synthetic_chunom_dir, config.chunom_max_img_w, config.chunom_max_img_h,
                             max_length=config.max_position_embeddings,
                             limit=config.limit, transform=transforms, mode=mode)
    return data
