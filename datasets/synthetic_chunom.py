import json
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list, bincount_app, resize_filling


class SyntheticChuNomImage(Dataset):
    def __init__(self, root, json_path, max_img_size, max_patch_size, max_length, limit, transform, repeat=3,
                 mode='training'):
        super().__init__()

        self.root = root
        self.max_img_size = max_img_size
        self.max_patch_size = max_patch_size
        self.transform = transform
        self.image_list_json = json.load(open(json_path))
        self.patch_image_list = []
        patch_image_list_json = []
        lvt_patch_image_list_json = []
        tok_patch_image_list_json = []

        for line in json.load(open(os.path.join(root, "train.json"))):
            if "Luc-Van-Tien" in line:
                lvt_patch_image_list_json.append(line)
            elif "tale-of-kieu" in line:
                tok_patch_image_list_json.append(line)
        patch_image_list_json.append(lvt_patch_image_list_json)
        patch_image_list_json.append(tok_patch_image_list_json)

        for sub_patch_list_json in patch_image_list_json:
            for _ in range(repeat):
                index = 0
                while index in range(len(sub_patch_list_json)):
                    group_size = random.randint(0, 20)
                    sub_patch_image_list = []
                    for grp_index in range(group_size):
                        if (index + grp_index) >= len(sub_patch_list_json):
                            break
                        sub_patch_image_list.append(sub_patch_list_json[index + grp_index])
                    self.patch_image_list.append(sub_patch_image_list)
                    index += group_size

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True)
        self.max_length = max_length + 1

    def __len__(self):
        return len(self.patch_image_list)

    def __getitem__(self, idx):
        if len(self.patch_image_list[idx]) == 0:
            image = Image.new('L', self.max_img_size, 255)
            image_ground_truth = ""
        else:
            # Combine patches into image
            image = get_combined_image(self.root, self.max_img_size, self.max_patch_size, self.patch_image_list[idx])
            # Get image ground truth
            image_ground_truth = get_image_ground_truth(self.root, self.patch_image_list[idx])

        image.save('synthetic_chunom.jpg')

        if self.transform:
            image = self.transform(image)
        max_img_w, max_img_h = self.max_img_size
        image = nested_tensor_from_tensor_list(image.unsqueeze(0), max_img_w, max_img_h)

        caption_encoded = self.tokenizer.encode_plus(
            image_ground_truth, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True,
            return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
                1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


def get_combined_image(root, max_img_size, max_patch_size, patch_image_list_dir):
    max_w, max_h = max_img_size
    combined_image = None
    indent = random.randint(int(max_h * 0.005), int(max_h * 0.01))
    temp_height = indent
    for i in range(len(patch_image_list_dir)):
        image_name = patch_image_list_dir[i].replace(".json", ".jpg")
        with Image.open(os.path.join(root, image_name)) as img:
            # Rotate image
            fill_color = bincount_app(img)
            if combined_image is None:
                combined_image = Image.new('RGB', (max_w, max_h), fill_color)
            img = img.rotate(90, fillcolor=fill_color, expand=1)
            img = img.convert('RGB')
            # Resize image
            resized_image = resize_filling(np.asarray(img), max_patch_size)
            resized_image = Image.fromarray(resized_image)
            # Combine line image
            max_patch_w, max_patch_h = max_patch_size
            width, height = resized_image.size
            if i % 2 == 0:
                combined_image.paste(resized_image, (indent, temp_height))
            else:
                combined_image.paste(resized_image, (max_w - indent - max_patch_w, temp_height))
                temp_height += height

    combined_image = combined_image.convert('L')

    return combined_image


def get_image_ground_truth(root, patch_image_list_dir):
    image_ground_truth = ""

    for i in range(len(patch_image_list_dir)):
        image_annotation = json.load(open(os.path.join(root, patch_image_list_dir[i])))
        image_ground_truth += " ".join(image_annotation[0]["hn_text"])

        if i % 2 == 0:
            image_ground_truth += "\t\t"
        else:
            image_ground_truth += "\n"

    return image_ground_truth


def build_dataset(config, transforms, mode='training'):
    if mode == 'training':
        train_json_path = os.path.join(config.synthetic_chunom_dir, "train.json")
        data = SyntheticChuNomImage(config.synthetic_chunom_dir, train_json_path,
                                    (config.chunom_max_img_w, config.chunom_max_img_h),
                                (config.chunom_max_patch_w, config.chunom_max_patch_h),
                                max_length=config.max_position_embeddings,
                                limit=config.limit, transform=transforms, mode=mode)
        return data

    elif mode == 'validation':
        val_json_path = os.path.join(config.synthetic_chunom_dir, "val.json")
        data = SyntheticChuNomImage(config.synthetic_chunom_dir, val_json_path,
                                    (config.chunom_max_img_w, config.chunom_max_img_h),
                                (config.chunom_max_patch_w, config.chunom_max_patch_h),
                                max_length=config.max_position_embeddings,
                                limit=config.limit, transform=transforms, mode=mode)
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")
