import csv
import glob
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list, bincount_app


class GeneratedChuNomImage(Dataset):
    def __init__(self, root, max_img_size, max_patch_size, font_dir, max_length, limit, transform, repeat=3,
                 mode='training'):
        super().__init__()

        self.root = root
        self.max_img_size = max_img_size
        self.max_patch_size = max_patch_size
        self.transform = transform
        self.font_dir = font_dir
        self.ground_truth_list = []

        num_patch_per_page = 20
        max_patch_len = 8

        with open(root, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            patch_ground_truth_list = []
            if header is not None:
                for line in reader:
                    patches = [line[0][i: i + max_patch_len] for i in range(0, len(line[0]), max_patch_len)]
                    for patch in patches:
                        patch_ground_truth_list.append(patch)

        self.ground_truth_list = [patch_ground_truth_list[i: i + num_patch_per_page] for i in
                                  range(0, len(patch_ground_truth_list), num_patch_per_page)]

        self.ground_truth_list = sorted(self.ground_truth_list)
        train_set_size = round(len(self.ground_truth_list) * 0.75)
        val_set_size = round(len(self.ground_truth_list) * 0.2)

        if mode == 'validation':
            self.ground_truth_list = self.ground_truth_list[train_set_size:train_set_size + val_set_size]
        elif mode == 'training':
            self.ground_truth_list = self.ground_truth_list[:train_set_size]
        else:
            self.ground_truth_list = self.ground_truth_list[train_set_size + val_set_size:]

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True)
        self.max_length = max_length + 1

    def __len__(self):
        return len(self.ground_truth_list)

    def __getitem__(self, idx):
        if len(self.ground_truth_list[idx]) == 0:
            image = Image.new('L', self.max_img_size, 255)
            image_ground_truth = ""
        else:
            # Generate image
            image = generate_image(self.max_img_size, self.max_patch_size, self.image_ground_truth_list[idx],
                                   self.font_dir)
            # Get image ground truth
            image_ground_truth = get_image_ground_truth(self.image_ground_truth_list[idx])
        image.save("generated.jpg")

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


def generate_image(image_size, patch_size, image_ground_truth_list, font_dir):
    max_w, max_h = image_size
    max_patch_w, max_patch_h = patch_size
    generated_image = Image.new('RGB', image_size, (255, 255, 255))
    indent = random.randint(int(max_h * 0.005), int(max_h * 0.01))
    temp_height = indent
    random_font_size = random.randint(25, 28)
    random_font = random.choice(glob.glob(os.path.join(font_dir, '*.[o|t]tf')))

    y = temp_height
    for idx in range(len(image_ground_truth_list)):
        # Generate patch
        patch = generate_patch(patch_size, image_ground_truth_list[idx], random_font, random_font_size)
        # Rotate patch
        fill_color = bincount_app(patch)
        patch = patch.rotate(90, fillcolor=fill_color, expand=1)
        patch = patch.convert('RGB')
        # Combine patch
        width, height = patch.size
        x = indent
        if idx % 2 == 0:
            generated_image.paste(patch, (x, y))
        else:
            generated_image.paste(patch, (max_w - indent - max_patch_w, y))
            y += height

    generated_image = generated_image.convert('L')

    return generated_image


def generate_patch(patch_size, patch_ground_truth, random_font, random_font_size):
    patch_h, patch_w = patch_size
    # Generate patch from its ground truth
    patch = Image.new('RGB', (patch_w, patch_h), (255, 255, 255))
    drawer = ImageDraw.Draw(patch)
    font = ImageFont.truetype(random_font, random_font_size)
    w, h = drawer.textsize(patch_ground_truth[0], font=font)
    y = random.randint(int(h * 3 / 100), int(h * 6 / 100))
    for char in patch_ground_truth:
        drawer.text(((patch_w - w) / 2, y), char, font=font, align='center', fill='#000')
        y = y + h

    return patch


def get_image_ground_truth(image_ground_truth_list):
    image_ground_truth = ""
    for i in range(len(image_ground_truth_list)):
        image_ground_truth += image_ground_truth_list[i]
        if i % 2 == 0:
            image_ground_truth += "\t\t"
        else:
            image_ground_truth += "\n"

    return image_ground_truth


def build_dataset(config, transforms, mode='training'):
    data = GeneratedChuNomImage(config.generated_chunom_dir,
                                (config.chunom_max_img_w, config.chunom_max_img_h),
                                (config.chunom_max_patch_w, config.chunom_max_patch_h),
                                config.chunom_font_dir,
                                max_length=config.max_position_embeddings,
                                limit=config.limit, transform=transforms, mode=mode)
    return data
