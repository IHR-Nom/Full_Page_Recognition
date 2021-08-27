import os
import random
import xml.etree.ElementTree as ET

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from . import utils
from .utils import nested_tensor_from_tensor_list, tokenizer

MAX_DIM = 299


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


class IAMImage(Dataset):
    def __init__(self, root, max_img_w, max_img_h, max_length, limit, transform, mode='training'):
        super().__init__()

        self.root = root
        self.max_img_w = max_img_w
        self.max_img_h = max_img_h
        self.transform = transform
        self.ground_truth_folder_dir = os.path.join(root, 'xml')
        self.image_list = []
        img_folders = ['formsA-D', 'formsE-H', 'formsI-Z']
        for img_folder in img_folders:
            for img in os.listdir(os.path.join(root, img_folder)):
                self.image_list.append(os.path.join(root, img_folder, img))

        self.image_list = sorted(self.image_list)
        train_set_size = round(len(self.image_list) * 0.75)
        val_set_size = round(len(self.image_list) * 0.2)

        if mode == 'validation':
            self.image_list = self.image_list[train_set_size:train_set_size + val_set_size]
        elif mode == 'training':
            self.image_list = self.image_list[:train_set_size]
        else:
            self.image_list = self.image_list[train_set_size + val_set_size:]

        self.tokenizer = tokenizer
        self.max_length = max_length + 1

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        with Image.open(os.path.join(self.root, self.image_list[idx])) as img:
            start_x, start_y, end_x, end_y, width, image_ground_truth = \
                get_image_size_and_ground_truth_of_form(self.image_list[idx], self.ground_truth_folder_dir)
            img = img.crop((start_x, start_y, end_x + width, end_y))
            img = img.convert('RGB')
            # width, height = img.size
            # Resize image
            # n_width, n_height = int(width / 2), int(height / 2)
            # resized_img = img.resize((n_width, n_height), Image.ANTIALIAS)
            # p_top_left, p_top_right = resized_img.getpixel((0, 0)), resized_img.getpixel((0, n_height - 1))
            # p_bot_left, p_bot_right = resized_img.getpixel((n_width - 1, 0)), resized_img.getpixel((n_width - 1, n_height - 1))
            #
            # avg_colour = np.stack([np.array(p_top_left), np.array(p_top_right), np.array(p_bot_left), np.array(p_bot_right)])
            # avg_colour = np.mean(avg_colour, axis=0).astype(np.int)
            # Add padding to image to get them into the same shape
            # padding_img = Image.new('RGB', (self.max_img_w, self.max_img_h), (0, 0, 0))
            # padding_img.paste(resized_img)
            # padding_img.save("iam.png")

            img = utils.resize_filling(np.asarray(img), (self.max_img_w, self.max_img_h))
            img = Image.fromarray(img)
            image = img.convert('L')

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0), self.max_img_w, self.max_img_h)

        caption_encoded = self.tokenizer.encode_plus(
            image_ground_truth, max_length=self.max_length, padding='max_length', return_attention_mask=True,
            return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
                1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


def get_image_size_and_ground_truth_of_form(image_name, ground_truth_folder_dir):
    ground_truth_folder_dir = ground_truth_folder_dir
    xml_root = ET.parse(
        os.path.join(ground_truth_folder_dir, os.path.basename(os.path.splitext(image_name)[0]) + ".xml")).getroot()
    image_ground_truth = ""

    width = int(xml_root.get("width"))
    image_handwritten_root = xml_root.find("handwritten-part")
    number_of_hw_line = len(image_handwritten_root.findall("line"))
    hw_line_index = 0
    start_x, end_x, start_y, end_y = 0, 0, 0, 0
    for line in image_handwritten_root.findall("line"):
        if hw_line_index == 0:
            start_x = int(line.get("asx"))
            start_y = int(line.get("asy"))

        image_ground_truth += line.get("text")

        hw_line_index += 1
        if hw_line_index == number_of_hw_line:
            end_x = int(line.get("dsx"))
            end_y = int(line.get("dsy"))

        if hw_line_index < number_of_hw_line:
            image_ground_truth += "\n"

    return start_x, start_y, end_x, end_y, width, image_ground_truth


def build_dataset(config, transforms, mode='training'):
    data = IAMImage(config.iam_dir, config.max_img_w, config.max_img_h, max_length=config.max_position_embeddings,
                    limit=config.limit, transform=transforms, mode=mode)
    return data
