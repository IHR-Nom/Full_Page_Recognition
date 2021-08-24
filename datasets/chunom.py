import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer

from . import utils
from .utils import nested_tensor_from_tensor_list, get_parent_folder


class ChuNomImage(Dataset):
    def __init__(self, root, max_img_w, max_img_h, max_length, limit, transform, mode='training'):
        super().__init__()

        self.root = root
        self.max_img_w = max_img_w
        self.max_img_h = max_img_h
        self.transform = transform
        self.image_list = []
        img_folders = ["Luc-Van-Tien", "tale-of-kieu"]
        for img_folder in img_folders:
            for img in os.listdir(os.path.join(root, img_folder, "images")):
                self.image_list.append(os.path.join(root, img_folder, "images", img))

        self.image_list = sorted(self.image_list)
        print(len(self.image_list))
        train_set_size = round(len(self.image_list) * 0.75)
        val_set_size = round(len(self.image_list) * 0.2)

        if mode == 'validation':
            self.image_list = self.image_list[train_set_size:train_set_size + val_set_size]
        elif mode == 'training':
            self.image_list = self.image_list[:train_set_size]
        else:
            self.image_list = self.image_list[train_set_size + val_set_size:]

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True)
        self.max_length = max_length + 1

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        parent_dir = get_parent_folder(self.image_list[idx], 2)
        image_name = os.path.relpath(self.image_list[idx], parent_dir)
        # Get image from original image
        with Image.open(self.image_list[idx]) as img:
            min_x, min_y, max_x, max_y = get_image_size(image_name, os.path.join(parent_dir, "bboxes.json"))
            img = img.crop((min_x, min_y, max_x, max_y))
            # Rotate image
            fill_color = utils.bincount_app(img)
            img = img.rotate(90, fillcolor=fill_color, expand=1)
            img = img.convert('RGB')
            img = utils.resize_filling(np.asarray(img), (self.max_img_w, self.max_img_h))
            img = Image.fromarray(img)
            image = img.convert('L')
            image.save(os.path.basename(os.path.splitext(image_name)[0]) + ".jpg")

        # Get image ground truth
        image_ground_truth = get_image_ground_truth(image_name, os.path.join(parent_dir, "annotation.json"))

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


def get_image_size(image_name, bounding_box_file):
    bbox_json = json.load(open(bounding_box_file))
    image_regions = [value for value in bbox_json["assets"].values() if value["asset"]["path"] == image_name][0][
        "regions"]
    image_points = [p["points"] for p in image_regions if p.get("tags")[0] == "Column"]
    image_points = [j for i in image_points for j in i]

    max_x = max(p.get("x") for p in image_points)
    max_y = max(p.get("y") for p in image_points)
    min_x = min(p.get("x") for p in image_points)
    min_y = min(p.get("y") for p in image_points)

    return min_x, min_y, max_x, max_y


def get_image_ground_truth(image_name, ground_truth_file):
    image_ground_truth = ""
    annotation_json = json.load(open(ground_truth_file))
    image_annotation = [value["annotations"] for value in annotation_json if value["img"] == image_name][0]

    for i in range(0, len(image_annotation)):
        if i % 2 != 0:
            image_ground_truth += "\t\t"
        elif i % 2 == 0 and i != 0:
            image_ground_truth += "\n"
        image_ground_truth += " ".join(image_annotation[i]["hn_text"])

    return image_ground_truth


def build_dataset(config, transforms, mode='training'):
    data = ChuNomImage(config.chunom_dir, config.chunom_max_img_w, config.chunom_max_img_h,
                       max_length=config.max_position_embeddings,
                       limit=config.limit, transform=transforms, mode=mode)
    return data
