import glob

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv
import xml.etree.ElementTree as ET

from PIL import Image
import numpy as np
import random
import os

from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list

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


def get_all_lines(image_name, ground_truth_folder_dir):
    ground_truth_folder_dir = ground_truth_folder_dir
    xml_root = ET.parse(
        os.path.join(ground_truth_folder_dir, os.path.basename(os.path.splitext(image_name)[0]) + ".xml")).getroot()
    results = []
    width = int(xml_root.get("width"))
    image_handwritten_root = xml_root.find("handwritten-part")
    for line in image_handwritten_root.findall("line"):
        start_x = int(line.get("asx"))
        start_y = int(line.get("asy"))

        image_ground_truth = line.get("text")

        end_x = int(line.get("dsx"))
        end_y = int(line.get("dsy"))

        results.append(((start_x, start_y, end_x + width, end_y), image_ground_truth))

    return results


class SyntheticIAMImage(Dataset):
    def __init__(self, root, max_img_w, max_img_h, max_length, limit, transform, repeat=3,
                 mode='training'):
        super().__init__()

        self.root = root
        self.max_img_w = max_img_w
        self.max_img_h = max_img_h
        self.transform = transform
        gt_folder_dir = os.path.join(root, 'xml')
        gt_lines = []
        img_folders = ['formsA-D', 'formsE-H', 'formsI-Z']
        for img_folder in img_folders:
            for img in os.listdir(os.path.join(root, img_folder)):
                img_path = os.path.join(root, img_folder, img)
                for (start_x, start_y, end_x, end_y), line_ground_truth in get_all_lines(img_path, gt_folder_dir):
                    gt_lines.append({
                        'img_path': img_path,
                        'line_location': (start_x, start_y, end_x, end_y),
                        'gt': line_ground_truth
                    })
        self.image_list = []
        for _ in range(repeat):
            random.shuffle(gt_lines)
            index = 0
            while index in range(len(gt_lines)):
                group_size = random.randint(0, 10)
                sub_image_list = []
                for grp_index in range(group_size):
                    if (index + grp_index) >= len(gt_lines):
                        break
                    sub_image_list.append(gt_lines[index + grp_index])
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
        images, gts = [], []
        for line in self.image_list[idx]:
            with Image.open(line['img_path']) as img:
                start_x, start_y, end_x, end_y = line['line_location']
                img = img.crop((start_x, start_y, end_x, end_y))
                img = img.convert('RGB')
            images.append(img)
            gts.append(line['gt'])
        image, image_ground_truth = combine_line_image_and_get_ground_truth(self.max_img_w, self.max_img_h,
                                                                            images, gts)
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


def combine_line_image_and_get_ground_truth(max_w, max_h, sub_line_images, ground_truth):
    if len(sub_line_images) == 0:
        return Image.new('L', (max_w, max_h), 255), ""

    combined_image = Image.new('RGB', (max_w, max_h), (255, 255, 255))
    indent_top = random.randint(int(max_h * 0.005), int(max_h * 0.01))
    temp_height = indent_top
    indent = 0
    ref_w, ref_h = int(max_w - indent * 2), int((max_h - indent_top) / 10)
    for image in sub_line_images:
        width, height = image.size
        ratio = min(ref_w / width, ref_h / height)
        resized_image = image.resize((int(width * ratio), int(height * ratio)), Image.ANTIALIAS)
        # Combine line image
        width, height = resized_image.size
        combined_image.paste(resized_image, (indent, temp_height))
        temp_height += height

    combined_image = combined_image.convert('L')

    return combined_image, '\n'.join(ground_truth)


def build_dataset(config, transforms, mode='training', repeat=3):
    data = SyntheticIAMImage(config.iam_dir, config.max_img_w, config.max_img_h,
                             max_length=config.max_position_embeddings,
                             limit=config.limit, transform=transforms, mode=mode, repeat=repeat)
    return data
