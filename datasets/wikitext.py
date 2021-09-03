import glob

import torch
import torchvision.transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

import numpy as np
import random
import os
import re
from PIL import Image, ImageDraw, ImageFont
from transformers import BertTokenizer

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


class WikiTextImage(Dataset):
    def __init__(self, root, max_img_w, max_img_h, font_dir, max_length, limit, transform, repeat=1,
                 mode='training'):
        super().__init__()

        self.root = root
        self.max_img_w = max_img_w
        self.max_img_h = max_img_h
        self.transform = transform
        self.font_dir = font_dir
        self.image_ground_truth = []

        with open(self.root, 'r') as file:
            paragraph = ""
            max_paragraph_len = 180
            paragraph_len = 0
            for line in file:
                if line.startswith(" \n"):
                    continue
                line = re.sub("[^ a-zA-Z0-9 \n () , .]|unk", "", line)
                words = line.split()
                if (paragraph_len + len(words)) > max_paragraph_len:
                    paragraph += ' '.join(words[: (max_paragraph_len - paragraph_len)])
                    for _ in range(repeat):
                        self.image_ground_truth.append(paragraph)
                    # max_paragraph_len = random.randint(1, 140)
                    # for char in paragraph:
                    #     tokenizer.add_char(char)
                    paragraph = " "
                    paragraph_len = 0
                else:
                    paragraph += line
                    paragraph_len += len(words)

        self.mode = mode

        self.tokenizer = tokenizer
        self.max_length = max_length + 1
        with Image.open('/data2/mvu/bg/bg.jpeg') as file:
            self.default_bg = file.convert('RGB')
        self.cropper = torchvision.transforms.RandomCrop((self.max_img_w, self.max_img_h))

    def __len__(self):
        if self.mode == 'training':
            return 30000
        return len(self.image_ground_truth)

    def __getitem__(self, idx):
        if self.mode == 'training':
            idx = random.randint(0, len(self.image_ground_truth) - 1)
        font_size_map = {
            'ReenieBeanie-Regular.ttf': (63, 66),
            'JustMeAgainDownHere-Regular.ttf': (55, 60),
            'TheGirlNextDoor-Regular.ttf': (45, 45),
            'Autography.otf': (52, 56)
        }

        # Generate image from its ground truth
        image = Image.new('RGB', (self.max_img_w, self.max_img_h), (255, 255, 255))
        drawer = ImageDraw.Draw(image)
        current_font_size, sentence_w = random.randint(50, 55), 0
        spacing = random.randint(1, 3) / 2.
        margin = 0
        margin_top = 0
        # random_font = random.choice(glob.glob(os.path.join(self.font_dir, '**', '*.[o|t]tf')))
        random_font = '/data2/mvu/fonts/veteran_typewriter/veteran typewriter.ttf'
        if os.path.basename(random_font) in font_size_map:
            min_size, max_size = font_size_map[os.path.basename(random_font)]
            current_font_size = random.randint(min_size, max_size)
        font = ImageFont.truetype(random_font, size=18)
        new_gt, current_line = [], []
        max_word_height = spacing
        for word in self.image_ground_truth[idx].split():
            line_dimension = drawer.textsize(' '.join(current_line + [word]), font=font)
            if line_dimension[1] > max_word_height:
                max_word_height = line_dimension[1]
            if line_dimension[0] + margin * 2 < self.max_img_w:
                current_line.append(word)
            else:
                new_gt.append(current_line)
                current_line = []
        if len(current_line) > 0:
            new_gt.append(current_line)
        y = margin_top
        final_gt = []
        for line in new_gt:
            if y + 2 * max_word_height >= self.max_img_h:
                break
            drawer.text((margin, y), ' '.join(line), font=font, align='left', fill='#000')
            final_gt.append(' '.join(line))
            y += max_word_height
        gt = '\n'.join(final_gt)

        image = image.convert('L')

        # if y + 2 * max_word_height < self.max_img_h:
        #     img = np.asarray(image)
        #     img[y:, :] = 0
        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0), self.max_img_w, self.max_img_h)
        if y + 2 * max_word_height < self.max_img_h:
            image.mask[0][y + max_word_height:, :] = True

        caption, cap_mask = self.tokenizer.encode([gt], max_length=self.max_length)
        cap_mask = (1 - cap_mask).type(torch.bool).squeeze(0)
        return image.tensors.squeeze(0), image.mask.squeeze(0), caption.squeeze(0), cap_mask


def build_dataset(config, transforms, mode='training'):
    if mode == 'training':
        train_dir = os.path.join(config.wikitext_dir, 'wiki.train.tokens')
        data = WikiTextImage(train_dir, config.max_img_w, config.max_img_h, config.font_dir,
                             max_length=config.max_position_embeddings,
                             limit=config.limit, transform=transforms, mode='training')
        return data

    elif mode == 'validation':
        val_dir = os.path.join(config.wikitext_dir, 'wiki.valid.tokens')
        data = WikiTextImage(val_dir, config.max_img_w, config.max_img_h, config.font_dir,
                             max_length=config.max_position_embeddings,
                             limit=config.limit, transform=transforms, mode='validation')
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")
