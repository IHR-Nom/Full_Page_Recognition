import glob

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

import numpy as np
import random
import os
import re
from PIL import Image, ImageDraw, ImageFont
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
            max_paragraph_len = 100
            max_line_len = 10
            paragraph_len = 0
            for line in file:
                if line.startswith(" \n"):
                    continue
                line = re.sub("[^ a-zA-Z0-9 \n () , .]|unk", "", line)
                words = line.split()
                if (paragraph_len + len(words)) > max_paragraph_len:
                    paragraph += ' '.join(words[: (max_paragraph_len - paragraph_len)])
                    paragraph_words = paragraph.split()
                    paragraph = " "
                    for i in range(0, len(paragraph_words), max_line_len):
                        paragraph += ' '.join(paragraph_words[i: i + max_line_len])
                        # paragraph += "\n"
                    for _ in range(repeat):
                        self.image_ground_truth.append(paragraph)
                    paragraph = " "
                    paragraph_len = 0
                else:
                    paragraph += line
                    paragraph_len += len(words)

        # train_set_size = round(len(self.image_ground_truth) * 0.2)
        #
        # if mode == 'validation':
        #     self.image_ground_truth = self.image_ground_truth[:]
        # if mode == 'training':
        #     self.image_ground_truth = self.image_ground_truth[: train_set_size]

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True)
        self.max_length = max_length + 1

    def __len__(self):
        return 30000

    def __getitem__(self, _):
        idx = random.randint(0, len(self.image_ground_truth) - 1)
        font_size_map = {
            'ReenieBeanie-Regular.ttf': (63, 66),
            'JustMeAgainDownHere-Regular.ttf': (55, 60),
            'TheGirlNextDoor-Regular.ttf': (45, 49)
        }

        # Generate image from its ground truth
        image = Image.new('RGB', (self.max_img_w, self.max_img_h), (255, 255, 255))
        drawer = ImageDraw.Draw(image)
        current_font_size, sentence_w = random.randint(50, 55), 0
        spacing = random.randint(1, 3) / 2.
        margin = random.randint(int(self.max_img_w * 5 / 100), int(self.max_img_w * 10 / 100))
        margin_top = random.randint(int(self.max_img_h * 3 / 100), int(self.max_img_h * 6 / 100))
        random_font = random.choice(glob.glob(os.path.join(self.font_dir, '**', '*.[o|t]tf')))
        if os.path.basename(random_font) in font_size_map:
            min_size, max_size = font_size_map[os.path.basename(random_font)]
            current_font_size = random.randint(min_size, max_size)
        font = ImageFont.truetype(random_font, current_font_size)
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
        for line in new_gt:
            drawer.text((margin, y), ' '.join(line), font=font, align='left', fill='#000')
            y += max_word_height
        gt = '\n'.join([' '.join(line) for line in new_gt])

        image = image.convert('L')

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0), self.max_img_w, self.max_img_h)

        caption_encoded = self.tokenizer.encode_plus(
            gt, max_length=self.max_length, pad_to_max_length=True,
            return_attention_mask=True, return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
                1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


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
