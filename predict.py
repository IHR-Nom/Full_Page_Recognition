import torch
import torchvision

from transformers import BertTokenizer
from PIL import Image
import argparse

from models import caption
from datasets import coco, utils, wikitext, iam
from configuration import Config
import os
import numpy as np

# parser = argparse.ArgumentParser(description='Image Captioning')
# parser.add_argument('--path', type=str, help='path to image', required=True)
# parser.add_argument('--v', type=str, help='version', default='v3')
# parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
# args = parser.parse_args()
# image_path = args.path
# version = args.v
# checkpoint_path = args.checkpoint

config = Config()

# if version == 'v1':
#     model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
# elif version == 'v2':
#     model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
# elif version == 'v3':
#     model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
# else:
print("Checking for checkpoint.")
if config.checkpoint is None:
    raise NotImplementedError('No model to chose from!')
else:
    if not os.path.exists(config.checkpoint):
        raise NotImplementedError('Give valid checkpoint path')
    print("Found checkpoint! Loading!")
    model, _ = caption.build_model(config)
    print("Loading Checkpoint...")
    checkpoint = torch.load(config.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(299),
        torchvision.transforms.ToTensor(),
        lambda x: x * 255.,
        lambda x: (x - 127.5) / 128.
    ])

image = Image.open("/data1/hong/datasets/iam/formsI-Z/p03-163.png")
start_x, start_y, end_x, end_y, width, image_ground_truth = iam.get_image_size_and_ground_truth_of_form("p03-163.png", "/data1/hong/datasets/iam/xml")
image = image.crop((start_x, start_y, end_x + width, end_y))
image = image.convert('RGB')
image = utils.resize_filling(np.asarray(image), (config.max_img_w, config.max_img_h))
image = Image.fromarray(image)
image = image.convert('L')
image = val_transform(image)
image = image.unsqueeze(0)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)


@torch.no_grad()
def evaluate():
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption

        caption[:, i + 1] = predicted_id[0]
        cap_mask[:, i + 1] = False

    return caption


output = evaluate()
result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
# result = tokenizer.decode(output[0], skip_special_tokens=True)
print(image_ground_truth)
print(result.capitalize())
