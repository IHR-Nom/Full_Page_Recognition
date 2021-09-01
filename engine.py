# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import sys

import torch
import tqdm

from models import utils


def train_one_epoch(model, criterion, data_loader,
                    optimizer, device, epoch, max_norm, steps_per_batch):
    model.train()
    criterion.train()

    epoch_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        count = 0
        for images, masks, caps, cap_masks in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])

            loss_value = loss.item()
            epoch_loss += loss_value

            # The loss needs to be scaled, since we are going to accumulate the gradients
            loss = loss / steps_per_batch

            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            loss.backward()
            count += 1
            if count == steps_per_batch:

                # if max_norm > 0:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()
                optimizer.zero_grad()
                count = 0

            pbar.update(1)
            pbar.set_description("Current train loss: %f" % loss_value)

    return epoch_loss / total


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    validation_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])

            validation_loss += loss.item()

            pbar.update(1)
        
    return validation_loss / total