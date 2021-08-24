import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from configuration import Config
from datasets import synthetic_iam, wikitext, iam, chunom
from engine import train_one_epoch, evaluate
from models import utils, caption
from imgaug import augmenters as iaa


def main(config):
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, criterion = caption.build_model(config)
    model.to(device)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    augment = iaa.Sequential([
            iaa.Sometimes(0.35, iaa.GaussianBlur(sigma=(0, 1.5))),
            iaa.Sometimes(0.35,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.02))])),
        ])
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(299),
        lambda x: np.asarray(x),
        lambda x: augment.augment_image(x),
        torchvision.transforms.ToTensor(),
        lambda x: x * 255.,
        lambda x: (x - 127.5) / 128.
    ])

    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(299),
        lambda x: np.asarray(x),
        torchvision.transforms.ToTensor(),
        lambda x: x * 255.,
        lambda x: (x - 127.5) / 128.
    ])

    # dataset_train = coco.build_dataset(config, mode='training')
    # dataset_val = coco.build_dataset(config, mode='validation')
    # print(f"Train: {len(dataset_train)}")
    # print(f"Valid: {len(dataset_val)}")
    #
    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    #
    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, config.batch_size, drop_last=True
    # )
    #
    # data_loader_train = DataLoader(
    #     dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    # data_loader_val = DataLoader(dataset_val, config.batch_size,
    #                              sampler=sampler_val, drop_last=False, num_workers=config.num_workers)
    # ChuNom dataset
    chunom_dataset_train = chunom.build_dataset(config, train_transform, mode='training')
    chunom_dataset_val = chunom.build_dataset(config, val_transform, mode='validation')
    chunom_dataset_test = iam.build_dataset(config, val_transform, mode='test')
    print(f"Chu Nom Train: {len(chunom_dataset_train)}")
    print(f"Chu Nom Valid: {len(chunom_dataset_val)}")
    # chunom_dataset_train[0]

    # # IAM dataset
    # iam_dataset_train = iam.build_dataset(config, train_transform, mode='training')
    # iam_dataset_val = iam.build_dataset(config, val_transform, mode='validation')
    # iam_dataset_test = iam.build_dataset(config, val_transform, mode='test')
    # print(f"IAM Train: {len(iam_dataset_train)}")
    # print(f"IAM Valid: {len(iam_dataset_val)}")
    # # iam_dataset_train[0]
    #
    # # Synthetic IAM dataset
    # synthetic_iam_dataset_train = synthetic_iam.build_dataset(config, train_transform, mode='training', repeat=10)
    # synthetic_iam_dataset_val = synthetic_iam.build_dataset(config, val_transform, mode='validation', repeat=10)
    # print(f"Synthetic IAM Train: {len(synthetic_iam_dataset_train)}")
    # print(f"Synthetic IAM Valid: {len(synthetic_iam_dataset_val)}")
    # # synthetic_iam_dataset_train[0]
    #
    # # WikiText 2 dataset
    # wikitext_dataset_train = wikitext.build_dataset(config, train_transform, mode='training')
    # # wikitext_dataset_val = wikitext.build_dataset(config, mode='validation')
    # print(f"WikiText Train: {len(wikitext_dataset_train)}")
    # # print(f"WikiText Valid: {len(wikitext_dataset_val)}")
    # # wikitext_dataset_train[0]
    #
    # dataset_train = torch.utils.data.ConcatDataset([iam_dataset_train, synthetic_iam_dataset_train, wikitext_dataset_train])
    # dataset_val = torch.utils.data.ConcatDataset([iam_dataset_val, synthetic_iam_dataset_val])
    #
    # print(f"Train: {len(dataset_train)}")
    # print(f"Valid: {len(dataset_val)}")
    #
    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    #
    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, config.batch_size, drop_last=True
    # )
    #
    # data_loader_train = DataLoader(
    #     dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    # data_loader_val = DataLoader(dataset_val, config.batch_size,
    #                              sampler=sampler_val, drop_last=False, num_workers=config.num_workers)
    # data_loader_test = DataLoader(iam_dataset_test, config.batch_size,
    #                               drop_last=False, num_workers=config.num_workers)
    #
    # # if os.path.exists(config.checkpoint):
    # #     print("Loading Checkpoint...")
    # #     checkpoint = torch.load(config.checkpoint, map_location='cpu')
    # #     model.load_state_dict(checkpoint['model'])
    # #     optimizer.load_state_dict(checkpoint['optimizer'])
    # #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    # #     config.start_epoch = checkpoint['epoch'] + 1
    #
    # print("Start Training..")
    # best_val = 99999
    # with open("test_file_list.txt", "w+") as test_file:
    #     test_file.writelines(iam_dataset_test.image_list)
    #
    # for epoch in range(config.start_epoch, config.epochs):
    #     print(f"Epoch: {epoch}")
    #     epoch_loss = train_one_epoch(
    #         model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
    #     lr_scheduler.step()
    #     print(f"Training Loss: {epoch_loss}")
    #
    #     validation_loss = evaluate(model, criterion, data_loader_val, device)
    #     if validation_loss < best_val:
    #         torch.save({
    #             'model': model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'lr_scheduler': lr_scheduler.state_dict(),
    #             'epoch': epoch,
    #         }, config.checkpoint)
    #         best_val = validation_loss
    #
    #     print(f"Validation Loss: {validation_loss}")
    #
    #     test_loss = evaluate(model, criterion, data_loader_test, device)
    #     print(f"Test Loss: {test_loss}")


if __name__ == "__main__":
    config = Config()
    main(config)
