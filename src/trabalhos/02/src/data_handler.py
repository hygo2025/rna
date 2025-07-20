# data_handler.py
import os
import torch
import torchvision.datasets as datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from config import Config


class DataHandler:
    def __init__(self, config: Config):
        self.config = config
        self._create_transforms()
        self._setup_datasets()
        self._create_dataloaders()

    def _create_transforms(self):
        self.train_transform = v2.Compose([
            v2.Resize(size=(self.config.IMG_SIZE, self.config.IMG_SIZE)),

            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(10),
            v2.ColorJitter(brightness=0.2, contrast=0.2),

            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.val_test_transform = v2.Compose([
            v2.Resize(size=(self.config.IMG_SIZE, self.config.IMG_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        print("Transformações de dados criadas.")

    def _setup_datasets(self):
        full_train_dataset = datasets.ImageFolder(os.path.join(self.config.processed_data_dir, 'train'),
                                                  transform=self.train_transform)
        self.test_dataset = datasets.ImageFolder(os.path.join(self.config.processed_data_dir, 'test'),
                                                 transform=self.val_test_transform)

        train_size = int(self.config.TRAIN_SPLIT_RATIO * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        self.train_subset, self.val_subset = random_split(full_train_dataset, [train_size, val_size])

        self.class_names = full_train_dataset.classes
        print(f"Datasets criados. Classes: {self.class_names}")
        print(
            f"Tamanho do treino: {len(self.train_subset)}, Validação: {len(self.val_subset)}, Teste: {len(self.test_dataset)}")

    def _create_dataloaders(self):
        self.train_loader = DataLoader(self.train_subset, batch_size=self.config.BATCH_SIZE, shuffle=True,
                                       num_workers=self.config.NUM_WORKERS, pin_memory=True)
        self.val_loader = DataLoader(self.val_subset, batch_size=self.config.BATCH_SIZE, shuffle=False,
                                     num_workers=self.config.NUM_WORKERS, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False,
                                      num_workers=self.config.NUM_WORKERS, pin_memory=True)
        print("DataLoaders criados com sucesso.")