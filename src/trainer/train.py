import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.utils.data import DataLoader

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.model import Model
from src.manage import DataManager

from tqdm import tqdm

import logging


class ModelTrainer:
    def __init__(
            self,
            config: DictConfig | ListConfig,
            device: torch.device = torch.device('cuda')
    ):
        self.config = config

        self.model = Model(
            config.in_channels,
            config.patch_size,
            config.img_H,
            config.img_W,
            config.depth,
            config.n_classes
        )

        self.device = device

    def _save(self, epoch):
        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), f"models/model_{epoch}.pt")

    def run(self):
        logger = logging.getLogger(__name__)
        data = DataManager()

        trainSet, testSet = data.get_data()

        trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=2)
        testloader = DataLoader(testSet, batch_size=64, shuffle=False, num_workers=2)

        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(200):
            train_loss = .0
            self.model.train()

            for (i, batch) in tqdm(enumerate(trainLoader), desc=f"Train epoch : {epoch}"):
                input, target = batch
                input = input.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()

                pred = self.model(input)

                loss = criterion(pred, target)

                train_loss += loss.item()

                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    logger.info(f"Train_loss = {train_loss / (i + 1)}")

            logger.info(f"train : {train_loss / len(trainSet) * 64}")

            with torch.no_grad():
                self.model.eval()

                val_loss = .0
                for (i, batch) in tqdm(enumerate(testloader), desc=f"Val epoch : {epoch}"):
                    input, target = batch
                    input = input.to(self.device)
                    target = target.to(self.device)

                    pred = self.model(input)

                    loss = criterion(pred, target)

                    val_loss += loss.item()

                logger.info(f"validate : {val_loss / len(testSet) * 64}")
            
            self._save(epoch)