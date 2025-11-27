import torch
import torch.nn as nn
import torch.optim as optim

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

    def run(self):
        logger = logging.getLogger(__name__)
        data = DataManager()

        trainSet, testSet = data.get_data()

        trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=2)
        testloader = DataLoader(testSet, batch_size=64, shuffle=False, num_workers=2)

        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        for epoch in range(50):
            train_loss = .0
            self.model.train()

            for (i, batch) in tqdm(enumerate(trainLoader), desc=f"Train epoch : {epoch}"):
                input, target = batch.to(self.device)

                pred = self.model(input)

                loss = criterion(pred, target)

                train_loss += loss.item()

                loss.backward()
                optimizer.step()

            logger.info(f"train : {train_loss}")

            with torch.no_grad():
                self.model.eval()

                val_loss = .0
                for (i, batch) in tqdm(enumerate(testloader), desc=f"Val epoch : {epoch}"):
                    input, target = batch.to(self.device)

                    pred = self.model(input)

                    loss = criterion(pred, target)

                    train_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                logger.info(f"validate : {val_loss}")