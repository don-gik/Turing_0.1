import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.manage import DataManager
from src.manage.config import CIFAR100_MEAN, CIFAR100_STD
from src.model import Model
from tqdm import tqdm


class Evaluator:
    def __init__(
            self,
            config: DictConfig | ListConfig,
            device: torch.device = torch.device('cuda'),
            load: str = "models/model_199.pt",
            output_root: str | os.PathLike[str] = "outputs/eval",
    ):
        self.config = config
        self.device = device
        self.output_dir = Path(output_root) / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = Model(
            config.in_channels,
            config.patch_size,
            config.img_H,
            config.img_W,
            config.depth,
            config.n_classes,
            dropout = config.dropout,
            forward_dropout = config.forward_dropout,
        ).to(self.device)

        state_dict = torch.load(load, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

        # Cached tensors for de-normalizing CIFAR100 samples when saving visualizations.
        self._mean = torch.tensor(CIFAR100_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CIFAR100_STD, device=self.device).view(1, 3, 1, 1)

    def run(self, lrp_samples: int = 8):
        logger = logging.getLogger(__name__)
        data = DataManager()

        _, testSet = data.get_data()

        testloader = DataLoader(testSet, batch_size=64, shuffle=False, num_workers=2)

        criterion = nn.CrossEntropyLoss()

        self.model.eval()
        loss, accuracy = self._evaluate(testloader, criterion)
        logger.info(f"validate loss: {loss:.4f}, accuracy: {accuracy * 100:.2f}%")

        lrp_paths = self._generate_lrp_examples(testSet, limit=lrp_samples)
        if lrp_paths:
            logger.info(f"LRP examples saved to {self.output_dir / 'lrp'}")

        return {
            "loss": loss,
            "accuracy": accuracy,
            "lrp_paths": [str(path) for path in lrp_paths],
        }

    def _evaluate(self, loader: DataLoader, criterion: nn.Module):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for (__, batch) in tqdm(enumerate(loader), desc="Evaluation", unit="batch"):
                input, target = batch
                input = input.to(self.device)
                target = target.to(self.device)

                pred = self.model(input)

                loss = criterion(pred, target)
                total_loss += loss.item() * input.size(0)

                predicted = torch.argmax(pred, dim=1)
                total_correct += (predicted == target).sum().item()
                total_samples += target.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        return avg_loss, accuracy

    def _generate_lrp_examples(self, dataset, limit: int = 8):
        """
        Generate a lightweight LRP-style relevance map using gradient x input
        for a handful of samples. Saves both the de-normalized input and the
        relevance heatmap for quick inspection.
        """
        logger = logging.getLogger(__name__)
        self.model.eval()

        num_samples = min(limit, len(dataset))
        lrp_dir = self.output_dir / "lrp"
        lrp_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []

        for idx in range(num_samples):
            image, label = dataset[idx]
            input = image.unsqueeze(0).to(self.device)
            input.requires_grad_(True)

            logits = self.model(input)
            predicted = int(torch.argmax(logits, dim=1).item())
            class_idx = int(label)
            if class_idx < 0 or class_idx >= logits.shape[1]:
                class_idx = predicted

            score = logits[0, class_idx]

            self.model.zero_grad(set_to_none=True)
            if input.grad is not None:
                input.grad.zero_()

            score.backward()

            relevance = (input.grad * input).sum(dim=1, keepdim=True)
            relevance = relevance.clamp(min=0)

            rel_max = relevance.max()
            rel_min = relevance.min()
            norm = (rel_max - rel_min).clamp(min=1e-8)
            relevance = (relevance - rel_min) / norm

            denorm = (input * self._std + self._mean).clamp(0, 1).detach().cpu()
            heatmap = relevance.detach().cpu().repeat(1, 3, 1, 1).clamp(0, 1)

            input_path = lrp_dir / f"sample_{idx:04d}_input_t{label}_p{predicted}.png"
            lrp_path = lrp_dir / f"sample_{idx:04d}_lrp_t{label}_p{predicted}.png"

            save_image(denorm, input_path)
            save_image(heatmap, lrp_path)

            logger.info(f"LRP sample {idx}: target={label}, pred={predicted}, saved={lrp_path}")
            saved_paths.append(lrp_path)

        return saved_paths
