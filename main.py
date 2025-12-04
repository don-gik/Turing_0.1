# Must be coded in future for easily testing and training.
import argparse
import json

# import subprocess
import logging
import os
from pathlib import Path

# import sys
import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.trainer import ModelTrainer
from src.evaluate import Evaluator


run_dir: Path = Path()
config_path = ""
config_name = ""
unkown_args: None | list[str] = None

LOGGING_CONFIG = "logging.conf"

def evaluate(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    evaluator = Evaluator(
        config = config.train,
        device = device,
        load = config.load
    )
    return evaluator.run()

@hydra.main(config_path="configs/", version_base=None)
def run(config):
    logging.config.fileConfig(  # type: ignore
        "logging.conf", defaults={"rundir": str(run_dir)}
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info("-------- Starting the program --------")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if getattr(config, "evaluate", False):
        evaluate(config)

    trainer = ModelTrainer(
        config = config.train,
        device = device
    )

    trainer.run()
    


if __name__ == "__main__":
    run()
