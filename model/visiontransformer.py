import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize
from torch import optim
import numpy as np
from torch.hub import tqdm

from dataclay import DataClayObject, activemethod

from .nn_modules import *


@dataclass
class ApplicationArgs:
    no_cuda: bool = False
    patch_size: int = 16
    latent_size: int = 768
    n_channels: int = 3
    num_heads: int = 12
    num_encoders: int = 12
    dropout: float = 0.1  # the example said int, but 0.1 is not an int AFAICT
    img_size: int = 224
    num_classes: int = 16
    epochs: int = 10
    lr: float = 1e-2  # same here, assuming float
    weight_decay: float = 3e-2  # technically an int, but I digress
    batch_size: int = 4
    dry_run: bool = False


class VisionTransformer(DataClayObject):
    # The attributes annotated here will be persisted by dataClay
    # (assuming make_persistent is called to the class, ofc)
    args: ApplicationArgs
    model: nn.Module

    # train_dataloader and valid_dataloader are not persisted
    # as they seem to be used transitoriously.
    # @abarcelo has no idea, so assume that I am messing up
    # the proper way of combining persistence / train stages

    def __init__(self, **args):
        self.args = ApplicationArgs(**args)

    def _train_fn(self, current_epoch):
        self.model.train()
        total_loss = 0.0
        tk = tqdm(self.train_dataloader, desc="EPOCH" + "[TRAIN]" + str(current_epoch + 1) + "/" + str(self.args.epochs))

        for t, data in enumerate(tk):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})
            if self.args.dry_run:
                break

        return total_loss / len(self.train_dataloader)

    def _eval_fn(self, current_epoch):
        self.model.eval()
        total_loss = 0.0
        tk = tqdm(self.valid_dataloader, desc="EPOCH" + "[VALID]" + str(current_epoch + 1) + "/" + str(self.args.epochs))

        for t, data in enumerate(tk):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})
            if self.args.dry_run:
                break

        return total_loss / len(self.valid_dataloader)

    @activemethod
    def train(self) -> tuple[float, float]:
        """This should be called *after* make_persistent, so it is done in dataClay space."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transforms = Compose([
            Resize((224, 224)),
            ToTensor()
        ])

        train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transforms)
        valid_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transforms)
        self.train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
        self.valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=True)

        self.model = ViT(self.args).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        # The example has this code in TrainEval.train_fn
        best_valid_loss = np.inf
        best_train_loss = np.inf
        for i in range(self.args.epochs):
            train_loss = self._train_fn(i)
            valid_loss = self._eval_fn(i)

            if valid_loss < best_valid_loss:
                torch.save(self.model.state_dict(), "best-weights.pt")
                print("Saved Best Weights")
                best_valid_loss = valid_loss
                best_train_loss = train_loss
        print(f"Training Loss : {best_train_loss}")
        print(f"Valid Loss : {best_valid_loss}")

        return best_train_loss, best_valid_loss
