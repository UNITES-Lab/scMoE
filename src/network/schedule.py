import torch
import numpy as np
from itertools import chain

from cytomoe.network.model import Model
from cytomoe.network.constants import *
from cytomoe.network.loss import CrossEntropyLoss


class Schedule:
    def __init__(self, name, model: Model, loss_weight):
        self.name = name
        self.best_loss = np.inf
        self.best_loss_term = None

        self.parameters = chain.from_iterable(
            [
                model.encoders.parameters(),
                model.fusers.parameters(),
                model.transformers.parameters(),
            ]
        )
        self.optimizer = torch.optim.Adam(self.parameters, lr=model.config[str_lr])
        # Collect Loss, including router loss
        self.losses = [
            CrossEntropyLoss(model, loss_weight),
        ]
        self.best_loss_term = str_cross_entropy_loss
    
    def step(self, train_model):
        if train_model:
            self.optimizer.zero_grad()

        total_loss = 0
        for loss in self.losses:
            total_loss += loss

        if train_model:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, 25)
            self.optimizer.step()
        return self.losses
    
    def check_and_save_best_model(self, model, losses, best_model_path, verbose=False):
        if self.best_loss_term is None:
            curr_loss = sum(losses.values())
    


