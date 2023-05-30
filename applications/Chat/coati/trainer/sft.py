import math
import os
import time
from typing import List, Optional

import torch
import torch.distributed as dist
import wandb
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import get_scheduler

from .base import Trainer
from .callbacks import Callback
from .strategies import ColossalAIStrategy, Strategy
from .utils import is_rank_0, to_device


class SFTTrainer(Trainer):
    """
        Trainer for supervised fine-tuning.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataloader: the dataloader to use for training
        eval_dataloader: the dataloader to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        tokenizer,
        strategy: Strategy,
        optim: Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader = None,
        max_epochs: int = 2,
        accumulation_steps: int = 8,
        steps_to_save_model: int = 500,
        callbacks: List[Callback] = [],
    ) -> None:
        if accumulation_steps > 1 and isinstance(strategy, ColossalAIStrategy) and strategy.stage == 3:
            raise ValueError("Accumulation steps are not supported in stage 3 of ColossalAI")
        super().__init__(strategy, max_epochs, callbacks=callbacks)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim

        self.accumulation_steps = accumulation_steps
        self.steps_to_save_model = steps_to_save_model
        num_update_steps_per_epoch = len(train_dataloader) // self.accumulation_steps
        max_steps = math.ceil(self.max_epochs * num_update_steps_per_epoch)

        self.scheduler = get_scheduler("cosine",
                                       self.optimizer,
                                       num_warmup_steps=math.ceil(max_steps * 0.03),
                                       num_training_steps=max_steps)

    def fit(self, logger, path: str, use_wandb: bool = False, project_name="Coati"):
        if use_wandb:
            wandb.init(project=project_name, name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            wandb.watch(self.model)
        total_loss = 0
        total_steps = len(self.train_dataloader) // self.accumulation_steps * self.max_epochs
        step_bar = tqdm(range(total_steps), desc=f'steps', disable=not is_rank_0())
        step_counter = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            for batch_id, batch in enumerate(self.train_dataloader):

                batch = to_device(batch, torch.cuda.current_device())
                outputs = self.model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])

                loss = outputs.loss

                loss = loss / self.accumulation_steps

                self.strategy.backward(loss, self.model, self.optimizer)

                total_loss += loss.item()

                # gradient accumulation
                if (batch_id + 1) % self.accumulation_steps == 0:
                    self.strategy.optimizer_step(self.optimizer)
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    if is_rank_0() and use_wandb:
                        wandb.log({
                            "loss": total_loss / self.accumulation_steps,
                            "lr": self.scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "batch_id": batch_id
                        })
                    total_loss = 0
                    step_bar.update()
                    step_counter += 1

                    if is_rank_0(
                    ) and self.steps_to_save_model is not None and step_counter != 0 and step_counter != total_steps and step_counter % self.steps_to_save_model == 0:
                        save_path = os.path.join(path, project_name + f"-step{step_counter}")
                        self.strategy.save_pretrained(self.model,
                                                      path=save_path,
                                                      only_rank0=True,
                                                      tokenizer=self.tokenizer)
                        logger.info(f"Model saved after {step_counter} step(s) at {save_path}")

            # Evaluation
            if self.eval_dataloader is not None:
                self.model.eval()
                with torch.no_grad():
                    loss_sum = 0
                    num_seen = 0
                    for batch in self.eval_dataloader:
                        batch = to_device(batch, torch.cuda.current_device())
                        outputs = self.model(batch["input_ids"],
                                             attention_mask=batch["attention_mask"],
                                             labels=batch["labels"])
                        loss = outputs.loss

                        loss_sum += loss.item()
                        num_seen += batch["input_ids"].size(0)

                    loss_mean = loss_sum / num_seen
                    if dist.get_rank() == 0:
                        logger.info(f'Eval Epoch {epoch}/{self.max_epochs} loss {loss_mean}')
