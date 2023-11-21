import torch
import random
import os
import numpy as np
import torchaudio
from hw_3 import waveglow
import utils
from torch import nn
from torch.nn.utils import clip_grad_norm_
from hw_3.utils.configs import FastSpeechSecondConfig as train_config
from hw_3.pitch_energy.synthesis import log_audios
from hw_3.pitch_energy.synthesis import synthesis, get_data
from hw_3.text import text_to_sequence
from tqdm import tqdm
import wandb

from hw_3.base import BaseTrainer
from hw_3.utils import inf_loop, MetricTracker
from hw_3.logger import WanDBWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader) * dataloaders.batch_expand_size
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch

        self.lr_scheduler = lr_scheduler
        self.log_step = 20

        self.train_metrics = MetricTracker(
            "loss", "grad norm", "mel_loss", "duration_loss", "pitch_loss", "energy_loss", writer=self.writer
        )

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        batch_idx = 0
        for batch in (
                tqdm(self.train_dataloader)
        ):
            for db in batch:
                batch_idx += 1

                # Get Data
                character = db["text"].long().to(train_config.device)
                mel_target = db["mel_target"].float().to(train_config.device)
                duration = db["duration"].int().to(train_config.device)
                pitch = db["pitch"].float().to(train_config.device)
                energy = db["energy"].float().to(train_config.device)
                mel_pos = db["mel_pos"].long().to(train_config.device)
                src_pos = db["src_pos"].long().to(train_config.device)
                max_mel_len = db["mel_max_len"]

                # Forward
                mel_output, duration_predictor_output, pitch_pred, energy_pred = self.model(
                    character,
                    src_pos,
                    mel_pos=mel_pos,
                    mel_max_length=max_mel_len,
                    length_target=duration, pitch_target=pitch, energy_target=energy)

                # Calc Loss
                mel_loss, duration_loss, pitch_loss, energy_loss = self.criterion(
                    mel_output,
                    duration_predictor_output,
                    pitch_pred,
                    energy_pred,
                    mel_target,
                    duration, pitch, energy)

                total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

                # Logger
                self.train_metrics.update("loss", total_loss.item())
                self.train_metrics.update("grad norm", self.get_grad_norm())
                self.train_metrics.update("mel_loss", mel_loss.item())
                self.train_metrics.update("duration_loss", duration_loss.item())
                self.train_metrics.update("pitch_loss", pitch_loss.item())
                self.train_metrics.update("energy_loss", energy_loss.item())

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                self._clip_grad_norm()

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()

                if batch_idx % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                    self.logger.debug("Train Epoch: {} {} Loss: {:.6f} mel loss: {:.6f} duration loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), total_loss.item(), mel_loss.item(), duration_loss.item(), pitch_loss.item(), energy_loss.item()
                    ))
                    self.writer.add_scalar("learning rate", self.lr_scheduler.get_last_lr()[0])
                    self._log_scalars(self.train_metrics)
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics
        self.evaluation()

        return log

    def evaluation(self):
        self.model.eval()
        WaveGlow = utils.get_WaveGlow()
        WaveGlow = WaveGlow.to(self.device)
        data_list, tests = get_data()
        table_labels = ["audio_waveglow", "duration", "pitch", "energy", "text"]
        container = []
        for speed in tqdm([0.8, 1., 1.3]):
            for pitch in [0.8, 1, 1.2]:
                for energy in [0.8, 1, 1.2]:
                    for i, phh in enumerate(data_list):
                        mel, mel_cuda = synthesis(self.model, phh, speed, pitch, energy)
                        os.makedirs("results", exist_ok=True)
                        path = f"results/s={speed}_{pitch}_{energy}_{i}_waveglow.wav"
                        waveglow.inference.inference(mel_cuda, WaveGlow,
                                                     f"results/s={speed}_{pitch}_{energy}_{i}_waveglow.wav")
                        container.append([wandb.Audio(path), speed, pitch, energy, tests[i]])

        table = wandb.Table(data=container, columns=table_labels)
        wandb.log({"examples": table})



    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    # def _log_audio(self, name, audio, sr):
    #     self.writer.add_audio(f"audio{name}", audio, sample_rate=sr)
