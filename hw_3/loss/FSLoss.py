import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, mel_target, duration_predictor_target):
        mel_loss = self.l1_loss(mel, mel_target)

        duration_predictor_loss = self.mse_loss(duration_predicted,
                                                duration_predictor_target.float())

        return mel_loss, duration_predictor_loss


class FastSpeechSecondLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_prediction, pitch_prediction,
                energy_prediction, mel_target, duration_predictor_target,
                pitch_target, energy_target):

        mel_loss = self.l1_loss(mel, mel_target)
        duration_predictor_loss = self.mse_loss(duration_prediction, torch.log1p(duration_predictor_target.float()))
        pitch_loss = self.mse_loss(pitch_prediction, pitch_target.float())
        energy_loss = self.mse_loss(energy_prediction, energy_target.float())

        return mel_loss, duration_predictor_loss, pitch_loss, energy_loss
