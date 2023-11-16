import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from hw_3.model.FastSpeechSecond.utils import Transpose, create_alignment


class VariancePredictor(nn.Module):
    def __init__(self, model_config):
        super(VariancePredictor).__init__()

        self.model_config = model_config
        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.variance_predictor_filter_size
        self.kernel = model_config.variance_predictor_kernel_size
        self.conv_output_size = model_config.variance_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length - output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = torch.tensor(torch.clamp(
                (torch.round(
                    torch.exp(duration_predictor_output) - 1) * alpha), min=0).int())
            output = self.LR(x, duration_predictor_output)

            mel_pos = torch.stack(
                [torch.Tensor([i + 1 for i in range(output.size(1))])]).long().to(x.device)

        return output, mel_pos


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        n_bins = model_config.n_bins
        pitch_min = model_config.pitch_min
        energy_min = model_config.energy_min
        pitch_max = model_config.pitch_max
        energy_max = model_config.energy_max

        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator(model_config)

        self.pitch_emb = nn.Embedding(n_bins, model_config.encoder_dim)
        self.energy_emb = nn.Embedding(n_bins, model_config.encoder_dim)

        self.pitch_bucket = nn.Parameter(torch.exp(torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins-1)),
                                         requires_grad=False)
        self.energy_bucket = nn.Parameter(torch.linspace(energy_min, energy_max, n_bins-1),
                                          requires_grad=False)

    def forward(self, x, alpha_d=1.0, alpha_p=1.0, alpha_e=1.0,
                dur_target=None, pitch_target=None, energy_target=None, max_len=None, mask=None):

        x, duration_predictor = self.length_regulator(x, alpha_d, dur_target, max_len)

        pitch_predictor = self.pitch_predictor(x)
        # energy_predictor = self.energy_predictor(x)

        if pitch_target is not None and energy_target is not None:
            pitch_value = self.pitch_emb(torch.bucketize(pitch_target, self.pitch_bucket))
            pitch_predictor.masked_fill(mask, 0.0)
        else:
            pitch_predictor = alpha_p * pitch_predictor
            pitch_value = self.pitch_emb(torch.bucketize(pitch_predictor, self.pitch_bucket))
        x = x + pitch_value

        energy_predictor = self.energy_predictor(x)
        if pitch_target is not None and energy_target is not None:
            energy_value = self.energy_emb(torch.bucketize(energy_target, self.energy_bucket))
            energy_predictor.masked_fill(mask, 0.0)
        else:
            energy_predictor = alpha_e * energy_predictor
            energy_value = self.energy_emb(torch.bucketize(energy_predictor, self.energy_bucket))
        x = x + energy_value

        return x, duration_predictor, pitch_predictor, energy_predictor

