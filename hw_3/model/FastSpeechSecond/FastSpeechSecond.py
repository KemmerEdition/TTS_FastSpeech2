import torch
from torch import nn
from hw_3.utils.configs import fast_speech_second_config
from hw_3.model.FastSpeechSecond.EncoderDecoder import Encoder, Decoder
from hw_3.model.FastSpeechSecond.VarianceAdaptor import VarianceAdaptor
from hw_3.model.FastSpeechSecond.utils import get_mask_from_lengths


class FastSpeechSecond(nn.Module):
    """ FastSpeechSecond """

    def __init__(self, model_config=fast_speech_second_config):
        super(FastSpeechSecond, self).__init__()
        self.model_config = model_config
        self.encoder = Encoder(self.model_config)
        self.var_adaptor = VarianceAdaptor(self.model_config)
        self.decoder = Decoder(self.model_config)

        self.mel_linear = nn.Linear(self.model_config.decoder_dim, self.model_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None,
                length_target=None, alpha=1.0, pitch_target=None, alpha_p=1.0,
                energy_target=None, alpha_e=1.0):
        x, non_pad_mask = self.encoder(src_seq, src_pos)
        if self.training:
            mask = (mel_pos == 0)
            output, duration_prediction, pitch_prediction, energy_prediction = self.var_adaptor(
                x, alpha, alpha_p, alpha_e, length_target, pitch_target, energy_target, mel_max_length, mask
            )
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return output, duration_prediction, pitch_prediction, energy_prediction
        else:
            output, mel_pos, pitch_prediction, energy_prediction = self.var_adaptor(
                x, alpha_d=alpha, alpha_p=alpha_p, alpha_e=alpha_e
            )
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return output

