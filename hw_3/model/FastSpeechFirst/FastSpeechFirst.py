import torch
from torch import nn
from hw_fs.model.FastSpeechFirst.EncoderDecoder import Encoder, Decoder
from hw_fs.model.FastSpeechFirst.blocks import LengthRegulator
from hw_fs.model.FastSpeechFirst.utils import get_mask_from_lengths
from hw_fs.utils.configs import fast_speech_config


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config=fast_speech_config):
        super(FastSpeech, self).__init__()
        self.model_config = model_config
        self.encoder = Encoder(self.model_config)
        self.length_regulator = LengthRegulator(self.model_config)
        self.decoder = Decoder(self.model_config)

        self.mel_linear = nn.Linear(self.model_config.decoder_dim, self.model_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        x, non_pad_mask = self.encoder(src_seq, src_pos)

        if self.training:
            output, duration_predictor_output = self.length_regulator(x, alpha, length_target, mel_max_length)
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return output, duration_predictor_output
        else:
            output, mel_pos = self.length_regulator(x, alpha)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return output
