import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from hw_fs.model.FastSpeechFirst.blocks import FFTBlock
from hw_fs.model.FastSpeechFirst.utils import get_attn_key_pad_mask, get_non_pad_mask


class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()

        self.model_config = model_config
        len_max_seq = model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.encoder_n_layer

        self.src_word_emb = nn.Embedding(
            model_config.vocab_size,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.fft_conv1d_kernel,
            model_config.fft_conv1d_padding,
            dropout=model_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, PAD=self.model_config.PAD)
        non_pad_mask = get_non_pad_mask(src_seq, PAD=self.model_config.PAD)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, model_config):

        super(Decoder, self).__init__()

        self.model_config = model_config
        len_max_seq = model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.decoder_n_layer

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.fft_conv1d_kernel,
            model_config.fft_conv1d_padding,
            dropout=model_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, PAD=self.model_config.PAD)
        non_pad_mask = get_non_pad_mask(enc_pos, PAD=self.model_config.PAD)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output
