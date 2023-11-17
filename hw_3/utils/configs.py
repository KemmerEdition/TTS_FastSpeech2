from dataclasses import dataclass
import torch


@dataclass
class PitchEnergyConfig:
    data_path = "./data/train.txt"
    wav_path = './data/LJSpeech-1.1/wavs'
    alignment_path = "./alignments"
    mel_ground_truth = "./mels"
    test_cleaners = ['english_cleaners']

    filter_length = 1024
    hop_length = 256
    win_length = 1024
    n_mels_channels = 80
    sampling_rate = 22050
    mel_fmin = 0.0
    mel_fmax = 8000.0
    n_fft = 1024
    power = 1


@dataclass
class FastSpeechConfig:
    num_mels: int = 80
    vocab_size: int = 300
    max_seq_len: int = 3000

    encoder_dim: int = 256
    encoder_n_layer: int = 4
    encoder_head: int = 2
    encoder_conv1d_filter_size: int = 1024

    decoder_dim: int = 256
    decoder_n_layer: int = 4
    decoder_head: int = 2
    decoder_conv1d_filter_size: int = 1024

    fft_conv1d_kernel: list = (9, 1)
    fft_conv1d_padding: list = (4, 0)

    duration_predictor_filter_size: int = 256
    duration_predictor_kernel_size: int = 3
    dropout: float = 0.1

    PAD: int = 0
    UNK: int = 1
    BOS: int = 2
    EOS: int = 3

    PAD_WORD: str = '<blank>'
    UNK_WORD: str = '<unk>'
    BOS_WORD: str = '<s>'
    EOS_WORD: str = '</s>'


@dataclass
class FastSpeechSecondConfig:
    num_mels: int = 80
    vocab_size: int = 300
    max_seq_len: int = 3000

    encoder_dim: int = 256
    encoder_n_layer: int = 4
    encoder_head: int = 2
    encoder_conv1d_filter_size: int = 1024

    decoder_dim: int = 256
    decoder_n_layer: int = 4
    decoder_head: int = 2
    decoder_conv1d_filter_size: int = 1024

    fft_conv1d_kernel: list = (9, 1)
    fft_conv1d_padding: list = (4, 0)
    test_cleaners = ['english_cleaners']

    n_bins = 256

    # with interpolation
    pitch_min = 60.6560
    # without interpolation
    # pitch_min = 0.0
    pitch_max = 861.0653
    energy_min = 0.0179
    energy_max = 314.9619
    batch_expand_size = 32

    variance_predictor_filter_size: int = 256
    variance_predictor_kernel_size: int = 3
    dropout: float = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PAD: int = 0
    UNK: int = 1
    BOS: int = 2
    EOS: int = 3

    PAD_WORD: str = '<blank>'
    UNK_WORD: str = '<unk>'
    BOS_WORD: str = '<s>'
    EOS_WORD: str = '</s>'


pitch_energy_config = PitchEnergyConfig()
fast_speech_config = FastSpeechConfig()
fast_speech_second_config = FastSpeechSecondConfig()