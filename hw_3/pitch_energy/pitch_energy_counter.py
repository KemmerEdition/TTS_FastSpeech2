from tqdm import tqdm
from hw_3.datasets.ljspeech_dataset import process_text
from hw_3.text import text_to_sequence
from hw_3.utils.configs import pitch_energy_config
import numpy as np
import pyworld as pw
import torchaudio
import torch
import time
import os


def pitch_energy_counter(config=pitch_energy_config):
    buffer = list()
    text = process_text(config.data_path)

    wav_paths = os.path.join('/'.join
                             (config.data_path.split('/')[:-1]), "LJSpeech-1.1/wavs")

    wav_names = sorted(os.listdir(wav_paths))
    start = time.perf_counter()
    for i in tqdm(range(len(text))):

        character = text[i][0:len(text[i]) - 1]
        character = np.array(
            text_to_sequence(character, config.test_cleaners))
        character = torch.from_numpy(character)

        duration = np.load(os.path.join(
            config.alignment_path, str(i) + ".npy"))
        duration = torch.from_numpy(duration)

        wav_tensor, sr = torchaudio.load(os.path.join(config.wav_path, wav_names[i]))
        wav_tensor = wav_tensor.squeeze().to(torch.float64)

        # get pitch
        pitch, t = pw.dio(wav_tensor.numpy(), config.sampling_rate,
                          frame_period=config.hop_length / config.sampling_rate * 1000)
        pitch = pw.stonemask(wav_tensor.numpy(), pitch, t, config.sampling_rate)
        # try without additional elements from article
        # pitch = pitch[: sum(duration)]
        # if np.sum(pitch != 0) <= 1:
        #     return None
        # with interpolation
        zero_res, non_zero_res = (pitch == 0), (pitch != 0)
        pitch[zero_res] = np.interp(np.argwhere(zero_res).squeeze(), np.argwhere(non_zero_res).squeeze(),
                                    pitch[non_zero_res])
        pitch = torch.tensor(pitch)

        # get energy
        spec = torchaudio.transforms.Spectrogram(n_fft=pitch_energy_config.n_fft,
                                                 win_length=pitch_energy_config.win_length,
                                                 hop_length=pitch_energy_config.hop_length,
                                                 power=pitch_energy_config.power)
        speca = spec(wav_tensor)
        energy = torch.norm(speca, p=2, dim=0)

        # mel_gt_target
        mel_cont = torchaudio.transforms.MelScale(n_mels=pitch_energy_config.n_mels_channels,
                                                  sample_rate=pitch_energy_config.sampling_rate,
                                                  f_min=pitch_energy_config.mel_fmin,
                                                  f_max=pitch_energy_config.mel_fmax,
                                                  n_stft=pitch_energy_config.n_fft // 2 + 1,
                                                  norm='slaney',
                                                  mel_scale='slaney')

        mel_cont_ = torch.log(torch.clamp(mel_cont(speca.float()), min=1e-5))
        mel_gt_targ = mel_cont_.transpose(-1, -2)
        buffer.append({"text": character,
                       "duration": duration,
                       "pitch": pitch,
                       "energy": energy,
                       "mel_target": mel_gt_targ})

    end = time.perf_counter()
    # saving results of count process
    np.save("pitch_energy_tensor.npy", buffer)
    print("cost {:.2f}s to load all data into buffer.".format(end - start))
    return buffer
