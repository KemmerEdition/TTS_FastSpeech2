import torch
import numpy as np
import os

import wandb

from hw_3.utils.configs import FastSpeechSecondConfig as train_config
from hw_3.text import text_to_sequence
import hw_3.waveglow
from hw_3.waveglow.inference import inference
from tqdm import tqdm


def synthesis(model, text, alpha=1.0, alpha_p=1.0, alpha_e=1.0):
    text = np.stack([text])
    src_pos = np.array([i + 1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)

    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha, alpha_p=alpha_p, alpha_e=alpha_e)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data():
    tests = [
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]
    data_list = list(text_to_sequence(test, train_config.test_cleaners) for test in tests)
    return data_list, tests


def get_WaveGlow():
    waveglow_path = os.path.join("waveglow", "pretrained_model")
    waveglow_path = os.path.join(waveglow_path, "waveglow_256channels.pt")
    wave_glow = torch.load(waveglow_path)['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    wave_glow.cuda().eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')

    return wave_glow


def log_audios(model, WaveGlow):
    data_list, tests = get_data()
    table_labels = ["audio_waveglow", "duration", "pitch", "energy", "text"]
    container = []
    for speed in tqdm([0.8, 1., 1.3]):
        for pitch in [0.8, 1, 1.2]:
            for energy in [0.8, 1, 1.2]:
                for i, phh in enumerate(data_list):
                    mel, mel_cuda = synthesis(model, phh, speed, pitch, energy)
                    os.makedirs("results", exist_ok=True)
                    path = f"results/s={speed}_{pitch}_{energy}_{i}_waveglow.wav"
                    waveglow.inference.inference(mel_cuda, WaveGlow, f"results/s={speed}_{pitch}_{energy}_{i}_waveglow.wav")
                    container.append([wandb.Audio(path), speed, pitch, energy, tests[i]])

    table = wandb.Table(data=container, columns=table_labels)
    return table
