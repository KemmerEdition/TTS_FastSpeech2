import os
import numpy as np
import torch
import time
from torch.utils.data import Dataset
from hw_fs.text import text_to_sequence
from tqdm import tqdm

# mini_config for fastspeech first
mel_ground_truth = "./mels"
data_path = "./data/train.txt"
alignment_path = "./alignments"
text_cleaners = ['english_cleaners']


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


# when fastspeech2 we don't need it
def get_data_to_buffer():
    buffer = list()
    text = process_text(data_path)

    start = time.perf_counter()
    for i in tqdm(range(len(text))):

        mel_gt_name = os.path.join(
            mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            alignment_path, str(i)+".npy"))
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


class BufferDataset(Dataset):
    def __init__(self):
        self.buffer = list(np.load("pitch_energy_tensor.npy", allow_pickle=True))
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]
