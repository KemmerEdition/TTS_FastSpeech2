import argparse
import json
import os
from pathlib import Path

import torch
import waveglow
import utils
import numpy as np
from hw_fs.text import text_to_sequence
from hw_fs.text import sequence_to_text
import audio
from tqdm import tqdm

import hw_fs.model as module_model
import hw_fs.datasets as dataset
# from hw_fs.trainer import Trainer
from hw_fs.utils import ROOT_PATH
from hw_fs.utils.object_loading import LJLoader
from hw_fs.utils.parse_config import ConfigParser

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    # datasets = config.init_obj(config["datasets"], dataset)
    # dataloaders = config.init_obj(config["dataloaders"], LJLoader, datasets=datasets)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    # from seminar 7 WaveGlow
    WaveGlow = utils.get_WaveGlow()
    WaveGlow = WaveGlow.to(device)

    def synthesis(model, text, alpha=1.0):
        text = np.array(text)
        text = np.stack([text])
        src_pos = np.array([i + 1 for i in range(text.shape[1])])
        src_pos = np.stack([src_pos])
        sequence = torch.from_numpy(text).long().to(device)
        src_pos = torch.from_numpy(src_pos).long().to(device)

        with torch.no_grad():
            mel = model.forward(sequence, src_pos, alpha=alpha)
        return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)

    def get_data():
        tests = [
            "I am very happy to see you again!",
            "Durian model is a very good speech synthesis!",
            "When I was twenty, I fell in love with a girl.",
            "I remove attention module in decoder and use average pooling to implement predicting r frames at once",
            "You can not improve your past, but you can improve your future. Once time is wasted, life is wasted.",
            "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old."
        ]
        data_list = list(text_to_sequence(test, ["english_cleaners"]) for test in tests)

        return data_list

    data_list = get_data()

    for speed in [0.8, 1., 1.3]:
        for i, phn in tqdm(enumerate(data_list)):
            mel, mel_cuda = synthesis(model, phn, speed)

            os.makedirs("results", exist_ok=True)

            audio.tools.inv_mel_spec(
                mel, f"results/s={speed}_{i}.wav"
            )

            waveglow.inference.inference(
                mel_cuda, WaveGlow,
                f"results/s={speed}_{i}_waveglow.wav"
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # # if `--test-data-folder` was provided, set it as a default test set
    # if args.test_data_folder is not None:
    #     test_data_folder = Path(args.test_data_folder).absolute().resolve()
    #     assert test_data_folder.exists()
    #     config.config["data"] = {
    #         "test": {
    #             "batch_size": args.batch_size,
    #             "num_workers": args.jobs,
    #             "datasets": [
    #                 {
    #                     "type": "CustomDirAudioDataset",
    #                     "args": {
    #                         "audio_dir": str(test_data_folder / "audio"),
    #                         "transcription_dir": str(
    #                             test_data_folder / "transcriptions"
    #                         ),
    #                     },
    #                 }
    #             ],
    #         }
    #     }
    #
    # assert config.config.get("data", {}).get("test", None) is not None
    # config["data"]["test"]["batch_size"] = args.batch_size
    # config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
