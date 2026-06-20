import os
from typing import override
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torchaudio
import torch.nn.functional as F
import torch

from eam.dataset.Dataset import Dataset, DatasetSplit
from eam.dataset.config import DatasetConfig
from eam.dataset.registry import register

@register
class AdmedVoiceDataset(Dataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.__load_corpus_from_disk()

    """Read corpus from disk, generate splits based on speakers"""
    def __load_corpus_from_disk(self):
        def get_duration_in_seconds(duration_str):
            h, m, s = map(float, duration_str.split(":"))
            return h * 3600 + m * 60 + s

        def get_file_path(example):
            source = example["source"]
            if source == "natural":
                source = "human_voices/human/human"
            elif source == "anonymization":
                source = "anoni/anoni"
            elif source == "synthesis":
                source = "synth/synth"
            else:
                raise ValueError(f"Unknown source type: {source}")

            cat_code = example["cat_code"]
            rec_place = example["rec_place"]
            speaker_id = example["speaker_id"]
            filename = example["filename"]
            file_path = (
                f"{self.config['dataset_path']}/{source}/cat_{cat_code}/{rec_place}-{speaker_id}/{filename}"
            )
            return file_path

        summary_path = os.path.join(self.config["dataset_path"], "corpus_summary_all.csv")
        corpus = pd.read_csv(summary_path, sep=";")

        # filer out non human
        corpus = corpus[corpus["source"] == "natural"]

        # filter out longer than 30s
        corpus["duration_sec"] = corpus["file_duration"].apply(get_duration_in_seconds)
        corpus = corpus[corpus["duration_sec"] <= 30.0]

        corpus["file_path"] = corpus.apply(get_file_path, axis=1)

        unique_speakers = corpus["speaker_id"].unique()
        train_speakers, heldout_speakers = train_test_split(
            unique_speakers,
            test_size=0.3,
            random_state=420,
        )

        val_speakers, test_speakers = train_test_split(
            heldout_speakers,
            test_size=0.5,
            random_state=67,
        )

        train_corpus = corpus[corpus["speaker_id"].isin(train_speakers)]
        val_corpus = corpus[corpus["speaker_id"].isin(val_speakers)]
        test_corpus = corpus[corpus["speaker_id"].isin(test_speakers)]
        train_ds = AdmedSplit(train_corpus, self.config)
        val_ds = AdmedSplit(val_corpus, self.config)
        test_ds = AdmedSplit(test_corpus, self.config)
        self.splits = {
                DatasetSplit.TRAIN: train_ds,
                DatasetSplit.VALIDATION: val_ds,
                DatasetSplit.TEST: test_ds
        }

class AdmedSplit(TorchDataset):
    def __init__(self, corpus_df: pd.DataFrame, config: DatasetConfig, max_seconds=30):
        self.corpus = corpus_df.reset_index(drop=True)
        self.config = config
        self.target_sr = 16000
        self.fixed_len = self.target_sr * max_seconds

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        item = self.corpus.iloc[idx]
        file_path = item["file_path"]

        waveform, sample_rate = torchaudio.load(file_path)

        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        # resample
        if sample_rate != self.target_sr:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sr
            )(waveform)
            sample_rate = self.target_sr

        length = waveform.size(0)

        if length > self.fixed_len:
            # random crop (better for training)
            start = torch.randint(0, length - self.fixed_len + 1, (1,)).item()
            waveform = waveform[start:start + self.fixed_len]
        else:
            # pad
            pad = self.fixed_len - length
            waveform = F.pad(waveform, (0, pad))

        return {
            self.config["audio_column"]: waveform,
            "sampling_rate": sample_rate,
            self.config["text_column"]: item["phrase"],
        }


def get_default_config() -> DatasetConfig:
    return DatasetConfig(
        dataset_name = "AdmedVoiceDataset",
        audio_column = "audio_array",
        text_column = "transcript",
        dataset_path = "/home/student5/datasets/admed_voice",
        batch_size = 32,
        shuffle_train = True,
    )
