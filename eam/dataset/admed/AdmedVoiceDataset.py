from collections import defaultdict
import os
import random
import re
import unicodedata
from tqdm.auto import tqdm
from typing import override
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset, DataLoader, Sampler
import torchaudio
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence

from eam.dataset.Dataset import Dataset, DatasetSplit
from eam.dataset.config import DatasetConfig
from eam.dataset.registry import register

def normalize_transcript(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

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

        corpus["transcript"] = corpus["phrase"].apply(normalize_transcript)
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
    @override
    def get_collate_function(self, pad_value=0.0):
        audio_key = self.config["audio_column"]
        text_key = self.config["text_column"]

        def collate(batch):
            waveforms = [b[audio_key] for b in batch]
            lengths = torch.tensor([w.size(0) for w in waveforms])

            padded_waveforms = pad_sequence(
                waveforms,
                batch_first=True,
                padding_value=pad_value
            )

            texts = [b[text_key] for b in batch]
            sample_rates = torch.tensor([b["sampling_rate"] for b in batch])

            return {
                audio_key: padded_waveforms,
                "lengths": lengths,
                "sampling_rate": sample_rates,
                text_key: texts,
            }

        return collate

    @override
    def get_train_sampler(self) -> Sampler | None:
        return UniqueTranscriptBatchSampler(self.splits[DatasetSplit.TRAIN], self.config["batch_size"], self.config["text_column"])


class AdmedSplit(TorchDataset):
    def __init__(self, corpus_df: pd.DataFrame, config: DatasetConfig):
        self.corpus = corpus_df.reset_index(drop=True)
        self.config = config
        self.target_sr = 16000

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

        return {
            self.config["audio_column"]: waveform,  # variable length now
            "sampling_rate": sample_rate,
            self.config["text_column"]: item["transcript"],
        }


class UniqueTranscriptBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, text_key):
        self.dataset = dataset
        self.batch_size = batch_size

        self.text_keys = []

        for i in tqdm(range(len(dataset)), desc="Indexing train dataset"):
            item = dataset[i]
            self.text_keys.append(item[text_key])

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        batch = []
        batch_keys = set()

        deferred = []

        while indices:
            idx = indices.pop()

            key = self.text_keys[idx]

            if key in batch_keys:
                deferred.append(idx)
                continue

            batch.append(idx)
            batch_keys.add(key)

            if len(batch) == self.batch_size:
                yield batch

                indices.extend(deferred)
                deferred.clear()

                random.shuffle(indices)

                batch = []
                batch_keys = set()

        if batch:
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size

def get_default_config() -> DatasetConfig:
    return DatasetConfig(
        dataset_name = "AdmedVoiceDataset",
        audio_column = "audio_array",
        text_column = "transcript",
        dataset_path = "/home/student5/datasets/admed_voice",
        batch_size = 32,
        shuffle_train = True,
    )
