import torch
import time
from tqdm.auto import tqdm
from queue import Queue, Full
from threading import Event


class EncoderPipeline:
    def __init__(self, encoder, loaders, stop_event: Event, device="cuda:0", queue_size=4):
        """
        loaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
        """
        self.encoder = encoder.to(device)
        self.device = torch.device(device)

        self.stop_event = stop_event
        self.queue = Queue(maxsize=queue_size)
        self.loaders = loaders

    def encode_batch(self, inputs) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder.encode(inputs)

    def prepare_cache(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        val_cache = []
        for batch in tqdm(self.loaders["val"], desc="Preparing validation cache"):
            encoded = self.encode_batch(batch).cpu()
            val_cache.append(encoded)
        torch.cuda.empty_cache()
        test_cache = []
        for batch in tqdm(self.loaders["test"], desc="Preparing test cache"):
            encoded = self.encode_batch(batch).cpu()
            test_cache.append(encoded)
        torch.cuda.empty_cache()
        return val_cache, test_cache

    def get_queue(self) -> tuple[Queue[torch.Tensor], int]:
        return self.queue, len(self.loaders["train"])


    """Runs the encoding pipeline indefinetly for train set. To be used in separate thread or process"""
    def run(self):
        self.encoder.eval()
        with torch.no_grad():
            while True:
                for batch in tqdm(self.loaders["train"], desc="Encoding train dataset"):
                    if self.stop_event.is_set():
                        return
                    inputs = batch
                    encoded = self.encode_batch(inputs)
                    encoded = encoded.detach().cpu().pin_memory()
                    while not self.stop_event.is_set():
                        try:
                            self.queue.put(encoded, timeout=1)
                            break
                        except Full:
                            time.sleep(0.1)

class DualEncoderPipeline:
    def __init__(self, text_encoder, audio_encoder, loaders, stop_event: Event, text_device="cuda:0", audio_device="cuda:1", queue_size=4):
        """
        loaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
        """
        self.text_encoder = text_encoder.to(text_device)
        self.text_device = torch.device(text_device)
        self.audio_encoder = audio_encoder.to(audio_device)
        self.audio_device = torch.device(audio_device)

        self.stop_event = stop_event
        self.queue = Queue(maxsize=queue_size)
        self.loaders = loaders
        self.audio_stream = torch.cuda.Stream(self.audio_device)
        self.text_stream = torch.cuda.Stream(self.text_device)


    def encode_batch(self, inputs):
        with torch.no_grad():
            with torch.cuda.stream(self.audio_stream):
                audio = self.audio_encoder.encode(inputs)

            with torch.cuda.stream(self.text_stream):
                text = self.text_encoder.encode(inputs)

        torch.cuda.synchronize(self.audio_device)
        torch.cuda.synchronize(self.text_device)

        return text, audio

    def prepare_cache(self) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
        val_cache = []
        for batch in tqdm(self.loaders["val"], desc="Preparing validation cache"):
            text, audio = self.encode_batch(batch)
            val_cache.append((text.cpu(), audio.cpu()))
        test_cache = []
        for batch in tqdm(self.loaders["test"], desc="Preparing test cache"):
            text, audio = self.encode_batch(batch)
            test_cache.append((text.cpu(), audio.cpu()))
        return val_cache, test_cache

    def get_queue(self) -> tuple[Queue[tuple[torch.Tensor, torch.Tensor]], int]:
        return self.queue, len(self.loaders["train"])

    def run(self):
        self.audio_encoder.eval()
        self.text_encoder.eval()
        with torch.no_grad():
            while True:
                for batch in tqdm(self.loaders["train"], desc="Encoding train dataset"):
                    if self.stop_event.is_set():
                        return
                    inputs = batch
                    text, audio = self.encode_batch(inputs)
                    text = text.detach().cpu().pin_memory()
                    audio = audio.detach().cpu().pin_memory()
                    while not self.stop_event.is_set():
                        try:
                            self.queue.put((text, audio), timeout=1)
                            break
                        except Full:
                            time.sleep(0.1)
