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
            return self.encoder.encode(inputs).to(self.device, non_blocking=True)

    def prepare_cache(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        val_cache = []
        for batch in tqdm(self.loaders["val"], desc="Preparing validation cache"):
            encoded = self.encode_batch(batch).cpu()
            val_cache.append(encoded)
        test_cache = []
        for batch in tqdm(self.loaders["test"], desc="Preparing test cache"):
            encoded = self.encode_batch(batch).cpu()
            test_cache.append(encoded)
        return val_cache, test_cache

    def get_queue(self) -> tuple[Queue[torch.Tensor], int]:
        return self.queue, len(self.loaders["train"])


    """Runs the encoding pipeline indefinetly for train set. To be used in separate thread or process"""
    def run(self):
        self.encoder.eval()
        with torch.no_grad():
            while True:
                for batch in tqdm(self.loaders["train"], desc="Encoding train dataset"):
                    inputs = batch
                    encoded = self.encode_batch(inputs)
                    encoded = encoded.detach().cpu().pin_memory()
                    while not self.stop_event.is_set():
                        try:
                            self.queue.put(encoded, timeout=1)
                            break
                        except Full:
                            time.sleep(0.1)
