import logging

import torch
import torchaudio

from pathlib import Path

from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class RuslanDataset(BaseDataset):
    """
    RUSLAN single-speaker Russian TTS dataset.

    Metadata format: ``<file_id>|<text>`` per line.
    Audio files: ``<wav_path>/<file_id>.wav``.

    Args:
        wav_path (str): path to directory with wav files.
        metadata_path (str): path to metadata CSV.
        part (str | None): dataset partition, e.g "train", "test".
            If None, the whole dataset will be used. Applies after shuffle and 
            filter operation. Default is None.
        test_size (int | float | None): If part is set, this parameter defines the 
            size of the test set. Default is None.
        segment_size (int | None): if set, crop random segment of this
            length (in samples) from each audio during training.
            None means use full audio.
    """

    def __init__(
        self,
        wav_path,
        metadata_path,
        part=None,
        test_size=None,
        segment_size=None,
        *args,
        **kwargs,
    ):
        self.wav_path = Path(wav_path)
        self.metadata_path = Path(metadata_path)
        self.part = part
        self.test_size = test_size
        self.segment_size = segment_size

        self._target_sr_for_index = kwargs.get("target_sr", 24000)

        index = self._create_index()
        super().__init__(index, *args, **kwargs)

        if self.part is not None and self.test_size is not None:
            if isinstance(self.test_size, float):
                split_idx = int(len(self._index) * (1 - self.test_size))
            elif isinstance(self.test_size, int):
                split_idx = len(self._index) - self.test_size
            else:
                raise ValueError(f"Invalid test_size: {self.test_size!r}. Must be int, float, or None.")
            if self.part == "train":
                self._index = self._index[:split_idx]
            elif self.part == "test":
                self._index = self._index[split_idx:]
            else:
                raise ValueError(
                    f"Invalid dataset part: {self.part!r}. "
                    "Valid values are 'train' or 'test', or None when no split is applied."
                )
        
    def _create_index(self):
        index = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                file_name, text = line.split("|", maxsplit=1)
                audio_path = self.wav_path / f"{file_name}.wav"
                if not audio_path.exists():
                    logger.warning(f"File not found, skipping: {audio_path}")
                    continue
                info = torchaudio.info(audio_path)
                audio_len_samples = int(
                    info.num_frames * self._target_sr_for_index / info.sample_rate
                )
                index.append({
                    "path": str(audio_path),
                    "text": text,
                    "audio_len": audio_len_samples,
                })
        logger.info(f"RuslanDataset: {len(index)} items loaded from {self.metadata_path}")
        return index

    def __getitem__(self, ind):
        """
        Load audio, compute spectrogram, optionally crop a random segment.
        No text encoding is performed.
        """
        data_dict = self._index[ind]
        audio = self.load_audio(data_dict["path"])  # (1, T)

        if self.segment_size is not None and audio.size(-1) >= self.segment_size:
            max_start = audio.size(-1) - self.segment_size
            start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
            audio = audio[:, start : start + self.segment_size]

        spectrogram = self.get_spectrogram(audio)

        instance_data = {
            "audio": audio,
            "spectrogram": spectrogram,
        }

        instance_data = self.preprocess_data(instance_data)
        return instance_data