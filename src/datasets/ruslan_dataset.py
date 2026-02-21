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
    Audio files: ``<data_dir>/<wav_subdir>/<file_id>.wav``.

    Expected layout of ``data_dir``::

        data_dir/
        ├── <subdir>/        # directory with wav files (first subdir found)
        └── <metadata>.csv   # metadata file (first *.csv found)

    Args:
        data_dir (str): path to directory containing the wav subdirectory
            and the metadata CSV file.
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
        data_dir,
        part=None,
        test_size=None,
        segment_size=None,
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)

        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV metadata file found in {data_dir}")
        self.metadata_path = csv_files[0]

        wav_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
        if not wav_dirs:
            raise FileNotFoundError(f"No wav subdirectory found in {data_dir}")
        self.wav_path = wav_dirs[0]
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
        audio, sr = self.load_audio(data_dict["path"])  # (1, T)

        if self.segment_size is not None and audio.size(-1) >= self.segment_size:
            max_start = audio.size(-1) - self.segment_size
            start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
            audio = audio[:, start : start + self.segment_size]

        spectrogram = self.get_spectrogram(audio)
        spectrogram = spectrogram.squeeze(0)

        instance_data = {
            "audio": audio,
            "spectrogram": spectrogram,
            "sr": sr,
        }

        instance_data = self.preprocess_data(instance_data)
        return instance_data
