import logging
import torch
import torchaudio

from pathlib import Path

from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, data_dir, segment_size=None, *args, **kwargs):
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "audio"
        self.transcriptions_dir = self.data_dir / "transcriptions"
        self.segment_size = segment_size

        self._target_sr_for_index = kwargs.get("target_sr", 22050)

        index = self._create_index()
        super().__init__(index, *args, **kwargs)

    def _create_index(self):
        index = []
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a"}

        for audio_path in sorted(self.audio_dir.iterdir()):
            if audio_path.suffix.lower() not in audio_extensions:
                continue

            text = ""
            if self.transcriptions_dir.exists():
                transc_path = self.transcriptions_dir / (audio_path.stem + ".txt")
                if transc_path.exists():
                    with transc_path.open(encoding="utf-8") as f:
                        text = f.read().strip()

            info = torchaudio.info(audio_path)
            audio_len = int(
                info.num_frames * self._target_sr_for_index / info.sample_rate
            )

            index.append({
                "path": str(audio_path),
                "text": text,
                "audio_len": audio_len,
            })

        logger.info(f"CustomDirAudioDataset: {len(index)} items from {self.data_dir}")
        return index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio, sr = self.load_audio(data_dict["path"])

        if self.segment_size is not None and audio.size(-1) >= self.segment_size:
            max_start = audio.size(-1) - self.segment_size
            start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
            audio = audio[:, start:start + self.segment_size]

        spectrogram = self.get_spectrogram(audio)
        spectrogram = spectrogram.squeeze(0)

        instance_data = {
            "audio": audio,
            "spectrogram": spectrogram,
            "text": data_dict["text"],
            "audio_path": data_dict["path"],
        }

        instance_data = self.preprocess_data(instance_data)
        return instance_data
