import hydra
import torch
import torchaudio
import warnings

from pathlib import Path
from hydra.utils import instantiate
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(config):
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    model = instantiate(config.model).to(device)

    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    mel_transform = instantiate(config.mel_transform).to(device)

    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sr = config.melspectrogram.sr

    if config.resynthesize:
        audio_dir = data_dir
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a"}
        audio_files = sorted(
            p for p in audio_dir.iterdir() if p.suffix.lower() in audio_extensions
        )

        for audio_path in tqdm(audio_files, desc="Resynthesizing"):
            audio, orig_sr = torchaudio.load(audio_path)
            audio = audio[:1, :]
            if orig_sr != sr:
                audio = torchaudio.functional.resample(audio, orig_sr, sr)

            with torch.no_grad():
                audio = audio.to(device)
                mel = mel_transform(audio)
                x = model.backbone(mel)
                audio_hat = model.head(x)

            out_path = output_dir / audio_path.name
            torchaudio.save(str(out_path), audio_hat.cpu(), sr)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()