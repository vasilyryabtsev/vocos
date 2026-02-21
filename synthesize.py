import hydra
import torch
import torchaudio
import warnings

from pathlib import Path
from hydra.utils import instantiate
from tqdm.auto import tqdm
from transformers import VitsModel, AutoTokenizer

from src.datasets.custom_dir_audio_dataset import CustomDirAudioDataset

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

    mel_transform = instantiate(config.mel_transform)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sr = config.melspectrogram.sr

    dataset = CustomDirAudioDataset(
        data_dir=config.data_dir,
        target_sr=sr,
        instance_transforms={"get_spectrogram": mel_transform},
    )
    if config.resynthesize:
        for item in tqdm(dataset, desc="Resynthesizing"):
            audio_path = Path(item["audio_path"])
            mel = item["spectrogram"].unsqueeze(0).to(device)

            with torch.no_grad():
                x = model.backbone(mel)
                audio_hat = model.head(x)

            out_path = output_dir / audio_path.name
            torchaudio.save(str(out_path), audio_hat.cpu(), sr)
            print(f"Saved: {out_path}")
    else:
        tts_model_id = config.get("tts_model_id", "facebook/mms-tts-rus")
        print(f"Loading TTS model: {tts_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(tts_model_id)
        tts_model = VitsModel.from_pretrained(tts_model_id).to(device)
        tts_model.eval()
        tts_sr = tts_model.config.sampling_rate

        original_dir = output_dir / "original"
        vocos_dir = output_dir / "tts_with_vocos"
        original_dir.mkdir(parents=True, exist_ok=True)
        vocos_dir.mkdir(parents=True, exist_ok=True)

        for item in tqdm(dataset, desc="TTS + Vocos"):
            text = item["text"]
            if not text:
                continue
            stem = Path(item["audio_path"]).stem

            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                tts_audio = tts_model(**inputs).waveform  # [1, T]

            if tts_sr != sr:
                tts_audio = torchaudio.functional.resample(tts_audio, tts_sr, sr)

            torchaudio.save(str(original_dir / f"{stem}.wav"), item["audio"].cpu(), sr)

            with torch.no_grad():
                mel = mel_transform(tts_audio.cpu()).to(device)
                x = model.backbone(mel)
                audio_hat = model.head(x)

            torchaudio.save(str(vocos_dir / f"{stem}.wav"), audio_hat.cpu(), sr)
            print(f"[{stem}] {text[:60]}...")


if __name__ == "__main__":
    main()
