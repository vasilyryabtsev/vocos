# Vocos

В данном репозитории реализуется вокодер Vocos (2024). Архитектура модели взята из одноименной статьи [VOCOS: CLOSING THE GAP BETWEEN TIME-DOMAIN
AND FOURIER-BASED NEURAL VOCODERS FOR HIGH-
QUALITY AUDIO SYNTHESIS](https://arxiv.org/pdf/2306.00814) и оригинальной имплементации статьи https://github.com/gemelo-ai/vocos.

---

## Установка

### Требования

- Python 3.12
- CUDA-совместимый GPU (рекомендуется) или CPU
- `venv`

> **Предупреждение:** Проект стабильно протестирован и запускается на **Python 3.12**. Работа на других версиях не гарантируется.

### Шаг 1. Клонирование репозитория

```bash
git clone https://github.com/vasilyryabtsev/vocos.git
cd vocos
```

### Шаг 2. Создание виртуального окружения

```bash
python3 -m venv .venv
```

### Шаг 3. Активация окружения

```bash
source .venv/bin/activate
```

### Шаг 4. Установка зависимостей

```bash
pip install -r requirements.txt
```

> **Примечание для GPU.** В `requirements.txt` уже прописаны CUDA 12.4 пакеты для `torch==2.5.1`. Если у вас другая версия CUDA или вы хотите CPU-only сборку, установите torch вручную перед `pip install -r requirements.txt`:
> ```bash
> # CPU only
> pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
> ```

### Альтернатива: запуск через uv (рекомендуется)

Если у вас установлен [`uv`](https://github.com/astral-sh/uv), можно не создавать окружение вручную.

**Синхронизация зависимостей с Python 3.12:**
```bash
uv sync --python 3.12
```

**Запуск скриптов:**
```bash
uv run --python 3.12 python3 train.py -cn=vocos_onebatchtest
uv run --python 3.12 python3 train.py -cn=vocos
uv run --python 3.12 python3 inference.py -cn=vocos_inference
```

---

## Обучение

### Подготовка данных

По умолчанию используется датасет [RUSLAN](https://ruslan-corpus.github.io/). Разместите файлы согласно структуре:

```
data/
├── ruslan/
│   └── RUSLAN/          # wav-файлы
└── metadata_RUSLAN_22200.csv
```

Конфигурацию датасета можно переопределить через [src/configs/datasets/](src/configs/datasets/).

### Запуск обучения

**Быстрый тест (один батч):**
```bash
python3 train.py -cn=vocos_onebatchtest
```

**Полное обучение:**
```bash
python3 train.py -cn=vocos
```

**Переопределение параметров конфига:**
```bash
python3 train.py -cn=vocos trainer.n_epochs=200 optimizer.lr=1e-4
```

Чекпоинты сохраняются в `saved/`. Трекинг экспериментов настраивается через [src/configs/writer/](src/configs/writer/) (Comet ML или W&B).

---

## Инференс

### Загрузка предобученных весов

```bash
python3 run.py
```

Веса будут скачаны в `pretrained/`. Если файл уже существует, повторная загрузка не производится.

Также можно указать путь вручную:
```bash
python3 run.py --checkpoint /path/to/dir
```

### Запуск инференса

```bash
python3 inference.py -cn=vocos_inference \
    inferencer.from_pretrained=pretrained/<checkpoint>.pth \
    datasets.inference.data_dir=<path/to/audio/dir>
```

Результаты сохраняются в `inference_output/`.

**Параметры:**
| Параметр | По умолчанию | Описание |
|---|---|---|
| `inferencer.from_pretrained` | `null` | Путь к чекпоинту `.pth` |
| `inferencer.device` | `cpu` | `cpu`, `cuda`, или `auto` |
| `inferencer.save_path` | `inference_output` | Папка для результатов |
| `datasets.inference.data_dir` | обязательный | Директория с входными wav-файлами |

---

## Ресинтез аудио

`synthesize.py` принимает директорию с аудиофайлами, прогоняет их через модель (аудио → мел-спектрограмма → аудио) и сохраняет результат.

### Запуск

```bash
python3 synthesize.py -cn=synthesize \
    data_dir=<path/to/data> \
    checkpoint_path=<path/to/checkpoint.pth>
```

Скрипт ожидает аудиофайлы в поддиректории `audio/` внутри `data_dir` (поддерживаются `.wav`, `.mp3`, `.flac`, `.m4a`). Результаты сохраняются в `synthesized/`.

**Параметры:**
| Параметр | По умолчанию | Описание |
|---|---|---|
| `data_dir` | обязательный | Директория с поддиректорией `audio/` |
| `checkpoint_path` | обязательный | Путь к чекпоинту `.pth` |
| `output_dir` | `synthesized` | Папка для сохранения результатов |
| `device` | `auto` | `cpu`, `cuda`, или `auto` |
| `resynthesize` | `true` | Запустить ресинтез аудиофайлов |
