# YOLO Projects

This workspace contains two separate YOLO detection projects.

## Requirements

- Python 3.8+
- CUDA-compatible GPU (NVIDIA) for training - highly recommended for Pokemon training
- CPU can be used but training will be significantly slower (2-4 hours vs 15-30 minutes)

## 1. Dog Detector

Located in `dog_detector/` folder.

Simple dog detection using pre-trained YOLOv8n model on COCO dataset.

**Quick start:**
```bash
cd dog_detector
pip install -r requirements.txt
python dog_detector.py --input dog.jpg
```

See `dog_detector/README.md` for details.

## 2. Pokemon Detector

Located in `pokemon_detector/` folder.

Custom Pokemon detection with model training capabilities.

**Quick start:**
```bash
cd pokemon_detector
pip install -r requirements.txt
python download_pokemon_dataset.py
python train_pokemon.py
python pokemon_detector.py --input your_image.jpg
```

See `pokemon_detector/POKEMON_TRAINING.md` for full training guide.

## Using VS Code Task Runner

This project includes pre-configured tasks for easy execution. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and type "Tasks: Run Task" to see available options:

**Dog Detector:**
- Dog: Run on image
- Dog: Run on image (Low confidence 0.3)
- Dog: Install Dependencies

**Pokemon Detector:**
- Pokemon: Download Dataset
- Pokemon: Train (50 epochs)
- Pokemon: Train (Quick 10 epochs)
- Pokemon: Detect on image
- Pokemon: Detect on image (Low confidence 0.3)
- Pokemon: Install Dependencies
- Pokemon: View Training Results

## Project Structure

```
YOLO/
├── dog_detector/
│   ├── dog_detector.py
│   ├── dog.jpg
│   ├── requirements.txt
│   └── README.md
│
├── pokemon_detector/
│   ├── download_pokemon_dataset.py
│   ├── train_pokemon.py
│   ├── pokemon_detector.py
│   ├── requirements.txt
│   └── POKEMON_TRAINING.md
│
└── .vscode/
    └── tasks.json          # Pre-configured VS Code tasks
```
