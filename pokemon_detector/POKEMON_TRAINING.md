# Pokemon Detection Training Guide

This guide shows you how to train a custom YOLOv8 model to detect Pokemon characters.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download Pokemon dataset:
```bash
python download_pokemon_dataset.py
```
You'll need a free Roboflow account. Sign up at roboflow.com and get your API key.

3. Train the model:
```bash
python train_pokemon.py
```

4. Test your trained model:
```bash
python pokemon_detector.py --input pokemon_image.jpg
```

## Detailed Steps

### Step 1: Get Roboflow API Key

- Go to https://roboflow.com
- Create a free account
- Navigate to Settings → API
- Copy your API key

### Step 2: Download Dataset

Run the download script and enter your API key when prompted:

```bash
python download_pokemon_dataset.py
```

This downloads a Pokemon classification dataset with images and labels in YOLO format. The dataset includes multiple Pokemon species with bounding box annotations.

### Step 3: Train the Model

Basic training (recommended for quick testing):
```bash
python train_pokemon.py
```

This trains for 50 epochs with YOLOv8n (nano model). Training takes 10-30 minutes on a decent GPU.

Custom training parameters:
```bash
python train_pokemon.py --epochs 100 --batch-size 8 --model-size s
```

Parameters:
- `--data`: Path to data.yaml (default: pokemon-classification-1/data.yaml)
- `--epochs`: Training epochs (default: 50)
- `--img-size`: Image size (default: 640)
- `--batch-size`: Batch size (default: 16, reduce to 8 or 4 if out of memory)
- `--model-size`: n (fastest), s, m, l, x (most accurate)

Training results are saved to `runs/detect/pokemon_detector/`

### Step 4: Use Your Trained Model

Detect Pokemon in images:
```bash
python pokemon_detector.py --input pokemon.jpg
```

Save results:
```bash
python pokemon_detector.py --input pokemon.jpg --output result.jpg
```

Use different confidence threshold:
```bash
python pokemon_detector.py --input pokemon.jpg --confidence 0.7
```

Use specific model checkpoint:
```bash
python pokemon_detector.py --input pokemon.jpg --model runs/detect/pokemon_detector/weights/best.pt
```

## Training Tips

**Out of memory errors:**
Reduce batch size:
```bash
python train_pokemon.py --batch-size 8
```

**Faster training:**
Use smaller model and fewer epochs:
```bash
python train_pokemon.py --epochs 30 --model-size n
```

**Better accuracy:**
Train longer with larger model:
```bash
python train_pokemon.py --epochs 100 --model-size s
```

**Monitor training:**
Training metrics and plots are automatically saved to `runs/detect/pokemon_detector/`:
- confusion_matrix.png - Shows classification accuracy
- results.png - Training curves (loss, precision, recall, mAP)
- val_batch*.jpg - Validation predictions

## Expected Results

After 50 epochs with YOLOv8n, you should see:
- mAP@0.5: 0.70-0.85 (70-85% accuracy)
- Training time: 15-30 minutes on GPU, 2-4 hours on CPU

Higher mAP means better detection accuracy.

## Project Files

```
YOLO/
├── download_pokemon_dataset.py   # Dataset download script
├── train_pokemon.py              # Training script
├── pokemon_detector.py           # Detection script for trained model
├── dog_detector.py               # Original dog detection (COCO)
├── requirements.txt              # Python dependencies
└── runs/
    └── detect/
        └── pokemon_detector/     # Training results
            └── weights/
                ├── best.pt       # Best model checkpoint
                └── last.pt       # Last epoch checkpoint
```

## Troubleshooting

**Dataset download fails:**
- Verify your API key is correct
- Check internet connection
- Make sure Roboflow account is active

**Training is very slow:**
- Check if GPU is being used (CUDA)
- Reduce batch size
- Use smaller model (nano instead of small)

**Poor detection results:**
- Train for more epochs
- Use larger model size
- Adjust confidence threshold
- Check if test images match training data style