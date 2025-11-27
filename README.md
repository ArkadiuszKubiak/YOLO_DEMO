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

## Training Results Explained

After training the Pokemon detector, results are saved to `pokemon_detector/runs/detect/pokemon_detector/`. Here's what each file means:

### Model Weights (weights/)
- **best.pt** - The best performing model during training based on validation mAP. **USE THIS FILE** for detection after training.
- **last.pt** - Model from the final training epoch. May not be the best performing. Use only if you want to continue training.

### Main Performance Dashboard
- **results.png** - Multi-panel chart showing how your model improved during training:
  - **train/box_loss** - How well model predicts bounding boxes (lower is better)
  - **train/cls_loss** - How well model classifies Pokemon types (lower is better)
  - **train/dfl_loss** - Distribution focal loss for box regression (lower is better)
  - **metrics/precision** - Percentage of correct predictions out of all predictions (higher is better)
  - **metrics/recall** - Percentage of Pokemon the model found out of all Pokemon in images (higher is better)
  - **metrics/mAP50** - Overall accuracy at 50% IoU threshold (higher is better, > 0.70 is good)
  - **metrics/mAP50-95** - Stricter accuracy metric across multiple thresholds (higher is better)

- **results.csv** - Same metrics as results.png but in spreadsheet format. Useful for creating custom charts.

### Precision-Recall Curves (How Good is the Model?)
- **BoxPR_curve.png** - Shows the trade-off between precision and recall:
  - **High curve** = model is good at finding Pokemon AND being correct
  - **Low curve** = model struggles with accuracy or detection
  - **Area under curve (AUC)** close to 1.0 is excellent

- **BoxP_curve.png** - Precision at different confidence thresholds:
  - Shows: if you set `--confidence 0.5`, what precision do you get?
  - **High precision** = fewer false positives (model doesn't detect things that aren't Pokemon)
  
- **BoxR_curve.png** - Recall at different confidence thresholds:
  - Shows: if you set `--confidence 0.5`, what recall do you get?
  - **High recall** = model finds most Pokemon in the image (fewer missed detections)

- **BoxF1_curve.png** - F1 score (balance between precision and recall):
  - **Peak of the curve** = optimal confidence threshold to use
  - If peak is at 0.4, use `--confidence 0.4` for best results

### Confusion Matrix (Which Pokemon Get Mixed Up?)
- **confusion_matrix.png** - Grid showing actual vs predicted classes:
  - **Diagonal cells** (top-left to bottom-right) = correct predictions (should be dark/high)
  - **Off-diagonal cells** = mistakes (should be light/low)
  - Example: if Pikachu row has high value in Raichu column, model confuses Pikachu with Raichu

- **confusion_matrix_normalized.png** - Same as above but shown as percentages (0-100%):
  - Easier to see which classes have the most problems
  - **100% on diagonal** = perfect classification for that Pokemon

### Visual Training Samples
- **train_batch0.jpg, train_batch1.jpg, train_batch2.jpg** - Sample images from your training dataset:
  - Shows what the model learned from (with bounding boxes)
  - Useful to verify dataset quality

- **train_batch1760.jpg, train_batch1761.jpg, train_batch1762.jpg** - Training samples from final epochs:
  - Shows later training examples
  - Can compare with early batches to see data augmentation variations

- **val_batch0_labels.jpg, val_batch1_labels.jpg, val_batch2_labels.jpg** - Validation images with **ground truth** labels:
  - Shows what the **correct** answer should be
  - Green boxes = actual Pokemon locations in the image

- **val_batch0_pred.jpg, val_batch1_pred.jpg, val_batch2_pred.jpg** - Validation images with **model predictions**:
  - Shows what the **model detected**
  - Compare with *_labels.jpg to see if model is accurate
  - Confidence scores shown on each box

- **labels.jpg** - Statistical analysis of your dataset:
  - **Class distribution** - How many of each Pokemon type (should be balanced)
  - **Box size distribution** - Sizes of bounding boxes (helps verify data quality)
  - **Box location heatmap** - Where Pokemon appear in images (should be spread out)

### What Good Training Looks Like
- **mAP@0.5** > 0.70 (70% overall accuracy) - Your model correctly detects 7 out of 10 Pokemon
- **Precision** > 0.80 (80%) - When model says "this is a Pikachu", it's right 8 out of 10 times
- **Recall** > 0.80 (80%) - Model finds 8 out of 10 Pokemon that are actually in the image
- **Loss curves** go down smoothly and flatten out (not zigzagging up and down)
- **Confusion matrix** has bright diagonal and dark off-diagonal cells
- **val_batch predictions** match closely with labels (boxes overlap well)

### Warning Signs
- **mAP < 0.50** - Model needs more training or better data
- **Precision high, Recall low** - Model is too cautious, increase training epochs or lower confidence
- **Recall high, Precision low** - Model detects too many false positives, need more training data
- **Loss curves still decreasing** - Model could improve with more epochs
- **Confusion matrix scattered** - Some Pokemon classes are too similar or mislabeled in dataset

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
│   ├── POKEMON_TRAINING.md
│   └── runs/
│       └── detect/
│           └── pokemon_detector/    # Training results here
│
└── .vscode/
    └── tasks.json          # Pre-configured VS Code tasks
```
