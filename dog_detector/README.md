# Dog Detection with YOLOv8n

Simple Python script for detecting dogs in images using the YOLOv8n model trained on COCO dataset.

## Requirements

Python 3.8 or newer is required. If you have a CUDA-compatible GPU, inference will be faster.

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

The YOLOv8n model (about 6 MB) downloads automatically on first run.

## Usage

Basic usage:

```bash
python dog_detector.py --input dog.jpg
```

The script opens a window showing the detected dogs. Press any key to close it.

To save the result:

```bash
python dog_detector.py --input dog.jpg --output result.jpg
```

Change detection sensitivity:

```bash
python dog_detector.py --input dog.jpg --confidence 0.7
```

Default confidence threshold is 0.5. Lower values detect more objects but increase false positives.

## Command Line Arguments

`--input` - Path to the input image (required)

`--output` - Path to save the annotated image (optional)

`--confidence` - Detection confidence threshold between 0 and 1 (default: 0.5)

## Implementation Details

The script uses YOLOv8n, the smallest and fastest variant of YOLOv8. It's pre-trained on the COCO dataset which includes 80 object classes. Dogs are class ID 16 in this dataset. The detector filters out all other classes and only reports dog detections.

## Notes

First run requires internet connection to download the model weights. The script displays results in a window using OpenCV. If you're running this on a headless server, you'll need to use the `--output` option instead of relying on the display functionality.
