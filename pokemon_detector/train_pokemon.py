"""
Train YOLOv8 Model on Pokemon Dataset
This script trains a YOLOv8 model to detect Pokemon characters.
"""

from ultralytics import YOLO
import argparse
import os


def train_pokemon_detector(data_yaml, epochs=50, img_size=640, batch_size=16, model_size='n'):
    """
    Train YOLOv8 model on Pokemon dataset.
    
    Args:
        data_yaml (str): Path to data.yaml file from downloaded dataset
        epochs (int): Number of training epochs (default: 50)
        img_size (int): Image size for training (default: 640)
        batch_size (int): Batch size (default: 16, reduce if out of memory)
        model_size (str): Model size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
    """
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        print(f"Error: data.yaml not found at {data_yaml}")
        print("\nPlease download the dataset first:")
        print("  python download_pokemon_dataset.py")
        return
    
    # Initialize model
    model_name = f'yolov8{model_size}.pt'
    print(f"\nInitializing YOLOv8{model_size.upper()} model...")
    model = YOLO(model_name)
    
    print(f"\nStarting training with parameters:")
    print(f"  Model: YOLOv8{model_size.upper()}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset: {data_yaml}")
    print("\n" + "="*60)
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='pokemon_detector',
        patience=10,  # Early stopping patience
        save=True,
        plots=True,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"\nResults saved to: runs/detect/pokemon_detector")
    print(f"Best model: runs/detect/pokemon_detector/weights/best.pt")
    print(f"Last model: runs/detect/pokemon_detector/weights/last.pt")
    print("\nTraining metrics and plots saved in the same directory.")
    print("\nTo use your trained model for detection:")
    print("  python pokemon_detector.py --model runs/detect/pokemon_detector/weights/best.pt --input your_image.jpg")
    
    return results


def main():
    """
    Main function to handle command-line arguments and start training.
    """
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 model on Pokemon detection dataset'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='pokemon-2/data.yaml',
        help='Path to data.yaml file (default: pokemon-2/data.yaml)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Image size for training (default: 640)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: 16, reduce if out of memory)'
    )
    
    parser.add_argument(
        '--model-size',
        type=str,
        choices=['n', 's', 'm', 'l', 'x'],
        default='n',
        help='Model size: n (nano), s (small), m (medium), l (large), x (xlarge). Default: n'
    )
    
    args = parser.parse_args()
    
    # Start training
    train_pokemon_detector(
        data_yaml=args.data,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        model_size=args.model_size
    )


if __name__ == '__main__':
    main()
