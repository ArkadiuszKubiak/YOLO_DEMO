"""
Pokemon Detection using Custom Trained YOLOv8
This script uses a custom trained YOLOv8 model to detect Pokemon characters in images.
"""

from ultralytics import YOLO
import cv2
import argparse
import os
import numpy as np
import random


class PokemonDetector:
    """
    A class to detect Pokemon using a custom trained YOLOv8 model.
    """
    
    def __init__(self, model_path='runs/detect/pokemon_detector/weights/best.pt', confidence_threshold=0.5):
        """
        Initialize the Pokemon detector with custom trained model.
        
        Args:
            model_path (str): Path to the trained model weights
            confidence_threshold (float): Minimum confidence score for detection (default: 0.5)
        """
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("\nPlease train the model first:")
            print("  1. Download dataset: python download_pokemon_dataset.py")
            print("  2. Train model: python train_pokemon.py")
            print("\nOr specify a different model path with --model argument")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        print("Model loaded successfully!")
    
    def detect_pokemon_in_image(self, image_path, save_path=None):
        """
        Detect Pokemon in a single image.
        
        Args:
            image_path (str): Path to the input image
            save_path (str, optional): Path to save the annotated image
            
        Returns:
            list: List of detection results for Pokemon
        """
        print(f"\nProcessing image: {image_path}")
        
        # Run inference on the image
        results = self.model(image_path, conf=self.confidence_threshold)
        
        # Collect all detections
        pokemon_detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                class_name = result.names[class_id]
                
                pokemon_detections.append({
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    'class_name': class_name,
                    'class_id': class_id
                })
        
        # Print detection results
        if pokemon_detections:
            print(f"Found {len(pokemon_detections)} Pokemon in the image!")
            for i, detection in enumerate(pokemon_detections, 1):
                print(f"  {i}. {detection['class_name']}: Confidence = {detection['confidence']:.2f}")
        else:
            print("No Pokemon detected in the image.")
        
        # Load original image for custom drawing
        annotated_frame = cv2.imread(image_path)
        
        # Define colors (excluding white) - BGR format
        colors = [
            (255, 0, 0),      # Blue
            (0, 255, 0),      # Green
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 0, 128),    # Purple
            (255, 165, 0),    # Orange
            (0, 128, 255),    # Light Blue
            (255, 192, 203),  # Pink
        ]
        
        # Draw custom bounding boxes with colors
        for i, detection in enumerate(pokemon_detections):
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, detection['bbox'])
            
            # Select color (cycle through colors if more detections than colors)
            color = colors[i % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA
            )
        
        # Save annotated image only if output path is specified
        if save_path:
            cv2.imwrite(save_path, annotated_frame)
            print(f"Annotated image saved to: {save_path}")
        
        # Try to display the image (may not work on headless systems)
        try:
            cv2.imshow('Pokemon Detection - Press any key to close', annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("Note: Image display not available (headless system). Check the saved file instead.")
        
        return pokemon_detections


def main():
    """
    Main function to handle command-line arguments and run Pokemon detection.
    """
    parser = argparse.ArgumentParser(
        description='Detect Pokemon in images using custom trained YOLOv8 model'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='runs/detect/pokemon_detector/weights/best.pt',
        help='Path to trained model weights (default: runs/detect/pokemon_detector/weights/best.pt)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save the annotated image'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for detection (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Initialize Pokemon detector
    try:
        detector = PokemonDetector(
            model_path=args.model,
            confidence_threshold=args.confidence
        )
        
        # Run detection on image
        detector.detect_pokemon_in_image(args.input, args.output)
        
    except FileNotFoundError as e:
        print(f"\n{e}")


if __name__ == '__main__':
    main()
