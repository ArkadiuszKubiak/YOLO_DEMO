"""
Dog Detection using YOLOv8n
This script uses Ultralytics YOLOv8n model pre-trained on COCO dataset
to detect dogs in images.
"""

from ultralytics import YOLO
import cv2
import argparse
import numpy as np


class DogDetector:
    """
    A class to detect dogs using YOLOv8n model trained on COCO dataset.
    In COCO dataset, dog class has ID 16.
    """
    
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the dog detector with YOLOv8n model.
        
        Args:
            model_name (str): Name of the YOLO model to use (default: 'yolov8n.pt')
            confidence_threshold (float): Minimum confidence score for detection (default: 0.5)
        """
        print(f"Loading {model_name} model...")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        # Dog class ID in COCO dataset is 16
        self.dog_class_id = 16
        print("Model loaded successfully!")
    
    def detect_dogs_in_image(self, image_path, save_path=None):
        """
        Detect dogs in a single image.
        
        Args:
            image_path (str): Path to the input image
            save_path (str, optional): Path to save the annotated image
            
        Returns:
            list: List of detection results for dogs
        """
        print(f"\nProcessing image: {image_path}")
        
        # Run inference on the image
        results = self.model(image_path, conf=self.confidence_threshold)
        
        # Filter results to only include dogs (class 16)
        dog_detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if detected class is dog
                if int(box.cls) == self.dog_class_id:
                    dog_detections.append({
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                        'class_name': 'dog'
                    })
        
        # Print detection results
        if dog_detections:
            print(f"Found {len(dog_detections)} dog(s) in the image!")
            for i, detection in enumerate(dog_detections, 1):
                print(f"  Dog {i}: Confidence = {detection['confidence']:.2f}")
        else:
            print("No dogs detected in the image.")
        
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
        for i, detection in enumerate(dog_detections):
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, detection['bbox'])
            
            # Select color (cycle through colors if more detections than colors)
            color = colors[i % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"dog {detection['confidence']:.2f}"
            
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
            cv2.imshow('Dog Detection - Press any key to close', annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("Note: Image display not available (headless system). Check the saved file instead.")
        
        return dog_detections


def main():
    """
    Main function to handle command-line arguments and run dog detection.
    """
    parser = argparse.ArgumentParser(
        description='Detect dogs in images using YOLOv8n model trained on COCO dataset'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image file'
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
    
    # Initialize dog detector
    detector = DogDetector(confidence_threshold=args.confidence)
    
    # Run detection on image
    detector.detect_dogs_in_image(args.input, args.output)


if __name__ == '__main__':
    main()
