"""
Download Pokemon Detection Dataset from Roboflow
This script downloads a Pokemon dataset in YOLO format for training.
"""

from roboflow import Roboflow
import os


def download_pokemon_dataset(api_key=None):
    """
    Download Pokemon dataset from Roboflow Universe.
    
    Args:
        api_key (str): Your Roboflow API key. Get it from roboflow.com (free account)
    """
    
    if api_key is None:
        print("\n" + "="*60)
        print("ROBOFLOW API KEY REQUIRED")
        print("="*60)
        print("\nTo download the dataset, you need a free Roboflow account:")
        print("1. Go to: https://roboflow.com")
        print("2. Sign up for free")
        print("3. Go to your account settings")
        print("4. Copy your API key")
        print("\nThen run this script with your API key:")
        print("python download_pokemon_dataset.py")
        print("\nOr set it in this script directly (line 35)")
        print("="*60 + "\n")
        
        # Prompt user for API key
        api_key = input("Enter your Roboflow API key (or press Enter to exit): ").strip()
        
        if not api_key:
            print("No API key provided. Exiting.")
            return
    
    print(f"\nInitializing Roboflow with API key...")
    
    try:
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        
        # Access Pokemon dataset from Roboflow Universe
        # This is a public Pokemon detection dataset with 1.1k images
        print("Accessing Pokemon dataset from Universe...")
        print("Dataset: Pokemon Object Detection (1.1k images)")
        
        project = rf.workspace("pokemon-h5b6t").project("pokemon-iqf8o")
        dataset = project.version(2).download("yolov8")
        
        print(f"\n✓ Dataset downloaded successfully!")
        print(f"✓ Location: {dataset.location}")
        print(f"\nDataset structure:")
        print(f"  - train/: Training images and labels")
        print(f"  - valid/: Validation images and labels")
        print(f"  - test/: Test images and labels")
        print(f"  - data.yaml: Dataset configuration file")
        print(f"\nYou can now train the model using:")
        print(f"  python train_pokemon.py")
        
        return dataset
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your API key is correct")
        print("2. Check your internet connection")
        print("3. Visit roboflow.com to verify your account")
        print("4. Try browsing universe.roboflow.com to find alternative Pokemon datasets")
        return None


if __name__ == '__main__':
    # Option 1: Set your API key here directly
    API_KEY = None  # Replace with your API key like: "your_api_key_here"
    
    # Option 2: Or set it as environment variable
    if API_KEY is None:
        API_KEY = os.environ.get('ROBOFLOW_API_KEY')
    
    download_pokemon_dataset(API_KEY)
