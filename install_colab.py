"""
Installation script for Google Colab
Run this script first to install all dependencies
"""

import subprocess
import sys

def install_packages():
    """Install required packages for HDM Quadratic"""
    packages = [
        'einops',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'tensorboard',
        'tensorly',
        'tensorly-torch',
        'tqdm',
        'transformers',
        'scikit-learn',  # For train_test_split in case needed
    ]
    
    print("Installing required packages...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\nAll packages installed successfully!")
    print("You can now run the model with:")
    print("  python main.py --config hdm_quadratic_fno.yml --doc quadratic_experiment")

if __name__ == "__main__":
    install_packages()