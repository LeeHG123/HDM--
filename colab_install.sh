#!/bin/bash
# Installation script for Google Colab

echo "Installing required packages for HDM Quadratic..."

pip install einops
pip install matplotlib
pip install numpy
pip install pandas
pip install scipy
pip install tensorboard
pip install tensorly
pip install tensorly-torch
pip install tqdm
pip install transformers
pip install scikit-learn
pip install glob2

echo ""
echo "Installation complete!"
echo ""
echo "You can now run the model with:"
echo "  python main.py --config hdm_quadratic_fno.yml --doc quadratic_experiment"
echo ""
echo "To generate samples:"
echo "  python main.py --config hdm_quadratic_fno.yml --doc quadratic_experiment --sample"