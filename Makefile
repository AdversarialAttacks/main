 lam.PHONY: help reqs reqs-cuda dl-covid dl-brain-tumor lambda-setup

## Show this help message
help:
	@echo ""
	@echo "Usage: make [target]"
	@echo "---"
	@echo "reqs"
	@echo "		Install Python Dependencies"
	@echo "reqs-cuda"
	@echo "		Install Python Dependencies with CUDA"
	@echo "dl-covid"
	@echo "		Download COVIDx-CXR4 Dataset"
	@echo "dl-brain-tumor"
	@echo "		Download Brain Tumor MRI Dataset"
	@echo "download_models"
	@echo "		Download Models from Weights and Biases"
	@echo "lambda-setup"
	@echo "		Setup Lambda Cloud for Training"
	@echo ""

## Install Python Dependencies
reqs:
	pip3 install -q -r requirements.txt

## Install Python Dependencies for Windows
reqs-cuda:
	pip3 install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	pip3 install -r requirements.txt

## Download COVIDx-CXR4 Dataset
dl-covid:
	kaggle datasets download -d andyczhao/covidx-cxr2 -p data/raw/COVIDX-CXR4 --unzip

## Download Brain Tumor MRI Dataset
dl-brain-tumor:
	kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri -p data/raw/Brain-Tumor-MRI --unzip 

## Download Models from Weights and Biases
download_models:
	PYTHONPATH=/home/ubuntu/main
	python3 src/utils/download.py

## Setup Lambda Cloud for Training 
## git clone https://github.com/AdversarialAttacks/main 
## cd main
## Drop .kaggle folder in the root of the project
lambda-setup:
	mkdir ../.kaggle
	mv kaggle.json ../.kaggle/kaggle.json
	make reqs
	make dl-brain-tumor
	make dl-covid
	make download_models

# Set the default goal to `help`
.DEFAULT_GOAL := help