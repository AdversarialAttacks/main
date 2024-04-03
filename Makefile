.PHONY: help reqs reqs-cuda dl-brain-tumor dl-covid

## Show this help message
help:
	@echo ""
	@echo "Usage: make [target]"
	@echo "---"
	@echo "reqs"
	@echo "		Install Python Dependencies"
	@echo "reqs-cuda"
	@echo "		Install Python Dependencies for CUDA"
	@echo "dl-covid"
	@echo "		Download COVIDx-CXR4 Dataset"
	@echo "dl-brain-tumor"
	@echo "		Download Brain Tumor MRI Dataset"
	@echo ""

## Install Python Dependencies
reqs:
	pip3 install -r requirements.txt

## Install Python Dependencies for CUDA
reqs-cuda:
	pip3 install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	pip3 install -r requirements-gpu.txt

## Download COVIDx-CXR4 Dataset
dl-covid:
	kaggle datasets download -d andyczhao/covidx-cxr2 -p data/raw/COVIDX-CXR4 --unzip

## Download Brain Tumor MRI Dataset
dl-brain-tumor:
	kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri -p data/raw/Brain-Tumor-MRI --unzip 

# Set the default goal to `help`
.DEFAULT_GOAL := help