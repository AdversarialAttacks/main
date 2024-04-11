 lam.PHONY: help reqs dl-brain-tumor dl-covid lambda-setup

## Show this help message
help:
	@echo ""
	@echo "Usage: make [target]"
	@echo "---"
	@echo "reqs"
	@echo "		Install Python Dependencies"
	@echo "dl-covid"
	@echo "		Download COVIDx-CXR4 Dataset"
	@echo "dl-brain-tumor"
	@echo "		Download Brain Tumor MRI Dataset"
	@echo "lambda-setup"
	@echo "		Setup Lambda Cloud for Training"
	@echo ""

## Install Python Dependencies
reqs:
	pip3 install -r requirements.txt --force-reinstall

## Install Python Dependencies for Windows
reqs-cuda-windows:
	pip3 install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	pip3 install -r requirements.txt

## Download COVIDx-CXR4 Dataset
dl-covid:
	kaggle datasets download -d andyczhao/covidx-cxr2 -p data/raw/COVIDX-CXR4 --unzip

## Download Brain Tumor MRI Dataset
dl-brain-tumor:
	kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri -p data/raw/Brain-Tumor-MRI --unzip 

## Setup Lambda Cloud for Training
lambda-setup:
	mkdir ../.kaggle
	mv kaggle.json ../.kaggle/kaggle.json
	make reqs
	make dl-brain-tumor
	make dl-covid

# Set the default goal to `help`
.DEFAULT_GOAL := help