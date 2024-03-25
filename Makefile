.PHONY: reqs reqs-cuda

## Install Python Dependencies
reqs:
	@pip3 install -r requirements.txt

## Install Python Dependencies for CUDA
reqs-cuda :
	@pip3 install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	@pip3 install -r requirements-cuda.txt
