# AI Image Upscaler (Web Interface)

A local web interface for upscaling images using AI models.

![image](https://github.com/user-attachments/assets/373b2866-a4d8-4972-8caa-2337f19a2424)

## Features

- Supports AI upscaling models through [spandrel](https://github.com/chaiNNer-org/spandrel/)
- Supports both .pth and .safetensors model formats
- Image resize functionality with various scaling methods
- Image information viewer
- Configurable through web interface
- Batch Upscaling

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU
- nvidia-smi
- [imagemagick](https://imagemagick.org/script/download.php)

## Installation

1. Clone this repository
2. Install PyTorch CUDA from https://pytorch.org/get-started/locally/
3. Install [imagemagick](https://imagemagick.org/script/download.php) and other requirements:
   ```
   pip install -r requirements.txt
   ```
4. Place your model files in the models directory

## Usage

1. Start the application:
   ```
   python web-upscaler.py
   ```

2. Open in browser:
   ```
   http://localhost:5000
   ```

## Configuration

Adjust settings through the web interface or `config.ini`:
- Model directory path
- VRAM usage limits
- Processing settings
- Default options

## Acknowledgements

- Uses [Spandrel](https://github.com/chaiNNer-org/spandrel) for model loading
- VRAM usage data from [@the-database](https://github.com/the-database)
