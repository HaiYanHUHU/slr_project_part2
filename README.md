# Sign Language Recognition Project

## project brief

This project implements a lightweight sign language recognition model.
The architecture design uses MobileNetV3 + BiLSTM + Attention.
Lightweight feature extraction is conducted through MobileNetV3.
BiLSTM is used for temporal modeling, and the Attention mechanism focuses on key frames to achieve the recognition of WLASL_100 sign language vocabulary.

## project structure


### 1. npm install

pip install -r requirements.txt


### 2. Verify installation

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"


## 3. Dataset
slr_project/
       ├── data
            └── WLASL_100
                └── test/
                └── train/
                └── validation/
            └── frames
                └── test/
                └── train/
                └── validation/