# ECoT-Franka-eval

## 1. 환경 설정
```bash
# Conda 환경 생성
conda create -n openvla python=3.10 -y
conda activate openvla

# PyTorch 설치 (GPU 환경에 맞게 수정 필요)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# OpenVLA 설치
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# FlashAttention 2 설치 (선택)
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

## 2. 모델 추출
