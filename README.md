# ECoT-Franka-eval

\section*{1. 환경 설정}
\begin{lstlisting}[language=bash]
conda create -n openvla python=3.10 -y
conda activate openvla

PyTorch 설치 (GPU 환경에 맞게 수정 필요)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

OpenVLA 설치
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# FlashAttention 2 설치 (선택)
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
\end{lstlisting}

\section*{2. 모델 추출}
\begin{lstlisting}[language=bash]
python extract_model.py \
  --src_ckpt /path/to/source/weights \
  --dst_ckpt ckpt/openvla-7b-ecot \
  --dtype bfloat16
\end{lstlisting}

\section*{3. Evaluation (ROS2-based)}
\begin{lstlisting}[language=bash]
python eval_ros.py \
  --model ckpt/openvla-7b-ecot \
  --task "pick pink cup and place on blue square" \
  --camera_topic /camera/color/image_raw \
  --depth_topic /camera/depth/image_raw \
  --robot_ip 192.168.1.100 \
  --hz 5
\end{lstlisting}
