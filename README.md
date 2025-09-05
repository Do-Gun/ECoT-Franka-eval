# ECoT-Franka-eval

\section*{ECoT-Franka-eval}

OpenVLA 기반 정책에 Embodied Chain-of-Thought(ECoT) 미세튜닝을 적용하고 Franka 환경에서 평가하는 스크립트 모음입니다.  

\subsection*{1. 환경 설정}

\begin{lstlisting}[language=bash]
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch (본인 GPU/OS에 맞게 조정 권장)
# 참고: https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
\end{lstlisting}

\subsubsection*{OpenVLA 설치}
\begin{lstlisting}[language=bash]
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .
cd ..
\end{lstlisting}

\subsubsection*{FlashAttention 2 설치}
\begin{lstlisting}[language=bash]
pip install packaging ninja
ninja --version; echo $?   # 0 이면 정상
pip install "flash-attn==2.5.5" --no-build-isolation
\end{lstlisting}

\subsection*{2. 모델 준비 (OpenVLA + ECoT Fintuning)}

\begin{lstlisting}[language=bash]
# 예시: 모델 추출/정리
python extract_model.py \
  --src_ckpt /path/to/source/weights \
  --dst_ckpt ckpt/openvla-7b-ecot \
  --dtype bfloat16
\end{lstlisting}

\subsection*{3. 평가 실행}

\begin{lstlisting}[language=bash]
# ROS 기반 평가
python eval_ros.py \
  --model ckpt/openvla-7b-ecot \
  --task "pick pink cup and place on blue square" \
  --camera_topic /camera/color/image_raw \
  --depth_topic /camera/depth/image_raw \
  --robot_ip 192.168.1.100 \
  --hz 5

# 오프라인 로그 평가
python eval_ros.py \
  --model ckpt/openvla-7b-ecot \
  --dataset ./datasets/bridge_samples \
  --num_episodes 50 \
  --save_metrics ./outputs/metrics.json
\end{lstlisting}

\subsection*{4. 권장 스펙}
\begin{itemize}
  \item GPU: 24GB VRAM 이상 권장 (7B 모델, bfloat16/float16)
  \item CUDA 12.x + 최신 NVIDIA 드라이버
  \item Python 3.10, PyTorch 최신 버전
\end{itemize}

\subsection*{5. 참고}
\begin{itemize}
  \item 대용량 체크포인트/데이터는 Git에 올리지 말고 외부 스토리지(Hugging Face Hub, Google Drive 등)에 보관하세요.
  \item FlashAttention 설치 문제 시: \texttt{pip cache remove flash\_attn} 후 재시도, 또는 해당 모듈 없이 실행.
\end{itemize}

