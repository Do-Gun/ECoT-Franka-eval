import torch # Import PyTorch library
from transformers import AutoModelForVision2Seq, AutoProcessor # Import model and processor classes from Hugging Face Transformers
import os # Import os module for operating system functionalities

# 1. Set the device for loading the model and processor
# Use CUDA if available, otherwise fall back to CPU.
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("경고: CUDA를 사용할 수 없습니다. CPU를 사용합니다. 성능이 현저히 느려질 수 있습니다.")

path_to_hf = "Embodied-CoT/ecot-openvla-7b-bridge"
# path_to_hf = "openvla/openvla-7b"
# 2. Load the pre-trained OpenVLA-7B base model and processor (bfloat16 type)
# Load the base OpenVLA-7B model from Hugging Face Hub.
print("OpenVLA-7B(+CoT Fine-tuning) Model (bfloat16) 및 프로세서를 로드 중...")
processor = AutoProcessor.from_pretrained(path_to_hf, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    path_to_hf,
    attn_implementation="flash_attention_2",  # Optional: use Flash Attention 2 for speedup
    torch_dtype=torch.bfloat16,              # Load model weights in bfloat16
    low_cpu_mem_usage=True,                  # Optimize CPU memory usage during loading
    trust_remote_code=True
)
# Move the loaded model to the specified device.
model.to(device)
print("기본 모델과 프로세서 로드 완료.")

# 3. Define the local directory to save the pre-trained model
# This directory will be created automatically if it doesn't exist.
output_dir = "/home/kist/openvla/ckpt/ecot-openvla-7b-bridge" # Define the folder name for saving the base model
# output_dir = "/home/kist/openvla/ckpt/openvla-7b-base-bfloat16" # Define the folder name for saving the base model

# 4. Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"저장 디렉토리 '{output_dir}'를 생성했습니다.")

# 5. Save the pre-trained model and processor to the local directory
# The save_pretrained() method saves the model's weights, configuration,
# and all necessary processor files to the specified folder.
print(f"OpenVLA-7B 기본 모델과 프로세서를 '{output_dir}'에 저장 중...")
model.save_pretrained(output_dir) # Save the base model
processor.save_pretrained(output_dir) # Save the processor
print(f"모델과 프로세서가 '{output_dir}'에 성공적으로 저장되었습니다.")

# --- Verify by loading the saved model from local (optional) ---
print(f"\n'{output_dir}'에서 모델과 프로세서를 다시 로드하여 확인 중...")
loaded_processor = AutoProcessor.from_pretrained(output_dir, trust_remote_code=True)
loaded_model = AutoModelForVision2Seq.from_pretrained(
    output_dir,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16, # Load with the same dtype as saved
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)
print("로컬에서 모델과 프로세서 로드 완료.")

# Set the loaded model to evaluation mode
loaded_model.eval()
print("로드된 모델이 평가 모드로 설정되었습니다.")
print("이제 인터넷 연결 없이, OpenVLA-7B 기본 모델을 로컬에서 바로 사용할 수 있습니다.")
