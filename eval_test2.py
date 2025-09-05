# Import PyTorch library
import torch
# Import model and processor classes from Hugging Face Transformers
from transformers import AutoModelForVision2Seq, AutoProcessor
# Import Pillow for image processing
from PIL import Image
# Import os module for operating system functionalities
import os
import sys

# Import distributed modules for FSDP
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

# Import transformer blocks from BOTH vision and language models for the wrap policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from timm.models.vision_transformer import Block as ViTBlock


# Set to True to use FSDP (for distributed or large model inference)
use_fsdp = True

# --- Robot Environment Simulation Functions ---
def get_from_camera():
    """Simulates getting an image from a robot's camera."""
    # This function simulates getting an image from a robot's camera.
    # If the image file doesn't exist, it creates a dummy image.
    image_path = "robot_scene_image.jpg"
    if os.path.exists(image_path):
        # print(f"Loading image from {image_path}")
        return Image.open(image_path)
    else:
        # print(f"Warning: '{image_path}' not found. Creating a dummy image.")
        return Image.new('RGB', (224, 224), color = (70, 130, 180))

def robot_act(action, *args, **kwargs):
    """Simulates a robot performing an action."""
    # This function simulates a robot performing a given action.
    print(f"Robot is attempting to perform action: {action}")
    print("Action executed (simulated).")

# --- Step 1: Distributed Environment Setup and Model Loading ---
def setup_and_load_model():
    """
    Sets up the distributed environment and loads the model to CPU RAM on each process.
    """
    # This function sets up the distributed environment.
    if 'LOCAL_RANK' not in os.environ:
        print("Error: Not running in a distributed environment. Please use `torch.distributed.run` or `accelerate launch`.")
        sys.exit(1)

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Initialize the process group for distributed communication.
    dist.init_process_group("nccl")
    # Set the target GPU for the current process.
    torch.cuda.set_device(local_rank)
    
    print(f"Process {rank}/{world_size} initialized on device cuda:{local_rank}.")
    
    # Use the local path where the model is saved.
    model_path = "home/kist/openvla/ckpt/openvla-7b-base-bfloat16"

    # Only rank 0 prints messages to avoid clutter.
    if rank == 0:
        print(f"\nLoading model and processor from local path '{model_path}' to CPU...")

    # Load the processor from the local path.
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Load the full model to CPU to prevent GPU OOM and compatibility issues.
    base_model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        device_map="cpu", 
    )
    
    if rank == 0:
        print("Model loaded to CPU successfully.")
        # ADDED: Calculate and print the model size in GB.
        total_params = sum(p.numel() for p in base_model.parameters())
        # bfloat16 uses 2 bytes per parameter.
        model_size_gb = (total_params * 2) / (1024 ** 3)
        print(f"Model size on CPU: {model_size_gb:.2f} GB")
    
    return base_model, processor, rank

# --- Step 2: FSDP Preparation ---
def prepare_and_shard_model(base_model, rank):
    """
    Wraps the CPU-loaded model with FSDP using CPU Offloading and a robust policy.
    This is the most stable method for large models that don't fit in a single GPU.
    """
    # This function prepares the model for FSDP.
    if rank == 0:
        print("Preparing model for FSDP with CPU Offloading...")
        
    # Define a robust wrapping policy for both vision and language transformer blocks.
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer, ViTBlock},
    )
    
    # Wrap the CPU-based model using the most stable FSDP configuration.
    vla_model = FSDP(
        base_model,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True), # Enable lazy loading from CPU to GPU.
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        # FINAL FIX: Removed 'use_orig_params=True' to allow FSDP's default, more robust
        # parameter flattening, which works better with CPU offloading.
    )
    
    # Wait for all processes to finish FSDP initialization.
    dist.barrier()

    if rank == 0:
        print("Model successfully wrapped with FSDP and sharded across GPUs.")
    
    return vla_model

# --- Step 3: Inference and Action Execution ---
def run_inference(vla_model, processor, rank):
    """
    Performs inference. Only rank 0 handles I/O and prints results.
    """
    # This function runs the inference.
    vla_model.eval()
    if rank == 0:
        print("\nModel set to evaluation mode.")
    
    # All processes get the same input data.
    image: Image.Image = get_from_camera()
    prompt = "In: What action should the robot take to {move the red block to the left}?\nOut:"

    if rank == 0:
        print(f"Processing prompt: '{prompt}'")

    # Keep inputs on the CPU. FSDP will handle moving data to the GPU internally.
    inputs = processor(prompt, image, return_tensors="pt")
    # The dtype conversion should also happen on the CPU.
    inputs = {k: v.to(dtype=torch.bfloat16) for k, v in inputs.items()}


    # Perform inference without calculating gradients.
    with torch.no_grad():
        action = vla_model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    # Wait for all processes to finish before printing the result.
    dist.barrier()
    if rank == 0:
        print(f"\nPredicted Action (7-DoF): {action}")
        robot_act(action)
        
# --- Main Execution Flow ---
if __name__ == "__main__":
    if use_fsdp:
        # FSDP execution path.
        base_model, processor, rank = setup_and_load_model()
        vla_model = prepare_and_shard_model(base_model, rank)
        run_inference(vla_model, processor, rank)

        # Clean up the distributed environment.
        dist.destroy_process_group()
    else:
        # Single-GPU execution path.
        print("FSDP is disabled. Script will run on a single GPU (or CPU).")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = "home/kist/openvla/ckpt/openvla-7b-base-bfloat16"
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        vla_model = base_model.to(device)
        vla_model.eval()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        image: Image.Image = get_from_camera()
        prompt = "In: What action should the robot take to {move the red block to the left}?\nOut:"
        inputs = processor(prompt, image, return_tensors="pt").to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            action = vla_model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        print(f"\nPredicted Action (7-DoF): {action}")
        robot_act(action)
