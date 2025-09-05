# Import PyTorch library
import torch
# Import model and processor classes from Hugging Face Transformers
from transformers import AutoModelForVision2Seq, AutoProcessor
# Import Pillow for image processing
from PIL import Image
# Import os module for operating system functionalities
import os

# --- Robot Environment Simulation Functions ---
def get_from_camera():
    """Simulates getting an image from a robot's camera."""
    # This function simulates getting an image from a robot's camera.
    # If the image file doesn't exist, it creates a dummy image.
    image_path = "robot_scene_image.jpg"
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        print(f"Warning: '{image_path}' not found. Creating a dummy image.")
        return Image.new('RGB', (224, 224), color=(70, 130, 180)) # SteelBlue color

def robot_act(action, *args, **kwargs):
    """Simulates a robot performing an action."""
    # This function simulates a robot performing a given action.
    print(f"Robot is attempting to perform action: {action}")
    print("Action executed (simulated).")

# --- Main Execution Flow ---
if __name__ == "__main__":
    print("Running in single-GPU mode. FSDP is disabled.")

    # --- Step 1: Setup and Model Loading ---
    
    # Define the target device for the model and tensors.
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the local path to the model.
    model_path = "home/kist/openvla/ckpt/openvla-7b-base-bfloat16"

    # Load the processor.
    print(f"Loading processor from '{model_path}'...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Load the model with memory-saving options.
    print(f"Loading model from '{model_path}'...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,      # Use bfloat16 for efficiency.
        low_cpu_mem_usage=True,          # Reduces RAM usage during loading.
        trust_remote_code=True
    )
    
    # Calculate and print the model size.
    total_params = sum(p.numel() for p in model.parameters())
    model_size_gb = (total_params * 2) / (1024 ** 3) # bfloat16 uses 2 bytes.
    print(f"Model size: {model_size_gb:.2f} GB")

    # Move the entire model to the selected device.
    print(f"Moving model to {device}...")
    model.to(device)
    
    # Set the model to evaluation mode.
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")

    # --- Step 2: Inference and Action Execution ---

    # Get image and prompt for the robot task.
    image: Image.Image = get_from_camera()
    prompt = "In: What action should the robot take to {move the red block to the left}?\nOut:"
    print(f"\nProcessing prompt: '{prompt}'")

    # Process inputs and move them to the same device as the model.
    inputs = processor(prompt, image, return_tensors="pt").to(device, dtype=torch.bfloat16)

    # Perform inference without calculating gradients.
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    # Print the predicted action and simulate execution.
    print(f"\nPredicted Action (7-DoF): {action}")
    robot_act(action)