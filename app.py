# app.py

import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import json
from PIL import Image

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Internal modules
from src.model.architecture import build_model
from src.model.gradcam import GradCAMPlusPlus as GradCAM
from src.utils.config import BEST_MODEL_PATH

# Load disease information
with open("disease_info.json", "r") as f:
    DISEASE_INFO = json.load(f)

# Load label mapping
with open("models/labels.json", "r") as f:
    idx_to_class = json.load(f)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(idx_to_class)

model = build_model(num_classes=num_classes, freeze_backbone=False)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

target_layer = model.features[-1]

# Sample images
SAMPLE_DIR = "sample_images"
sample_choices = sorted(os.listdir(SAMPLE_DIR))

# Utility Functions
def beautify_name(raw_classname):
    parts = raw_classname.split("___")
    if len(parts) >= 3:
        plant = parts[1].title()
        disease = parts[2].replace("_", " ").replace("(", "").replace(")", "").replace("__", " ").title()
        return plant, disease
    else:
        return "Unknown", "Unknown"

def generate_gradcam(model, input_tensor, target_layer):
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(input_tensor)
    return cam

def preprocess_image(image):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def predict(image):
    if image is None:
        return None, None, None, None, None

    model.eval()
    image_tensor = preprocess_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    top3_indices = probs.argsort()[-3:][::-1]
    top3_classes = [(idx_to_class[str(idx)], float(probs[idx])) for idx in top3_indices]

    pred_class, pred_prob = top3_classes[0]
    plant, disease = beautify_name(pred_class)

    # Format Top-3 nicely
    top3_text = ""
    for i, (c, p) in enumerate(top3_classes, 1):
        c_plant, c_disease = beautify_name(c)
        top3_text += f"{i}. {c_plant} - {c_disease} ({p*100:.2f}%)\n"

    # GradCAM
    cam = generate_gradcam(model, image_tensor, target_layer)

    img_np = np.array(image) / 255.0

    # 🔥 Resize uploaded image to 224x224 for matching heatmap
    img_np_resized = cv2.resize(img_np, (224, 224))

    img_gray = cv2.cvtColor(np.uint8(img_np_resized * 255), cv2.COLOR_RGB2GRAY) / 255.0
    img_gray_3ch = np.stack([img_gray]*3, axis=-1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_PLASMA)
    heatmap = np.float32(heatmap) / 255

    overlay = heatmap + img_gray_3ch
    overlay = overlay / np.max(overlay)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(overlay)
    ax.axis("off")
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='plasma'), orientation='horizontal', pad=0.05, ax=ax)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low Focus', 'Medium Focus', 'High Focus'])
    plt.tight_layout()

    cam_path = "cam_output.png"
    fig.savefig(cam_path)
    plt.close(fig)

    # Health Status + Disease Info
    if "healthy" in pred_class.lower():
        health_status = f"{plant} - Healthy"
        identified_disease = "None"
        disease_info_text = "✅ No disease detected."
    else:
        health_status = f"{plant} - Diseased"
        identified_disease = disease
        disease_data = DISEASE_INFO.get(pred_class, {})
        disease_info_text = f"""
**Symptoms:** {disease_data.get('symptoms', 'No information available.')}

**Causes:** {disease_data.get('causes', 'No information available.')}

**Disease Cycle:** {disease_data.get('disease_cycle', 'No information available.')}

**Care & Treatment:** {disease_data.get('care_treatment', 'No information available.')}

[Learn more on Wikipedia]({disease_data.get('wiki_url', '#')})
"""

    alert = None
    if pred_prob < 0.6:
        alert = "⚠️ Low confidence in prediction! Please verify manually."

    return health_status, identified_disease, top3_text, alert, cam_path, disease_info_text

def load_sample_image(sample_name):
    img_path = os.path.join(SAMPLE_DIR, sample_name)
    img = Image.open(img_path).convert("RGB")
    return img

# Interface
title = "CropGuard: Leaf Disease Detector"
copyright_text = "© 2025 Made by [Arka Mitra](https://github.com/mitraarka27)"
instruction_text = """
Upload a clear image of a **potato**, **tomato**, or **grape** leaf.

CropGuard will predict:
- Whether the leaf is **healthy** or **diseased**.
- The likely disease (if any).
- Where the model focused its attention.

⚡ **Note**: Currently supports only **Potato, Tomato, Grape** leaves.
"""

with gr.Blocks(theme="default") as app:
    with gr.Row():
        gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
    gr.Markdown("<p style='text-align: center;'>© 2025 Made by <a href='https://github.com/mitraarka27' target='_blank'>Arka Mitra</a></p>")
    with gr.Row():
        with gr.Column(scale=2):
            upload = gr.Image(
                type="pil",
                sources=["upload", "webcam", "clipboard"],
                label="Upload, Capture, or Paste Leaf Image"
            )
            gr.Markdown("**OR** choose from sample images below:")
            sample_dropdown = gr.Dropdown(choices=sample_choices, label="Select a Sample Image")
            load_btn = gr.Button("Load Sample Image")
            predict_btn = gr.Button("Predict", variant="primary")
            gr.Markdown(instruction_text)
            alert_box = gr.Textbox(label="Prediction Alert", lines=2, interactive=False)
            top3_preds = gr.Textbox(label="Top-3 Predictions", lines=5, interactive=False)
        with gr.Column(scale=3):
            health_status = gr.Label(label="Plant Health Status")
            disease_name = gr.Label(label="Identified Disease (includes details)")
            disease_info = gr.Markdown()
            heatmap = gr.Image(label="Model Focus Heatmap")

    load_btn.click(
        fn=load_sample_image,
        inputs=[sample_dropdown],
        outputs=[upload]
    )

    predict_btn.click(
        fn=predict,
        inputs=[upload],
        outputs=[health_status, disease_name, top3_preds, alert_box, heatmap, disease_info]
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=True)