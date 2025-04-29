# app.py

import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import json

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Internal modules
from src.model.architecture import build_model
from src.model.gradcam import GradCAMPlusPlus as GradCAM
from src.utils.config import BEST_MODEL_PATH
from src.data.augment import AugmentationPipeline
from src.data.dataset import PlantVillageDataset

# Load disease information
with open("disease_info.json", "r") as f:
    DISEASE_INFO = json.load(f)

# Load model and classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

augment = AugmentationPipeline()
test_ds = PlantVillageDataset(root_dir="data/split/test", transform=augment.get_transforms("val"))

model = build_model(num_classes=len(test_ds.class_to_idx), freeze_backbone=False)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

class_to_idx = test_ds.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
target_layer = model.features[-1]

def beautify_name(raw_classname):
    parts = raw_classname.split("___")
    if len(parts) >= 3:
        plant = parts[1].title()
        disease = parts[2].replace("_", " ").title()
        return plant, disease
    else:
        return "Unknown", "Unknown"

def generate_gradcam(model, input_tensor, target_layer):
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(input_tensor)
    return cam

def predict(image):
    if image is None:
        return None, None, None, None, None, None

    model.eval()

    # Resize captured image for webcam compatibility
    image = image.resize((224, 224))

    transform = augment.get_transforms("val")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    top3_indices = probs.argsort()[-3:][::-1]
    top3_classes = [(idx_to_class[idx], float(probs[idx])) for idx in top3_indices]

    pred_class, pred_prob = top3_classes[0]
    plant, disease = beautify_name(pred_class)

    # Format Top-3 Predictions nicely
    top3_text = ""
    for i, (c, p) in enumerate(top3_classes, 1):
        c_plant, c_disease = beautify_name(c)
        top3_text += f"{i}. {c_plant} - {c_disease} ({p*100:.2f}%)\n"

    # GradCAM
    cam = generate_gradcam(model, image_tensor, target_layer)

    img_np = np.array(image) / 255.0
    img_gray = cv2.cvtColor(np.uint8(img_np * 255), cv2.COLOR_RGB2GRAY) / 255.0
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

    # Health Status and Disease Info
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

# Layout
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
    with gr.Row():
        gr.Markdown(
            "<p style='text-align: center;'>© 2025 Made by <a href='https://github.com/mitraarka27' target='_blank'>Arka Mitra</a></p>"
        )
    with gr.Row():
        with gr.Column(scale=2):
            upload = gr.Image(
                type="pil",
                sources=["upload", "webcam"],
                label="Upload or Capture Leaf Image"
            )
            gr.Markdown(instruction_text)
            predict_btn = gr.Button("Predict", variant="primary")
            alert = gr.Textbox(label="Prediction Alert", lines=2, interactive=False)
            top3_preds = gr.Textbox(label="Top-3 Predictions", lines=5, interactive=False)
        with gr.Column(scale=3):
            health_status = gr.Label(label="Plant Health Status")
            disease_name = gr.Label(label="Identified Disease (includes details below)")
            disease_info = gr.Markdown()
            heatmap = gr.Image(label="Model Focus Heatmap")

    predict_btn.click(
        fn=predict,
        inputs=[upload],
        outputs=[health_status, disease_name, top3_preds, alert, heatmap, disease_info]
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)