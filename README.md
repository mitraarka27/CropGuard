# CropGuard: Leaf Disease Detector

Use online at HuggingFace Spaces at: https://huggingface.co/spaces/mitraarka27/CropGuard

**CropGuard** is a lightweight, deployable machine learning app that detects **leaf diseases** in **Potato**, **Tomato**, and **Grape** plants from user-uploaded or captured images.

Built using **PyTorch**, **Gradio**, **Docker**, and **Hugging Face Spaces**, it provides the following capabilities:

- Upload or capture a leaf image
- Predict plant health status
- Identify likely disease (if any)
- Visualize model attention using **GradCAM++** heatmaps
- Provide quick disease information and treatment suggestions

<img width="1154" alt="Screenshot 2025-04-28 at 4 24 27 PM" src="https://github.com/user-attachments/assets/7e8c8b53-aa62-4803-a7a9-e4398753106b" />

---

## Project Structure

```
CropGuard/
├── app.py                # Gradio app
├── Dockerfile            # Docker container definition
├── requirements.txt      # Python dependencies
├── notebooks/            # Step-by-step project development
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_validation.ipynb
│   └── 04_gradcam_visualization.ipynb
├── src/                  # Source code (organized into modules)
│   ├── app/
│   ├── data/
│   ├── model/
│   └── utils/
├── sample_images/        # Few test images (optional for demo)
├── disease_info.json     # Disease descriptions
└── README.md             # (this file)
```

---

## Notebooks Overview

| Notebook | Purpose |
|:-|:-|
| `01_data_preprocessing.ipynb` | Download PlantVillage dataset, clean and split into train/val/test sets |
| `02_model_training.ipynb` | Set up data augmentation, train MobileNetV2 model, monitor training curves |
| `03_model_validation.ipynb` | Evaluate model performance, generate metrics, confusion matrix |
| `04_gradcam_visualization.ipynb` | Generate GradCAM++ heatmaps to visualize model focus |

---

## How to Run Locally

1. **Clone the repo:**

```bash
git clone https://github.com/mitraarka27/CropGuard.git
cd CropGuard
```

2. **Create a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Launch the app:**

```bash
python app.py
```

It will be available at [http://localhost:7860](http://localhost:7860).

---

## How to Build and Run with Docker

```bash
docker build -t cropguard-app .
docker run -p 7860:7860 cropguard-app
```

---

## Web Deployment

Easily deployable on:

- Hugging Face Spaces
- DockerHub

---

## Sample Images

We provide a few **sample leaf images** in the `sample_images/` directory so users can test the model even without their own images.

---

## License

MIT License.

---

## Acknowledgments

- **Dataset:** [PlantVillage Dataset](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)
- **Base Model:** [MobileNetV2 (pretrained on ImageNet)](https://arxiv.org/abs/1801.04381)
- **Visualization:** GradCAM++

---

## Author

Made by **[Arka Mitra](https://github.com/mitraarka27)** © 2025.

---

