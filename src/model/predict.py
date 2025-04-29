import torch
from PIL import Image

def predict_single_image(model, image_path, transform, class_idx_to_name, device):
    """
    Predict the class of a single image.
    
    Args:
        model: Trained model
        image_path (str): Path to the image
        transform: Transformations to apply
        class_idx_to_name (dict): Mapping from class index to class name
        device: torch.device
    """
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # Add batch dimension
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

    predicted_class = class_idx_to_name[pred.item()]
    return predicted_class