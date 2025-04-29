# src/model/gradcam.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Hook to capture activations and gradients
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        loss = output[0, class_idx]
        loss.backward(retain_graph=True)

        # GradCAM++ calculation
        grads = self.gradients  # [batch, channels, height, width]
        activations = self.activations

        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3

        sum_grads = torch.sum(grads, dim=(2, 3), keepdim=True)

        eps = 1e-8  # Avoid divide-by-zero
        alpha_numer = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_grads * grads_power_3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_numer / alpha_denom

        weights = (alphas * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1).squeeze()

        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + eps)
        return cam