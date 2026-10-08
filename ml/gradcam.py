"""
Grad-CAM Explainability Module for AttentionCNN.

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize
which spatial regions and cytological features (e.g., nucleus chromatin, cytoplasm borders)
most influence the network's multi-class cervical diagnosis.

Reference:
Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization", ICCV 2017.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.attention_model import get_attention_model, AttentionCNN
from ml.dataset import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, get_transforms

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "attention_cnn.pth"
CELLS_DIR = PROJECT_ROOT / "data" / "cells"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam"
REPORT_PATH = PROJECT_ROOT / "outputs" / "gradcam_examples.json"

class GradCAM:
    """Computes Grad-CAM heatmaps for a given convolutional layer in AttentionCNN."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        """Clean up PyTorch hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float, List[float]]:
        """Generate a 224x224 Grad-CAM heatmap for an input image.

        Args:
            input_tensor: (1, 3, 224, 224) normalized input tensor.
            target_class: Integer class index to explain. If None, uses top predicted class.

        Returns:
            heatmap_norm: 2D numpy array (224, 224) normalized to [0, 1].
            predicted_class_idx: Integer index of predicted class.
            predicted_prob: Probability (confidence) for predicted class.
            all_probs: Probabilities for all 6 classes.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_prob = probabilities[predicted_class_idx].item()
        all_probs = probabilities.detach().cpu().tolist()

        # Target score to differentiate
        class_to_explain = predicted_class_idx if target_class is None else target_class
        score = logits[0, class_to_explain]

        # Backward pass
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

        # 1. Global average pooling of gradients along spatial dimensions (H, W)
        # alpha_k: Importance weight for feature channel k
        alpha_k = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # 2. Weighted linear combination of forward activation maps
        cam = torch.sum(alpha_k * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)

        # 3. Apply ReLU (keep features with positive influence on target class)
        cam = torch.relu(cam)

        # 4. Upsample to original image resolution (224, 224)
        cam = nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # 5. Min-max normalization into [0, 1]
        denom = cam.max() - cam.min()
        if denom > 1e-8:
            heatmap_norm = (cam - cam.min()) / denom
        else:
            heatmap_norm = np.zeros_like(cam)

        return heatmap_norm, predicted_class_idx, predicted_prob, all_probs

def colormap_jet(heatmap: np.ndarray) -> np.ndarray:
    """Pure-NumPy Jet colormap mapping a [0, 1] 2D float array to (H, W, 3) RGB uint8."""
    v = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(3.0 * v - 2.5), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(3.0 * v - 1.5), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(3.0 * v - 0.5), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

def create_gradcam_panel(
    original_img: Image.Image,
    heatmap: np.ndarray,
    true_class: str,
    pred_class: str,
    prob: float
) -> Image.Image:
    """Builds a clear 3-panel visualization:
    [Original Cell Crop] | [Grad-CAM Heatmap] | [Overlay on Cell]
    with an informative header banner.
    """
    orig_rgb = np.array(original_img.convert("RGB"))
    heatmap_rgb = colormap_jet(heatmap)

    # Alpha overlay: 60% original image + 40% heatmap
    overlay_rgb = (0.60 * orig_rgb + 0.40 * heatmap_rgb).astype(np.uint8)

    orig_pil = Image.fromarray(orig_rgb)
    heat_pil = Image.fromarray(heatmap_rgb)
    over_pil = Image.fromarray(overlay_rgb)

    # Panel dimensions
    w, h = 224, 224
    header_h = 56
    margin = 8
    total_w = w * 3 + margin * 4
    total_h = h + header_h + margin * 2

    canvas = Image.new("RGB", (total_w, total_h), color=(245, 246, 250))
    draw = ImageDraw.Draw(canvas)

    # Paste the 3 panels
    canvas.paste(orig_pil, (margin, header_h + margin))
    canvas.paste(heat_pil, (margin * 2 + w, header_h + margin))
    canvas.paste(over_pil, (margin * 3 + w * 2, header_h + margin))

    # Header texts
    status = "CORRECT" if true_class == pred_class else "MISCLASSIFIED"
    status_color = (39, 174, 96) if true_class == pred_class else (192, 57, 43)

    title_text = f"True: {true_class}   |   Pred: {pred_class} ({prob*100:.1f}%)   [{status}]"
    draw.text((margin + 4, 10), title_text, fill=status_color)

    # Sub-panel labels
    draw.text((margin + 4, header_h - 16), "1. Original Cell Crop", fill=(70, 70, 70))
    draw.text((margin * 2 + w + 4, header_h - 16), "2. Grad-CAM Activation", fill=(70, 70, 70))
    draw.text((margin * 3 + w * 2 + 4, header_h - 16), "3. Diagnostic Overlay", fill=(70, 70, 70))

    return canvas

def generate_representative_examples(num_per_class: int = 2) -> List[Dict]:
    """Select representative test cells from each class and generate Grad-CAM visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and checkpoint
    model = get_attention_model(num_classes=len(CLASS_NAMES))
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Target the last convolutional block before spatial attention
    target_layer = model.block4
    gradcam = GradCAM(model, target_layer)
    transform = get_transforms()

    test_dir = CELLS_DIR / "test"
    records = []

    print("=" * 65)
    print("GENERATING GRAD-CAM EXPLAINABILITY VISUALIZATIONS")
    print("=" * 65)

    sample_index = 1
    for class_name in CLASS_NAMES:
        class_folder = test_dir / class_name
        if not class_folder.exists():
            continue

        image_files = sorted(list(class_folder.glob("*.png")))[:num_per_class]

        for img_path in image_files:
            orig_img = Image.open(img_path).convert("RGB")
            tensor_img = transform(orig_img).unsqueeze(0)  # (1, 3, 224, 224)

            # Generate Grad-CAM for the predicted class
            heatmap, pred_idx, prob, all_probs = gradcam.generate(tensor_img)
            pred_class = CLASS_NAMES[pred_idx]

            # Build 3-panel visualization
            panel = create_gradcam_panel(
                original_img=orig_img,
                heatmap=heatmap,
                true_class=class_name,
                pred_class=pred_class,
                prob=prob
            )

            # Save visualization image
            out_filename = f"gradcam_{class_name}_{sample_index:02d}.png"
            out_path = OUTPUT_DIR / out_filename
            panel.save(out_path, format="PNG")

            record = {
                "sample_id": sample_index,
                "crop_filename": img_path.name,
                "crop_path": str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "true_class": class_name,
                "predicted_class": pred_class,
                "predicted_probability": round(prob, 4),
                "is_correct": (class_name == pred_class),
                "all_class_probabilities": {
                    cls_name: round(p, 4) for cls_name, p in zip(CLASS_NAMES, all_probs)
                },
                "heatmap_stats": {
                    "min": round(float(heatmap.min()), 4),
                    "max": round(float(heatmap.max()), 4),
                    "mean": round(float(heatmap.mean()), 4),
                },
                "output_visualization": f"outputs/gradcam/{out_filename}",
            }
            records.append(record)

            print(f"[{sample_index:02d}/12] {img_path.name:15} | True: {class_name:10} | "
                  f"Pred: {pred_class:10} ({prob*100:5.1f}%) | Saved -> {out_filename}")
            sample_index += 1

    gradcam.remove_hooks()

    # Save JSON report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "target_layer": "model.block4 (Last Convolutional Feature Map)",
            "total_examples": len(records),
            "examples": records
        }, f, indent=2)

    print("\n" + "=" * 65)
    print(f"Successfully generated {len(records)} Grad-CAM visualizations!")
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 65)
    return records

if __name__ == "__main__":
    generate_representative_examples(num_per_class=2)
