# Explainable and Uncertainty-Aware Deep Learning for Multi-Class Cervical Cell Classification

A research prototype designed to classify cervical cell images across multiple diagnostic classes, while providing visual explainability (e.g., Grad-CAM heatmaps) and predictive uncertainty estimation to support trustworthy clinical decision-making.

---

## Project Structure

```text
cervical-ai/
├── data/       # Cervical-cell image dataset
├── ml/         # Machine learning and deep learning pipelines
├── backend/    # FastAPI web service
├── frontend/   # React + TypeScript user interface
├── models/     # Saved model weights and checkpoints
├── outputs/    # Predictions, Grad-CAM heatmaps, and evaluation metrics
└── README.md   # Project overview and documentation
```

---

## Directory Descriptions

- **`data/`**: Stores raw and preprocessed cervical cell image datasets organized by class.
- **`ml/`**: Contains code for data loading, preprocessing, model architecture, training loops, evaluation metrics, explainability (Grad-CAM), and uncertainty estimation (e.g., Monte Carlo Dropout / Ensembles).
- **`backend/`**: Hosts the FastAPI server that loads the trained models and serves prediction endpoints to the frontend.
- **`frontend/`**: The web application built with React and TypeScript, providing an interface to upload cell images, view classification scores, inspect heatmaps, and see uncertainty estimates.
- **`models/`**: Houses exported model weights, checkpoints, and configuration files.
- **`outputs/`**: Holds output artifacts generated during runs, such as saliency maps, confusion matrices, ROC curves, calibration plots, and inference logs.
