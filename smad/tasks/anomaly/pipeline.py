from pathlib import Path
import torch
from torch.utils.data import DataLoader
from smad.models.utils import load_trained_model

MODEL_DIR = Path(__file__).parent.parent.resolve() / 'models/trained'

class AE_SVM_AnomalyDetection:
    def __init__(model: str, dataloader: DataLoader):
        model = load_trained_model(MODEL_DIR / model) 