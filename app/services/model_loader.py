import joblib
import torch
import torch.nn as nn
import torchvision.models as models

from app.core.config import settings


class MLModels:
    def __init__(self):
        print("Loading AI pipeline into memory...")

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 7)
        self.cnn = nn.Sequential(*list(model.children())[:-1])

        self.cnn.load_state_dict(
            torch.load(settings.RESNET_PATH, map_location=self.device)
        )
        self.cnn = self.cnn.to(self.device)
        self.cnn.eval()

        self.svm = joblib.load(settings.SVM_PATH)
        self.main_scaler = joblib.load(settings.MAIN_SCALER_PATH)
        self.age_scaler = joblib.load(settings.AGE_SCALER_PATH)
        self.meta_cols = joblib.load(settings.META_COLS_PATH)
        self.class_names = joblib.load(settings.CLASS_NAMES_PATH)
        print("Pipeline successfully loaded and ready for inference.")


ml_models = MLModels()
