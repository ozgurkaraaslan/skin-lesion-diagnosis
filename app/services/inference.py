import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from app.services.model_loader import ml_models


class InferenceService:
    def __init__(self):

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _remove_hair(self, image_cv2):
        """The DullRazor algorithm applied via OpenCV"""
        grayScale = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        return cv2.inpaint(image_cv2, mask, 1, cv2.INPAINT_TELEA)

    def process_and_predict(
        self, image_bytes: bytes, age: float, sex: str, localization: str
    ):

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        img_clean = self._remove_hair(img_cv2)
        img_resized = cv2.resize(img_clean, (224, 224))
        img_tensor = self.transform(img_resized).unsqueeze(0).to(ml_models.device)

        with torch.no_grad():
            visual_features = ml_models.cnn(img_tensor)
            visual_features = (
                visual_features.view(visual_features.size(0), -1).cpu().numpy()
            )

        meta_dict = {"sex": [sex], "localization": [localization]}
        df_meta = pd.DataFrame(meta_dict)
        meta_encoded = pd.get_dummies(df_meta)

        meta_encoded = meta_encoded.reindex(
            columns=ml_models.meta_cols[1:], fill_value=0
        )

        # Scale age
        scaled_age = ml_models.age_scaler.transform([[age]])
        meta_encoded.insert(0, "age_scaled", scaled_age.flatten())

        clinical_features = meta_encoded.to_numpy(dtype=np.float64)

        fused_features = np.hstack((visual_features, clinical_features))
        fused_clean = np.nan_to_num(fused_features, nan=0.0, posinf=1e6, neginf=-1e6)

        fused_scaled = ml_models.main_scaler.transform(fused_clean)
        fused_scaled = np.clip(fused_scaled, -10, 10)

        probabilities = ml_models.svm.predict_proba(fused_scaled)[0]
        prediction_idx = np.argmax(probabilities)
        predicted_class = ml_models.class_names[prediction_idx]

        confidence_scores = {
            ml_models.class_names[i]: float(probabilities[i])
            for i in range(len(ml_models.class_names))
        }

        return {
            "diagnosis": predicted_class,
            "confidence": float(probabilities[prediction_idx]),
            "all_probabilities": confidence_scores,
        }


inference_service = InferenceService()
