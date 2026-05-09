from pathlib import Path

# Dynamically find the root directory (two levels up from core/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "saved_models"


class Settings:
    PROJECT_NAME: str = "Skin Lesion Diagnosis API"
    VERSION: str = "1.0.0"

    # Model Paths
    RESNET_PATH = MODEL_DIR / "resnet50_extractor.pth"
    SVM_PATH = MODEL_DIR / "svm_model.joblib"
    MAIN_SCALER_PATH = MODEL_DIR / "main_feature_scaler.joblib"
    AGE_SCALER_PATH = MODEL_DIR / "age_scaler.joblib"
    META_COLS_PATH = MODEL_DIR / "metadata_columns.joblib"
    CLASS_NAMES_PATH = MODEL_DIR / "class_names.joblib"


settings = Settings()
