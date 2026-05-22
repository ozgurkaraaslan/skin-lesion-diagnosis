from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.inference import inference_service

router = APIRouter()


@router.post("/predict")
async def predict_lesion(
    file: Annotated[UploadFile, File(...)],
    age: Annotated[float, Form(...)],
    sex: Annotated[str, Form(...)],
    localization: Annotated[str, Form(...)],
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_data = await file.read()

        # Pass all data directly into the mathematical pipeline
        result = inference_service.process_and_predict(
            image_data, age, sex, localization
        )

        return {
            "filename": file.filename,
            "status": "success",
            "prediction": result["diagnosis"],
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"],
        }

    except Exception as e:
        raise RuntimeError(
            f"Data process failed during image conversion or ML inference: {str(e)}"
        ) from e
