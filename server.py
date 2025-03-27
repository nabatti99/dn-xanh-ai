from typing import *
import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging
from PIL import Image
from model import check_can_classification, classify_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    try:
        image_contents = [(await file.read()) for file in files]
        images = [Image.open(io.BytesIO(content)) for content in image_contents]

        should_classifications = check_can_classification(images)
        if not any(should_classifications):
            return JSONResponse(content={"should_classification": False})

        predictions = classify_image(images)
        return JSONResponse(
            content={
                "should_classification": True,
                "predictions": predictions,
            }
        )

    except Exception as e:
        logger.error(f"Error processing images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)
