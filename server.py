import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision.transforms import v2
from typing import List
from PIL import Image
import io
from model import WasteClassificationModel

app = FastAPI()

device = torch.device("cpu")
model = WasteClassificationModel()
model.load_state_dict(
    torch.load("train_loss_best_cpu.pt", map_location=device, weights_only=True)
)
model.eval()

class_names = open("class_names.txt").read().splitlines()

pil_transform = v2.Compose(
    [
        v2.Resize(size=(256, 256)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


def read_transform(file) -> Image.Image:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = pil_transform(image)
    return image


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    try:
        print(len(files))
        tensors = [read_transform(await file.read()) for file in files]
        input_batch: torch.Tensor = torch.stack(tensors)

        outputs = model(input_batch)

        _, predictionIds = torch.max(outputs, 1)
        predictions = [class_names[id] for id in predictionIds]

        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)
