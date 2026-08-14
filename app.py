import io
from pathlib import Path

import torch
import torch.nn as nn

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
from torchvision import transforms


# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(
    title="MNIST CNN Prediction API",
    description="API for predicting handwritten digits using a trained PyTorch CNN model.",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# CNN MODEL
# Must match the architecture used during training
# =========================================================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 5 * 5,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                64,
                10
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

MODEL_PATH = Path("mnist_cnn.pth")


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "mnist_cnn.pth not found. "
        "Please put the .pth model in the same folder as app.py."
    )


model = CNN().to(device)


try:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

except TypeError:

    # Compatibility with older PyTorch versions

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )


# If the saved file contains:
# {"state_dict": ...}

if (
    isinstance(checkpoint, dict)
    and "state_dict" in checkpoint
):

    checkpoint = checkpoint["state_dict"]


model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

transform = transforms.Compose([

    # Convert image to grayscale
    transforms.Grayscale(
        num_output_channels=1
    ),

    # MNIST images are 28 x 28
    transforms.Resize(
        (28, 28)
    ),

    # Convert image to tensor
    transforms.ToTensor()
])


# =========================================================
# HOME ENDPOINT
# =========================================================

@app.get("/")
def home():

    return {

        "message": "MNIST CNN API is running",

        "model": "mnist_cnn.pth",

        "device": str(device),

        "prediction_endpoint": "/predict",

        "documentation": "/docs"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "model": MODEL_PATH.name,

        "device": str(device)
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Check file type
    # -----------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):

        raise HTTPException(

            status_code=400,

            detail="Please upload an image file."
        )


    try:

        # -------------------------------------------------
        # Read uploaded image
        # -------------------------------------------------

        contents = await file.read()


        image = Image.open(
            io.BytesIO(contents)
        )


        # Convert to grayscale

        image = image.convert("L")


        # -------------------------------------------------
        # Preprocess image
        # -------------------------------------------------

        image_tensor = transform(image)


        # Add batch dimension
        #
        # Before:
        # [1, 28, 28]
        #
        # After:
        # [1, 1, 28, 28]

        image_tensor = image_tensor.unsqueeze(0)


        # Move to CPU/GPU

        image_tensor = image_tensor.to(device)


        # -------------------------------------------------
        # Make prediction
        # -------------------------------------------------

        with torch.no_grad():

            output = model(
                image_tensor
            )


            # Convert logits to probabilities

            probabilities = torch.softmax(
                output,
                dim=1
            )


            # Get predicted digit

            predicted_digit = torch.argmax(
                probabilities,
                dim=1
            ).item()


            # Get confidence

            confidence = (
                probabilities[
                    0,
                    predicted_digit
                ].item()
                * 100
            )


        # -------------------------------------------------
        # Return result
        # -------------------------------------------------

        return {

            "success": True,

            "filename": file.filename,

            "predicted_digit": predicted_digit,

            "confidence": round(
                confidence,
                2
            )
        }


    except Exception as e:

        raise HTTPException(

            status_code=400,

            detail=f"Could not process image: {str(e)}"
        )


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "app:app",

        host="0.0.0.0",

        port=8000,

        reload=True
    )
