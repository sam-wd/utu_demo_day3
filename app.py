import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms
import io

app = FastAPI(title="MNIST CNN API", version="1.0")

# -----------------------------
# CNN architecture
# -----------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# -----------------------------
# Load saved model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "mnist_cnn.pth"

model = CNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Same preprocessing used for MNIST
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


# -----------------------------
# API endpoints
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "MNIST CNN API is running",
        "endpoint": "/predict"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")

        # Preprocess
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)

            predicted_digit = torch.argmax(
                probabilities, dim=1
            ).item()

            confidence = probabilities[
                0, predicted_digit
            ].item() * 100

        return {
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
