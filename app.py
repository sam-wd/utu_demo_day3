import torch
import torch.nn as nn
from flask import Flask, request, render_template_string
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>MNIST CNN Classifier</title>
<style>
body { font-family: Arial; text-align:center; margin-top:50px; }
.box { width:500px; margin:auto; padding:30px; border:1px solid #ddd; border-radius:10px; }
button { padding:10px 25px; margin-top:15px; cursor:pointer; }
.result { font-size:28px; font-weight:bold; margin-top:20px; }
</style>
</head>
<body>
<div class="box">
<h1>MNIST CNN Classifier</h1>
<form method="POST" enctype="multipart/form-data">
<input type="file" name="image" accept="image/*" required><br>
<button type="submit">Predict Digit</button>
</form>
{% if prediction is not none %}
<div class="result">Predicted Digit: {{ prediction }}</div>
<p>Confidence: {{ confidence }}%</p>
{% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file).convert("L")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            predicted = torch.argmax(probabilities, dim=1).item()
            confidence = round(probabilities[0, predicted].item() * 100, 2)

        prediction = predicted

    return render_template_string(
        HTML, prediction=prediction, confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
