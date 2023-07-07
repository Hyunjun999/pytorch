import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from imagenet1000_labels import label_dict

app = Flask(__name__)


class VGG11(nn.Module):
    def __init__(self, num_classes=1000) -> None:
        super(VGG11, self).__init__()
        self.features = models.vgg11(pretrained=False).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_model(path):
    model = VGG11()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


path = "./vgg11-bbd30ac9.pth"
model = load_model(path)
print(model)


def preprocess(img):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    img = transform(img).unsqueeze(0)
    return img


# API Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = request.files["image"]
    img = Image.open(img)
    img = preprocess(img)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs.data, 1)
        label_num = int(pred.item())
        class_name = label_dict[label_num]
        predict = str(class_name)

    return jsonify({"prediction": predict}), 200


if __name__ == "__main__":
    app.run(debug=True)
