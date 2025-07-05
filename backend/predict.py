import torch
from torchvision import models, transforms
from PIL import Image
import io

# 載入模型
model = models.resnet18(pretrained=True)
model.eval()

# ImageNet 類別名稱
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
import urllib.request
labels = urllib.request.urlopen(LABELS_URL).read().decode("utf-8").split("\n")

# 圖片轉換
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)  # 加 batch 維度
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)
    return labels[predicted.item()]