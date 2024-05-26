- 👋 Hi, I’m @DeveloperAi0
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
DeveloperAi0/DeveloperAi0 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    input_image = Image.open(image_path)
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

image_path = 'input_image.jpg'
input_batch = preprocess_image(image_path)

with torch.no_grad():
    output = model(input_batch)

print(output)
