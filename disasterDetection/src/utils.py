from PIL import Image

from torchvision import transforms

from .config import IMAGE_SIZE

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


def preprocess_image(image):

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor