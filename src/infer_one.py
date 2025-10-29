import sys, torch
from PIL import Image
from torchvision import transforms
import timm

CKPT = "checkpoints/deit_small_patch16_224_best.pth"
IMG = sys.argv[1] if len(sys.argv) > 1 else "example.jpg"

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

ckpt = torch.load(CKPT, map_location=device)
classes = ckpt["classes"]

model = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=len(classes))
model.load_state_dict(ckpt["model"])
model.eval().to(device)

img = Image.open(IMG).convert("RGB")
x = tf(img).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(x)
pred = logits.argmax(1).item()
print(f"Pred: {classes[pred]}")
