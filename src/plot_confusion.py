import torch, itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import transforms, datasets
import timm
import torch.nn as nn

CKPT = "checkpoints/deit_tiny_patch16_224_best.pth"
DATA = "data/flowers"
IMG_SIZE = 224
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# data
mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
tf = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(mean,std)])
ds = datasets.ImageFolder(DATA, transform=tf)
# même split simple: reprends 80/20 comme à l'entraînement
n = len(ds); n_val = int(0.2*n); n_train = n - n_val
_, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

# model
ckpt = torch.load(CKPT, map_location=device)
classes = ckpt['classes']
model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=len(classes))
model.load_state_dict(ckpt['model']); model.eval().to(device)

y_true, y_pred = [], []
with torch.no_grad():
  for x, y in val_loader:
      x = x.to(device)
      p = model(x).argmax(1).cpu()
      y_pred += p.tolist(); y_true += y.tolist()

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, interpolation='nearest')
ax.set_title("Confusion Matrix")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha='right'); ax.set_yticklabels(classes)
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
print("saved -> results/confusion_matrix.png")
