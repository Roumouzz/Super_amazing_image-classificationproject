# Vision Transformer Image Classification

- Fine-tuning d’un modèle ViT/DeiT sur un dataset Kaggle de fleurs
- Entraînement local simple avec PyTorch + timm

## Installation rapide
Voir `requirements.txt`, puis:


## Lancer l’entraînement
python src/train_vit.py --data_dir data/flowers --model deit_small_patch16_224 --epochs 15
