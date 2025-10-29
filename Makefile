PY=python

setup:
	$(PY) -m pip install -r requirements.txt

data:
	bash scripts/download_data.sh

train:
	$(PY) src/train_vit.py --data_dir data/flowers --model deit_small_patch16_224 --epochs 15 --batch_size 32

train_tiny:
	$(PY) src/train_vit.py --data_dir data/flowers --model deit_tiny_patch16_224 --epochs 15 --batch_size 32

eval:
	# l'éval est déjà incluse en fin d'entraînement, placeholder si tu ajoutes un script dédié
	@echo "Eval incluse dans train_vit.py. Ajouter un script eval si besoin."
