mkdir -p scripts
cat > scripts/download_data.sh << 'EOF'
#!/usr/bin/env bash
set -e
mkdir -p data
kaggle datasets download -d alxmamaev/flowers-recognition -p data
unzip -o data/flowers-recognition.zip -d data/flowers
rm -f data/flowers-recognition.zip
echo "OK: data/flowers prêt"
EOF
chmod +x scripts/download_data.sh
