#!/usr/bin/env bash
set -o errexit

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
mkdir -p staticfiles
python manage.py collectstatic --no-input
python manage.py migrate

# Pre-download the model during build so it's ready at runtime
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"