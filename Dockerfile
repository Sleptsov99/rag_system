FROM python:3.11-slim

WORKDIR /app

# System deps needed by some Python packages (tokenizers, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first (avoids downloading 2GB CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data so container starts faster
RUN python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"

COPY . .

CMD ["python3", "bot.py"]
