# Use official lightweight Python image
FROM python:3.10-slim

# Install system dependencies needed by OpenCV, curl, and certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy dependency list from the backend folder
COPY backend/requirements.txt .

# Install dependencies (using CPU version of PyTorch to save ~2.5GB of space)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ /app/backend/

# Copy the models definition from the root src directory so python can find it
COPY src/ /app/src/

# Copy the weights file if it exists in the build context (wildcard prevents failure if missing)
COPY convnextv2_large_*.pth* /app/

# If weights were not in build context, download directly from GitHub Release
RUN if [ ! -f /app/convnextv2_large_clean.pth ] && [ ! -f /app/convnextv2_large_epoch_25_ema.pth ]; then \
        echo "Downloading 757MB model weights from GitHub Release..."; \
        curl -L -o /app/convnextv2_large_clean.pth https://github.com/ymprajapati0110/Diabetic-Retinopathy/releases/download/v1.0.0/convnextv2_large_clean.pth; \
    fi

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI using Uvicorn with dynamic PORT support for cloud hosts
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
