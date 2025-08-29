FROM tensorflow/tensorflow:2.9.1-gpu

# Set working directory
WORKDIR /app

# Install system dependencies including matplotlib build deps
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies - FIXED ORDER
RUN pip install --upgrade pip
RUN pip install --no-cache-dir numpy==1.23.2
RUN pip install --no-cache-dir matplotlib==3.5.3
RUN pip install --no-cache-dir seaborn==0.12.0
RUN pip install --no-cache-dir scikit-learn==1.1.1
RUN pip install --no-cache-dir pandas==1.4.4
RUN pip install --no-cache-dir mlflow==1.26.1

# Copy project files
COPY scripts/ ./scripts/
COPY . .

# Create directories
RUN mkdir -p /app/data /app/models /app/mlflow

# Set environment variables
ENV MLFLOW_TRACKING_URI=/app/mlflow
ENV PYTHONPATH=/app

# Use both ENTRYPOINT and CMD for better flexibility
ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--epochs", "50"]