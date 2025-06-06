# Dockerfile (Debian-slim hardened version)

# Use stable and minimal Debian-based Python image
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] fpdf \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create writable reports directory
RUN mkdir -p /app/reports

# Expose FastAPI port
EXPOSE 8000

# Default command
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
