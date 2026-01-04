# Deployment Guide

Complete guide for deploying the Stock Analysis Multi-Agent System to production.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Scaling](#scaling)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

## Overview

The system can be deployed in multiple ways:

1. **Local Development:** Direct Python execution
2. **Docker:** Containerized deployment
3. **Cloud:** AWS, GCP, Azure deployment
4. **Kubernetes:** Orchestrated container deployment

## Prerequisites

### System Requirements

- **CPU:** 4+ cores recommended
- **RAM:** 16GB minimum, 32GB recommended
- **GPU:** Optional but recommended for faster inference
  - NVIDIA GPU with 8GB+ VRAM
  - CUDA 11.8+ and cuDNN
- **Storage:** 50GB+ for models and data
- **OS:** Linux (Ubuntu 22.04 recommended), macOS, or Windows with WSL2

### Software Requirements

- Python 3.11+
- Docker 20.10+ (for containerized deployment)
- NVIDIA Docker (for GPU support)
- Git

### API Keys

Required environment variables:

```bash
# Anthropic (for Judge)
ANTHROPIC_API_KEY=your_key_here

# Optional: OpenAI (alternative Judge)
OPENAI_API_KEY=your_key_here

# Optional: News API
NEWS_API_KEY=your_key_here

# Optional: Weights & Biases (for training monitoring)
WANDB_API_KEY=your_key_here
```

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/stock-agent-system-final.git
cd stock-agent-system-final
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure System

```bash
# Copy example config
cp config/system.yaml.example config/system.yaml

# Edit configuration
nano config/system.yaml
```

### 5. Download Models

```bash
# Option 1: Download pre-trained models
python scripts/download_models.py

# Option 2: Train your own models
python training/sft/train_news_agent.py --config config/sft/news_agent.yaml
```

### 6. Run API Server

```bash
# Development mode with auto-reload
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

# Production mode
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### 7. Test Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "use_supervisor": false}'
```

## Docker Deployment

### 1. Build Docker Image

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=config/system.yaml

# Run server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build image:

```bash
docker build -t stock-agent-system:latest .
```

### 2. Run Container

```bash
docker run -d \
  --name stock-agent-api \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  stock-agent-system:latest
```

### 3. Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - CONFIG_PATH=config/system.yaml
    volumes:
      - ./config:/app/config
      - ./models:/app/models
      - ./data:/app/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped
```

Run with Docker Compose:

```bash
docker-compose up -d
```

### 4. GPU Support

For GPU support, use NVIDIA Docker:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ... rest of Dockerfile
```

Run with GPU:

```bash
docker run -d \
  --gpus all \
  --name stock-agent-api \
  -p 8000:8000 \
  stock-agent-system:latest
```

## Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance

```bash
# Launch EC2 instance (g4dn.xlarge for GPU)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g4dn.xlarge \
  --key-name your-key \
  --security-groups stock-agent-sg

# SSH into instance
ssh -i your-key.pem ubuntu@ec2-instance-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2

# Deploy application
git clone https://github.com/yourusername/stock-agent-system-final.git
cd stock-agent-system-final
docker-compose up -d
```

#### 2. ECS (Elastic Container Service)

Create `task-definition.json`:

```json
{
  "family": "stock-agent-system",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-ecr-repo/stock-agent-system:latest",
      "memory": 16384,
      "cpu": 4096,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "CONFIG_PATH",
          "value": "config/system.yaml"
        }
      ],
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:anthropic-key"
        }
      ]
    }
  ]
}
```

Deploy to ECS:

```bash
# Create ECR repository
aws ecr create-repository --repository-name stock-agent-system

# Build and push image
docker build -t stock-agent-system:latest .
docker tag stock-agent-system:latest your-ecr-repo/stock-agent-system:latest
docker push your-ecr-repo/stock-agent-system:latest

# Create ECS cluster
aws ecs create-cluster --cluster-name stock-agent-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster stock-agent-cluster \
  --service-name stock-agent-service \
  --task-definition stock-agent-system \
  --desired-count 2 \
  --launch-type FARGATE
```

### GCP Deployment

#### Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/your-project/stock-agent-system

# Deploy to Cloud Run
gcloud run deploy stock-agent-api \
  --image gcr.io/your-project/stock-agent-system \
  --platform managed \
  --region us-central1 \
  --memory 16Gi \
  --cpu 4 \
  --set-env-vars CONFIG_PATH=config/system.yaml \
  --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest
```

### Azure Deployment

#### Container Instances

```bash
# Create resource group
az group create --name stock-agent-rg --location eastus

# Create container registry
az acr create --resource-group stock-agent-rg \
  --name stockagentregistry --sku Basic

# Build and push image
az acr build --registry stockagentregistry \
  --image stock-agent-system:latest .

# Deploy container
az container create \
  --resource-group stock-agent-rg \
  --name stock-agent-api \
  --image stockagentregistry.azurecr.io/stock-agent-system:latest \
  --cpu 4 \
  --memory 16 \
  --ports 8000 \
  --environment-variables CONFIG_PATH=config/system.yaml \
  --secure-environment-variables ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
```

## Configuration

### Environment Variables

```bash
# Required
export ANTHROPIC_API_KEY=your_key_here

# Optional
export OPENAI_API_KEY=your_key_here
export NEWS_API_KEY=your_key_here
export WANDB_API_KEY=your_key_here
export CONFIG_PATH=config/system.yaml
export LOG_LEVEL=INFO
```

### System Configuration

Edit `config/system.yaml`:

```yaml
agents:
  news:
    enabled: true
    model_path: models/news_agent_v1
    
  technical:
    enabled: true
    model_path: models/technical_agent_v1
    
  fundamental:
    enabled: true
    model_path: models/fundamental_agent_v1

supervisor:
  enabled: false  # Enable after training

strategist:
  model_path: models/strategist_v1
  max_position_size: 0.10
  max_drawdown: 0.15
```

## Monitoring

### Application Monitoring

#### Prometheus & Grafana

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'stock-agent-api'
    static_configs:
      - targets: ['api:8000']
```

Add to `docker-compose.yml`:

```yaml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Logging

Configure structured logging:

```python
# utils/logging_setup.py
from loguru import logger

logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
```

### Health Checks

```bash
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Scaling

### Horizontal Scaling

#### Load Balancer Configuration

```nginx
# nginx.conf
upstream api_backend {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Kubernetes Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: stock-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: stock-agent-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Scaling

Adjust resources in deployment:

```yaml
resources:
  requests:
    memory: "16Gi"
    cpu: "4"
  limits:
    memory: "32Gi"
    cpu: "8"
```

## Security

### API Security

1. **Add Authentication:**

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

2. **Rate Limiting:**

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze(request: Request, ...):
    ...
```

3. **HTTPS/TLS:**

```bash
# Generate SSL certificate
certbot certonly --standalone -d your-domain.com

# Configure nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ...
}
```

### Secrets Management

Use cloud secret managers:

```python
# AWS Secrets Manager
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']
```

## Troubleshooting

### Common Issues

**Out of Memory:**
```bash
# Increase Docker memory limit
docker run -m 16g ...

# Or adjust model settings
config:
  fp16: true
  gradient_checkpointing: true
```

**Slow Inference:**
```bash
# Enable GPU
docker run --gpus all ...

# Use quantization
config:
  load_in_4bit: true
```

**Connection Timeout:**
```bash
# Increase timeout in nginx
proxy_read_timeout 300s;
proxy_connect_timeout 300s;
```

### Logs

```bash
# Docker logs
docker logs stock-agent-api

# Follow logs
docker logs -f stock-agent-api

# Kubernetes logs
kubectl logs -f deployment/stock-agent-api
```

## Backup & Recovery

### Database Backup

```bash
# Backup trajectory database
cp data/experience_library.db backups/experience_library_$(date +%Y%m%d).db
```

### Model Backup

```bash
# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

## Support

For deployment issues:
- GitHub Issues: [repository]/issues
- Email: devops@example.com
- Documentation: [repository]/docs
