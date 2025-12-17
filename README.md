# DevSecOps Framework for Machine Learning

## A Comprehensive Security-Integrated MLOps Pipeline

[![CI/CD Pipeline](https://github.com/awsarv/mlops-automation/actions/workflows/devsecops-pipeline.yaml/badge.svg)](https://github.com/awsarv/mlops-automation/actions)
[![Security Scan](https://img.shields.io/badge/security-scanned-green.svg)](./docs/ml-threat-model.md)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

---

## Overview

This project implements a **DevSecOps Framework for Machine Learning** that integrates automated security auditing and threat modeling into cloud-native CI/CD/CT pipelines. It demonstrates best practices for securing ML systems from development to production.

**Dissertation Project**: M.Tech in AI/ML, BITS Pilani
**Author**: Arvind Kumar (2023AC05606)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DevSecOps ML Pipeline                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  GitHub ──► GitHub Actions ──► ECR ──► EKS (Kubernetes)                     │
│                  │                          │                               │
│            ┌─────┴─────┐              ┌─────┴─────┐                         │
│            │ Security  │              │ Runtime   │                         │
│            │  Gates    │              │ Security  │                         │
│            ├───────────┤              ├───────────┤                         │
│            │ Bandit    │              │ OPA       │                         │
│            │ Trivy     │              │ Network   │                         │
│            │ pip-audit │              │ Policies  │                         │
│            └───────────┘              └───────────┘                         │
│                                             │                               │
│                                             ▼                               │
│            MLflow ◄──────────────────► Prometheus + Grafana                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Security Integration
- **SAST**: Bandit for Python code analysis
- **SCA**: pip-audit & Safety for dependency scanning
- **Container Security**: Trivy & Grype for image scanning
- **Policy Enforcement**: OPA/Gatekeeper for Kubernetes policies
- **Network Security**: Kubernetes NetworkPolicies

### ML Pipeline
- **Data Versioning**: DVC with S3 backend
- **Experiment Tracking**: MLflow
- **Model Training**: scikit-learn (Linear Regression, Decision Tree)
- **API Service**: FastAPI with Prometheus metrics

### Infrastructure
- **Container Registry**: Amazon ECR (scan-on-push)
- **Orchestration**: Amazon EKS (Kubernetes)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana

---

## Repository Structure

```
.
├── .github/workflows/          # CI/CD pipeline definitions
│   └── devsecops-pipeline.yaml # Main DevSecOps pipeline
├── docs/                       # Documentation
│   ├── framework-architecture.md
│   └── ml-threat-model.md      # STRIDE threat analysis
├── k8s/                        # Kubernetes manifests
│   ├── base/                   # Application manifests
│   └── monitoring/             # Prometheus & Grafana
├── security/                   # Security configurations
│   └── policies/               # OPA Rego policies
├── src/                        # Application source code
│   ├── api.py                  # FastAPI inference service
│   ├── train.py                # Model training script
│   └── data_prep.py            # Data preparation
├── Dockerfile                  # Multi-stage secure build
├── eks-cluster.yaml            # EKS cluster configuration
└── requirements.txt            # Python dependencies (pinned)
```

---

## Quick Start

### Prerequisites
- AWS CLI configured
- kubectl installed
- Docker installed
- Python 3.10+

### 1. Clone Repository
```bash
git clone https://github.com/awsarv/mlops-automation.git
cd mlops-automation
git checkout prod
```

### 2. Create EKS Cluster
```bash
eksctl create cluster -f eks-cluster.yaml
```

### 3. Deploy Application
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/base/namespace.yaml
kubectl apply -f k8s/base/

# Deploy monitoring
kubectl apply -f k8s/monitoring/
```

### 4. Access Services
```bash
# Get API endpoint
kubectl get svc -n devsecops-mlops housing-api

# Get Grafana endpoint
kubectl get svc -n monitoring grafana
```

---

## Security Scanning

### Run Locally
```bash
# Install security tools
pip install bandit pip-audit safety

# Code analysis
bandit -r src/

# Dependency scan
pip-audit

# Container scan (requires Trivy)
trivy image <your-image>
```

### Pipeline Security Gates
The CI/CD pipeline includes:
1. **Bandit** - Python SAST
2. **pip-audit** - Dependency CVE check
3. **Trivy** - Container vulnerabilities
4. **Grype** - Additional container scan
5. **OPA** - Kubernetes policy compliance

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Model inference |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI |

### Example Request
```bash
curl -X POST "http://<api-endpoint>/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984,
    "AveBedrms": 1.023,
    "Population": 322.0,
    "AveOccup": 2.555,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

---

## Monitoring

### Prometheus Metrics
- `predictions_total` - Total prediction count
- `inference_latency_seconds` - Prediction latency histogram
- `process_*` - Process metrics

### Grafana Dashboards
- API Request Rate
- Latency Percentiles (p50, p90, p95, p99)
- Error Rate
- Resource Usage

---

## Documentation

- [Framework Architecture](./docs/framework-architecture.md)
- [ML Threat Model (STRIDE)](./docs/ml-threat-model.md)
- [OPA Security Policies](./security/policies/)

---

## AWS Resources

| Resource | Name | Purpose |
|----------|------|---------|
| S3 | devsecops-mlops-artifacts-* | Model storage |
| ECR | devsecops-mlops/housing-api | Container images |
| EKS | devsecops-mlops-cluster | Kubernetes cluster |

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/security-enhancement`)
3. Run security scans locally
4. Commit changes
5. Push and create PR

---

## License

MIT License - see [LICENSE](./LICENSE)

---

## Author

**Arvind Kumar**
M.Tech in AI/ML, BITS Pilani
Student ID: 2023AC05606

**Supervisor**: Dr. Sheela Verma (IIT BHU)
**Examiner**: Dr. Pratibha Verma (Turing, USA)
