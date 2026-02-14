# DevSecOps Framework for Machine Learning

## Fraud Detection with Security-Integrated MLOps Pipeline

[![CI/CD Pipeline](https://github.com/awsarv/mlops-automation/actions/workflows/devsecops-pipeline.yaml/badge.svg)](https://github.com/awsarv/mlops-automation/actions)
[![Security Scan](https://img.shields.io/badge/security-scanned-green.svg)](./docs/ml-threat-model.md)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

---

## Overview

This project implements a **DevSecOps Framework for Machine Learning** that integrates automated security auditing and threat modeling into cloud-native CI/CD/CT pipelines. It demonstrates best practices for securing ML systems from development to production.

A production-ready implementation demonstrating enterprise-grade ML security practices and cloud-native deployment strategies.

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
- **Experiment Tracking**: MLflow
- **Model Training**: scikit-learn (Random Forest, Logistic Regression)
- **Use Case**: Credit Card Fraud Detection (Binary Classification)
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
│   ├── fraud_api.py            # FastAPI fraud detection service
│   ├── fraud_train.py          # Model training script
│   └── fraud_data_prep.py      # Synthetic data generation
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
kubectl get svc -n devsecops-mlops fraud-api

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
| `/predict` | POST | Fraud detection inference |
| `/metrics` | GET | Prometheus metrics |
| `/model/info` | GET | Model information |
| `/stats` | GET | Prediction statistics |
| `/docs` | GET | Swagger UI |

### Example Fraud Detection Request
```bash
curl -X POST "http://<api-endpoint>/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1250.50,
    "hour": 3,
    "day_of_week": 5,
    "merchant_category": 12,
    "distance_from_home": 85.3,
    "distance_from_last_transaction": 45.2,
    "ratio_to_median_purchase": 4.5,
    "repeat_retailer": 0,
    "used_chip": 0,
    "used_pin": 0,
    "online_order": 1
  }'
```

### Example Response
```json
{
  "transaction_id": "TXN-20251217123456789",
  "fraud_probability": 0.8723,
  "is_fraudulent": true,
  "risk_level": "HIGH",
  "recommendation": "BLOCK transaction. Require additional verification.",
  "timestamp": "2025-12-17T12:34:56.789Z"
}
```

---

## Monitoring

### Prometheus Metrics
- `fraud_predictions_total` - Total prediction count by status and result
- `fraud_inference_latency_seconds` - Prediction latency histogram
- `fraud_transactions_flagged_total` - Flagged fraud count
- `fraud_model_confidence` - Model confidence distribution
- `transaction_amount_dollars` - Transaction amount histogram

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
| ECR | devsecops-mlops/fraud-api | Container images |
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

## About

This project demonstrates production-grade DevSecOps practices for ML systems, combining security automation with cloud-native deployment on AWS EKS. Ideal for learning modern MLOps workflows and security-first development.
