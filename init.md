# DevSecOps MLOps Framework - Setup & Operations Guide

## Project Overview

**Project**: DevSecOps Framework for Machine Learning Pipeline
**Use Case**: Credit Card Fraud Detection API
**Institution**: BITS Pilani (M.Tech Dissertation)
**Repository**: https://github.com/awsarv/mlops-automation
**AWS Account**: 760829799650
**Region**: ap-south-1 (Mumbai)

---

## Current Status (As of January 2026)

### All Components Deployed and Working

| Component | Status | Details |
|-----------|--------|---------|
| EKS Cluster | **ACTIVE** | `devsecops-mlops-cluster` with 2 nodes (t3.small) |
| Node Group | **ACTIVE** | `mlops-workers` (2 nodes, scalable 1-3) |
| ECR Repository | **ACTIVE** | `devsecops-mlops/fraud-api` with scan-on-push |
| S3 Bucket | **ACTIVE** | `devsecops-mlops-artifacts-760829799650` |
| Fraud Detection API | **RUNNING** | 2 pods with HPA (2-5 replicas) |
| Prometheus | **RUNNING** | Metrics collection in `monitoring` namespace |
| Grafana | **RUNNING** | Dashboards for API monitoring |
| CI/CD Pipeline | **WORKING** | GitHub Actions with 7 security stages |

---

## Live Endpoints

### Fraud Detection API
```
http://af7adef5a6b3e44aaa659ca8a1da7d54-393184583.ap-south-1.elb.amazonaws.com
```

**Test Commands:**
```bash
# Health check
curl http://af7adef5a6b3e44aaa659ca8a1da7d54-393184583.ap-south-1.elb.amazonaws.com/health

# Fraud prediction
curl -X POST http://af7adef5a6b3e44aaa659ca8a1da7d54-393184583.ap-south-1.elb.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 5000.00,
    "hour": 3,
    "day_of_week": 6,
    "merchant_category": 5,
    "distance_from_home": 150.0,
    "distance_from_last_transaction": 200.0,
    "ratio_to_median_purchase": 15.0,
    "repeat_retailer": 0,
    "used_chip": 0,
    "used_pin": 0,
    "online_order": 1
  }'

# Model info
curl http://af7adef5a6b3e44aaa659ca8a1da7d54-393184583.ap-south-1.elb.amazonaws.com/model/info

# Metrics (Prometheus format)
curl http://af7adef5a6b3e44aaa659ca8a1da7d54-393184583.ap-south-1.elb.amazonaws.com/metrics
```

### Monitoring
```
Prometheus: http://aa01c44a85ddc4d90bff37c2a0efbe9d-577859957.ap-south-1.elb.amazonaws.com:9090
Grafana:    http://a4dc8087d8efc4ed4be1abd025c65569-1664369899.ap-south-1.elb.amazonaws.com:3000
```
**Grafana Credentials**: admin / (check grafana-secrets in monitoring namespace)

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Repository                         │
│  (Code Push) ──► GitHub Actions CI/CD Pipeline                  │
│                                                                  │
│  Stage 1: Security Scan (Bandit, pip-audit, Safety)             │
│  Stage 2: Lint & Test (Flake8, pytest)                          │
│  Stage 3: Build & Container Scan (Trivy, Grype)                 │
│  Stage 4: ML Model Training (scikit-learn, MLflow)              │
│  Stage 5: Deploy to EKS                                         │
│  Stage 6: OPA Policy Validation                                 │
│  Stage 7: Security Report Generation                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         AWS Cloud                                │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │     ECR      │    │      S3      │    │  CloudWatch  │      │
│  │ (Container   │    │   (Model     │    │   (Logs)     │      │
│  │  Registry)   │    │  Artifacts)  │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    EKS Cluster                            │  │
│  │  ┌─────────────────────┐  ┌─────────────────────┐        │  │
│  │  │ devsecops-mlops NS  │  │   monitoring NS     │        │  │
│  │  │ - Fraud API (2 pods)│  │ - Prometheus        │        │  │
│  │  │ - HPA (2-5 replicas)│  │ - Grafana           │        │  │
│  │  │ - NetworkPolicy     │  │                     │        │  │
│  │  └─────────────────────┘  └─────────────────────┘        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Security Tools Integrated

| Tool | Purpose | Stage |
|------|---------|-------|
| **Bandit** | Python SAST (Static Analysis) | CI/CD Stage 1 |
| **pip-audit** | Dependency vulnerability scanning | CI/CD Stage 1 |
| **Safety** | Python dependency security check | CI/CD Stage 1 |
| **Trivy** | Container vulnerability scanner | CI/CD Stage 3 |
| **Grype** | Additional container security | CI/CD Stage 3 |
| **OPA/Rego** | Policy-as-code for Kubernetes | CI/CD Stage 6 |
| **NetworkPolicy** | Network segmentation | Runtime |
| **IRSA** | Pod-level IAM roles | Runtime |

---

## Quick Commands

### Check Cluster Status
```bash
# Update kubeconfig
aws eks update-kubeconfig --region ap-south-1 --name devsecops-mlops-cluster

# Check nodes
kubectl get nodes

# Check all pods
kubectl get pods -A

# Check application pods
kubectl get pods -n devsecops-mlops

# Check monitoring pods
kubectl get pods -n monitoring

# Check services (get LoadBalancer URLs)
kubectl get svc -A
```

### Trigger CI/CD Pipeline
```bash
# Manual trigger
gh workflow run devsecops-pipeline.yaml --ref prod

# Check pipeline status
gh run list --workflow=devsecops-pipeline.yaml

# View specific run
gh run view <run-id>
```

### View Logs
```bash
# API logs
kubectl logs -l app=fraud-api -n devsecops-mlops -f

# Prometheus logs
kubectl logs -l app=prometheus -n monitoring

# Grafana logs
kubectl logs -l app=grafana -n monitoring
```

---

## Cost Management

### Current Estimated Costs (When Running)

| Resource | Cost/Month | Notes |
|----------|------------|-------|
| EKS Control Plane | ~$73 | Fixed cost |
| EC2 (2x t3.small) | ~$30 | On-demand instances |
| EBS Storage | ~$5 | 20GB per node |
| Load Balancers (3) | ~$50 | NLB for API + monitoring |
| ECR/S3/CloudWatch | ~$5 | Minimal storage |
| **Total (Running)** | **~$165** | |

### Stop Resources to Minimize Costs

#### Option 1: Scale Down Nodes Only (Saves ~$30/month)
```bash
# Scale node group to 0
eksctl scale nodegroup --cluster=devsecops-mlops-cluster \
  --name=mlops-workers --nodes=0 --nodes-min=0 --region=ap-south-1

# Verify
kubectl get nodes
```

#### Option 2: Delete Cluster Completely (Saves ~$165/month)
```bash
# Delete the EKS cluster (takes 10-15 minutes)
eksctl delete cluster --name=devsecops-mlops-cluster --region=ap-south-1

# Verify deletion
aws eks list-clusters --region ap-south-1
```

#### Option 3: Full Cleanup (Delete Everything)
```bash
# 1. Delete EKS cluster
eksctl delete cluster --name=devsecops-mlops-cluster --region=ap-south-1

# 2. Delete ECR images (optional - keep repository)
aws ecr batch-delete-image --repository-name devsecops-mlops/fraud-api \
  --image-ids "$(aws ecr list-images --repository-name devsecops-mlops/fraud-api \
  --query 'imageIds[*]' --output json)" --region ap-south-1

# 3. Empty S3 bucket (optional - keep models)
# aws s3 rm s3://devsecops-mlops-artifacts-760829799650 --recursive
```

---

## Resume/Restart Resources

### Recreate Cluster from Scratch
```bash
cd /home/arv/project/dissertation/mlops-automation

# 1. Create EKS cluster (15-20 minutes)
eksctl create cluster -f eks-cluster.yaml

# 2. Update kubeconfig
aws eks update-kubeconfig --region ap-south-1 --name devsecops-mlops-cluster

# 3. Create IAM service account for S3 access
eksctl create iamserviceaccount \
  --name fraud-api-sa \
  --namespace devsecops-mlops \
  --cluster devsecops-mlops-cluster \
  --region ap-south-1 \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
  --override-existing-serviceaccounts \
  --approve

# 4. Deploy application (replace IMAGE_TAG with latest)
sed 's|ECR_REGISTRY|760829799650.dkr.ecr.ap-south-1.amazonaws.com|g; s|IMAGE_TAG|latest|g' \
  k8s/base/deployment.yaml > /tmp/deployment.yaml
kubectl apply -f k8s/base/namespace.yaml
kubectl apply -f k8s/base/service.yaml
kubectl apply -f /tmp/deployment.yaml
kubectl apply -f k8s/base/hpa.yaml
kubectl apply -f k8s/base/networkpolicy.yaml

# 5. Deploy monitoring
kubectl apply -f k8s/monitoring/

# 6. Verify
kubectl get pods -A
kubectl get svc -A
```

### Scale Up Existing Cluster (If Only Scaled Down)
```bash
# Scale node group back up
eksctl scale nodegroup --cluster=devsecops-mlops-cluster \
  --name=mlops-workers --nodes=2 --nodes-min=1 --region=ap-south-1

# Wait for nodes
kubectl get nodes -w

# Pods will automatically restart
kubectl get pods -n devsecops-mlops -w
```

---

## CI/CD Pipeline Details

### Pipeline Stages

1. **security-scan**: Bandit SAST, pip-audit, Safety
2. **lint-and-test**: Flake8 linting, pytest with coverage
3. **build-and-scan**: Docker build, Trivy, Grype, ECR push
4. **train-model**: Data generation, model training, S3 upload
5. **deploy**: EKS deployment with rollout verification
6. **policy-check**: OPA Kubernetes and ML security policies
7. **security-report**: Consolidated security artifacts (90-day retention)

### GitHub Secrets Required
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION (ap-south-1)
AWS_ACCOUNT_ID (760829799650)
ECR_REPOSITORY (devsecops-mlops/fraud-api)
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `eks-cluster.yaml` | EKS cluster configuration |
| `.github/workflows/devsecops-pipeline.yaml` | CI/CD pipeline (344 lines) |
| `k8s/base/deployment.yaml` | Application deployment |
| `k8s/base/service.yaml` | LoadBalancer service |
| `k8s/base/hpa.yaml` | Horizontal Pod Autoscaler |
| `k8s/base/networkpolicy.yaml` | Network security rules |
| `k8s/monitoring/` | Prometheus & Grafana |
| `security/policies/*.rego` | OPA security policies |
| `src/fraud_api.py` | FastAPI application (322 lines) |
| `src/fraud_train.py` | ML model training |
| `Dockerfile` | Multi-stage secure build |
| `docs/framework-architecture.md` | Architecture documentation |
| `docs/ml-threat-model.md` | STRIDE threat analysis |

---

## Troubleshooting

### Pods Not Starting
```bash
kubectl describe pod <pod-name> -n devsecops-mlops
kubectl logs <pod-name> -n devsecops-mlops
kubectl logs <pod-name> -c download-model -n devsecops-mlops  # Init container
```

### Model Download Failing
```bash
# Check S3 bucket
aws s3 ls s3://devsecops-mlops-artifacts-760829799650/models/

# Check service account IAM role
kubectl describe sa fraud-api-sa -n devsecops-mlops
```

### Pipeline Failing
```bash
# Check GitHub Actions
gh run list --workflow=devsecops-pipeline.yaml
gh run view <run-id> --log
```

---

## Viva Demonstration Flow

1. **Show Architecture** - Explain the DevSecOps pipeline diagram
2. **Show GitHub Actions** - Trigger pipeline, explain 7 stages
3. **Show Security Scans** - Bandit, Trivy reports in artifacts
4. **Show Kubernetes** - `kubectl get pods -A`, explain deployments
5. **Test API** - Run curl commands, show fraud detection working
6. **Show Monitoring** - Open Grafana dashboards
7. **Show OPA Policies** - Explain kubernetes.rego and ml-security.rego
8. **Show Threat Model** - docs/ml-threat-model.md STRIDE analysis

---

## Contact & Repository

- **GitHub**: https://github.com/awsarv/mlops-automation
- **Documentation**: See `docs/` folder

---

*Last Updated: January 2026*
