# ML Threat Model: STRIDE Analysis for DevSecOps MLOps Pipeline

## 1. System Overview

This document provides a comprehensive threat model for the DevSecOps ML Pipeline implementing California Housing Price Prediction. The analysis follows the STRIDE methodology adapted for Machine Learning systems.

### 1.1 Architecture Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DevSecOps ML Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  GitHub  │───►│  GitHub  │───►│   ECR    │───►│   EKS    │              │
│  │   Repo   │    │ Actions  │    │ Registry │    │ Cluster  │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│       │               │               │               │                     │
│       │         ┌─────┴─────┐         │         ┌─────┴─────┐              │
│       │         │ Security  │         │         │ Runtime   │              │
│       │         │  Scans    │         │         │ Security  │              │
│       │         └───────────┘         │         └───────────┘              │
│       │                               │                                     │
│       ▼                               ▼                                     │
│  ┌──────────┐                   ┌──────────┐                               │
│  │  MLflow  │                   │ S3 Bucket│                               │
│  │ Registry │                   │ (Models) │                               │
│  └──────────┘                   └──────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow

1. **Code Push** → GitHub Repository
2. **CI/CD Trigger** → GitHub Actions
3. **Security Scanning** → Bandit, Trivy, pip-audit, OPA
4. **Container Build** → Docker Image
5. **Container Push** → Amazon ECR
6. **Model Training** → MLflow Tracking
7. **Model Storage** → S3 Bucket
8. **Deployment** → EKS Kubernetes Cluster
9. **Inference** → FastAPI Service
10. **Monitoring** → Prometheus + Grafana

---

## 2. STRIDE Threat Analysis

### 2.1 Spoofing (Identity)

| Threat ID | Threat Description | Component | Likelihood | Impact | Risk Level |
|-----------|-------------------|-----------|------------|--------|------------|
| S-01 | Attacker spoofs GitHub webhook to trigger malicious builds | GitHub Actions | Medium | High | High |
| S-02 | Compromised developer credentials push malicious code | GitHub Repo | Medium | Critical | Critical |
| S-03 | Fake training data injection via compromised data source | Data Ingestion | Low | High | Medium |
| S-04 | Spoofed API requests to inference endpoint | FastAPI Service | Medium | Medium | Medium |
| S-05 | Impersonation of MLflow tracking server | MLflow | Low | High | Medium |

**Mitigations Implemented:**
- ✅ GitHub branch protection rules
- ✅ Webhook signature verification
- ✅ IAM role-based access control
- ✅ API authentication (planned)
- ✅ Service account isolation in Kubernetes

---

### 2.2 Tampering (Data Integrity)

| Threat ID | Threat Description | Component | Likelihood | Impact | Risk Level |
|-----------|-------------------|-----------|------------|--------|------------|
| T-01 | Model poisoning via tampered training data | Data Pipeline | Medium | Critical | Critical |
| T-02 | Modification of model weights in transit/storage | S3/ECR | Low | Critical | High |
| T-03 | Container image tampering in registry | ECR | Low | Critical | Medium |
| T-04 | CI/CD pipeline manipulation | GitHub Actions | Low | Critical | High |
| T-05 | Kubernetes manifest tampering | K8s Deployment | Low | High | Medium |
| T-06 | Log tampering to hide attacks | CloudWatch | Low | Medium | Low |

**Mitigations Implemented:**
- ✅ Data integrity checksums (SHA-256)
- ✅ ECR image scanning on push
- ✅ Signed commits (recommended)
- ✅ Immutable infrastructure
- ✅ S3 versioning and encryption
- ✅ OPA policy enforcement

---

### 2.3 Repudiation (Accountability)

| Threat ID | Threat Description | Component | Likelihood | Impact | Risk Level |
|-----------|-------------------|-----------|------------|--------|------------|
| R-01 | Untracked model version changes | MLflow | Medium | High | High |
| R-02 | Missing audit trail for data access | S3 | Medium | Medium | Medium |
| R-03 | Lack of deployment change history | EKS | Low | Medium | Low |
| R-04 | Inference requests without logging | FastAPI | Medium | Medium | Medium |
| R-05 | Training job execution without records | GitHub Actions | Low | Medium | Low |

**Mitigations Implemented:**
- ✅ MLflow experiment tracking
- ✅ Git commit history
- ✅ CloudWatch logging
- ✅ S3 access logging
- ✅ Prometheus metrics collection
- ✅ SQLite request logging in API

---

### 2.4 Information Disclosure (Confidentiality)

| Threat ID | Threat Description | Component | Likelihood | Impact | Risk Level |
|-----------|-------------------|-----------|------------|--------|------------|
| I-01 | Model extraction via repeated API queries | FastAPI Service | Medium | High | High |
| I-02 | Training data leakage from model | ML Model | Medium | High | High |
| I-03 | Secrets exposure in CI/CD logs | GitHub Actions | Medium | Critical | Critical |
| I-04 | Unauthorized access to S3 model artifacts | S3 Bucket | Low | High | Medium |
| I-05 | Container image layer inspection revealing secrets | ECR | Low | High | Medium |
| I-06 | API response leaking internal information | FastAPI | Medium | Medium | Medium |

**Mitigations Implemented:**
- ✅ GitHub Secrets for credentials
- ✅ IAM least privilege access
- ✅ S3 bucket policies
- ✅ Network policies in Kubernetes
- ✅ Rate limiting (planned)
- ✅ Input/output sanitization

---

### 2.5 Denial of Service (Availability)

| Threat ID | Threat Description | Component | Likelihood | Impact | Risk Level |
|-----------|-------------------|-----------|------------|--------|------------|
| D-01 | API endpoint overwhelmed by requests | FastAPI Service | High | High | High |
| D-02 | Resource exhaustion during training | Training Job | Medium | Medium | Medium |
| D-03 | Kubernetes cluster resource starvation | EKS | Medium | High | High |
| D-04 | S3 rate limiting triggered | S3 | Low | Medium | Low |
| D-05 | CI/CD pipeline queue exhaustion | GitHub Actions | Low | Low | Low |

**Mitigations Implemented:**
- ✅ Horizontal Pod Autoscaler (HPA)
- ✅ Resource limits/requests in K8s
- ✅ Rate limiting (planned)
- ✅ Circuit breaker patterns (planned)
- ✅ Health checks and auto-restart

---

### 2.6 Elevation of Privilege (Authorization)

| Threat ID | Threat Description | Component | Likelihood | Impact | Risk Level |
|-----------|-------------------|-----------|------------|--------|------------|
| E-01 | Container escape to host | EKS Node | Low | Critical | High |
| E-02 | Service account privilege escalation | Kubernetes | Low | Critical | High |
| E-03 | CI/CD pipeline privilege abuse | GitHub Actions | Low | High | Medium |
| E-04 | IAM role assumption attack | AWS IAM | Low | Critical | High |
| E-05 | Unauthorized model deployment | EKS | Low | High | Medium |

**Mitigations Implemented:**
- ✅ Non-root containers
- ✅ Read-only root filesystem (where possible)
- ✅ Dropped capabilities
- ✅ Network policies
- ✅ RBAC in Kubernetes
- ✅ IAM roles for service accounts (IRSA)
- ✅ OPA/Gatekeeper policies

---

## 3. ML-Specific Threats

### 3.1 Adversarial Attacks

| Threat | Description | Mitigation |
|--------|-------------|------------|
| Evasion Attack | Crafted inputs that cause misclassification | Input validation, anomaly detection |
| Poisoning Attack | Malicious data injected during training | Data validation, provenance tracking |
| Model Inversion | Extracting training data from model | Differential privacy (future) |
| Membership Inference | Determining if data was in training set | Model regularization |

### 3.2 Supply Chain Attacks

| Threat | Description | Mitigation |
|--------|-------------|------------|
| Malicious Dependencies | Compromised Python packages | pip-audit, Safety checks |
| Backdoored Base Images | Trojan in container base | Trivy scanning, trusted registries |
| Compromised ML Frameworks | Malicious sklearn/pandas | Version pinning, vulnerability scanning |

---

## 4. Risk Matrix Summary

```
                    IMPACT
              Low    Medium    High    Critical
         ┌────────┬─────────┬────────┬──────────┐
    High │   L    │    M    │   H    │    C     │
L        ├────────┼─────────┼────────┼──────────┤
I   Med  │   L    │    M    │   M    │    H     │
K        ├────────┼─────────┼────────┼──────────┤
E   Low  │   L    │    L    │   M    │    M     │
L        └────────┴─────────┴────────┴──────────┘
I
H
O   Risk Levels: L=Low, M=Medium, H=High, C=Critical
O
D
```

### Critical Risks (Require Immediate Attention):
1. **S-02**: Compromised developer credentials
2. **T-01**: Model poisoning via tampered training data
3. **I-03**: Secrets exposure in CI/CD logs

### High Risks (Require Near-term Attention):
1. **S-01**: GitHub webhook spoofing
2. **T-02**: Model weight tampering
3. **I-01**: Model extraction attacks
4. **D-01**: API denial of service
5. **E-01**: Container escape

---

## 5. Security Controls Matrix

| Control | STRIDE Coverage | Status |
|---------|-----------------|--------|
| Bandit (SAST) | T, I, E | ✅ Implemented |
| Trivy (Container Scan) | T, I, E | ✅ Implemented |
| pip-audit (Dependencies) | T, I | ✅ Implemented |
| OPA/Gatekeeper | S, T, E | ✅ Implemented |
| Network Policies | S, I, D | ✅ Implemented |
| RBAC | S, E | ✅ Implemented |
| Resource Limits | D | ✅ Implemented |
| MLflow Tracking | R | ✅ Implemented |
| CloudWatch Logging | R, I | ✅ Implemented |
| S3 Encryption | I | ✅ Implemented |
| IAM Least Privilege | S, E | ✅ Implemented |
| Health Checks | D | ✅ Implemented |
| HPA Autoscaling | D | ✅ Implemented |

---

## 6. Recommendations

### Immediate (Before Production):
1. Enable GitHub branch protection with required reviews
2. Implement API rate limiting
3. Add authentication to inference endpoint
4. Enable S3 access logging

### Short-term (Next Sprint):
1. Implement model input validation
2. Add anomaly detection for inference requests
3. Set up alerting for security events
4. Conduct penetration testing

### Long-term (Roadmap):
1. Implement differential privacy for training
2. Add model watermarking
3. Deploy Falco for runtime security
4. Implement zero-trust networking

---

## 7. Compliance Mapping

| Requirement | Framework | Controls |
|-------------|-----------|----------|
| Access Control | NIST 800-53 AC | IAM, RBAC, Network Policies |
| Audit Logging | NIST 800-53 AU | CloudWatch, MLflow, Prometheus |
| Data Protection | GDPR Art. 32 | Encryption, Access Controls |
| Secure Development | OWASP | SAST, Container Scanning |
| Supply Chain | SLSA Level 2 | Dependency Scanning, Signed Commits |

---

## Document Information

- **Version**: 1.0
- **Last Updated**: December 2025
- **Author**: Arvind Kumar
- **Review Status**: Draft for Mid-Semester Report
