# DevSecOps Framework for Machine Learning
## Architecture Documentation

### 1. Executive Summary

This document describes the architecture of a comprehensive DevSecOps framework designed specifically for Machine Learning systems deployed in cloud-native environments. The framework integrates automated security auditing, threat modeling, and continuous security enforcement throughout the ML pipeline lifecycle.

**Key Features:**
- Automated security scanning at every pipeline stage
- ML-specific threat modeling using STRIDE methodology
- Policy-as-code enforcement with OPA/Gatekeeper
- Comprehensive monitoring and observability
- Cloud-native deployment on Amazon EKS

---

### 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                    DevSecOps ML Pipeline Architecture                                 │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────────────┐    │
│  │   GitHub    │     │              GitHub Actions CI/CD                        │    │
│  │    Repo     │────►│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │    │
│  │  (Source)   │     │  │ Bandit  │ │pip-audit│ │  Trivy  │ │  Grype  │       │    │
│  └─────────────┘     │  │ (SAST)  │ │ (SCA)   │ │(Container)│ │(Container)│     │    │
│                      │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │    │
│                      │       │           │           │           │             │    │
│                      │       └───────────┴───────────┴───────────┘             │    │
│                      │                        │                                 │    │
│                      │                   Security Gate                          │    │
│                      │                        │                                 │    │
│                      └────────────────────────┼─────────────────────────────────┘    │
│                                               │                                      │
│                                               ▼                                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │   Amazon    │     │   Amazon    │     │   Amazon    │     │   Amazon    │        │
│  │     S3      │◄────│    ECR      │────►│    EKS      │────►│ CloudWatch  │        │
│  │  (Models)   │     │  (Images)   │     │  (Runtime)  │     │  (Logs)     │        │
│  └─────────────┘     └─────────────┘     └──────┬──────┘     └─────────────┘        │
│                                                  │                                   │
│                      ┌───────────────────────────┼───────────────────────────┐      │
│                      │         Kubernetes Cluster                            │      │
│                      │  ┌─────────────┐    ┌─────────────┐    ┌───────────┐ │      │
│                      │  │  Fraud API  │    │ Prometheus  │    │  Grafana  │ │      │
│                      │  │  (FastAPI)  │───►│ (Metrics)   │───►│(Dashboard)│ │      │
│                      │  └─────────────┘    └─────────────┘    └───────────┘ │      │
│                      │         │                                             │      │
│                      │         │           ┌─────────────┐                   │      │
│                      │         └──────────►│   MLflow    │                   │      │
│                      │                     │ (Tracking)  │                   │      │
│                      │                     └─────────────┘                   │      │
│                      └───────────────────────────────────────────────────────┘      │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Component Details

#### 3.1 Source Code Management
| Component | Technology | Purpose |
|-----------|------------|---------|
| Version Control | GitHub | Source code, IaC, configs |
| Branch Protection | GitHub | Enforce code review |
| Secret Management | GitHub Secrets | Secure credential storage |

#### 3.2 CI/CD Pipeline (GitHub Actions)
| Stage | Tools | Security Focus |
|-------|-------|----------------|
| Code Analysis | Bandit | Python SAST |
| Dependency Scan | pip-audit, Safety | SCA, CVE detection |
| Container Scan | Trivy, Grype | Image vulnerabilities |
| Policy Check | OPA | K8s policy compliance |
| Build | Docker | Multi-stage, non-root |
| Push | Amazon ECR | Scan-on-push enabled |
| Deploy | kubectl | GitOps deployment |

#### 3.3 Container Registry (Amazon ECR)
- **Image Scanning**: Automatic vulnerability scanning on push
- **Immutable Tags**: Prevent tag overwriting
- **Encryption**: AES-256 at rest
- **IAM Access**: Role-based pull/push permissions

#### 3.4 Kubernetes Platform (Amazon EKS)
| Component | Configuration | Security Feature |
|-----------|---------------|------------------|
| Cluster | EKS v1.29 | Managed control plane |
| Nodes | t3.medium (Spot) | Cost-optimized workers |
| Networking | VPC CNI | Network policies |
| Storage | EBS CSI | Encrypted volumes |
| Auth | IRSA | Pod-level IAM roles |

#### 3.5 Application Layer
| Service | Technology | Features |
|---------|------------|----------|
| API | FastAPI | /predict, /health, /metrics, /model/info, /stats |
| Model | scikit-learn | Random Forest (Fraud Classification) |
| Tracking | MLflow | Experiment logging |
| Data | S3 | Model artifact storage |

#### 3.6 Monitoring Stack
| Tool | Purpose | Metrics |
|------|---------|---------|
| Prometheus | Metrics collection | Latency, errors, throughput |
| Grafana | Visualization | Dashboards, alerts |
| CloudWatch | Logs | API and cluster logs |

---

### 4. Security Controls Implementation

#### 4.1 Static Analysis (SAST)
```yaml
Tool: Bandit
Scope: Python source code
Checks:
  - SQL injection
  - Command injection
  - Hardcoded secrets
  - Insecure deserialization
```

#### 4.2 Software Composition Analysis (SCA)
```yaml
Tool: pip-audit + Safety
Scope: Python dependencies
Checks:
  - Known CVEs
  - Outdated packages
  - License compliance
```

#### 4.3 Container Security
```yaml
Tool: Trivy + Grype
Scope: Docker images
Checks:
  - OS vulnerabilities
  - Application vulnerabilities
  - Misconfigurations
  - Secret detection
```

#### 4.4 Runtime Security
```yaml
Tool: OPA/Gatekeeper
Scope: Kubernetes resources
Policies:
  - No privileged containers
  - Non-root user required
  - Resource limits mandatory
  - Health checks required
  - No latest tag
```

#### 4.5 Network Security
```yaml
Implementation: Kubernetes NetworkPolicy
Rules:
  - Ingress: Load balancer only
  - Egress: DNS + HTTPS only
  - Pod-to-pod: Restricted
```

---

### 5. Data Flow Security

```
┌──────────┐    HTTPS    ┌──────────┐    HTTPS    ┌──────────┐
│  Client  │────────────►│   ALB    │────────────►│   Pod    │
└──────────┘             └──────────┘             └──────────┘
                                                       │
                              ┌────────────────────────┤
                              │                        │
                              ▼                        ▼
                         ┌──────────┐            ┌──────────┐
                         │   S3     │            │ CloudWatch│
                         │ (Models) │            │  (Logs)   │
                         └──────────┘            └──────────┘
```

**Security Measures:**
1. TLS termination at ALB
2. mTLS between services (optional)
3. S3 server-side encryption
4. CloudWatch log encryption
5. VPC private subnets for pods

---

### 6. Deployment Configuration

#### 6.1 Kubernetes Resources
```
k8s/
├── base/
│   ├── namespace.yaml       # Namespace isolation
│   ├── deployment.yaml      # Application pods
│   ├── service.yaml         # Load balancer
│   ├── serviceaccount.yaml  # IRSA configuration
│   ├── hpa.yaml             # Auto-scaling
│   └── networkpolicy.yaml   # Network isolation
└── monitoring/
    ├── prometheus-*.yaml    # Metrics collection
    └── grafana-*.yaml       # Visualization
```

#### 6.2 Security Configurations
| Resource | Security Setting |
|----------|-----------------|
| Pod | runAsNonRoot: true |
| Pod | allowPrivilegeEscalation: false |
| Pod | readOnlyRootFilesystem: false |
| Pod | capabilities: drop ALL |
| Container | Resource limits defined |
| Container | Health checks configured |
| Service | LoadBalancer with security groups |

---

### 7. Technology Stack Summary

| Layer | Technology | Version |
|-------|------------|---------|
| **Cloud** | AWS | - |
| **Orchestration** | Amazon EKS | 1.29 |
| **Container Runtime** | containerd | Latest |
| **Container Registry** | Amazon ECR | - |
| **Object Storage** | Amazon S3 | - |
| **CI/CD** | GitHub Actions | v4 |
| **Security Scanning** | Bandit, Trivy, pip-audit | Latest |
| **Policy Engine** | OPA | 0.61.0 |
| **ML Framework** | scikit-learn | 1.4.0 |
| **ML Tracking** | MLflow | 2.12.2 |
| **API Framework** | FastAPI | 0.109.2 |
| **Monitoring** | Prometheus | 2.48.0 |
| **Visualization** | Grafana | 10.2.2 |

---

### 8. Cost Optimization

| Resource | Optimization | Savings |
|----------|-------------|---------|
| EKS Nodes | Spot Instances | ~70% |
| Container Images | Multi-stage builds | Smaller size |
| S3 | Intelligent Tiering | Variable |
| CloudWatch | Log retention policies | Reduced storage |

**Estimated Monthly Cost (Student Account):**
- EKS Control Plane: ~$73
- EC2 Spot (2x t3.medium): ~$30
- S3 + ECR: ~$5
- **Total**: ~$108/month

---

### 9. Future Enhancements

1. **Falco**: Runtime threat detection
2. **ArgoCD**: GitOps deployment
3. **Vault**: Secret management
4. **Service Mesh**: Istio for mTLS
5. **Model Monitoring**: Data drift detection
6. **A/B Testing**: Canary deployments

---

### 10. References

- [STRIDE Threat Modeling](./ml-threat-model.md)
- [OPA Policies](../security/policies/)
- [Kubernetes Manifests](../k8s/)
- [CI/CD Pipeline](../.github/workflows/)

---

**Document Version**: 1.0
**Last Updated**: December 2025
**Author**: Arvind Kumar (2023AC05606)
