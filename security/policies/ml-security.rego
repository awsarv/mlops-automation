# OPA Security Policies for ML-Specific Security
# DevSecOps Framework for Machine Learning

package ml.security

# ML Model Security Policies

# Deny models without integrity verification
deny[msg] {
    input.kind == "MLModel"
    not input.spec.integrity.checksum
    msg := "ML Model must have integrity checksum defined"
}

# Deny models from untrusted sources
trusted_registries := [
    "760829799650.dkr.ecr.ap-south-1.amazonaws.com",
    "devsecops-mlops-artifacts-760829799650.s3.ap-south-1.amazonaws.com"
]

deny[msg] {
    input.kind == "MLModel"
    registry := input.spec.source.registry
    not registry_trusted(registry)
    msg := sprintf("ML Model source '%v' is not from a trusted registry", [registry])
}

registry_trusted(registry) {
    some i
    startswith(registry, trusted_registries[i])
}

# Deny models without version tracking
deny[msg] {
    input.kind == "MLModel"
    not input.metadata.labels.version
    msg := "ML Model must have version label"
}

# Data Security Policies

# Deny datasets without encryption
deny[msg] {
    input.kind == "Dataset"
    not input.spec.encryption.enabled
    msg := "Dataset must have encryption enabled"
}

# Deny datasets without access logging
deny[msg] {
    input.kind == "Dataset"
    not input.spec.audit.logging
    msg := "Dataset must have access logging enabled"
}

# Training Security Policies

# Deny training jobs without resource quotas
deny[msg] {
    input.kind == "TrainingJob"
    not input.spec.resources.limits
    msg := "Training job must have resource limits defined"
}

# Deny training jobs with excessive permissions
deny[msg] {
    input.kind == "TrainingJob"
    input.spec.serviceAccount.admin == true
    msg := "Training job must not use admin service account"
}

# Inference Security Policies

# Require rate limiting on inference endpoints
deny[msg] {
    input.kind == "InferenceService"
    not input.spec.rateLimit
    msg := "Inference service must have rate limiting configured"
}

# Require input validation on inference endpoints
deny[msg] {
    input.kind == "InferenceService"
    not input.spec.inputValidation.enabled
    msg := "Inference service must have input validation enabled"
}
