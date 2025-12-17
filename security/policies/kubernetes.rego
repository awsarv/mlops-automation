# OPA Security Policies for Kubernetes ML Deployments
# DevSecOps Framework for Machine Learning

package kubernetes

# Deny containers running as root
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.runAsNonRoot
    msg := sprintf("Container '%v' must not run as root", [container.name])
}

# Deny containers without resource limits
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.resources.limits
    msg := sprintf("Container '%v' must have resource limits defined", [container.name])
}

# Deny privileged containers
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    container.securityContext.privileged == true
    msg := sprintf("Container '%v' must not be privileged", [container.name])
}

# Deny containers that allow privilege escalation
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    container.securityContext.allowPrivilegeEscalation == true
    msg := sprintf("Container '%v' must not allow privilege escalation", [container.name])
}

# Require health checks
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.livenessProbe
    msg := sprintf("Container '%v' must have a liveness probe", [container.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.readinessProbe
    msg := sprintf("Container '%v' must have a readiness probe", [container.name])
}

# Deny latest tag
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    endswith(container.image, ":latest")
    msg := sprintf("Container '%v' must not use 'latest' tag", [container.name])
}

# Require specific labels
deny[msg] {
    input.kind == "Deployment"
    not input.metadata.labels.app
    msg := "Deployment must have 'app' label"
}

# Deny hostNetwork
deny[msg] {
    input.kind == "Deployment"
    input.spec.template.spec.hostNetwork == true
    msg := "Deployment must not use host network"
}

# Deny hostPID
deny[msg] {
    input.kind == "Deployment"
    input.spec.template.spec.hostPID == true
    msg := "Deployment must not use host PID namespace"
}
