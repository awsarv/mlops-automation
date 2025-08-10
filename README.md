# MLOps Automation Project – Housing Price Prediction

## 📌 Project Overview
This project implements a complete **MLOps pipeline** for housing price prediction, covering:
- Data versioning  
- Model training & experiment tracking  
- API service deployment with Docker  
- CI/CD automation using GitHub Actions  
- Logging & monitoring with Prometheus and Grafana  

It follows best practices for **reproducibility**, **automation**, and **observability**.

---

## 🗂 Repository Structure
```
.
├── data/                         # Raw & processed datasets
├── models/                       # Saved model files
├── notebooks/                    # Jupyter notebooks for EDA & experiments
├── src/                          # Source code for training & API
├── mlruns/                        # MLflow experiment tracking
├── grafana/provisioning/         # Dashboards & datasources
├── .github/workflows/            # CI/CD pipelines
├── Dockerfile                    # API container definition
└── README.md                     # Project documentation
```

---

## 🚀 Features Implemented

### **Part 1 – Repository & Data Versioning** ✅
- Clean GitHub repo structure
- Dataset loading & preprocessing
- Optional dataset tracking with **DVC**
- Version-controlled code and data

### **Part 2 – Model Development & Experiment Tracking** ✅
- Trained **Linear Regression** & **Decision Tree** models
- Logged parameters, metrics, and artifacts in **MLflow**
- Registered the best model for deployment

### **Part 3 – API & Docker Packaging** ✅
- Built a **FastAPI** prediction service
- Accepts JSON input and returns model predictions
- Containerized using **Docker**

### **Part 4 – CI/CD with GitHub Actions** ✅
- Automated **linting & testing** on code push
- Built & pushed Docker image to Docker Hub
- Deployment-ready via shell script or `docker run`

### **Part 5 – Logging & Monitoring** ✅
- Logged incoming requests and predictions
- Integrated **Prometheus metrics** at `/metrics`
- Configured **Grafana dashboard** for real-time API monitoring

### **Part 6 – Summary & Demo** ✅
- One-page architecture summary
- Recorded 5-min demo video showcasing:
  - Training
  - API usage
  - Monitoring dashboard

### **Bonus Features** 🎯
- Input validation using **Pydantic**
- End-to-end monitoring with **Prometheus & Grafana**
- Hooks for automated retraining on new data

---

## 🖥 Architecture Diagram

```mermaid
flowchart TD
    A[Dataset] --> B[Preprocessing]
    B --> C[Model Training + MLflow]
    C --> D[Best Model Registry]
    D --> E[FastAPI Prediction API]
    E --> F[Docker Container]
    F --> G[Deployment: EC2 / Local]
    G --> H[Prometheus Metrics]
    H --> I[Grafana Dashboard]

---

## 📊 Monitoring & Observability
- **Prometheus** scrapes `/metrics` endpoint from FastAPI
- **Grafana** displays API request rate, prediction latency, error counts
- Pre-configured dashboard JSON in `grafana/provisioning/dashboards/`

---

## 🐳 Docker Image
Public Docker Hub Repository:  
[https://hub.docker.com/r/mlopsdemo/housing-api](https://hub.docker.com/r/mlopsdemo/housing-api)

**Pull the latest image:**
```bash
docker pull mlopsdemo/housing-api:latest
```

**Run the container locally:**
```bash
docker run -d -p 8000:8000 mlopsdemo/housing-api:latest
```

**Access API:**
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Metrics endpoint: [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

## ⚡ Steps to Test End-to-End
1. **Clone the repo**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Run Docker container**
   ```bash
   docker run -d -p 8000:8000 mlopsdemo/housing-api:latest
   ```

3. **Test API prediction**
   ```bash
   curl -X POST "http://localhost:8000/predict"    -H "Content-Type: application/json"    -d '{"feature1": 1.2, "feature2": 3.4, "feature3": 5.6}'
   ```

4. **Check monitoring metrics**
   - Prometheus: `http://<server-ip>:9090`
   - Grafana Dashboard: `http://<server-ip>:3000`

5. **View MLflow UI**
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```
   Open in browser: `http://<server-ip>:5000`

---

## 📄 Evaluation Mapping
| Task | Implementation | Status |
|------|----------------|--------|
| Part 1 – Repo & Data Versioning | GitHub + DVC | ✅ |
| Part 2 – Model Development & Tracking | MLflow + 2 Models | ✅ |
| Part 3 – API & Docker | FastAPI + Docker | ✅ |
| Part 4 – CI/CD | GitHub Actions | ✅ |
| Part 5 – Logging & Monitoring | Prometheus + Grafana | ✅ |
| Part 6 – Summary & Demo | README + Video | ✅ |
| Bonus | Validation + Retraining Hooks | ✅ |
