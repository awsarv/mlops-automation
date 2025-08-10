# MLOps Pipeline for California Housing Price Prediction

## Overview
This project implements an end-to-end MLOps workflow for predicting median housing prices using the California Housing dataset.  
It covers dataset versioning, experiment tracking, CI/CD automation, containerized deployment, and monitoring.

---

## Architecture & Workflow

1. **Data Versioning**
   - Dataset loaded via `fetch_california_housing()` and saved as `data/california_housing.csv`.
   - Tracked using **DVC** with remote storage in AWS S3 (`mlops-grp13/mlops-app/files`).
   - Changes to dataset automatically reflected in Git/DVC.

2. **Model Training & Experiment Tracking**
   - Trains **Linear Regression** and **Decision Tree Regressor**.
   - Tracks parameters, metrics (MSE, RMSE, R²), and artifacts in **MLflow**.
   - Best model registered in MLflow Model Registry as **BestHousingModel**.
   - Model artifacts versioned with DVC.

3. **API & Deployment**
   - **FastAPI** REST API for predictions.
   - `POST /predict` accepts JSON input and returns predicted median house value (in $100,000 units).
   - `POST /retrain` triggers retraining and updates model in MLflow & DVC.
   - Dockerized and deployed locally/EC2.

4. **CI/CD Automation**
   - **GitHub Actions** workflow:
     - Runs linting/tests on push.
     - Builds Docker image and pushes to Docker Hub (`mlopsdemo/housing-api`).
     - Deploys container using `docker-compose` or `docker run`.

5. **Logging & Monitoring**
   - All prediction requests and outputs logged to:
     - `logs/api.log` (file)
     - SQLite database (`logs/api_requests.db`)
   - **Prometheus** scrapes `/metrics` for request count, latency, etc.
   - **Grafana** dashboard for real-time visualization.

---

## Tech Stack
- **Language:** Python 3
- **ML:** scikit-learn, Pandas, MLflow
- **Infra:** Docker, DVC, GitHub Actions, AWS S3, EC2
- **Monitoring:** Prometheus, Grafana
- **API:** FastAPI

---

## Key Endpoints
- `POST /predict` — Predict median house value
- `POST /retrain` — Retrain model with latest data
- `/docs` — Swagger UI for API testing
- `/metrics` — Prometheus metrics endpoint

---

## Links
- **GitHub Repo:** `https://github.com/awsarv/mlops-automation`
- **MLflow UI:** `http://<server-ip>:5000`
- **Grafana Dashboard:** `http://<server-ip>:3000`
