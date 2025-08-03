# MLops Automation with Enhanced MLflow Tracking & FastAPI Deployment

A comprehensive MLops pipeline for California Housing price prediction featuring advanced MLflow experiment tracking, FastAPI deployment, and monitoring with Prometheus/Grafana.

## 🚀 Project Overview

This project demonstrates a complete MLops workflow including:

- **Data Preparation**: Automated California Housing dataset loading
- **Model Training**: Linear Regression & Decision Tree with hyperparameter tuning
- **Experiment Tracking**: Comprehensive MLflow integration with rich metrics and visualizations
- **Model Deployment**: FastAPI-based REST API with prediction endpoint
- **Monitoring**: Prometheus metrics collection and Grafana dashboards
- **Containerization**: Docker-based deployment with docker-compose

## 📊 Enhanced MLflow Features

### Model Metrics & Visualizations

- **Learning Curves**: Training vs validation performance over dataset sizes
- **Residual Analysis**: Prediction error distributions and scatter plots
- **Feature Correlation Matrix**: Heatmap of feature relationships
- **Validation Curves**: Hyperparameter optimization visualization (for Decision Trees)
- **Distribution Analysis**: Target distribution, actual vs predicted, residuals
- **Feature Importance**: Tree-based model feature ranking
- **Cross-Validation Metrics**: Robust performance estimation with CV folds

### System Metrics

- **CPU Usage**: Real-time CPU utilization tracking during training
- **Memory Usage**: RAM consumption monitoring
- **Training Time**: Model training duration measurement
- **Inference Time**: Prediction latency per sample
- **Overfitting Analysis**: Training vs test performance ratios

### Advanced Analytics

- **Data Profiling**: Dataset statistics (mean, std, min, max, skewness, kurtosis)
- **Model Comparison**: Side-by-side performance across configurations
- **Hyperparameter Grid Search**: Automated testing of multiple max_depth values
- **Feature Scaling Experiments**: Scaled vs unscaled feature comparison

## 🛠️ Quick Start

### 1. Prerequisites & Setup

```bash
# Clone the repository
git clone <repository-url>
cd mlops-automation

# Install dependencies
pip install -r requirements.txt

# Generate California Housing dataset
python src/data_prep.py
```

### 2. MLflow Experiment Tracking

#### Start MLflow Server

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

#### Run Enhanced Model Training

```bash
cd src
python train.py --max_depth_list 3 5 7 10 15 --test_size 0.2 --cv_folds 5
```

### 3. Model Deployment & API

#### Build and Deploy with Docker

```bash
# Build the Docker image
docker build -t mlopsdemo/housing-api:latest .

# Run with docker-compose (includes Prometheus & Grafana)
docker-compose up -d
```

#### API Endpoints

- **Prediction**: `POST http://localhost:8000/predict`
- **Metrics**: `GET http://localhost:8000/metrics` (Prometheus format)
- **Retrain**: `POST http://localhost:8000/retrain` (demo endpoint)

### 4. Monitoring & Dashboards

- **MLflow UI**: <http://localhost:5000> (experiment tracking)
- **API Documentation**: <http://localhost:8000/docs> (FastAPI Swagger)
- **Prometheus**: <http://localhost:9090> (metrics collection)
- **Grafana**: <http://localhost:3000> (visualization dashboards)

## 📋 Command Line Arguments & API Usage

### Training Script Options

The enhanced `train.py` script supports these arguments:

```bash
python src/train.py [OPTIONS]

Options:
  --max_depth_list     List of max depths to test (default: [3, 5, 7, 10, 15])
  --test_size          Test set size ratio (default: 0.2)
  --random_state       Random seed for reproducibility (default: 42)
  --cv_folds          Cross-validation folds (default: 5)
```

### Examples

```bash
# Quick test with few hyperparameters
python src/train.py --max_depth_list 5 10 --cv_folds 3

# Comprehensive hyperparameter search
python src/train.py --max_depth_list 1 3 5 7 10 15 20 25 --cv_folds 10

# Different data split
python src/train.py --test_size 0.3 --random_state 123
```

### API Usage Examples

#### Making Predictions

```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "MedInc": 5.0,
       "HouseAge": 10.0,
       "AveRooms": 6.5,
       "AveBedrms": 1.2,
       "Population": 3000.0,
       "AveOccup": 3.0,
       "Latitude": 34.0,
       "Longitude": -118.0
     }'
```

```python
# Using Python requests
import requests

url = "http://localhost:8000/predict"
data = {
    "MedInc": 5.0,
    "HouseAge": 10.0,
    "AveRooms": 6.5,
    "AveBedrms": 1.2,
    "Population": 3000.0,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0
}

response = requests.post(url, json=data)
prediction = response.json()["prediction"]
```

## 📊 What You'll See in MLflow UI

### Experiment: "California-Housing-Regression-Enhanced"

#### Run Types

1. **LinearRegression-Standard**: Basic linear regression with original features
2. **LinearRegression-Scaled**: Linear regression with standardized features
3. **DecisionTreeRegressor-MaxDepth-X**: Decision trees with various max_depth values
4. **Best-Model-Register**: Best performing model registration

#### Model Metrics

- **Training/Test Metrics**: MSE, MAE, R² for both training and test sets
- **Cross-validation Scores**: Mean and standard deviation across CV folds
- **Learning Curves**: Step-wise training progress with different dataset sizes
- **Validation Curves**: Hyperparameter optimization curves (Decision Trees)
- **Overfitting Analysis**: Training vs test performance ratios
- **Inference Performance**: Average prediction time per sample

#### Runtime Metrics

- **Resource Utilization**: CPU percentage, memory usage, available memory
- **Performance Timing**: Training time, total inference time
- **Advanced Statistics**: Target skewness, kurtosis, correlation metrics

#### Artifacts & Visualizations

- **Learning Curve Plots**: Training progress visualization (plots/)
- **Residual Plots**: Error analysis with scatter plots and histograms (plots/)
- **Correlation Heatmaps**: Feature relationship analysis (advanced_plots/)
- **Validation Curves**: Hyperparameter tuning visualization (advanced_plots/)
- **Distribution Analysis**: Target, predictions, and residuals (advanced_plots/)
- **Model Files**: Trained models and preprocessing scalers

#### Parameters & Tags

- **Hyperparameters**: max_depth, feature_scaling, model_type
- **Experiment Metadata**: Dataset info, framework details, training info
- **Selection Criteria**: Best model selection method and total experiments

## 🏗️ Project Architecture

### Directory Structure

```text
mlops-automation/
├── src/                          # Source code
│   ├── train.py                  # Enhanced training with MLflow tracking
│   ├── api.py                    # FastAPI deployment server
│   ├── data_prep.py              # Dataset preparation utility
│   ├── DecisionTreeRegressor.py  # Decision tree model wrapper
│   └── LinearRegression.py       # Linear regression model wrapper
├── models/                       # Trained model artifacts
│   └── best_model.pkl           # Best performing model
├── data/                         # Dataset storage
│   └── california_housing.csv   # California housing dataset
├── grafana/                      # Grafana configuration
│   └── provisioning/            # Dashboard and datasource configs
├── logs/                         # API logs and request database
├── mlruns/                       # MLflow experiment tracking data
├── docker-compose.yaml          # Multi-service deployment
├── Dockerfile                   # API containerization
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

### Technology Stack

- **ML Framework**: scikit-learn
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI
- **Monitoring**: Prometheus + Grafana
- **Visualization**: matplotlib, seaborn
- **Containerization**: Docker
- **Database**: SQLite (API request logging)
- **Data Processing**: pandas, numpy

## 🔄 MLops Workflow

### 1. Data Preparation

- Automated California Housing dataset download via `data_prep.py`
- Dataset profiling and statistics logging to MLflow
- Train/test split with configurable ratios

### 2. Model Development & Training

- **Linear Regression**: Standard and feature-scaled variants
- **Decision Tree**: Multiple max_depth hyperparameter testing
- Cross-validation for robust performance estimation
- Comprehensive metric logging and visualization generation

### 3. Model Evaluation & Selection

- Cross-validation MSE-based model selection
- Performance comparison across all trained models
- Best model registration in MLflow Model Registry
- Local model artifact saving to `models/` directory

### 4. Model Deployment

- FastAPI-based REST API with automatic model loading
- Request/response logging with SQLite database
- Prometheus metrics exposure for monitoring
- Docker containerization for consistent deployment

### 5. Monitoring & Observability

- Prometheus metrics collection from API
- Grafana dashboards for visualization
- Request logging and prediction tracking
- System resource monitoring during training

## 📈 Key Features & Improvements

| Component | Features | Implementation |
|-----------|----------|----------------|
| **Data Pipeline** | Automated dataset loading, profiling, scaling | `data_prep.py`, feature preprocessing |
| **Model Training** | Linear Regression, Decision Trees, hyperparameter tuning | Modular `train.py` with enhanced evaluation |
| **Experiment Tracking** | 25+ metrics, visualizations, model registry | Comprehensive MLflow integration |
| **Model Deployment** | REST API, request logging, health monitoring | FastAPI with Pydantic validation |
| **Monitoring** | Prometheus metrics, Grafana dashboards | Container-based monitoring stack |
| **Containerization** | Docker images, multi-service orchestration | Docker + docker-compose setup |

### MLflow Enhancements

- **Metrics**: 25+ comprehensive metrics vs basic MSE/R²
- **Visualizations**: 6 plot types (learning curves, residuals, correlation, etc.)
- **System Monitoring**: Real-time CPU/memory tracking during training
- **Model Comparison**: Cross-validation based selection with performance analysis
- **Artifacts**: Plots, models, scalers automatically logged
- **Hyperparameter Tuning**: Grid search with validation curve analysis

### API Features

- **Prediction Endpoint**: JSON-based house price prediction
- **Request Logging**: File and SQLite database logging
- **Prometheus Metrics**: Prediction counter and custom metrics
- **Error Handling**: Comprehensive validation and error responses
- **Documentation**: Auto-generated OpenAPI/Swagger docs

## 🎯 Use Cases & Applications

1. **ML Model Development**: Compare algorithms and hyperparameters systematically
2. **Production Deployment**: REST API ready for integration with web/mobile apps
3. **Performance Monitoring**: Track model and system metrics in production
4. **Experiment Management**: Organize and compare training runs with rich metadata
5. **Resource Optimization**: Monitor system usage during training and inference
6. **Educational Purposes**: Complete MLops pipeline demonstration
7. **Proof of Concept**: Template for housing price prediction systems

## 🔍 Troubleshooting & Common Issues

### MLflow Issues

#### Empty Metrics in UI

1. Ensure MLflow server is running: `mlflow ui --host 0.0.0.0 --port 5000`
2. Check experiment name: "California-Housing-Regression-Enhanced"
3. Verify data file exists: `data/california_housing.csv`
4. Run data preparation: `python src/data_prep.py`

#### Missing Visualizations

1. Install visualization dependencies: `pip install matplotlib seaborn`
2. Check matplotlib backend compatibility (automatic Windows handling)
3. Verify artifact logging permissions and disk space

### API Deployment Issues

#### Docker Build Failures

1. Ensure model file exists: `models/best_model.pkl`
2. Check Docker daemon is running
3. Verify all dependencies in requirements.txt

#### API Connection Errors

1. Check port availability: `netstat -an | findstr 8000`
2. Verify container is running: `docker ps`
3. Check logs: `docker logs housing-api`

### Training Issues

#### Memory/Performance Problems

1. Reduce hyperparameter grid size: `--max_depth_list 3 5 7`
2. Decrease CV folds: `--cv_folds 3`
3. Monitor system resources during training

#### Missing Dependencies

```bash
# Install all required packages
pip install scikit-learn pandas mlflow joblib fastapi uvicorn pydantic prometheus-client matplotlib seaborn psutil numpy
```

### File Path Issues

#### Data Not Found

- Ensure data file exists: `data/california_housing.csv`
- Run data preparation: `python src/data_prep.py`
- Check current working directory when running scripts

#### Model Loading Errors

- Train a model first: `python src/train.py`
- Check model file location: `models/best_model.pkl`
- Verify model saving path configuration

## 📞 Support & Documentation

### Getting Help

For issues or questions about the MLops automation pipeline:

1. **Check the Documentation**: Review this README and inline code comments
2. **MLflow UI**: Check experiment logs and error messages at <http://localhost:5000>
3. **API Documentation**: View auto-generated docs at <http://localhost:8000/docs>
4. **Console Output**: Review training logs and error messages during execution
5. **Dependencies**: Ensure all packages are installed: `pip install -r requirements.txt`

### Development & Extension

This project serves as a foundation for:

- **Custom Models**: Add new algorithms in separate modules following the existing pattern
- **Enhanced APIs**: Extend FastAPI with additional endpoints and functionality
- **Advanced Monitoring**: Implement custom Grafana dashboards and Prometheus metrics
- **Data Pipelines**: Integrate with different datasets and preprocessing steps
- **Production Deployment**: Scale with Kubernetes, cloud platforms, or serverless functions

### File Structure for Development

- **Models**: Add new model types in `src/` following `LinearRegression.py` pattern
- **API Endpoints**: Extend `src/api.py` with additional routes and functionality
- **Monitoring**: Customize `grafana/` configurations for specific dashboards
- **Training**: Modify `src/train.py` for different experiments and evaluation metrics

---

## 🎉 Getting Started Summary

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Prepare Data**: `python src/data_prep.py`
3. **Start MLflow**: `mlflow ui --host 0.0.0.0 --port 5000`
4. **Train Models**: `python src/train.py --max_depth_list 3 5 7`
5. **Deploy API**: `docker-compose up -d`
6. **Explore Results**: Visit <http://localhost:5000> for MLflow UI

Your MLops pipeline is now ready with comprehensive experiment tracking, model deployment, and monitoring capabilities! 🚀
