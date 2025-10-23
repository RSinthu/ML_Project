# 💎 Gemstone Price Prediction

A machine learning web application that predicts gemstone prices based on their physical properties and quality attributes using multiple regression models. **Now live on Microsoft Azure with Docker containerization!**

🌐 **Live Demo**: [https://project001-g0hqf5brccbtcpfx.canadacentral-01.azurewebsites.net/predict](https://project001-g0hqf5brccbtcpfx.canadacentral-01.azurewebsites.net/predict)

## 🚀 Project Overview

This project implements an end-to-end machine learning pipeline for predicting gemstone prices. The system evaluates multiple regression algorithms and automatically selects the best performing model based on R² score. The application is containerized with Docker and deployed on Microsoft Azure App Service, featuring a user-friendly web interface for real-time price predictions.

## 📊 Dataset Features

The model uses the following gemstone characteristics:

### Physical Properties
- **Carat**: Weight of the gemstone
- **Depth**: Total depth percentage (z / mean(x, y))  
- **Table**: Width of top facet relative to widest point
- **X, Y, Z**: Length, width, and depth dimensions in mm

### Quality Attributes
- **Cut**: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- **Color**: Diamond color grade (D, E, F, G, H, I, J)
- **Clarity**: Measurement of clarity (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)

## 🏗️ Project Architecture

```
d:\Project\Deployment\
├── app.py                      # Flask web application (runs on port 8000)
├── requirements.txt            # Project dependencies
├── setup.py                   # Package setup
├── Dockerfile                  # Docker container configuration
├── .dockerignore              # Docker ignore patterns
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Data loading and splitting
│   │   ├── data_transformation.py # Preprocessing pipeline
│   │   └── model_trainer.py       # Model training and evaluation
│   ├── pipeline/
│   │   ├── predict_pipeline.py    # Prediction pipeline
│   │   └── train_pipeline.py      # Training pipeline (if exists)
│   ├── data/
│   │   └── gemstone.csv           # Dataset (if exists)
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging configuration
│   └── utils.py                   # Utility functions
├── static/
│   └── style.css                  # Web interface styling
├── templates/
│   └── index.html                 # Web interface template
└── artifacts/                     # Generated models and reports
    ├── model.pkl                  # Best trained model
    ├── preprocessor.pkl           # Data preprocessing pipeline
    └── model_evaluation_report.txt # Detailed model performance
```

## 🤖 Machine Learning Models

The system evaluates the following regression algorithms:

1. **Linear Regression**: Basic linear relationship modeling
2. **Lasso Regression**: Linear regression with L1 regularization
3. **Ridge Regression**: Linear regression with L2 regularization
4. **Decision Tree Regressor**: Non-linear tree-based model
5. **Random Forest Regressor**: Ensemble of decision trees
6. **XGBoost Regressor**: Gradient boosting algorithm

Each model undergoes hyperparameter tuning using GridSearchCV for optimal performance.

## 📈 Model Evaluation

The system generates comprehensive evaluation reports including:

- **Performance Metrics**: R² Score, MAE, RMSE for train/test sets
- **Overfitting Analysis**: Training vs testing performance comparison
- **Model Rankings**: Sorted by R² score performance
- **Best Parameters**: Optimal hyperparameters for each model

## 🐳 Docker Containerization

The application is fully containerized using Docker for consistent deployment across environments.

### Docker Configuration

- **Base Image**: `python:3.13-slim`
- **Port**: 80 (exposed), mapped to internal port 8000
- **Environment**: Production-optimized with minimal dependencies
- **System Dependencies**: `libgomp1` for XGBoost/scikit-learn optimization

### Docker Build & Run

1. **Build the Docker image**
```bash
docker build -t gemstone-predictor .
```

2. **Run the container locally**
```bash
docker run -p 8000:80 gemstone-predictor
```

3. **Access the application**
- Local: `http://localhost:8000`
- Container health check available


## ☁️ Azure Deployment

The application is deployed on **Microsoft Azure App Service** with Docker container support:

- **Deployment URL**: https://project001-g0hqf5brccbtcpfx.canadacentral-01.azurewebsites.net/predict
- **Region**: Canada Central
- **Platform**: Azure App Service (Container)
- **Runtime**: Docker + Python Flask application
- **Port Mapping**: 80 (Azure) → 8000 (Application)


## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- Docker (for containerized deployment)
- pip package manager

### Local Development Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Deployment
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install the package**
```bash
pip install -e .
```

### Docker Development Setup

1. **Build and run with Docker**
```bash
# Build the image
docker build -t gemstone-predictor .

# Run the container
docker run -p 8000:80 gemstone-predictor
```

## 🚀 Usage

### Accessing the Live Application

**🌐 Visit**: [https://project001-g0hqf5brccbtcpfx.canadacentral-01.azurewebsites.net/predict](https://project001-g0hqf5brccbtcpfx.canadacentral-01.azurewebsites.net/predict)

1. Enter gemstone specifications in the form
2. Click "Calculate Price" to get the prediction
3. View the estimated price instantly

### Running Locally

#### Option 1: Direct Python Execution
```bash
python app.py
```

#### Option 2: Docker Container
```bash
docker run -p 8000:80 gemstone-predictor
```

#### Option 3: Docker Development Mode
```bash
# Build for development
docker build -t gemstone-predictor-dev .

# Run with volume mounting for development
docker run -p 8000:80 -v $(pwd)/artifacts:/app/artifacts gemstone-predictor-dev
```

**Access the application**: `http://localhost:8000`

### Making Predictions Programmatically

```python
from src.pipeline.predict_pipeline import Predict_Pipeline, CustomData

# Create prediction pipeline
predictor = Predict_Pipeline()

# Prepare input data
data = CustomData(
    carat=0.50,
    depth=61.5,
    table=55.0,
    x=3.95,
    y=3.98,
    z=2.43,
    cut='Ideal',
    color='E',
    clarity='VS2'
)

# Convert to DataFrame
df = data.get_data_as_dataframe()

# Make prediction
prediction = predictor.predict(df)
print(f"Predicted Price: ${prediction[0]:.2f}")
```

## 📊 Model Performance

The system automatically selects the best performing model based on R² score. Typical performance metrics:

- **R² Score**: 0.85-0.95 (varies by model)
- **MAE**: Mean Absolute Error in price prediction
- **RMSE**: Root Mean Square Error
- **Overfitting Check**: Training vs Test performance analysis

## 🔧 Configuration

### Data Paths
- Raw data: `src/data/gemstone.csv` (if available)
- Processed data: `artifacts/`
- Models: `artifacts/model.pkl`
- Preprocessor: `artifacts/preprocessor.pkl`

### Application Configuration
- **Local Port**: 8000 (as configured in [`app.py`](app.py))
- **Docker Port**: 80 (exposed), 8000 (internal)
- **Host**: 0.0.0.0 (allows external connections)
- **Debug Mode**: Disabled in production

### Hyperparameter Grids
The system uses predefined hyperparameter grids for each model, which can be customized in [`ModelTrainer.initiate_model_training()`](src/components/model_trainer.py).

## 📝 Logging

The application includes comprehensive logging:
- Log files: `logs/MM_DD_YYYY_HH_MM_SS.log`
- Logging configuration: [`src/logger.py`](src/logger.py)
- Custom exceptions: [`src/exception.py`](src/exception.py)
- Docker logs: `docker logs <container-id>`
- Azure Application Insights integration

## 🎨 Web Interface Features

- **Modern UI**: Gradient backgrounds and smooth animations
- **Responsive Design**: Works on desktop and mobile devices
- **Form Validation**: Real-time input validation
- **Interactive Elements**: Hover effects and smooth transitions
- **Results Display**: Clear price prediction with formatting
- **Cross-browser Compatibility**: Works across all modern browsers

## 🔍 Data Preprocessing

The preprocessing pipeline includes:

1. **Numerical Features**: Median imputation + Standard scaling
2. **Categorical Features**: Most frequent imputation + One-hot encoding
3. **Feature Engineering**: Automatic categorical/numerical detection
4. **Pipeline Persistence**: Saves preprocessing steps for inference

## 📋 Dependencies

Core dependencies from [`requirements.txt`](requirements.txt):
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms
- `xgboost`: Gradient boosting framework
- `flask`: Web framework
- `flask-cors`: Cross-origin resource sharing
- `seaborn`: Statistical visualization
- `dill`: Serialization library

## 🌐 API Endpoints

### Main Endpoints
- **Home**: `/` - Renders the main prediction interface
- **Predict**: `/predict` - Handles both GET and POST requests for predictions

### Request Format (POST to /predict)
```json
{
  "carat": 0.50,
  "depth": 61.5,
  "table": 55.0,
  "x": 3.95,
  "y": 3.98,
  "z": 2.43,
  "cut": "Ideal",
  "color": "E",
  "clarity": "VS2"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker: `docker build -t test-build .`
5. Add tests if applicable
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Sinthujan**
- Email: sinthujanp2135@gmail.com
- Project deployed on Microsoft Azure with Docker

## 🎯 Future Enhancements

- [ ] Add more advanced ensemble methods — Implement stacking, blending, and voting ensembles (cross-validated meta-models and stacked regressors) to combine strengths of multiple base learners.
- [ ] Implement feature importance analysis — Provide global and local importance using model importances, permutation importance, and SHAP summary plots; persist ranked feature reports.
- [ ] Add model interpretability features — Integrate SHAP/LIME explanations, partial dependence plots, and per-prediction explanation endpoints for transparency.
- [ ] Add API documentation with Swagger — Add OpenAPI/Swagger UI (e.g., flask-restx or flask-smorest) to document endpoints, request/response schemas, and examples.
- [ ] Implement model retraining pipeline — Automate scheduled or data-triggered retraining, validation, model versioning/registry updates, and safe rollout (canary/validation) of new models.
- [ ] Add data drift detection — Monitor feature and target distributions (PSI, KS test), log drift metrics, raise alerts, and optionally trigger retraining when thresholds are exceeded.

---

🌐 **Live Application**: [https://project001-g0hqf5brccbtcpfx.canadacentral-01.azurewebsites.net/predict](https://project001-g0hqf5brccbtcpfx.canadacentral-01.azurewebsites.net/predict)

🐳 **Containerized with Docker** for consistent deployment across all environments

For questions or issues, please open an issue in the repository or contact the author directly.