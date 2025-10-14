# 💎 Gemstone Price Prediction

A machine learning web application that predicts gemstone prices based on their physical properties and quality attributes using multiple regression models.

## 🚀 Project Overview

This project implements an end-to-end machine learning pipeline for predicting gemstone prices. The system evaluates multiple regression algorithms and automatically selects the best performing model based on R² score. It includes a user-friendly web interface for real-time price predictions.

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
├── app.py                      # Flask web application
├── requirements.txt            # Project dependencies
├── setup.py                   # Package setup
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Data loading and splitting
│   │   ├── data_transformation.py # Preprocessing pipeline
│   │   └── model_trainer.py       # Model training and evaluation
│   ├── pipeline/
│   │   ├── predict_pipeline.py    # Prediction pipeline
│   │   └── train_pipeline.py      # Training pipeline
│   ├── data/
│   │   └── gemstone.csv           # Dataset
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

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Installation Steps

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

## 🚀 Usage

### Training the Model

1. **Run the complete training pipeline**
```python
from src.pipeline.train_pipeline import TrainPipeline

# Initialize and run training
pipeline = TrainPipeline()
pipeline.run_pipeline()
```

This will:
- Load and split the gemstone dataset
- Apply data preprocessing and feature engineering
- Train and evaluate multiple ML models
- Save the best model and preprocessing pipeline
- Generate a comprehensive evaluation report

### Running the Web Application

1. **Start the Flask server**
```bash
python app.py
```

2. **Access the web interface**
- Open your browser and go to `http://localhost:5000`
- Enter gemstone specifications in the form
- Click "Calculate Price" to get the prediction

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
- Raw data: `src/data/gemstone.csv`
- Processed data: `artifacts/`
- Models: `artifacts/model.pkl`
- Preprocessor: `artifacts/preprocessor.pkl`

### Hyperparameter Grids
The system uses predefined hyperparameter grids for each model, which can be customized in [`ModelTrainer.initiate_model_training()`](src/components/model_trainer.py).

## 📝 Logging

The application includes comprehensive logging:
- Log files: `logs/MM_DD_YYYY_HH_MM_SS.log`
- Logging configuration: [`src/logger.py`](src/logger.py)
- Custom exceptions: [`src/exception.py`](src/exception.py)

## 🎨 Web Interface Features

- **Modern UI**: Gradient backgrounds and smooth animations
- **Responsive Design**: Works on desktop and mobile devices
- **Form Validation**: Real-time input validation
- **Interactive Elements**: Hover effects and smooth transitions
- **Results Display**: Clear price prediction with formatting

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
- `seaborn`: Statistical visualization
- `dill`: Serialization library

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Sinthujan**
- Email: sinthujanp2135@gmail.com

## 🎯 Future Enhancements

- [ ] Add more advanced ensemble methods
- [ ] Implement feature importance analysis
- [ ] Add model interpretability features
- [ ] Include confidence intervals for predictions
- [ ] Add API documentation with Swagger
- [ ] Implement model retraining pipeline
- [ ] Add data drift detection

---

For questions or issues, please open an issue in the repository or contact the author directly.