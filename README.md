# 🏦 Bank Marketing Prediction ML Project

[![Python](https://img.shields.io/badge/Python-3.12.0-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green.svg)](https://xgboost.readthedocs.io/)

An advanced machine learning project that predicts customer subscription to bank term deposits using telemarketing campaign data. Features a stacked ensemble model achieving 85.85% accuracy and an interactive Streamlit web application for real-time predictions.

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🛠️ Technology Stack](#️-technology-stack)
- [📊 Dataset Information](#-dataset-information)
- [🚀 Installation & Setup](#-installation--setup)
- [🎮 Usage](#-usage)
- [🏗️ Project Architecture](#️-project-architecture)
- [🤖 Model Details](#-model-details)
- [📈 Performance Metrics](#-performance-metrics)
- [🎨 Web Application](#-web-application)
- [📁 Project Structure](#-project-structure)
- [🔧 Development](#-development)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📞 Contact](#-contact)

## 🎯 Project Overview

This project addresses the challenge of predicting customer subscription to term deposits in a Portuguese banking institution's telemarketing campaign. The solution combines advanced machine learning techniques with an intuitive web interface to help banks optimize their marketing efforts and improve conversion rates.

### Business Problem
- **Challenge**: Low conversion rates in telemarketing campaigns (11.7% success rate)
- **Goal**: Predict which customers are likely to subscribe to term deposits
- **Impact**: Reduce marketing costs and improve campaign effectiveness

### Solution Approach
- **Data-Driven**: Analyzes historical campaign data with 45,211 customer records
- **ML-Powered**: Stacked ensemble model combining XGBoost, Random Forest, and HistGradientBoosting
- **User-Friendly**: Interactive web application for real-time predictions
- **Scalable**: Modular architecture for easy deployment and maintenance

## ✨ Key Features

### 🔬 Machine Learning
- **Stacked Ensemble Model**: Combines multiple algorithms for superior performance
- **Class Imbalance Handling**: Advanced techniques for imbalanced datasets
- **Hyperparameter Optimization**: Fine-tuned model parameters for optimal results
- **Cross-Validation**: Robust evaluation using 5-fold cross-validation

### 🌐 Web Application
- **Interactive Interface**: Modern, responsive design with gradient styling
- **Real-Time Predictions**: Instant results with confidence scores
- **Simplified Input**: Streamlined form with 7 key predictive features
- **Visual Feedback**: Success animations and detailed customer summaries

### 📊 Analytics & Insights
- **Performance Metrics**: Comprehensive evaluation (Accuracy, AUC, Precision, Recall, F1-Score)
- **Feature Importance**: Analysis of key predictive factors
- **Confusion Matrix**: Detailed classification performance breakdown
- **Business Impact**: ROI-focused metrics and recommendations

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.12.0**: Primary programming language
- **Streamlit 1.28.0**: Web application framework
- **scikit-learn 1.3.0**: Machine learning library
- **XGBoost 1.7.6**: Gradient boosting framework

### Supporting Libraries
- **pandas 2.0.3**: Data manipulation and analysis
- **numpy 1.24.3**: Numerical computing
- **joblib**: Model serialization
- **plotly**: Interactive visualizations

### Development Tools
- **Jupyter Notebook**: Exploratory data analysis
- **VS Code**: Integrated development environment
- **Git**: Version control
- **Virtual Environment**: Dependency management

## 📊 Dataset Information

### Source
- **Dataset**: Bank Marketing Dataset (UCI Machine Learning Repository)
- **Institution**: Portuguese banking institution
- **Campaign Type**: Direct telemarketing campaigns for term deposits

### Dataset Characteristics
- **Records**: 45,211 customer interactions
- **Features**: 16 predictive variables + 1 target variable
- **Time Period**: May 2008 - November 2010
- **Target Variable**: `deposit` (subscription to term deposit)

### Feature Categories

#### Demographic Features
- `age`: Customer age (18-95 years)
- `job`: Occupation (12 categories: admin, blue-collar, entrepreneur, etc.)
- `marital`: Marital status (married, single, divorced)
- `education`: Education level (primary, secondary, tertiary, unknown)

#### Financial Features
- `balance`: Average yearly balance in euros
- `default`: Credit default status (yes/no)
- `housing`: Housing loan status (yes/no)
- `loan`: Personal loan status (yes/no)

#### Campaign Features
- `contact`: Contact communication type (cellular, telephone)
- `day`: Last contact day of month (1-31)
- `month`: Last contact month (jan-dec)
- `duration`: Last contact duration in seconds (0-4918)
- `campaign`: Number of contacts during this campaign (1-63)
- `pdays`: Days since last contact (-1 if not previously contacted)
- `previous`: Number of contacts before this campaign (0-275)
- `poutcome`: Outcome of previous marketing campaign

### Data Quality
- **Missing Values**: None (clean dataset)
- **Data Types**: Mixed (categorical: 10, numerical: 7)
- **Class Distribution**: Highly imbalanced (88.3% no, 11.7% yes)
- **Outliers**: Present in balance, duration, and campaign features

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/bank-marketing-prediction.git
cd bank-marketing-prediction
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Download Dataset
- Place `bank_dataset.csv` in the project root directory
- Dataset available from UCI Machine Learning Repository

#### 5. Train the Model (Optional)
```bash
python model_training_best.py
```
This will generate `best_final_model.pkl` and `scaler.pkl`

#### 6. Run the Application
```bash
streamlit run app.py
```

#### 7. Access the Application
Open your browser and navigate to: `http://localhost:8501`

## 🎮 Usage

### Web Application Interface

#### 1. **Header Section**
- Project title with gradient background
- Brief description of the prediction system

#### 2. **Input Sidebar**
Enter customer information using the simplified form:

- **Age**: Customer age (18-100 years)
- **Bank Balance**: Current account balance in rupees
- **Job**: Occupation category
- **Housing Loan**: Whether customer has housing loan
- **Contact Type**: Preferred contact method
- **Contact Duration**: Length of last contact in seconds
- **Previous Campaign Outcome**: Result of previous marketing attempt
- **Previous Contacts**: Number of prior contacts

#### 3. **Prediction Results**
- **Binary Prediction**: Subscribe / Not Subscribe
- **Confidence Scores**: Probability percentages for both outcomes
- **Celebration Animation**: Balloons for successful predictions
- **Customer Summary**: Formatted display of all input data

#### 4. **Advanced Analytics** (Expandable)
- Model accuracy and AUC scores
- Confusion matrix visualization
- Classification report details

### API Usage (Programmatic)

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('best_final_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare input data
input_data = {
    'age': 35,
    'balance': 1500,  # Scaled value
    'job': 0,  # Encoded job category
    'housing': 0,  # Encoded housing status
    'contact': 0,  # Encoded contact type
    'duration': 250,
    'poutcome': 0,  # Encoded previous outcome
    'previous': 0
}

# Create DataFrame and preprocess
input_df = pd.DataFrame([input_data])
input_df[['age', 'balance', 'duration', 'previous']] = scaler.transform(
    input_df[['age', 'balance', 'duration', 'previous']]
)

# Make prediction
prediction = model.predict(input_df)[0]
confidence = model.predict_proba(input_df)[0]

print(f"Prediction: {'Subscribe' if prediction == 1 else 'Not Subscribe'}")
print(f"Confidence: {confidence[prediction]*100:.1f}%")
```

## 🏗️ Project Architecture

### Model Architecture
```
Input Data (16 features)
    │
    ├── Categorical Encoding (LabelEncoder)
    │
    ├── Numerical Scaling (StandardScaler)
    │
    └── Feature Selection (7 key features)
        │
        ├── Base Learners
        │   ├── XGBoost Classifier
        │   ├── Random Forest Classifier
        │   └── HistGradientBoosting Classifier
        │
        └── Meta-Learner (Logistic Regression)
            │
            └── Final Prediction
```

### Application Architecture
```
Streamlit App
    │
    ├── Frontend Layer
    │   ├── Header Component
    │   ├── Sidebar Input Form
    │   └── Results Display
    │
    ├── Backend Layer
    │   ├── Model Loading (joblib)
    │   ├── Data Preprocessing
    │   └── Prediction Engine
    │
    └── Data Layer
        ├── Model Files (.pkl)
        └── Scaler Objects
```

## 🤖 Model Details

### Algorithm Selection

#### Base Models
1. **XGBoost Classifier**
   - Gradient boosting framework
   - Handles missing values and outliers
   - Built-in regularization

2. **Random Forest Classifier**
   - Ensemble of decision trees
   - Robust to overfitting
   - Feature importance ranking

3. **HistGradientBoosting Classifier**
   - Histogram-based gradient boosting
   - Faster training than traditional GBM
   - Memory efficient

#### Meta-Learner
- **Logistic Regression**: Simple, interpretable meta-classifier
- Combines base model predictions with original features
- Final probability calibration

### Hyperparameter Configuration

#### XGBoost Parameters
```python
{
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 1,
    'scale_pos_weight': 1.0
}
```

#### Random Forest Parameters
```python
{
    'n_estimators': 300,
    'max_depth': 20,
    'min_samples_split': 3,
    'class_weight': 'balanced'
}
```

#### HistGradientBoosting Parameters
```python
{
    'max_iter': 300,
    'learning_rate': 0.05,
    'max_depth': 8
}
```

### Training Process
1. **Data Preparation**: Encoding, scaling, train-test split
2. **Base Model Training**: Individual model optimization
3. **Stacking Assembly**: Meta-learner training with cross-validation
4. **Model Evaluation**: Comprehensive performance assessment
5. **Model Serialization**: Save trained model and preprocessing objects

## 📈 Performance Metrics

### Overall Performance
- **Accuracy**: 85.85%
- **AUC-ROC**: 0.925
- **Precision**: 0.62
- **Recall**: 0.48
- **F1-Score**: 0.54

### Class-Specific Performance
```
              precision    recall  f1-score   support

       No        0.97      0.98      0.97      7799
      Yes        0.59      0.48      0.53       524

accuracy                           0.94      8323
macro avg       0.78      0.73      0.75      8323
weighted avg     0.94      0.94      0.94      8323
```

### Confusion Matrix
```
Predicted:     No    Yes
Actual: No   7628   171
        Yes   275   249
```

### Business Impact Analysis
- **True Positives (249)**: Correctly identified potential subscribers
- **False Negatives (275)**: Missed opportunities (cost: lost revenue)
- **False Positives (171)**: Unnecessary marketing efforts (cost: campaign expenses)
- **True Negatives (7628)**: Efficient resource allocation

## 🎨 Web Application

### Design Philosophy
- **Modern UI**: Gradient backgrounds and rounded corners
- **Responsive Design**: Works on desktop and mobile devices
- **Intuitive UX**: Simplified input form with smart defaults
- **Visual Feedback**: Animations and color-coded results

### Key Components

#### Header Section
```css
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 1rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}
```

#### Interactive Elements
- **Hover Effects**: Button animations and transitions
- **Form Validation**: Input constraints and error handling
- **Loading States**: Progress indicators during prediction
- **Success Animations**: Balloon celebrations for positive predictions

#### Customer Summary Display
- **Grid Layout**: Organized information presentation
- **Currency Formatting**: Proper INR display with commas
- **Responsive Cards**: White background with subtle shadows

## 📁 Project Structure

```
bank-marketing-prediction/
│
├── 📄 app.py                          # Main Streamlit application
├── 📄 model_training_best.py          # Model training script
├── 📄 preprocessing.py                # Data preprocessing utilities
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # Project documentation
│
├── 📊 bank_dataset.csv                # Raw dataset (45,211 records)
├── 🤖 best_final_model.pkl            # Trained stacked ensemble model
├── 🔧 scaler.pkl                      # Feature scaling object
│
├── 📁 .venv/                          # Virtual environment
├── 📁 __pycache__/                    # Python cache files
└── 📁 .git/                           # Git repository (if applicable)
```

### File Descriptions

#### Core Files
- **`app.py`**: Complete web application with UI, prediction logic, and styling
- **`model_training_best.py`**: Model development, training, and evaluation pipeline
- **`preprocessing.py`**: Data cleaning, encoding, and transformation utilities

#### Data Files
- **`bank_dataset.csv`**: Original UCI Bank Marketing dataset
- **`best_final_model.pkl`**: Serialized trained model (StackingClassifier)
- **`scaler.pkl`**: Serialized StandardScaler for feature normalization

#### Configuration Files
- **`requirements.txt`**: All Python package dependencies with versions
- **`.gitignore`**: Git ignore patterns for sensitive/large files

## 🔧 Development

### Local Development Setup

#### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
```

#### 2. Data Preparation
```bash
# Run preprocessing
python preprocessing.py

# Train model
python model_training_best.py
```

#### 3. Application Development
```bash
# Run in development mode
streamlit run app.py --server.headless true --server.port 8501
```

### Testing & Validation

#### Model Testing
```python
# Quick model validation
python -c "
import joblib
model = joblib.load('best_final_model.pkl')
print('Model loaded successfully')
print('Model type:', type(model).__name__)
"
```

#### Application Testing
```bash
# Test Streamlit app
streamlit run app.py

# Check for import errors
python -c "import streamlit, sklearn, xgboost; print('All imports successful')"
```

### Code Quality

#### Best Practices Implemented
- **Modular Design**: Separate concerns across files
- **Error Handling**: Try-except blocks for robust execution
- **Documentation**: Comprehensive comments and docstrings
- **Reproducibility**: Fixed random seeds and version control

#### Performance Optimization
- **Caching**: `@st.cache_resource` for expensive operations
- **Parallel Processing**: `n_jobs=-1` for model training
- **Memory Efficiency**: Optimized data structures and algorithms

## 🤝 Contributing

We welcome contributions to improve the Bank Marketing Prediction project!

### Ways to Contribute
- 🐛 **Bug Reports**: Submit issues for bugs or unexpected behavior
- ✨ **Feature Requests**: Suggest new features or improvements
- 🔧 **Code Contributions**: Submit pull requests with enhancements
- 📖 **Documentation**: Improve documentation and tutorials

### Contribution Process

#### 1. Fork the Repository
```bash
git clone https://github.com/your-username/bank-marketing-prediction.git
cd bank-marketing-prediction
```

#### 2. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

#### 3. Make Changes
- Follow existing code style and conventions
- Add tests for new functionality
- Update documentation as needed

#### 4. Commit Changes
```bash
git add .
git commit -m "Add: Brief description of your changes"
```

#### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

### Guidelines
- **Code Style**: Follow PEP 8 Python conventions
- **Commits**: Use clear, descriptive commit messages
- **Testing**: Ensure all tests pass before submitting
- **Documentation**: Update README for significant changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Bank Marketing Prediction Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 📞 Contact

### Project Maintainers
- **Name**: [Your Name]
- **Email**: [your.email@example.com]
- **LinkedIn**: [Your LinkedIn Profile]
- **GitHub**: [@your-username](https://github.com/your-username)

### Academic Supervisor (if applicable)
- **Name**: [Supervisor Name]
- **Email**: [supervisor.email@university.edu]
- **Department**: [Department Name]

### Support
- **Issues**: [GitHub Issues](https://github.com/your-username/bank-marketing-prediction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/bank-marketing-prediction/discussions)
- **Documentation**: [Project Wiki](https://github.com/your-username/bank-marketing-prediction/wiki)

### Acknowledgments
- **Dataset Source**: UCI Machine Learning Repository
- **Institution**: University/College Name
- **Special Thanks**: Mentors, peers, and contributors

---

## 🎯 Quick Start Commands

```bash
# One-line setup and run
git clone <repository-url> && cd bank-marketing-prediction && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && streamlit run app.py
```

## 🔍 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check if model files exist
ls -la *.pkl

# Re-train model if needed
python model_training_best.py
```

#### Package Installation Issues
```bash
# Update pip
pip install --upgrade pip

# Install specific versions
pip install scikit-learn==1.3.0 xgboost==1.7.6 streamlit==1.28.0
```

#### Port Already in Use
```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Or use different port
streamlit run app.py --server.port 8502
```

---

**⭐ Star this repository if you found it helpful!**

**📧 Questions? Feel free to open an issue or start a discussion!**
Run the command-line version:

```
python predict.py
```

Then enter the values when prompted:
- **Age**: (press Enter for default 35, or type a number)
- **Job**: (press Enter for default Admin, or type a job)
- **Balance**: (press Enter for default 50000, or type an amount)

### Option 3: Full Pipeline
Run the entire pipeline:

```
python main.py
```

This will:
1. Explore the data and generate visualizations
2. Preprocess the data
3. Train the models
4. Evaluate the models
5. Run a prediction example

## Files

- `app.py`: Interactive web application with dropdown options
- `predict.py`: Command-line prediction interface
- `data_exploration.py`: Data loading and visualization
- `preprocessing.py`: Data cleaning and preparation
- `model_training.py`: Model training and evaluation
- `main.py`: Main script to run the project
- `requirements.txt`: Python dependencies

## Model Performance

The XGBoost model achieves the best performance with:
- **Accuracy**: 84.55%
- **AUC Score**: 0.9202
- **Balanced predictions** for both subscription and non-subscription cases

## Simplified Interface

The web app uses **feature importance analysis** to only ask for the most predictive customer attributes:
1. **Contact Type** (20% importance)
2. **Contact Duration** (16% importance) 
3. **Previous Outcome** (12% importance)
4. **Housing Loan** (11% importance)
5. **Previous Contacts** (7% importance)
6. **Age** and **Balance** (basic customer info)

This reduces input complexity while maintaining high prediction accuracy!

## Usage

Install dependencies first:

```
pip install -r requirements.txt
```

Run the project pipeline:

```
python main.py
```

Launch the separate web page interface:

```
streamlit run app.py
```

This will open a browser page where you can enter customer details and see:
- prediction result
- confidence score
- model used
- accuracy
- confusion matrix
- classification report

## Results

After running, check the console output for model accuracies and classification reports. Visualizations will be saved as PNG files.