# Insurance Charges ML Analysis

Comprehensive machine learning analysis workflow for predicting insurance charges based on demographic and health factors.

## Overview

This project implements an intensive ML analysis using multiple machine learning models to predict insurance charges. The analysis includes:

- **Data Exploration**: Statistical summaries, correlation analysis, and visualizations
- **Multiple ML Models**: 
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor (SVR)
  - XGBoost (if available)
  - Neural Network / MLP (if available)
- **Comprehensive Evaluation**: R² Score, MAE, RMSE, MAPE metrics
- **Visualizations**: Model comparisons, predictions vs actual, residuals analysis, feature importance
- **Detailed Report**: Conclusions and recommendations based on results

## Dataset

The insurance dataset contains the following features:
- `age`: Age of the insured person
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index
- `children`: Number of children/dependents
- `smoker`: Smoking status (yes/no)
- `region`: Geographic region
- `charges`: Insurance charges (target variable)

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Set up Kaggle credentials** (if downloading dataset automatically):
   - Install Kaggle CLI: `pip install kaggle`
   - Place your `kaggle.json` API credentials file in `~/.kaggle/`
   - Or the script will use `kagglehub` which handles authentication automatically

## Usage

### Quick Start

Simply run the main analysis script:

```bash
python ml_analysis.py
```

The script will:
1. Automatically download the dataset from Kaggle (if `kagglehub` is installed)
2. Load and explore the data
3. Preprocess the data (encode categorical variables, scale features)
4. Train multiple ML models
5. Evaluate all models with comprehensive metrics
6. Generate visualizations
7. Create a detailed analysis report

### Output

All results are saved in the `ml_results/` directory:

- **`correlation_heatmap.png`**: Correlation matrix of numeric features
- **`eda_visualizations.png`**: Exploratory data analysis plots
- **`model_comparison.csv`**: Performance metrics for all models
- **`model_comparison_metrics.png`**: Bar charts comparing model performance
- **`best_model_analysis_*.png`**: Detailed analysis of the best performing model
- **`feature_importance.png`**: Feature importance plots for tree-based models
- **`analysis_report.txt`**: Comprehensive text report with conclusions
- **`results.json`**: Machine-readable results in JSON format

### Using Local Dataset

If you already have the `insurance.csv` file, place it in the project directory and the script will automatically detect it.

## Model Performance

The script evaluates models using multiple metrics:
- **R² Score**: Coefficient of determination (higher is better, max 1.0)
- **MAE**: Mean Absolute Error in dollars (lower is better)
- **RMSE**: Root Mean Squared Error in dollars (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)

## Results Interpretation

After running the analysis:

1. **Check the console output** for real-time progress and key metrics
2. **Review `analysis_report.txt`** for comprehensive conclusions
3. **Examine visualizations** in the `ml_results/` directory
4. **Compare models** using `model_comparison.csv`

The report includes:
- Dataset overview and statistics
- Key insights from exploratory analysis
- Model performance comparison
- Best model identification
- Feature importance analysis
- Business insights and recommendations

## Next Steps

After reviewing the results, you can:

1. **Implement model improvements** based on recommendations in the report
2. **Tune hyperparameters** for better performance
3. **Try additional models** or ensemble methods
4. **Deploy the best model** for production use

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

## Troubleshooting

- **ModuleNotFoundError**: Install missing packages with `pip install -r requirements.txt`
- **Dataset not found**: Ensure `kagglehub` is installed or place `insurance.csv` in the project directory
- **Memory issues**: Reduce dataset size or model complexity if working with limited resources

## License

This project is for educational purposes.

