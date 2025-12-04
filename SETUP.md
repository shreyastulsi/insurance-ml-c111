# Setup Instructions for New Environment

Follow these steps to set up a fresh environment and run the ML analysis project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Step-by-Step Setup

### 1. Navigate to Project Directory

```bash
cd /path/to/c111-final-project
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
```

**On Windows:**
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your command prompt.

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy
- kagglehub

### 5. (Optional) Set Up Kaggle Credentials

If you want the script to automatically download the dataset from Kaggle:

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New Token" to download `kaggle.json`
4. Place it in `~/.kaggle/kaggle.json` (or `C:\Users\<username>\.kaggle\kaggle.json` on Windows)

**Note:** If you don't set up Kaggle credentials, the script will look for a local `insurance.csv` file in the project directory.

### 6. Run the Analysis

```bash
python ml_analysis.py
```

The script will:
- Download the dataset (if kagglehub is configured) or use local file
- Perform complete ML analysis
- Generate all visualizations and reports
- Save results to `ml_results/` directory

## Alternative: Using Local Dataset

If you already have `insurance.csv`, simply place it in the project root directory:

```
c111-final-project/
├── insurance.csv
├── ml_analysis.py
├── requirements.txt
└── ...
```

The script will automatically detect and use it.

## Deactivate Virtual Environment

When you're done working:

```bash
deactivate
```

## Troubleshooting

### Issue: `ModuleNotFoundError`
**Solution:** Make sure you activated the virtual environment and installed requirements:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: Kaggle dataset download fails
**Solution:** Either:
1. Set up Kaggle API credentials (see step 5 above), OR
2. Download `insurance.csv` manually from Kaggle and place it in the project directory

### Issue: XGBoost errors (if you add it later)
**Solution:** On macOS, install OpenMP:
```bash
brew install libomp
```

## Project Structure

After running, you'll have:

```
c111-final-project/
├── ml_analysis.py          # Main analysis script
├── requirements.txt        # Python dependencies
├── insurance.csv           # Dataset (if using local file)
├── ml_results/             # Generated results
│   ├── analysis_report.txt
│   ├── model_comparison.csv
│   ├── correlation_heatmap.png
│   ├── eda_visualizations.png
│   ├── model_comparison_metrics.png
│   ├── feature_importance.png
│   ├── best_model_analysis_*.png
│   └── results.json
└── venv/                   # Virtual environment (don't commit this)
```

## Quick Start Summary

```bash
# 1. Create and activate venv
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run analysis
python ml_analysis.py
```

That's it! 🚀

