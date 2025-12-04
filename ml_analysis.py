from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.neural_network import MLPRegressor
    NEURAL_NET_AVAILABLE = True
except ImportError:
    NEURAL_NET_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

def load_data(csv_path):
    return pd.read_csv(csv_path)

def explore_data(df, output_dir):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].hist(df["charges"], bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Distribution of Insurance Charges")
    axes[0, 0].set_xlabel("Charges")
    axes[0, 0].set_ylabel("Frequency")
    
    df.boxplot(column="charges", by="smoker", ax=axes[0, 1])
    axes[0, 1].set_title("Charges Distribution by Smoker Status")
    axes[0, 1].set_xlabel("Smoker")
    axes[0, 1].set_ylabel("Charges")
    
    df.boxplot(column="charges", by="region", ax=axes[1, 0])
    axes[1, 0].set_title("Charges Distribution by Region")
    axes[1, 0].set_xlabel("Region")
    axes[1, 0].set_ylabel("Charges")
    
    for smoker_status in ["yes", "no"]:
        subset = df[df["smoker"] == smoker_status]
        axes[1, 1].scatter(subset["age"], subset["charges"], alpha=0.6,
                          label="Smoker" if smoker_status == "yes" else "Non-Smoker")
    axes[1, 1].set_title("Age vs Charges (by Smoker Status)")
    axes[1, 1].set_xlabel("Age")
    axes[1, 1].set_ylabel("Charges")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "eda_visualizations.png", dpi=150)
    plt.close()

def preprocess_data(df):
    df_processed = df.copy()
    categorical_cols = ["sex", "smoker", "region"]
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    X = df_processed.drop("charges", axis=1).values
    y = df_processed["charges"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
    }
    
    if NEURAL_NET_AVAILABLE:
        models["Neural Network (MLP)"] = MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1
        )
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def evaluate_models(models, X_test, y_test, output_dir):
    results = {}
    predictions = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        results[name] = {"R² Score": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}
    
    results_df = pd.DataFrame(results).T.sort_values("R² Score", ascending=False)
    results_df.to_csv(output_dir / "model_comparison.csv")
    
    return results, predictions

def visualize_results(results, predictions, models, y_test, output_dir):
    results_df = pd.DataFrame(results).T
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    results_df["R² Score"].sort_values(ascending=True).plot(kind="barh", ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("R² Score Comparison")
    axes[0, 0].set_xlabel("R² Score")
    axes[0, 0].grid(axis="x", alpha=0.3)
    
    results_df["RMSE"].sort_values(ascending=False).plot(kind="barh", ax=axes[0, 1], color="coral")
    axes[0, 1].set_title("RMSE Comparison (Lower is Better)")
    axes[0, 1].set_xlabel("RMSE ($)")
    axes[0, 1].grid(axis="x", alpha=0.3)
    
    results_df["MAE"].sort_values(ascending=False).plot(kind="barh", ax=axes[1, 0], color="mediumseagreen")
    axes[1, 0].set_title("MAE Comparison (Lower is Better)")
    axes[1, 0].set_xlabel("MAE ($)")
    axes[1, 0].grid(axis="x", alpha=0.3)
    
    results_df["MAPE"].sort_values(ascending=False).plot(kind="barh", ax=axes[1, 1], color="gold")
    axes[1, 1].set_title("MAPE Comparison (Lower is Better)")
    axes[1, 1].set_xlabel("MAPE (%)")
    axes[1, 1].grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_metrics.png", dpi=150)
    plt.close()
    
    best_model = max(results.items(), key=lambda x: x[1]["R² Score"])[0]
    y_pred_best = predictions[best_model]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].scatter(y_test, y_pred_best, alpha=0.6, edgecolors="black", linewidths=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect Prediction")
    axes[0, 0].set_xlabel("Actual Charges ($)")
    axes[0, 0].set_ylabel("Predicted Charges ($)")
    axes[0, 0].set_title(f"Predictions vs Actual - {best_model}")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    residuals = y_test - y_pred_best
    axes[0, 1].scatter(y_pred_best, residuals, alpha=0.6, edgecolors="black", linewidths=0.5)
    axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[0, 1].set_xlabel("Predicted Charges ($)")
    axes[0, 1].set_ylabel("Residuals ($)")
    axes[0, 1].set_title(f"Residuals Plot - {best_model}")
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[1, 0].set_xlabel("Residuals ($)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title(f"Distribution of Residuals - {best_model}")
    axes[1, 0].axvline(x=0, color="r", linestyle="--", lw=2)
    axes[1, 0].grid(alpha=0.3)
    
    if SCIPY_AVAILABLE:
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, "Q-Q plot requires scipy", ha="center", va="center", transform=axes[1, 1].transAxes)
    axes[1, 1].set_title(f"Q-Q Plot of Residuals - {best_model}")
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"best_model_analysis_{best_model.replace(' ', '_')}.png", dpi=150)
    plt.close()
    
    tree_models = {name: model for name, model in models.items() if hasattr(model, "feature_importances_")}
    
    if tree_models:
        n_models = len(tree_models)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        feature_names = ["age", "sex", "bmi", "children", "smoker", "region"]
        
        for idx, (name, model) in enumerate(tree_models.items()):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            axes[idx].barh(range(len(feature_names)), importances[indices], color="steelblue")
            axes[idx].set_yticks(range(len(feature_names)))
            axes[idx].set_yticklabels([feature_names[i] for i in indices])
            axes[idx].set_xlabel("Feature Importance")
            axes[idx].set_title(f"Feature Importance - {name}")
            axes[idx].grid(axis="x", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=150)
        plt.close()

def generate_report(df, results, models, output_dir):
    best_name, best_metrics = max(results.items(), key=lambda x: x[1]["R² Score"])
    
    report = []
    report.append("=" * 80)
    report.append("INSURANCE CHARGES PREDICTION - COMPREHENSIVE ML ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    report.append("DATASET OVERVIEW")
    report.append("-" * 80)
    report.append(f"Total samples: {len(df)}")
    report.append(f"Features: {', '.join(df.columns[:-1])}")
    report.append(f"Target variable: charges")
    report.append(f"Average charges: ${df['charges'].mean():,.2f}")
    report.append(f"Standard deviation: ${df['charges'].std():,.2f}")
    report.append(f"Min charges: ${df['charges'].min():,.2f}")
    report.append(f"Max charges: ${df['charges'].max():,.2f}")
    report.append("")
    
    report.append("KEY INSIGHTS FROM EXPLORATORY DATA ANALYSIS")
    report.append("-" * 80)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if "charges" in numeric_cols:
        correlations = df[numeric_cols].corr()["charges"].abs().sort_values(ascending=False)
        report.append("Feature correlations with charges:")
        for feature, corr in correlations.items():
            if feature != "charges":
                report.append(f"  {feature}: {corr:.4f}")
    report.append("")
    report.append(f"Smoker impact:")
    smoker_avg = df[df["smoker"] == "yes"]["charges"].mean()
    non_smoker_avg = df[df["smoker"] == "no"]["charges"].mean()
    report.append(f"  Average charges for smokers: ${smoker_avg:,.2f}")
    report.append(f"  Average charges for non-smokers: ${non_smoker_avg:,.2f}")
    report.append(f"  Difference: ${smoker_avg - non_smoker_avg:,.2f} ({((smoker_avg/non_smoker_avg - 1) * 100):.1f}% higher)")
    report.append("")
    
    report.append("MODELS EVALUATED")
    report.append("-" * 80)
    for name in models.keys():
        report.append(f"  • {name}")
    report.append("")
    
    report.append("MODEL PERFORMANCE COMPARISON")
    report.append("-" * 80)
    results_df = pd.DataFrame(results).T.sort_values("R² Score", ascending=False)
    for name, metrics in results_df.iterrows():
        report.append(f"\n{name}:")
        report.append(f"  R² Score:  {metrics['R² Score']:.4f}")
        report.append(f"  MAE:       ${metrics['MAE']:,.2f}")
        report.append(f"  RMSE:      ${metrics['RMSE']:,.2f}")
        report.append(f"  MAPE:      {metrics['MAPE']:.2%}")
    report.append("")
    
    report.append("BEST MODEL")
    report.append("-" * 80)
    report.append(f"Model: {best_name}")
    report.append(f"R² Score: {best_metrics['R² Score']:.4f}")
    report.append(f"Mean Absolute Error: ${best_metrics['MAE']:,.2f}")
    report.append(f"Root Mean Squared Error: ${best_metrics['RMSE']:,.2f}")
    report.append(f"Mean Absolute Percentage Error: {best_metrics['MAPE']:.2%}")
    report.append("")
    
    if hasattr(models[best_name], "feature_importances_"):
        importances = models[best_name].feature_importances_
        feature_names = ["age", "sex", "bmi", "children", "smoker", "region"]
        importance_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        report.append("Feature Importance (Best Model):")
        for feature, importance in importance_pairs:
            report.append(f"  {feature}: {importance:.4f}")
        report.append("")
    
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    report_path = output_dir / "analysis_report.txt"
    report_path.write_text(report_text)
    
    json_results = {
        "best_model": best_name,
        "best_model_metrics": best_metrics,
        "all_model_metrics": results,
        "dataset_stats": {
            "n_samples": len(df),
            "mean_charges": float(df["charges"].mean()),
            "std_charges": float(df["charges"].std()),
        },
    }
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(json_results, indent=2))

def main():
    script_dir = Path(__file__).parent
    output_dir = Path("ml_results")
    output_dir.mkdir(exist_ok=True)

    # Expect a local insurance.csv next to this script.
    csv_path = script_dir / "insurance.csv"
    if not csv_path.exists():
        print("ERROR: insurance.csv not found next to ml_analysis.py")
        sys.exit(1)

    df = load_data(csv_path)
    explore_data(df, output_dir)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    models = train_models(X_train, y_train)
    results, predictions = evaluate_models(models, X_test, y_test, output_dir)
    visualize_results(results, predictions, models, y_test, output_dir)
    generate_report(df, results, models, output_dir)

if __name__ == "__main__":
    main()
