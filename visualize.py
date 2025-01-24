import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import sys
import os
import shutil


def load_test_data(test_data_path):
    """Load and prepare test data"""
    test_data_dir = os.path.dirname(test_data_path) + '/'
    shutil.unpack_archive(test_data_path, extract_dir=test_data_dir)

    x_price_test = np.load(test_data_dir + 'x_price_test.npy')
    x_news_test = list(np.load(test_data_dir + 'x_news_test.npy'))
    y_test = np.load(test_data_dir + 'y_test.npy')

    return x_price_test, x_news_test, y_test


def get_predictions(model, x_price_test, x_news_test):
    """Get model predictions"""
    return model.predict([x_price_test] + x_news_test)


def calculate_metrics(y_test, y_pred, label_scl):
    """Calculate various error metrics"""
    # Inverse transform the scaled values
    y_test_orig = label_scl.inverse_transform(y_test)
    y_pred_orig = label_scl.inverse_transform(y_pred)

    # Calculate relative errors
    rel_errors = np.abs(y_test_orig - y_pred_orig) / y_test_orig * 100

    # Calculate percentage of predictions within different error margins
    within_1_percent = np.mean(rel_errors <= 1, axis=0) * 100
    within_5_percent = np.mean(rel_errors <= 5, axis=0) * 100
    within_10_percent = np.mean(rel_errors <= 10, axis=0) * 100

    # Mean absolute percentage error
    mape = np.mean(rel_errors, axis=0)

    return {
        'y_test_orig': y_test_orig,
        'y_pred_orig': y_pred_orig,
        'rel_errors': rel_errors,
        'within_1_percent': within_1_percent,
        'within_5_percent': within_5_percent,
        'within_10_percent': within_10_percent,
        'mape': mape
    }


def plot_prediction_comparison(metrics, company_names):
    """Plot actual vs predicted values for each company"""
    n_companies = len(company_names)
    fig, axes = plt.subplots(n_companies, 1, figsize=(15, 5 * n_companies))
    if n_companies == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(metrics['y_test_orig'][:, i], label='Actual', color='blue', alpha=0.7)
        ax.plot(metrics['y_pred_orig'][:, i], label='Predicted', color='red', alpha=0.7)
        ax.set_title(f'Stock Price Predictions for {company_names[i]}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Stock Price')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
    plt.close()


def plot_error_distribution(metrics, company_names):
    """Plot error distribution for each company"""
    plt.figure(figsize=(15, 8))

    for i, company in enumerate(company_names):
        sns.kdeplot(metrics['rel_errors'][:, i], label=company)

    plt.title('Distribution of Relative Errors by Company')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.close()


def plot_accuracy_bars(metrics, company_names):
    """Plot accuracy metrics for each company"""
    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(company_names))
    width = 0.25

    ax.bar(x - width, metrics['within_1_percent'], width, label='Within 1%', color='green')
    ax.bar(x, metrics['within_5_percent'], width, label='Within 5%', color='blue')
    ax.bar(x + width, metrics['within_10_percent'], width, label='Within 10%', color='orange')

    ax.set_ylabel('Percentage of Predictions')
    ax.set_title('Prediction Accuracy by Error Margin')
    ax.set_xticks(x)
    ax.set_xticklabels(company_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig('accuracy_bars.png')
    plt.close()


def create_summary_report(metrics, company_names):
    """Create a summary report of all metrics"""
    report = pd.DataFrame({
        'Company': company_names,
        'MAPE (%)': metrics['mape'].round(2),
        'Within 1%': metrics['within_1_percent'].round(2),
        'Within 5%': metrics['within_5_percent'].round(2),
        'Within 10%': metrics['within_10_percent'].round(2)
    })

    report.to_csv('performance_summary.csv', index=False)
    return report


def main():
    if len(sys.argv) != 4:
        print("Usage: python visualize.py test_data.zip model.h5 company_names.txt")
        sys.exit(1)

    # Load data and model
    test_data_path = sys.argv[1]
    model_path = sys.argv[2]
    company_names_file = sys.argv[3]

    # Load company names
    with open(company_names_file, 'r') as f:
        company_names = [line.strip() for line in f.readlines()]

    # Load scalers
    price_scl = joblib.load('Tmp/price_scl.scl')
    label_scl = joblib.load('Tmp/label_scl.scl')

    # Load model with custom objects
    custom_objects = {
        'mse': MeanSquaredError(),
        'mae': 'mean_absolute_error'
    }
    model = load_model(model_path, custom_objects=custom_objects)

    # Load data and get predictions
    x_price_test, x_news_test, y_test = load_test_data(test_data_path)
    y_pred = get_predictions(model, x_price_test, x_news_test)

    # Calculate metrics and create visualizations
    metrics = calculate_metrics(y_test, y_pred, label_scl)

    plot_prediction_comparison(metrics, company_names)
    plot_error_distribution(metrics, company_names)
    plot_accuracy_bars(metrics, company_names)

    # Create and display summary report
    report = create_summary_report(metrics, company_names)
    print("\nPerformance Summary:")
    print(report.to_string(index=False))

    print("\nVisualization files created:")
    print("- prediction_comparison.png")
    print("- error_distribution.png")
    print("- accuracy_bars.png")
    print("- performance_summary.csv")


if __name__ == "__main__":
    main()