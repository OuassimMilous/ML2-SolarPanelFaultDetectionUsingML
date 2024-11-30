import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import ModelsOptimized

# Define file paths
data_files = ['data.csv']  # List of datasets to combine
results_json_path = os.path.join('logs', 'performance_final.json')  # Path for saving performance results
mean_plot_path = os.path.join('figures', 'mean_performance_plot_final.png')  # Path for saving performance plot

# Ensure necessary directories exist
os.makedirs('logs', exist_ok=True)  # Create 'logs' directory if not exists
os.makedirs('tests', exist_ok=True)  # Create 'tests' directory if not exists

# Clear or create the results.json file
with open(results_json_path, 'w') as f:
    json.dump({}, f)  # Empty the file or create it if not present

# Function to load and preprocess the dataset
def load_and_preprocess_data(files):
    # Load and concatenate datasets into a single DataFrame
    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs, ignore_index=True)
    
    # Separate numeric columns for imputation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Impute missing numeric values with the mean of each column
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Convert the 'faulty' column to numerical values (0 for 'No', 1 for 'Yes')
    df['faulty'] = df['faulty'].map({'No': 0, 'Yes': 1})
    
    # Define features (X) and target (y)
    X = df.drop(columns=['annotation_file', 'panel_index', 'faulty'])  # Features excluding irrelevant columns
    y = df['faulty']  # Target column
    
    # Normalize the features using standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Initialize dictionaries to track parameter values and results
best_param_counts = {
    'SVM': {'C': {}, 'class_weight': {}, 'degree': {}, 'gamma': {}, 'kernel': {}},  # Track hyperparameter counts for SVM
}

# Initialize a dictionary to store model evaluation metrics
all_results = {'SVM': {'F1-Score': [], 'Recall': [], 'Precision': [], 'Accuracy': []}}

# Load and preprocess data
X_scaled, y = load_and_preprocess_data(data_files)

# Set up K-fold cross-validation (10 folds)
k = 10
kf = KFold(n_splits=k, shuffle=True)

# Run SVM model with K-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X_scaled), 1):
    print(f"Starting fold {fold} of {k}...")

    # Split data into training and test sets for the current fold
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Use the SVM model from ModelsOptimized for evaluation
    model_name = 'SVM'
    model_script = ModelsOptimized.svm  # Reference to the SVM model function
    accuracy, precision, recall, f1, best_params = model_script(X_train, X_test, y_train, y_test)
    
    # Append evaluation metrics to the results dictionary
    all_results[model_name]['F1-Score'].append(f1)
    all_results[model_name]['Recall'].append(recall)
    all_results[model_name]['Precision'].append(precision)
    all_results[model_name]['Accuracy'].append(accuracy)
    
    # Track hyperparameter counts for the current fold
    for param, value in best_params.items():
        if param in best_param_counts[model_name]:
            if value not in best_param_counts[model_name][param]:
                best_param_counts[model_name][param][value] = 1
            else:
                best_param_counts[model_name][param][value] += 1
    
    # Save the updated results to the JSON file
    with open(results_json_path, 'w') as f:
        json.dump({'results': all_results, 'best_param_counts': best_param_counts}, f, indent=4)
    
    print(f"{model_name} - Fold {fold}: F1={f1:.2f}, Recall={recall:.2f}, Precision={precision:.2f}, Accuracy={accuracy:.2f}")

print(f"Results saved to {results_json_path}")

# Calculate the mean results for the SVM model
mean_results = {model: {metric: np.mean(values) for metric, values in metrics.items()} for model, metrics in all_results.items()}

# Prepare data for plotting
fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure for the plot

# Bar positions and width for the metrics
ind = np.arange(len(mean_results))  # Indices for bars
width = 0.2  # Width of each bar

# Extract means for each evaluation metric
f1_means = [mean_results[model]['F1-Score'] for model in mean_results]
recall_means = [mean_results[model]['Recall'] for model in mean_results]
precision_means = [mean_results[model]['Precision'] for model in mean_results]
accuracy_means = [mean_results[model]['Accuracy'] for model in mean_results]

# Plot bars for each metric
bars1 = ax.bar(ind - 1.5 * width, accuracy_means, width, label='Accuracy', color='skyblue')
bars2 = ax.bar(ind - 0.5 * width, precision_means, width, label='Precision', color='orange')
bars3 = ax.bar(ind + 0.5 * width, recall_means, width, label='Recall', color='green')
bars4 = ax.bar(ind + 1.5 * width, f1_means, width, label='F1-Score', color='red')

# Plot best parameter counts using vertical bars
best_param_width = 0.1  # Width for the best parameter bars
param_counts = [sum(len(param_dict) for param_dict in best_param_counts[model].values()) for model in best_param_counts]

# Set labels, title, and ticks for the plot
ax.set_ylabel('Scores / Counts')
ax.set_xlabel('Model')
ax.set_title('SVM Model Comparison: Accuracy, Precision, Recall, F1-Score & Best Param Count')
ax.set_xticks(ind)
ax.set_xticklabels(mean_results.keys())

# Add legends for the plot
ax.legend(title="Metrics", loc='upper left', bbox_to_anchor=(1, 1))

# Annotate bars with the actual values
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{bar.get_height():.2f}', 
                va='bottom', ha='center', color='black', fontsize=10)

# Adjust layout to ensure the plot fits well
plt.tight_layout()

# Save the plot to the specified file
fig.savefig(mean_plot_path, dpi=300)

# Display the plot
plt.show()

print(f"Mean performance plot saved to {mean_plot_path}")
