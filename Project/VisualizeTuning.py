import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Load the JSON data
results_json_path = os.path.join('logs', 'performance_all_models.json')
with open(results_json_path, 'r') as f:
    data = json.load(f)

# Define a function to plot best parameters for each model
def plot_best_params(data):
    models = list(data['best_param_counts'].keys())
    num_models = len(models)

    # Create a horizontal grid of subplots (1 row, number of models columns)
    fig, axs = plt.subplots(1, num_models, figsize=(15, 5))

    # In case there is only one model, convert axs to a list
    if num_models == 1:
        axs = [axs]

    # Define a colormap for different categories
    colormap = mcolors.TABLEAU_COLORS
    colors = list(colormap.values())

    # Loop through each model to plot its best parameters
    for idx, model in enumerate(models):
        ax = axs[idx]
        best_params = data['best_param_counts'][model]

        # Prepare the data for plotting
        labels = []
        counts = []
        color_map = {}  # To store color for each parameter category
        colors_to_use = []

        # Assign a unique color for each category (e.g., 'C', 'n_estimators')
        color_idx = 0
        for param, values in best_params.items():
            for value, count in values.items():
                labels.append(f"{param}: {value}")
                counts.append(count)

                # Use the parameter name as the category and assign a color
                if param not in color_map:
                    color_map[param] = colors[color_idx % len(colors)]
                    color_idx += 1
                colors_to_use.append(color_map[param])

        # Plotting vertical bars with the assigned colors
        ax.bar(labels, counts, color=colors_to_use)
        ax.set_title(f"Parameters for {model}")  # Removed 'Best' from the title
        ax.set_xlabel('Parameter')
        ax.set_ylabel('Count')

        # Rotate x-axis labels to make them vertical
        ax.tick_params(axis='x', labelrotation=90)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3)
    
    # Save the plot
    save_path = './figures/hyperparameter_performance.png'
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# Create the plot
plot_best_params(data)
