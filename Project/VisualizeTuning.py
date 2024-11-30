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
    models = list(data['best_param_counts'].keys())  # Extract models from the data
    num_models = len(models)  # Count the number of models

    # Create a horizontal grid of subplots (1 row, number of models columns)
    fig, axs = plt.subplots(1, num_models, figsize=(15, 5))

    # In case there is only one model, convert axs to a list to maintain consistent indexing
    if num_models == 1:
        axs = [axs]

    # Define a colormap for different categories
    colormap = mcolors.TABLEAU_COLORS
    colors = list(colormap.values())  # Get the available colors from the colormap

    # Loop through each model to plot its best parameters
    for idx, model in enumerate(models):
        ax = axs[idx]  # Get the subplot for the current model
        best_params = data['best_param_counts'][model]  # Get the best parameter counts for the model

        # Prepare the data for plotting
        labels = []  # Store the labels for the bars
        counts = []  # Store the counts for the bars
        color_map = {}  # Map for storing color assignments for each parameter category
        colors_to_use = []  # List to store colors for each parameter

        # Assign a unique color for each category (e.g., 'C', 'n_estimators')
        color_idx = 0
        for param, values in best_params.items():
            for value, count in values.items():
                labels.append(f"{param}: {value}")  # Label each parameter with its value
                counts.append(count)  # Append the count of occurrences for the parameter

                # Assign a unique color to each parameter category
                if param not in color_map:
                    color_map[param] = colors[color_idx % len(colors)]
                    color_idx += 1
                colors_to_use.append(color_map[param])  # Add the assigned color

        # Plotting vertical bars with the assigned colors
        ax.bar(labels, counts, color=colors_to_use)  # Create the bar plot
        ax.set_title(f"Parameters for {model}")  # Set the title for the subplot
        ax.set_xlabel('Parameter')  # Set the x-axis label
        ax.set_ylabel('Count')  # Set the y-axis label

        # Rotate x-axis labels to make them vertical for better readability
        ax.tick_params(axis='x', labelrotation=90)

    # Adjust spacing between subplots to avoid overlap
    plt.subplots_adjust(wspace=0.3)
    
    # Save the plot to a file
    save_path = './figures/hyperparameter_performance.png'
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.savefig(save_path)  # Save the plot as a PNG file
    plt.show()  # Display the plot

# Create the plot
plot_best_params(data)
