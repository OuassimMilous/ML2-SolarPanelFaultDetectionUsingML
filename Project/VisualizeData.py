import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the panel data from the CSV file
df = pd.read_csv("data.csv")

# Custom color palette for distinguishing faulty vs. non-faulty panels
custom_palette = {'Yes': 'red', 'No': 'blue'}

# Select relevant variables (columns) to visualize for pairwise relationships
selected_columns = [
    'mean_temperature',  # Mean temperature of the panel
    'temperature_standard_deviation',  # Standard deviation of the temperature
    'max_temperature',  # Maximum temperature recorded in the panel
    'temperature_skewness',  # Skewness of the temperature distribution
    'temperature_range',  # Temperature range (max - min)
    'temperature_kurtosis'  # Kurtosis of the temperature distribution
]

# Create pairplot to visualize correlations between the selected variables
sns.pairplot(
    df,
    hue='faulty',  # Color points based on the 'faulty' column (red for 'Yes', blue for 'No')
    palette=custom_palette,  # Apply custom color palette for faulty/non-faulty panels
    diag_kind='kde',  # Use Kernel Density Estimate (KDE) plot for the diagonal to show distribution
    vars=selected_columns,  # Only plot the selected columns
    plot_kws={'alpha': 0.5, 's': 50}  # Set transparency and point size for better visualization
)

# Adjust the title and axis labels for better readability
plt.suptitle("Pairwise Scatter Plots for Panel Data", fontsize=16, y=1.02)  # Set the title above the plot
plt.subplots_adjust(top=0.92)  # Adjust the title position to prevent overlap with the plot

# Adjust axis labels and ticks for better readability
for label in plt.gca().get_xticklabels():
    label.set_fontsize(10)  # Set a smaller font size for x-axis labels
for label in plt.gca().get_yticklabels():
    label.set_fontsize(10)  # Set a smaller font size for y-axis labels

# Ensure the 'figures' directory exists, and create it if necessary
os.makedirs('figures', exist_ok=True)

# Define the file path to save the plot
plot_filename = 'figures/Features Plot.png'

# Save the plot as a PNG image in the 'figures' directory with high resolution
plt.savefig(plot_filename, dpi=300)  # Save the plot at 300 DPI for better quality

# Optionally, display the plot
plt.show()

# Print the file path where the plot has been saved
print(f"Plot saved to {plot_filename}")
