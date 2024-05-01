import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import logging
import os

# log set up
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sample dataset
data = pd.DataFrame({
    'Bandwidth': [3.27, 3.90, 3.62, 3.90, 3.63, 3.26, 3.60, 2.72, 3.62, 3.88, 3.43, 2.66, 3.21, 3.05, 3.25, 3.72, 2.96, 3.34, 2.86, 3.48, 3.47, 3.49, 3.16, 3.49, 3.56, 3.11, 2.89, 2.77, 3.18, 3.32, 3.05, 2.67, 3.53, 2.91, 3.23, 3.83, 2.92, 2.76, 3.48, 3.33, 2.62, 3.78, 3.44, 3.00, 3.39, 2.76, 2.80, 3.52, 3.69, 3.82, 2.91, 3.87, 2.66, 2.65, 3.17, 3.14, 2.68, 2.77, 3.81, 2.97, 3.75, 2.99, 3.86, 3.04, 2.90, 3.53, 2.84, 3.68, 2.61, 3.07, 3.57, 3.34, 2.97, 3.35, 3.36, 3.27, 2.99, 2.61, 3.44, 3.82, 3.65, 3.09, 3.79, 3.82, 3.31, 3.06, 2.76, 3.35, 3.15, 3.20, 2.74, 3.61, 2.90, 2.98, 2.62, 3.26, 2.65, 3.69, 2.97, 3.16],
    'Delay': [740, 14, 28, 10, 37, 49, 58, 34, 54, 58, 26, 43, 14, 33, 27, 59, 25, 35, 40, 41, 35, 58, 23, 11, 17, 17, 15, 58, 22, 51, 60, 18, 58, 56, 44, 49, 44, 42, 59, 25, 33, 16, 36, 18, 25, 52, 50, 10, 32, 20, 42, 47, 57, 47, 48, 15, 39, 40, 16, 26, 28, 60, 18, 26, 45, 20, 26, 50, 39, 22, 29, 32, 11, 45, 38, 18, 41, 19, 27, 36, 42, 33, 24, 36, 22, 42, 39, 21, 10, 60, 23, 18, 26, 26, 48, 56, 59, 59, 57, 50], # delay difference
    'QoE': [7.50, 7.15, 7.52, 7.03, 7.21, 7.62, 7.77, 7.45, 7.14, 7.58, 7.21, 7.79, 7.09, 7.32, 7.67, 7.85, 7.10, 7.48, 7.50, 7.31, 7.50, 7.45, 7.47, 7.40, 7.14, 7.34, 7.17, 7.07, 7.71, 7.01, 7.33, 7.76, 7.56, 7.27, 7.27, 7.72, 7.65, 7.03, 7.44, 7.64, 7.66, 7.06, 7.44, 7.50, 7.05, 7.81, 7.59, 7.62, 7.76, 7.87, 7.80, 7.61, 7.60, 7.60, 7.62, 7.22, 7.31, 7.87, 7.71, 7.52, 7.18, 7.21, 7.65, 7.13, 7.35, 7.12, 7.05, 7.62, 7.05, 7.58, 7.00, 7.51, 7.44, 7.46, 7.28, 7.13, 7.82, 7.44, 7.10, 7.60, 7.68, 7.47, 7.80, 7.07, 7.67, 7.78, 7.28, 7.79, 7.52, 7.19, 7.41, 7.18, 7.09, 7.81, 7.72, 7.81, 7.63, 7.06, 7.15, 7.45]
})


# linear regression model
X = data[['Bandwidth', 'Delay']]
y = data['QoE']
model = LinearRegression()
model.fit(X, y)

logging.info("Model training is complete. Let's generate the synthetic data.")

# synthetic data generation
def generate_synthetic_data(num_samples=100000, bandwidth_range=(2.33, 4)):
    logging.info(f"Generating {num_samples} synthetic data points...")
    bandwidths = np.random.uniform(bandwidth_range[0], bandwidth_range[1], num_samples)
    delays = np.random.choice(data['Delay'], size=num_samples, replace=True)
    predictions = model.predict(pd.DataFrame({
        'Bandwidth': bandwidths,
        'Delay': delays
    }))
    synthetic_data = pd.DataFrame({
        'Bandwidth': bandwidths,
        'Delay': delays,
        'QoE': predictions
    })
    logging.info("Synthetic data generation complete.")
    return synthetic_data

'''
def generate_synthetic_data(num_samples=10):
    logging.info(f"Generating {num_samples} synthetic data...")
    bandwidths = np.random.randint(10, 100, size=num_samples)
    delays = np.random.randint(10, 100, size=num_samples)
    predictions = model.predict(np.column_stack([bandwidths, delays]))
    synthetic_data = pd.DataFrame({
        'Bandwidth': bandwidths,
        'Delay': delays,
        'QoE': predictions
    })
    logging.info("Synthetic data generation is complete.")
    return synthetic_data

'''



# call to generate data
synthetic_data = generate_synthetic_data()

# Output
output_directory = './'
bandwidth_delay_file_path = os.path.join(output_directory, 'bandwidth_delay.txt')
qoe_file_path = os.path.join(output_directory, 'qoe.txt')

# I need two-decimal
synthetic_data[['Bandwidth', 'Delay']].to_csv(bandwidth_delay_file_path, index=False, header=False, sep=',', float_format='%.2f')
synthetic_data['QoE'].to_csv(qoe_file_path, index=False, header=False, float_format='%.2f')

logging.info(f"Data has been saved to files at {output_directory} with two decimal formatting.")

# Just in case!!
plt.scatter(data['Bandwidth'], data['QoE'], color='blue', label='Original Data')
plt.scatter(synthetic_data['Bandwidth'], synthetic_data['QoE'], color='red', label='Synthetic Data', alpha=0.5)
plt.xlabel('Bandwidth')
plt.ylabel('QoE')
plt.legend()
plt.show()