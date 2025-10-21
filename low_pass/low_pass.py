import matplotlib.pyplot as plt
import csv

# Initialize empty lists to store data
x_data, y_data, z_data = [], [], []

# Read data from the CSV file
with open('../data/linear_acc.csv', 'r') as file:
    reader = csv.reader(file)
    # next(reader)  # Uncomment if CSV has header
    for row in reader:
        x_data.append(float(row[0]) * 9.81)  # Convert to m/s²
        y_data.append(float(row[1]) * 9.81)
        z_data.append(float(row[2]) * 9.81)

# Initialize filter outputs
x_out, y_out, z_out = [x_data[0]], [y_data[0]], [z_data[0]]

# Cutoff frequency (Hz)
w_c = 3.7
# Sampling period (s)
T_s = 0.018

alpha = (T_s*w_c)/(2 + T_s*w_c)
gamma = (2 - T_s*w_c)/(2 + T_s*w_c)

# Apply first-order low-pass filter
for i in range(1, len(x_data)):
    x_out.append(alpha * (x_data[i] + x_data[i-1]) + gamma * x_out[i-1])
    y_out.append(alpha * (y_data[i] + y_data[i-1]) + gamma * y_out[i-1])
    z_out.append(alpha * (z_data[i] + z_data[i-1]) + gamma * z_out[i-1])

# Number of samples to plot (first few seconds)
N = 500  

# Create subplots
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(x_data[:N], label='Original X', alpha=0.3)
plt.plot(x_out[:N], label='Filtered X', alpha=0.9)
plt.legend()
plt.ylabel('Acceleration (m/s²)')
plt.title('Low-Pass Filter Effect (First {} samples)'.format(N))

plt.subplot(3, 1, 2)
plt.plot(y_data[:N], label='Original Y', alpha=0.3)
plt.plot(y_out[:N], label='Filtered Y', alpha=0.9)
plt.legend()
plt.ylabel('Acceleration (m/s²)')

plt.subplot(3, 1, 3)
plt.plot(z_data[:N], label='Original Z', alpha=0.3)
plt.plot(z_out[:N], label='Filtered Z', alpha=0.9)
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Acceleration (m/s²)')

plt.tight_layout()
plt.show()
