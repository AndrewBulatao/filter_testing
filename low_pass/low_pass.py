import matplotlib.pyplot as plt
import csv

# Initialize empty lists to store data
x_grav_data, y_grav_data, z_grav_out = [], [], []

# Read data from the CSV file
with open('../data/gravity.csv', 'r') as file:
    reader = csv.reader(file)
    # next(reader)  # Uncomment if CSV has header
    for row in reader:
        x_grav_data.append(float(row[0]) * 9.81)  # Convert to m/s²
        y_grav_data.append(float(row[1]) * 9.81)
        z_grav_out.append(float(row[2]) * 9.81)

# Initialize filter outputs
x_grav_out, y_grav_out, z_grav_out = [x_grav_data[0]], [y_grav_data[0]], [z_grav_out[0]]

# Cutoff frequency (Hz)
w_c = 3.7
# Sampling period (s)
T_s = 0.018

alpha = (T_s*w_c)/(2 + T_s*w_c)
gamma = (2 - T_s*w_c)/(2 + T_s*w_c)

# Apply first-order low-pass filter
for i in range(1, len(x_grav_data)):
    x_grav_out.append(alpha * (x_grav_data[i] + x_grav_data[i-1]) + gamma * x_grav_out[i-1])
    y_grav_out.append(alpha * (y_grav_data[i] + y_grav_data[i-1]) + gamma * y_grav_out[i-1])
    z_grav_out.append(alpha * (z_grav_out[i] + z_grav_out[i-1]) + gamma * z_grav_out[i-1])

# Number of samples to plot (first few seconds)
N = 6000  

# Create subplots
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(x_grav_data[:N], label='Original X', alpha=0.3)
plt.plot(x_grav_out[:N], label='Filtered X', alpha=0.9)
plt.legend()
plt.ylabel('Acceleration (m/s²)')
plt.title('Low-Pass Filter Effect (First {} samples)'.format(N))

plt.subplot(3, 1, 2)
plt.plot(y_grav_data[:N], label='Original Y', alpha=0.3)
plt.plot(y_grav_out[:N], label='Filtered Y', alpha=0.9)
plt.legend()
plt.ylabel('Acceleration (m/s²)')

plt.subplot(3, 1, 3)
plt.plot(z_grav_out[:N], label='Original Z', alpha=0.3)
plt.plot(z_grav_out[:N], label='Filtered Z', alpha=0.9)
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Acceleration (m/s²)')

plt.tight_layout()
plt.show()
