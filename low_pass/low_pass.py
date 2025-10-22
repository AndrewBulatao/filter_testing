import matplotlib.pyplot as plt
import csv

# Initialize empty lists to store data
x_grav_data, y_grav_data, z_grav_data = [], [], []
x_acc_data, y_acc_data, z_acc_data = [], [], []

# Read data from the CSV file
with open('../data/grav_acc.csv', 'r') as file:
    reader = csv.reader(file)
    # next(reader)  # Uncomment if CSV has header
    # gravity in G so conver to m/s²
    for row in reader:
        x_grav_data.append(float(row[0]) * 9.81)  
        y_grav_data.append(float(row[1]) * 9.81)
        z_grav_data.append(float(row[2]) * -9.81)
        x_acc_data.append(float(row[3]) * 9.81)   
        y_acc_data.append(float(row[4]) * 9.81)
        z_acc_data.append(float(row[5]) * -9.81)

# ---- LOW-PASS FILTER IMPLEMENTATION ----
# Initialize filter outputs
x_grav_out, y_grav_out, z_grav_out = [x_grav_data[0]], [y_grav_data[0]], [z_grav_data[0]]
x_acc_out, y_acc_out, z_acc_out = [x_acc_data[0]], [y_acc_data[0]], [z_acc_data[0]]

# Filter parameters
# Cutoff frequency and sampling period
w_c = 3.7   
T_s = 0.018 

alpha = (T_s*w_c)/(2 + T_s*w_c)
gamma = (2 - T_s*w_c)/(2 + T_s*w_c)

# Apply first-order low-pass filter
for i in range(1, len(x_grav_data)):
    # gravity    
    x_grav_out.append(alpha * (x_grav_data[i] + x_grav_data[i-1]) + gamma * x_grav_out[i-1])
    y_grav_out.append(alpha * (y_grav_data[i] + y_grav_data[i-1]) + gamma * y_grav_out[i-1])
    z_grav_out.append(alpha * (z_grav_data[i] + z_grav_data[i-1]) + gamma * z_grav_out[i-1])
    
    # acceleration
    x_acc_out.append(alpha * (x_acc_data[i] + x_acc_data[i-1]) + gamma * x_acc_out[i-1])
    y_acc_out.append(alpha * (y_acc_data[i] + y_acc_data[i-1]) + gamma * y_acc_out[i-1])
    z_acc_out.append(alpha * (z_acc_data[i] + z_acc_data[i-1]) + gamma * z_acc_out[i-1])


# ---- SUBTRACTING GRAVITY FROM ACCELERATION DATA ----
x_lin_acc = [a - g for a, g in zip(x_acc_out, x_grav_out)]
y_lin_acc = [a - g for a, g in zip(y_acc_out, y_grav_out)]
z_lin_acc = [a - g for a, g in zip(z_acc_out, z_grav_out)]

# Number of samples to plot
N = 100
print("Acceleration before", y_acc_out[:N])
print("Gravity", y_grav_out[:N])
print("Acceleration after", y_lin_acc[:N])
print("---")

# ---- PLOTTING ----
plt.figure(figsize=(12, 10))

# X-axis
plt.subplot(3, 1, 1)
plt.plot(x_acc_out[:N], label='Raw Acc X', alpha=0.3)
plt.plot(x_lin_acc[:N], label='Linear Acc X', alpha=0.9)
plt.legend()
plt.ylabel('Acceleration (m/s²)')
plt.title('Linear Acceleration after Subtracting Gravity (First {} samples)'.format(N))

# Y-axis
plt.subplot(3, 1, 2)
plt.plot(y_acc_out[:N], label='Raw Acc Y', alpha=0.3)
plt.plot(y_lin_acc[:N], label='Linear Acc Y', alpha=0.9)
plt.legend()
plt.ylabel('Acceleration (m/s²)')

# Z-axis
plt.subplot(3, 1, 3)
plt.plot(z_acc_out[:N], label='Raw Acc Z', alpha=0.3)
plt.plot(z_lin_acc[:N], label='Linear Acc Z', alpha=0.9)
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Acceleration (m/s²)')

plt.tight_layout()
plt.show()
