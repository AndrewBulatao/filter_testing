import matplotlib.pyplot as plt
import csv

# ---- READ ACCELERATION DATA ----
x_acc_data, y_acc_data, z_acc_data = [], [], []

with open('../data/individual/acceleration.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x_acc_data.append(float(row[0]))   
        y_acc_data.append(float(row[1]))
        z_acc_data.append(float(row[2]))

# ---- LOW-PASS FILTER PARAMETERS ----
w_c = 3.7     
T_s = 0.018   

alpha = (T_s*w_c)/(2 + T_s*w_c)
gamma = (2 - T_s*w_c)/(2 + T_s*w_c)

# ---- INITIALIZE FILTER OUTPUTS ----
x_acc_out = [x_acc_data[0]]
y_acc_out = [y_acc_data[0]]
z_acc_out = [z_acc_data[0]]

# ---- APPLY LOW-PASS FILTER ----
for i in range(1, len(x_acc_data)):
    x_acc_out.append(alpha * (x_acc_data[i] + x_acc_data[i-1]) + gamma * x_acc_out[i-1])
    y_acc_out.append(alpha * (y_acc_data[i] + y_acc_data[i-1]) + gamma * y_acc_out[i-1])
    z_acc_out.append(alpha * (z_acc_data[i] + z_acc_data[i-1]) + gamma * z_acc_out[i-1])

# ---- RAW vs FILTERED ACCELERATION ----
x_lin_acc = x_acc_out  
y_lin_acc = y_acc_out
z_lin_acc = z_acc_out

# ---- PLOTTING ----
N = 500 

plt.figure(figsize=(12, 10))

# X-axis
plt.subplot(3, 1, 1)
plt.plot(x_acc_data[:N], label='Raw Acc X', alpha=0.3)
plt.plot(x_lin_acc[:N], label='Filtered Acc X', alpha=0.9)
plt.legend()
plt.ylabel('Acceleration (m/s²)')
plt.title(f'Gravity Comparison (First {N} samples)')

# Y-axis
plt.subplot(3, 1, 2)
plt.plot(y_acc_data[:N], label='Raw Acc Y', alpha=0.3)
plt.plot(y_lin_acc[:N], label='Filtered Acc Y', alpha=0.9)
plt.legend()
plt.ylabel('Acceleration (m/s²)')

# Z-axis
plt.subplot(3, 1, 3)
plt.plot(z_acc_data[:N], label='Raw Acc Z', alpha=0.3)
plt.plot(z_lin_acc[:N], label='Filtered Acc Z', alpha=0.9)
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Acceleration (m/s²)')

plt.tight_layout()
plt.show()
