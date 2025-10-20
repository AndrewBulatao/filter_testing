import matplotlib.pyplot as plt
import csv

# Initialize empty lists to store data
x_data, y_data, z_data = [], [], []

# Read data from the CSV file
with open('data/linear_acc.csv', 'r') as file:
    reader = csv.reader(file)
    #next(reader)  # Skip the header row
    for row in reader:
        x_data.append(float(row[0]) * 9.81) # Convert to m/s²
        y_data.append(float(row[1])* 9.81)
        z_data.append(float(row[2])* 9.81)

x_out, y_out, z_out = [], [], []

x_out.append(x_data[0])
y_out.append(y_data[0])
z_out.append(z_data[0])

w_c = 3.7
T_s = 0.018

alpha = (T_s*w_c)/(2+T_s*w_c)
gamma = (2-T_s*w_c)/(2+T_s*w_c)

for i in range(1, len(x_data)):
    x_out.append(alpha * (x_data[i] + x_data[i-1]) + gamma * x_out[i - 1])
    y_out.append(alpha * (y_data[i] + y_data[i-1]) + gamma * y_out[i - 1])
    z_out.append(alpha * (z_data[i] + z_data[i-1]) + gamma * z_out[i - 1])

# Plotting
plt.figure(figsize=(10,6))
plt.plot(x_data, label='Original X')
plt.plot(x_out, label='Filtered X')

plt.plot(y_data, label='Original Y')
plt.plot(y_out, label='Filtered Y')

plt.plot(z_data, label='Original Z')
plt.plot(z_out, label='Filtered Z')

plt.legend()
plt.xlabel('Sample')
plt.ylabel('Acceleration (m/s²)')
plt.title('Low-Pass Filter Effect')
plt.show()
