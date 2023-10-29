import numpy as np
import matplotlib.pyplot as plt

# Define a function to plot data
def plot_data(ax, data, marker_color):
    ax.scatter(data[:, 0], data[:, 1], s=20, c=marker_color, edgecolors='black')
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='-')

# Create a grid of x and y values
grid_x, grid_y = np.meshgrid(np.arange(-10, 11, 1), np.arange(-10, 11, 1))
x_arr = grid_x.reshape(-1)
y_arr = grid_y.reshape(-1)
data_arr = np.stack([x_arr, y_arr], axis=1)

# Define marker colors
marker_color = ['black'] * data_arr.shape[0]

# 1st quadrant
indices = (data_arr[:, 0] > 0) & (data_arr[:, 1] > 0)
marker_color = ['red' if ind else marker_color[i] for i, ind in enumerate(indices)]

# 2nd quadrant
indices = (data_arr[:, 0] < 0) & (data_arr[:, 1] > 0)
marker_color = ['green' if ind else marker_color[i] for i, ind in enumerate(indices)]

# 3rd quadrant
indices = (data_arr[:, 0] < 0) & (data_arr[:, 1] < 0)
marker_color = ['blue' if ind else marker_color[i] for i, ind in enumerate(indices)]

# 4th quadrant
indices = (data_arr[:, 0] > 0) & (data_arr[:, 1] < 0)
marker_color = ['yellow' if ind else marker_color[i] for i, ind in enumerate(indices)]

# Plot the original data
fig, ax = plt.subplots(figsize=(3, 3))
plot_data(ax, data_arr, marker_color)

# Save the figure
fig.tight_layout()
fig.savefig('data_points.pdf', dpi=300)

# Apply sigmoid transformation to the data
data_arr_sigmoid = 1 / (1 + np.exp(-data_arr))

# Plot data after sigmoid transformation
fig, ax = plt.subplots(figsize=(3, 3))
plot_data(ax, data_arr_sigmoid, marker_color)

# Save the figure
fig.tight_layout()
fig.savefig('data_points_after_sigmoid.pdf', dpi=300)

# Apply tanh transformation to the data
data_arr_tanh = np.tanh(data_arr)

# Plot data after tanh transformation
fig, ax = plt.subplots(figsize=(3, 3))
plot_data(ax, data_arr_tanh, marker_color)

# Save the figure
fig.tight_layout()
fig.savefig('data_points_after_tanh.pdf', dpi=300)

# Apply ReLU transformation to the data
data_arr_relu = np.maximum(0, data_arr)

# Plot data after ReLU transformation
fig, ax = plt.subplots(figsize=(3, 3))
plot_data(ax, data_arr_relu, marker_color)

# Save the figure
fig.tight_layout()
fig.savefig('data_points_after_relu.pdf', dpi=300)

# Show the plots
plt.show()
