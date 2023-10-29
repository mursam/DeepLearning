import numpy as np
import matplotlib.pyplot as plt

# Set font size in plots
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8})

# Define candidate functions
def h1(s):
    return -np.minimum(1, s)

def h2(s):
    return 0.5 * s + 0.5

def h3(s):
    return np.minimum(s, 0.2 * s)

def h4(s):
    return np.where(s >= 0, np.minimum(s, 0.2 * s), np.maximum(s, 0.2 * s))

s_arr = np.arange(-8, 8.01, 0.01)
h1_arr = h1(s_arr)
h2_arr = h2(s_arr)
h3_arr = h3(s_arr)
h4_arr = h4(s_arr)

def plot_h(temp_ax, s, h, text_str):
    # Plot h
    temp_ax.plot(s, h, color='red')
    temp_ax.text(0.2, 0.9, text_str, fontsize=8, horizontalalignment='left', verticalalignment='top', transform=temp_ax.transAxes)

    # Decorate the axis
    temp_ax.set_xticks(np.arange(-8, 8.01, 2))
    temp_ax.set_axisbelow(True)
    temp_ax.grid(True, which='major', linestyle='--', color='lightgrey', alpha=0.5)

fig, ax = plt.subplots(1, 4, figsize=(6, 2))

# Plot h1
plot_h(ax[0], s_arr, h1_arr, r'$h_1$')

# Plot h2
plot_h(ax[1], s_arr, h2_arr, r'$h_2$')

# Plot h3
plot_h(ax[2], s_arr, h3_arr, r'$h_3$')

# Plot h4
plot_h(ax[3], s_arr, h4_arr, r'$h_4$')

# Save figure
fig.tight_layout()
fig.savefig('candidate_functions.pdf', dpi=300)
