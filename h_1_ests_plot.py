import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

# Setting the starting and ending generation for the simulations
start_generation = 1
end_generation = 1000

# Setting the parameters used in the simulations
h_1, z = (10, 0.5)
S = 5  # this is the expected value of your size biased distribution

# Minimum threshold of data points required to compute variance
minimum_threshold = 4900

# Computing q1 and q2 values which are used in derived variance computation for method 3
q1 = ((h_1*z)/(1 + h_1*z))**(S-1)
q2 = ((1 + (h_1 - 1)*z)/(1 + h_1*z))**(S-1)

# Filenames for the saved simulation results
file_names = [
    f"simulation_results_m1_z{z}_h{h_1}.pkl",
    f"simulation_results_m2_z{z}_h{h_1}.pkl",
    f"simulation_results_m3_z{z}_h{h_1}.pkl"
]

# Functions to calculate derived variance for each method
derived_variance_generations = [
    lambda h_1, z, generation: ((h_1 * (z + 1) * (h_1 * z * ((h_1 - 1) * z + 1) + 1)) / (generation * z ** 2)),
    lambda h_1, z, generation: (((h_1*(1 + z)*(1 + h_1*z*(1 + (-1 + h_1)*z)))) / (generation * (S-1) * (z ** 2))),
    lambda h_1, z, generation: (h_1*(-2 + q1 + q2)*(-1 + q1 + h_1*(-1 + q2)*(1 + (-1 + h_1)*z)))/(generation * (S-1) * (-1 + q1)*(-1 + q2)*z),
]
# Colors for each method in the plot
colors = ["#8B0000", "#00008B", "#006400"]
labels = ["Method 1", "Method 2", "Method 3"]

# Specify which methods you want to plot (0-based index)
selected_methods = [1, 2]

# Create labels and filename based on selected methods
selected_labels = [labels[i] for i in selected_methods]
plot_title = f"$\hat{{h_1}}$ Variance ${{\cdot}}$ Generation for {', '.join(selected_labels)}, $h_1$={h_1}, z={z}"
plot_filename = f"h_1Variance_Generation_{'_'.join(selected_labels)}_h1_{h_1}_z_{z}.png"

# Array for generations
generations = np.arange(start_generation, end_generation + 1)

# Adjust the figure size
plt.figure(figsize=(10, 6))

# Set plot style
sns.set_style('whitegrid')

# Loop over selected methods, calculate observed and derived variance and plot
for i in selected_methods:
    file_name = file_names[i]
    derived_variance = derived_variance_generations[i]
    with open(file_name, "rb") as f:
        ests = pickle.load(f)
    
    # Extract h_1 estimates from the simulation results
    p_12_ests, p_21_ests, E_D_1_ests, E_D_2_ests, N_1_ests, h_1_ests, z_ests = ests
    h_1_ests = np.array(h_1_ests[start_generation - 1:])

    # Convert inf to nan
    h_1_ests[np.isinf(h_1_ests)] = np.nan

    # Only calculate variance where count of non-NaN values exceeds the threshold
    valid_counts = np.count_nonzero(~np.isnan(h_1_ests), axis=1)
    obs_var = np.full_like(generations, np.nan, dtype=float)
    obs_var[valid_counts >= minimum_threshold] = np.nanvar(h_1_ests[valid_counts >= minimum_threshold], axis=1)

    # Compute derived variance
    der_var = np.array([derived_variance(h_1, z, gen) for gen in generations])

    # Multiply variance with generations for plotting
    obs_var_times_gen = obs_var * generations
    der_var_times_gen = der_var * generations

    # Plot observed and derived variance * generations
    plt.plot(generations, obs_var_times_gen, color=colors[i], linewidth=2, label=f"Observed Variance x Generation {labels[i]}")
    plt.plot(generations, der_var_times_gen, color=colors[i], linestyle='dashed', linewidth=2, alpha=0.7, label=f"Asymptotic Variance x Generation {labels[i]}")

# Set plot labels, legend, and title
plt.title(plot_title, fontsize=18)
plt.xlabel("Generation", fontsize=14)
plt.ylabel("${\hat{h_1}}$ variance ${\cdot}$ generation", fontsize=14)
plt.legend(fontsize=12, loc='upper right')

# Set y-axis limit
plt.ylim(0, None)

# Arrange layout for better spacing
plt.tight_layout()

# Save the plot as a png file
plt.savefig(plot_filename)

# Show the plot
plt.show()
