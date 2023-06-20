import numpy as np
import pickle

class Individual:
    def __init__(self, individual_type, degree, links_to_group_1, links_to_group_2):
        self.individual_type = individual_type
        self.degree = degree
        self.links_to_group_1 = links_to_group_1
        self.links_to_group_2 = links_to_group_2

def simulate_RDS(N_1, N_2, h_1, h_2, E_D_1, E_D_2, n, k, x, method):
    p_1_to_2 = N_2 * E_D_2 / (N_2 * E_D_2 + N_1 * E_D_1 * h_1)
    p_2_to_1 = N_1 * E_D_1 / (N_1 * E_D_1 + N_2 * E_D_2 * h_2)
    
    harmonic_degree = np.zeros(2)

    def generate_individual_properties(N_1, N_2, E_D_1, E_D_2, h_1, h_2, type):

        individual_type = np.random.choice([1, 2], p=[N_1, N_2]) if type == 3 else type

        #This is the size biased degree distribution
        degree = np.random.poisson(4) + 2 if individual_type == 1 else np.random.poisson(4) + 2
        
        if individual_type == 1:
            links_to_group_2 = np.random.binomial(degree-1, p_1_to_2)
            links_to_group_1 = (degree-1) - links_to_group_2
        else:
            links_to_group_1 = np.random.binomial(degree-1, p_2_to_1)
            links_to_group_2 = (degree-1) - links_to_group_1
        return Individual(individual_type, degree, links_to_group_1, links_to_group_2)

    def generate_k_links(individual, k, method):
        invited_individuals = []
        for _ in range(k):
            total_links = individual.links_to_group_1 + individual.links_to_group_2
            if total_links == 0:
                break
            if method == 3:
                # Method 3: Tries to recruit someone from the different type if possible
                if individual.individual_type == 1 and individual.links_to_group_2 > 0:
                    invited_individual_type = 2
                    individual.links_to_group_2 -= 1
                elif individual.individual_type == 2 and individual.links_to_group_1 > 0:
                    invited_individual_type = 1
                    individual.links_to_group_1 -= 1
                else: 
                    weights = [individual.links_to_group_1, individual.links_to_group_2]
                    invited_individual_type = np.random.choice([1, 2], p=np.array(weights) / total_links)
                    if invited_individual_type == 1:
                        individual.links_to_group_1 -= 1
                    else:
                        individual.links_to_group_2 -= 1
            else:
                # Method 1 and 2: Does not try to recruit someone from the different type
                weights = [individual.links_to_group_1, individual.links_to_group_2]
                invited_individual_type = np.random.choice([1, 2], p=np.array(weights) / total_links)
                if invited_individual_type == 1:
                    individual.links_to_group_1 -= 1
                else:
                    individual.links_to_group_2 -= 1

            invited_individual = generate_individual_properties(N_1, N_2, E_D_1, E_D_2, h_1, h_2, invited_individual_type)
            invited_individuals.append(invited_individual)
        return invited_individuals
    
    current_generation = [generate_individual_properties(N_1, N_2, E_D_1, E_D_2, h_1, h_2, 3) for _ in range(n)]
    data_total_individuals = np.zeros((x, 1))
    data_type_counts = np.zeros((x, 2))
    data_invitations = np.zeros((x, 2, 2))
    data_harmonic_degree = np.zeros((x, 2)) # To store harmonic means for each generation
    data_invitations_links = np.zeros((x, 2, 2))

    degree_type_1 = []  # New line
    degree_type_2 = []  # New line

    for gen in range(x):
        new_generation = []
        results = {
            'total_individuals': 0,
            'type_counts': np.zeros(2),
            'invitations': np.zeros((2, 2)),
            'harmonic_degree': np.zeros(2), # We do not need to copy a harmonic_degree array anymore
            'invitations_links': np.zeros((2, 2))
        }

        for individual in current_generation:
            invited_individuals = generate_k_links(individual, k, method)
            for invited_individual in invited_individuals:
                # Add degree of each invited individual based on their type
                if invited_individual.individual_type == 1:
                    degree_type_1.append(invited_individual.degree)
                else:
                    degree_type_2.append(invited_individual.degree)

                results['invitations'][individual.individual_type - 1][invited_individual.individual_type - 1] += 1
                results['total_individuals'] += 1
                results['type_counts'][invited_individual.individual_type - 1] += 1
                results['invitations_links'][invited_individual.individual_type - 1][0] += invited_individual.links_to_group_1
                results['invitations_links'][invited_individual.individual_type - 1][1] += invited_individual.links_to_group_2
            new_generation.extend(invited_individuals)

        current_generation = new_generation

        # Compute harmonic mean at the end of each generation and store them in results['harmonic_degree']
        results['harmonic_degree'][0] = len(degree_type_1) / sum(1.0/i for i in degree_type_1) if degree_type_1 else 0
        results['harmonic_degree'][1] = len(degree_type_2) / sum(1.0/i for i in degree_type_2) if degree_type_2 else 0
        data_total_individuals[gen] = results['total_individuals']
        data_type_counts[gen] = results['type_counts']
        data_invitations[gen] = results['invitations']
        data_harmonic_degree[gen] = results['harmonic_degree'] # Save harmonic mean of degree for each type in each generation
        data_invitations_links[gen] = results['invitations_links']

    
    if method == 1:
        data_links = data_invitations
    else:
        data_links= data_invitations_links
        
    return data_total_individuals, data_type_counts, data_links, data_harmonic_degree # Return data_harmonic_degree

def calculate_estimators(data_total_individuals, data_type_counts, data_invitations, data_harmonic_degree, i):

    total_individuals = data_total_individuals[:i+1].sum(axis=0)
    type_counts = data_type_counts[:i+1].sum(axis=0)
    invitations = data_invitations[:i+1].sum(axis=0)
    harmonic_degree = data_harmonic_degree[:i+1].mean(axis=0)

    p_11_count = invitations[0][0]
    p_12_count = invitations[0][1]
    p_21_count = invitations[1][0]
    p_22_count = invitations[1][1]

    with np.errstate(divide='ignore', invalid='ignore'):
        # Instead of computing the average degree, we simply use the calculated harmonic mean
        E_D_1_est = harmonic_degree[0]
        E_D_2_est = harmonic_degree[1]

        try:
            p_12_est = p_12_count / (p_12_count + p_11_count)
        except ZeroDivisionError:
            p_12_est = float('nan')

        try:
            p_21_est = p_21_count / (p_21_count + p_22_count)
        except ZeroDivisionError:
            p_21_est = float('nan')

        try:
            z_est = (((p_11_count + p_12_count) * p_21_count) / ((p_21_count + p_22_count) * p_12_count))*(E_D_2_est/E_D_1_est )
        except ZeroDivisionError:
            z_est = float('nan')

        try:
            N_1_est = p_21_est / (p_12_est + p_21_est)
        except ZeroDivisionError:
            N_1_est = float('nan')

        try:
            h_1_est = (p_11_count * (p_22_count + p_21_count)) / (p_21_count * (p_11_count + p_12_count))
        except ZeroDivisionError:
            h_1_est = float('nan')

    return p_12_est, p_21_est, E_D_1_est, E_D_2_est, N_1_est, h_1_est, z_est



def run_multiple_simulations(simulations, steps):
    p_12_ests = [[] for _ in range(steps)]
    p_21_ests = [[] for _ in range(steps)]
    E_D_1_ests = [[] for _ in range(steps)]
    E_D_2_ests = [[] for _ in range(steps)]
    N_1_ests = [[] for _ in range(steps)]
    h_1_ests = [[] for _ in range(steps)]
    z_ests = [[] for _ in range(steps)]

    for i in range(simulations):
        print(i)
        data_total_individuals, data_type_counts, data_links, data_degree_mean = simulate_RDS(N_1, N_2, h_1, h_2, E_D_1, E_D_2, seeds, coupons, steps, method)
        for i in range(steps):
            p_12_est, p_21_est, E_D_1_est, E_D_2_est, N_1_est, h_1_est, z_est = calculate_estimators(data_total_individuals, data_type_counts, data_links, data_degree_mean, i)
            p_12_ests[i].append(p_12_est)
            p_21_ests[i].append(p_21_est)
            E_D_1_ests[i].append(E_D_1_est)
            E_D_2_ests[i].append(E_D_2_est)
            N_1_ests[i].append(N_1_est)
            h_1_ests[i].append(h_1_est)
            z_ests[i].append(z_est)

    return p_12_ests, p_21_ests, E_D_1_ests, E_D_2_ests, N_1_ests, h_1_ests, z_ests



# Define your sets of parameters you want to simulate
parameters = [
     (1, 1),
     (0.2, 0.2),
     (5, 0.2), 
     (5, 5),
     (0.6, 2)
]

# Average size biased degree for both groups
E_S_1 = 6
E_S_2 = 6

# Average degree for bothh groups, in this case it is the harmonic mean of Pois(E_S_1 - 2) + 2
E_D_1 = 5.302
E_D_2 = 5.302   
# Constants for seeds, coupons, generations, and simulations
seeds = 1
coupons = 1
generations = 1000
simulations = 5000

for h_1, z in parameters:
    # Population ratios
    print(h_1, z)
    N_1 = z / (z + 1)
    N_2 = 1 - N_1

    # Homophily of group 2 
    h_2 = 1 + (E_D_1 / E_D_2) * (N_1 / N_2) * h_1 - (E_D_1 / E_D_2) * (N_1 / N_2)

    for method in range(1, 4):  # Loop over the 3 methods
        ests = run_multiple_simulations(simulations, generations)

        # Generate the filename
        filename = f"simulation_results_m{method}_z{z}_h{h_1}.pkl"

        # Save the results to a file
        with open(filename, "wb") as f:
            pickle.dump(ests, f)
