import numpy as np
import matplotlib.pyplot as plt

def simulate_node_traces(total_nodes, min_online, max_online, oscillation_time, simulation_duration, time_step=0.1):
    """
    Simulate node churn and record per-node online intervals (traces).

    Each node gets:
      - A random phase phi_i (determining its preferred time within the cycle)
      - A bias beta_i (drawn from a skewed distribution so that most nodes have a negative bias,
        meaning they are online only near the peak; a few have positive bias and stay online longer)

    At each time t the target number of online nodes is:
       desired_online = min_online + fraction*(max_online - min_online)
    where fraction = (sin(2pi t/T) + 1)/2.
    
    The effective score for node i at time t is:
       score_i(t) = sin(2pi t/T + phi_i) + beta_i.
    
    The nodes with the top "desired_online" scores are marked online.
    
    Parameters:
      total_nodes       : int   - Total number of nodes.
      min_online        : int   - Minimum number of nodes online.
      max_online        : int   - Maximum number of nodes online.
      oscillation_time  : float - Period of the oscillatory (diurnal) pattern.
      simulation_duration: float- Total simulation time.
      time_step         : float - Time resolution of the simulation.
      
    Returns:
      times       : np.array of time points.
      online_counts: list of the number of online nodes at each time step.
      traces      : dict mapping node id -> list of (online_start, online_end) intervals.
    """
    # Each node gets a random phase in [0, 2pi)
    phases = np.random.uniform(0, 2 * np.pi, total_nodes)
    
    # Draw a bias for each node from a Beta distribution.
    # Beta(2,5) is skewed toward lower values; shift it so that the mean is near 0.
    raw_bias = np.random.beta(2, 5, total_nodes)  # values in [0,1]
    beta_mean = 2 / (2 + 5)  # approx 0.2857
    biases = raw_bias - beta_mean  # now most biases are negative; a few will be positive.
    
    times = np.arange(0, simulation_duration, time_step)
    online_counts = []  # to record total online nodes over time

    # For each node, record the intervals (start, end) during which it is online.
    traces = {i: [] for i in range(total_nodes)}
    
    # Keep track of current online status and the start time of the current online period.
    current_status = np.zeros(total_nodes, dtype=bool)  # False: offline, True: online
    current_start = np.full(total_nodes, np.nan)

    for t in times:
        # Compute fraction based on diurnal sine curve.
        fraction = (np.sin(2 * np.pi * t / oscillation_time) + 1) / 2  # in [0,1]
        desired_online = min_online + fraction * (max_online - min_online)
        desired_online = int(round(desired_online))
        
        # Compute each node's effective score.
        scores = np.sin(2 * np.pi * t / oscillation_time + phases) + biases
        # Sort nodes in descending order of score.
        sorted_indices = np.argsort(scores)[::-1]
        # Mark the top 'desired_online' nodes as online.
        online_nodes = np.zeros(total_nodes, dtype=bool)
        online_nodes[sorted_indices[:desired_online]] = True

        online_counts.append(desired_online)

        # Check for transitions for each node.
        for i in range(total_nodes):
            if online_nodes[i] and not current_status[i]:
                # Transition from offline to online.
                current_status[i] = True
                current_start[i] = t
            elif (not online_nodes[i]) and current_status[i]:
                # Transition from online to offline.
                traces[i].append((current_start[i], t))
                current_status[i] = False
                current_start[i] = np.nan

    # If a node is still online at the end, close its interval.
    for i in range(total_nodes):
        if current_status[i]:
            traces[i].append((current_start[i], simulation_duration))

    return times, online_counts, traces

if __name__ == '__main__':
    # Simulation parameters
    total_nodes = 100
    min_online = 20
    max_online = 80
    oscillation_time = 3600         # e.g. 2-hour diurnal cycle
    simulation_duration = 7200      # simulate for 2 time units (hours)
    time_step = 1

    # Run the simulation.
    times, online_counts, traces = simulate_node_traces(
        total_nodes, min_online, max_online, oscillation_time, simulation_duration, time_step
    )

    # Plot the overall number of online nodes over time.
    plt.figure(figsize=(10, 5))
    plt.plot(times, online_counts, label="Online Nodes")
    plt.xlabel("Time")
    plt.ylabel("Number of Online Nodes")
    plt.title("Simulated Network Churn with Heterogeneous Durations")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Example: Print the first 5 nodes' traces.
    for i in range(10):
        print(f"Node {i} online intervals:")
        for (start, end) in traces[i]:
            print(f"  from t = {start:.2f} to t = {end:.2f}")

    # Compute the average online duration (in seconds) across all nodes
    all_online_durations = []
    for intervals in traces.values():
        for (start, end) in intervals:
            all_online_durations.append(end - start)

    if all_online_durations:
        average_duration = sum(all_online_durations) / len(all_online_durations)
        print(f"Average online duration per interval: {average_duration:.2f} seconds")
    else:
        print("No online intervals recorded.")