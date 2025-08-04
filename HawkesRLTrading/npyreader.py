import numpy as np
import glob
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib.pyplot as plt

data = np.load("/Users/alirazajafree/researchprojects/probabilistictests/probabilisticalone.npy")


file_pattern = "/Users/alirazajafree/30JuneCopy/Market Impact/Price path/TWAP/*.npy"
file_list = glob.glob(file_pattern)
data_list = [np.load(f) for f in file_list]

times = np.load('/Users/alirazajafree/30JuneCopy/Market Impact/Price path/times.npy')


# times = data[0]

# portfolios = data[1]

# boundaryIndices = [0]

# for i in range(1, len(times)):
#     if(times[i] < times[i-1]):
#         boundaryIndices.append(i)
# print(boundaryIndices)
# logReturns = []

# for i in range(len(boundaryIndices)-1):
#     episodicValues = [portfolios[boundaryIndices[i]], portfolios[boundaryIndices[i+1]]]
#     logReturns.extend(np.diff(np.log(episodicValues)))

# logReturns = np.array(logReturns)
# std = np.std(logReturns)
# mean = np.mean(logReturns)

# sharpe = mean/std
# print(f"bad sharpe {sharpe}")

def getSharpeNoEpisodeBoundaries(data, window_size=100):
    """
    Calculate Sharpe ratio for single episode data using rolling windows.
    
    Parameters:
    data: numpy array with shape (2, n) where data[0] is time, data[1] is portfolio value
    window_size: size of rolling window for calculating returns
    
    Returns:
    sharpe: Sharpe ratio
    ann_sharpe: Annualized Sharpe ratio
    """
    times = data[0]
    portfolio_values = data[1]
    
    # Method 1: Use rolling windows to calculate returns
    log_returns = []
    
    for i in range(window_size, len(portfolio_values), window_size):
        start_val = portfolio_values[i - window_size]
        end_val = portfolio_values[i]
        if start_val > 0:  # Avoid log(0) or negative values
            log_returns.append(np.log(end_val / start_val))
    
    # If we don't have enough data for rolling windows, use consecutive periods
    if len(log_returns) < 2:
        log_returns = []
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i-1] > 0:
                log_returns.append(np.log(portfolio_values[i] / portfolio_values[i-1]))
    
    log_returns = np.array(log_returns)
    
    if len(log_returns) == 0:
        print("No valid returns calculated")
        return 0, 0
    
    # Calculate Sharpe ratio
    mean_return = np.mean(log_returns)
    std_return = np.std(log_returns)
    
    if std_return == 0:
        sharpe = 0
    else:
        sharpe = mean_return / std_return
    
    # Annualize (assuming your time units and scaling)
    ann_sharpe = sharpe * np.sqrt(6.5 * 12 * 252)
    
    print(f"Single Episode Sharpe: {sharpe}")
    print(f"Single Episode Ann_sharpe: {ann_sharpe}")
    
    return sharpe, ann_sharpe


def getSharpe():
    arr = data
    episode_boundaries = np.where(np.diff(arr[0]) <0)[0]
    start_idxs = episode_boundaries[:-1] + 1
    end_idxs = episode_boundaries[1:]
    log_ret2 = []
    for s, e in zip(start_idxs, end_idxs):
        log_ret2.append(np.log(arr[1][e]/arr[1][s]))
    sharpe = np.mean(log_ret2)/np.std(log_ret2)
    ann_sharpe = sharpe*np.sqrt(6.5*12*252)
    print(f"Sharpe: {sharpe}")
    print(f"Ann_sharpe: {ann_sharpe}")

def aggregatePricePaths():
    plt.figure(figsize=(12, 8))
    
    for i, data in enumerate(data_list):
        plt.plot(data, color = 'lightblue', alpha=0.7, label=f"Run {i+1}")
    # Find the maximum length among all datasets
    max_len = max([d.shape[0] for d in data_list])

    # Pad shorter arrays with np.nan for proper averaging
    padded = np.full((len(data_list), max_len), np.nan)
    for i, d in enumerate(data_list):
        padded[i, :d.shape[0]] = d

    # Compute mean across runs, ignoring nan
    mean_path = np.nanmean(padded, axis=0)

    # Remove NaN values for curve fitting
    valid_mask = ~np.isnan(mean_path)
    x_valid = np.arange(len(mean_path))[valid_mask]
    y_valid = mean_path[valid_mask]

    #plot mean
    plt.plot(mean_path, color='orange', linewidth=2, label='Average')
 
    plt.xlabel("Time step")
    plt.ylabel("Percentage change in midprice")
    plt.title("Aggregated Price Paths from Multiple Runs")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("", dpi=300, bbox_inches='tight')
    plt.show()



def aggregatePricePaths_decay():
    plt.figure(figsize=(12, 8))
    
    time_threshold_idx = np.where(times >= 1300)[0]
    if len(time_threshold_idx) == 0:
        print("No data after 1300 seconds")
        return
    
    start_idx = time_threshold_idx[0]
    filtered_data_list = []
    
    for i, data in enumerate(data_list):
        # Filter data after 1300 seconds
        if len(data) > start_idx:
            filtered_data = data[start_idx:]
            filtered_times = times[start_idx:start_idx + len(filtered_data)]
            filtered_data_list.append(filtered_data)
            plt.plot(filtered_data, color='lightblue', alpha=0.7, label=f"Run {i+1}")
    
    # Find the maximum length among filtered datasets
    if filtered_data_list:
        max_len = max([d.shape[0] for d in filtered_data_list])
        
        # Pad shorter arrays with np.nan for proper averaging
        padded = np.full((len(filtered_data_list), max_len), np.nan)
        for i, d in enumerate(filtered_data_list):
            padded[i, :d.shape[0]] = d
        
        # Compute mean across runs, ignoring nan
        mean_path = np.nanmean(padded, axis=0)
        
        # Plot mean with actual times
        mean_times = times[start_idx:start_idx + len(mean_path)]
        plt.plot(mean_path, color='orange', linewidth=2, label='Average')
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Percentage change in midprice")
    plt.title("Aggregated Price Paths from Multiple Runs (After 1300 seconds)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/alirazajafree/30JuneCopy/Market Impact/Price Decay/TWAP/TWAP_price_decay_unfitted_with_time_1.png", dpi=300, bbox_inches='tight')
    plt.show()

def aggregatePricePaths_decay_fitted_propogatormodel():
    time_threshold_idx = np.where(times >= 1300)[0]
    if len(time_threshold_idx) == 0:
        print("No data after 1300 seconds")
        return
    
    start_idx = time_threshold_idx[0]
    filtered_data_list = []
    
    for i, data in enumerate(data_list):
        # Filter data after 1300 seconds
        if len(data) > start_idx:
            filtered_data = data[start_idx:]
            filtered_data_list.append(filtered_data)
    if not filtered_data_list:
        return
    
    max_len = max([d.shape[0] for d in filtered_data_list])
        
    # Pad shorter arrays with np.nan for proper averaging
    padded = np.full((len(filtered_data_list), max_len), np.nan)
    for i, d in enumerate(filtered_data_list):
        padded[i, :d.shape[0]] = d
    
    # Compute mean across runs, ignoring nan
    mean_path = np.nanmean(padded, axis=0)
    mean_times = times[start_idx:start_idx + len(mean_path)]

    # Remove NaN values for fitting
    valid_mask = ~np.isnan(mean_path)
    t_fit = mean_times[valid_mask]
    y_fit = mean_path[valid_mask]

    # Calculate z values - since we normalized by subtracting 100, use total time range
    z = [(t-100)/1200 for t in mean_times]  # Total time range is 1200s (100 to 1300)
    z_fit = [(t-100)/1200 for t in t_fit]
    
    # Filter to only include z between 1 and 2 inclusive
    z_mask = np.array([(z_val >= 1.0) and (z_val <= 2.0) for z_val in z])
    z_fit_mask = np.array([(z_val >= 1.0) and (z_val <= 2.0) for z_val in z_fit])
    
    # Apply the mask to filter data
    z_filtered = np.array(z)[z_mask]
    mean_path_filtered = mean_path[z_mask]
    z_fit_filtered = np.array(z_fit)[z_fit_mask]
    y_fit_filtered = y_fit[z_fit_mask]
    
    print(f"Original data points: {len(z_fit)}")
    print(f"Filtered data points (1 ≤ z ≤ 2): {len(z_fit_filtered)}")
    print(f"z range: {min(z_fit_filtered):.3f} to {max(z_fit_filtered):.3f}")

    plt.figure(figsize=(12, 8))
    
    # Plot only the average on z-axis (filtered)
    plt.plot(z_filtered, mean_path_filtered, color='orange', linewidth=3, label='Average', alpha=0.8)

    # Define the propagator model varying both β and a
    def propagator_model(z_array, a, beta):
        result = []
        for z_val in z_array:
            if z_val <= 0:
                # Handle edge case where z <= 0
                term = 0
            elif z_val >= 1:
                # When z >= 1, both terms are positive
                term = z_val**(1-beta) - (z_val-1)**(1-beta)
            else:
                # When 0 < z < 1, (z-1) is negative
                # Use absolute value and handle sign carefully
                term1 = z_val**(1-beta)
                # For (z-1)^(1-beta) where z-1 < 0 and 1-beta is fractional
                # Use: |z-1|^(1-beta) * (-1)^(1-beta) = (1-z)^(1-beta) * (-1)
                term2 = (1-z_val)**(1-beta) * (-1)
                term = term1 - term2
            
            result.append(a * term)
        return np.array(result)
    
    try:
        # Check if we have enough filtered data points
        if len(z_fit_filtered) < 3:
            print("Not enough data points in z range [1, 2] for fitting")
            return
            
        # Fit both parameters 'a' and 'β' using filtered data
        initial_guess = [0.000175*np.sqrt(1200)-0.000272, 0.22]  # Initial guess for a and β
        
        # Bounds: a > 0, 0.1 < β < 1.9
        bounds = ([0, 0.1], [np.inf, 1.9])
        
        # Fit the propagator model using filtered data
        popt, pcov = curve_fit(propagator_model, z_fit_filtered, y_fit_filtered, 
                              p0=initial_guess, bounds=bounds, maxfev=5000)
        
        a_fit, beta_fit = popt
        
        # Generate fitted curve for plotting (only for z between 1 and 2)
        z_smooth = np.linspace(min(z_fit_filtered), max(z_fit_filtered), 100)
        y_propagator_fit = propagator_model(z_smooth, a_fit, beta_fit)
        
        plt.plot(z_smooth, y_propagator_fit, color='red', linewidth=2, linestyle='--',
                label=f'Propagator Model: {a_fit:.4f} * (z^{1-beta_fit:.3f} - (z-1)^{1-beta_fit:.3f})')
        
        # Calculate R-squared using filtered data
        y_fitted_points = propagator_model(z_fit_filtered, a_fit, beta_fit)
        ss_res = np.sum((y_fit_filtered - y_fitted_points) ** 2)
        ss_tot = np.sum((y_fit_filtered - np.mean(y_fit_filtered)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"Propagator Model Fit - R²: {r_squared:.4f}")
        print(f"Parameters: a={a_fit:.4f}, β={beta_fit:.3f}")
        print(f"Equation: y = {a_fit:.4f} * (z^{1-beta_fit:.3f} - (z-1)^{1-beta_fit:.3f})")
        print(f"Where z = (t-100)/1200, constrained to [1, 2]")
        
        # Add R² to plot
        plt.text(0.02, 0.98, f'R² = {r_squared:.4f}\nβ = {beta_fit:.3f}\na = {a_fit:.4f}\n1 ≤ z ≤ 2', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Calculate parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov))
        print(f"Parameter uncertainties: a ± {param_errors[0]:.4f}, β ± {param_errors[1]:.3f}")
        
    except Exception as e:
        print(f"Propagator model fit failed: {e}")
        import traceback
        traceback.print_exc()

    plt.xlabel("z = (t-100)/1200")
    plt.ylabel("Percentage change in midprice")
    plt.title("TWAP Price Impact Decay - Propagator Model Fit (1 ≤ z ≤ 2)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(1.0, 2.0)  # Set x-axis limits to show only the filtered range
    plt.tight_layout()
    # plt.savefig("/Users/alirazajafree/30JuneCopy/Market Impact/Price Decay/TWAP/TWAP_price_decay_propagator_z_1_to_2.png", 
    #             dpi=300, bbox_inches='tight')
    plt.show()
    

    


def aggregatePricePaths_decay_fitted():
    time_threshold_idx = np.where(times >= 1300)[0]
    if len(time_threshold_idx) == 0:
        print("No data after 1300 seconds")
        return
    
    start_idx = time_threshold_idx[0]
    filtered_data_list = []
    
    for i, data in enumerate(data_list):
        # Filter data after 1300 seconds
        if len(data) > start_idx:
            filtered_data = data[start_idx:]
            filtered_data_list.append(filtered_data)
    if not filtered_data_list:
        return
    
    max_len = max([d.shape[0] for d in filtered_data_list])
        
    # Pad shorter arrays with np.nan for proper averaging
    padded = np.full((len(filtered_data_list), max_len), np.nan)
    for i, d in enumerate(filtered_data_list):
        padded[i, :d.shape[0]] = d
    
    # Compute mean across runs, ignoring nan
    mean_path = np.nanmean(padded, axis=0)
    mean_times = times[start_idx:start_idx + len(mean_path)]

    # Remove NaN values for fitting
    valid_mask = ~np.isnan(mean_path)
    t_fit = mean_times[valid_mask]
    y_fit = mean_path[valid_mask]

    # Use original time values (no normalization)
    # t_normalized = t_fit - t_fit[0]  # REMOVED
    
    # --- Plot 1: Power Law Fit ---
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs
    for i, data in enumerate(filtered_data_list):
        filtered_times = times[start_idx:start_idx + len(data)]
        plt.plot(filtered_times, data, color='lightblue', alpha=0.3)
    
    # Plot mean
    plt.plot(mean_times, mean_path, color='orange', linewidth=3, label='Average', alpha=0.8)
    
    # Power law fit: y = a * t^(-b) using original time values
    def power_law(t, a, b):
        return a * np.power(t, -b)  # Changed to t^(-b)
    
    try:
        power_params, power_cov = curve_fit(power_law, t_fit, y_fit, 
                                          p0=[y_fit[0] * t_fit[0]**0.5, 0.5], 
                                          bounds=([0, 0], [np.inf, 2]))
        
        power_fitted = power_law(t_fit, *power_params)
        plt.plot(t_fit, power_fitted, color='red', linewidth=2, linestyle='--',
                label=f'Power Law: {power_params[0]:.4f} * t^(-{power_params[1]:.3f})')
        
        # Calculate R-squared
        ss_res = np.sum((y_fit - power_fitted) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared_power = 1 - (ss_res / ss_tot)
        
        print(f"Power Law - R²: {r_squared_power:.4f}")
        print(f"Power Law parameters: a={power_params[0]:.4f}, b={power_params[1]:.3f}")
        
        # Add R² to plot
        plt.text(0.02, 0.98, f'R² = {r_squared_power:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    except Exception as e:
        print(f"Power law fit failed: {e}")
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Percentage change in midprice")
    plt.title("TWAP Price Impact Decay - Power Law Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("/Users/alirazajafree/30JuneCopy/Market Impact/Price Decay/TWAP/TWAP_price_decay_power_law.png", 
    #             dpi=300, bbox_inches='tight')
    plt.show()

    # --- Plot 2: Exponential Decay Fit ---
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs
    for i, data in enumerate(filtered_data_list):
        filtered_times = times[start_idx:start_idx + len(data)]
        plt.plot(filtered_times, data, color='lightblue', alpha=0.3)
    
    # Plot mean
    plt.plot(mean_times, mean_path, color='orange', linewidth=3, label='Average', alpha=0.8)
    
    # Exponential decay fit: y = a * exp(-b*t) + c using original time values
    def exponential_decay(t, a, b, c):
        return a * np.exp(-b * t) + c
    
    try:
        # Initial guess for original time values
        initial_a = y_fit[0] - y_fit[-1]
        initial_b = 0.0001  # Much smaller decay rate for large time values
        initial_c = y_fit[-1]
        
        exp_params, exp_cov = curve_fit(exponential_decay, t_fit, y_fit,
                                       p0=[initial_a, initial_b, initial_c],
                                       bounds=([0, 0, -np.inf], [np.inf, 0.01, np.inf]))
        
        exp_fitted = exponential_decay(t_fit, *exp_params)
        plt.plot(t_fit, exp_fitted, color='green', linewidth=2, linestyle='-.',
                label=f'Exponential: {exp_params[0]:.4f}*exp(-{exp_params[1]:.6f}*t) + {exp_params[2]:.4f}')
        
        # Calculate R-squared
        ss_res = np.sum((y_fit - exp_fitted) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared_exp = 1 - (ss_res / ss_tot)
        
        print(f"Exponential Decay - R²: {r_squared_exp:.4f}")
        print(f"Exponential parameters: a={exp_params[0]:.4f}, b={exp_params[1]:.6f}, c={exp_params[2]:.4f}")
        
        # Add R² to plot
        plt.text(0.02, 0.98, f'R² = {r_squared_exp:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    except Exception as e:
        print(f"Exponential decay fit failed: {e}")
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Percentage change in midprice")
    plt.title("TWAP Price Impact Decay - Exponential Decay Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("/Users/alirazajafree/30JuneCopy/Market Impact/Price Decay/TWAP/TWAP_price_decay_exponential.png", 
    #             dpi=300, bbox_inches='tight')
    plt.show()

def price_quantity_graph_predecay():
    # Since TWAP is linear with time step window size, the executed quantity over time can be assumed to be linear
    time_threshold_idx = np.where(times <= 1300)[0]
    if len(time_threshold_idx) == 0:
        print("No data before 1300 seconds")
        return
    
    end_idx = time_threshold_idx[-1]
    filtered_data_list = []
    
    for i, data in enumerate(data_list):
        # Filter data after 1300 seconds
        filtered_data = data[:end_idx]
        filtered_data_list.append(filtered_data)
    if not filtered_data_list:
        return
    
    
    # Compute mean across runs, ignoring nan
    mean_path = np.nanmean(filtered_data_list, axis=0)
    mean_times = times[:end_idx][:len(mean_path)]

    plt.figure(figsize=(12, 8))
    
    # Plot individual runs
    for i, data in enumerate(filtered_data_list):
        filtered_times = times[0:len(data)]
        plt.plot(filtered_times, data, color='lightblue', alpha=0.3)
    
    # Plot mean
    plt.plot(mean_times, mean_path, color='orange', linewidth=3, label='Average', alpha=0.8)

    # plt.show()

    # Remove NaN values for fitting
    valid_mask = ~np.isnan(mean_path)
    t_fit = mean_times[valid_mask]
    y_fit = mean_path[valid_mask]
    
    # Fit a square root function: y = a * sqrt(t) + b
    def sqrt_func(t, a, b):
        return a * np.sqrt(t) + b

    try:
        # Use only t > 0 to avoid sqrt(0) issues
        mask = t_fit > 0
        popt, pcov = curve_fit(sqrt_func, t_fit[mask], y_fit[mask], p0=[1, 0])
        y_sqrt_fit = sqrt_func(t_fit, *popt)
        plt.plot(t_fit, y_sqrt_fit, color='red', linewidth=2, linestyle='--',
                 label=f'Sqrt Fit: {popt[0]:.4f} * sqrt(t) + {popt[1]:.4f}')
        # Calculate R-squared
        ss_res = np.sum((y_fit[mask] - sqrt_func(t_fit[mask], *popt)) ** 2)
        ss_tot = np.sum((y_fit[mask] - np.mean(y_fit[mask])) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"Square Root Fit - R²: {r_squared:.4f}")
        print(f"Square Root parameters: a={popt[0]:.4f}, b={popt[1]:.4f}")
        plt.legend()
        plt.xlabel("Quantity")
        plt.ylabel("Price impact")
        plt.title("Price vs Quantity (Square Root Fit)")

        plt.show()
    except Exception as e:
        print(f"Square root fit failed: {e}")


def price_quantity_graph_predecay_mortised():
    """
    Same as price_quantity_graph_predecay but sampling data every 50 seconds with tolerance for float imprecision
    """
    # Since TWAP is linear with time step window size, the executed quantity over time can be assumed to be linear
    time_threshold_idx = np.where(times <= 1300)[0]
    if len(time_threshold_idx) == 0:
        print("No data before 1300 seconds")
        return
    
    end_idx = time_threshold_idx[-1]
    
    # Find indices for every 50 seconds with tolerance
    target_times = np.arange(0, 1300, 50)  # [0, 50, 100, 150, ..., 1250]
    tolerance = 0.5  # Allow ±0.5 seconds tolerance
    
    sampled_indices = []
    for target_time in target_times:
        # Find the closest time within tolerance
        time_diffs = np.abs(times[:end_idx] - target_time)
        closest_idx = np.argmin(time_diffs)
        
        if time_diffs[closest_idx] <= tolerance:
            sampled_indices.append(closest_idx)
        else:
            print(f"Warning: No time found within {tolerance}s of target time {target_time}")
    
    print(f"Found {len(sampled_indices)} sample points every ~50 seconds")
    print(f"Sample times: {times[sampled_indices]}")
    
    filtered_data_list = []
    
    for i, data in enumerate(data_list):
        # Filter data to only include sampled indices (before 1300 seconds)
        if len(data) > max(sampled_indices):
            filtered_data = data[sampled_indices]
            filtered_data_list.append(filtered_data)
        else:
            # If data is shorter, take what we can
            valid_indices = [idx for idx in sampled_indices if idx < len(data)]
            if valid_indices:
                filtered_data = data[valid_indices]
                filtered_data_list.append(filtered_data)
    
    if not filtered_data_list:
        print("No filtered data available")
        return
    
    # Compute mean across runs, ignoring nan
    mean_path = np.nanmean(filtered_data_list, axis=0)
    mean_times = times[sampled_indices[:len(mean_path)]]

    plt.figure(figsize=(12, 8))
    
    # Plot individual runs with markers since we have fewer points
    for i, data in enumerate(filtered_data_list):
        sample_times = times[sampled_indices[:len(data)]]
        plt.plot(sample_times, data, color='lightblue', alpha=0.5, marker='o', markersize=3)
    
    # Plot mean with larger markers
    plt.plot(mean_times, mean_path, color='orange', linewidth=3, marker='o', markersize=6,
             label='Average (50s intervals)', alpha=0.8)

    # Remove NaN values for fitting
    valid_mask = ~np.isnan(mean_path)
    t_fit = mean_times[valid_mask]
    y_fit = mean_path[valid_mask]
    
    # Fit a square root function: y = a * sqrt(t) + b
    def sqrt_func(t, a, b):
        return a * np.sqrt(t + 1e-6) + b  # Add small epsilon to avoid sqrt(0)

    try:
        # Use only t > 0 to avoid sqrt(0) issues
        mask = t_fit > 0
        if np.sum(mask) > 2:  # Need at least 3 points for fitting
            popt, pcov = curve_fit(sqrt_func, t_fit[mask], y_fit[mask], p0=[0.001, 0])
            
            # Generate smooth curve for plotting
            t_smooth = np.linspace(min(t_fit[mask]), max(t_fit[mask]), 100)
            y_sqrt_fit_smooth = sqrt_func(t_smooth, *popt)
            
            plt.plot(t_smooth, y_sqrt_fit_smooth, color='red', linewidth=2, linestyle='--',
                     label=f'√ Fit: {popt[0]:.6f} * √t + {popt[1]:.6f}')
            
            # Calculate R-squared
            y_sqrt_fit_points = sqrt_func(t_fit[mask], *popt)
            ss_res = np.sum((y_fit[mask] - y_sqrt_fit_points) ** 2)
            ss_tot = np.sum((y_fit[mask] - np.mean(y_fit[mask])) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"Square Root Fit (50s intervals) - R²: {r_squared:.4f}")
            print(f"Square Root parameters: a={popt[0]:.6f}, b={popt[1]:.6f}")
            
            # Add R² to plot
            plt.text(0.02, 0.98, f'R² = {r_squared:.4f}\n50s sampling', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            print("Not enough valid data points for fitting")
            
    except Exception as e:
        print(f"Square root fit failed: {e}")

    plt.xlabel("Quantity")
    plt.ylabel("Price Impact (%)")
    plt.title("TWAP Price Impact Build-up - 50 Second Intervals (√t Fit)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/alirazajafree/30JuneCopy/Market Impact/Price path/TWAP/TWAP_predecay_50s_intervals.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return mean_times, mean_path, sampled_indices



def price_quantity_graph_predecay_mortised_starting_at_0():
    """
    Same as price_quantity_graph_predecay but sampling data every 50 seconds with tolerance for float imprecision
    """
    # Since TWAP is linear with time step window size, the executed quantity over time can be assumed to be linear
    time_threshold_idx = np.where(times <= 1350)[0]
    if len(time_threshold_idx) == 0:
        print("No data before 1300 seconds")
        return
    
    end_idx = time_threshold_idx[-1]
    
    # Find indices for every 50 seconds with tolerance
    target_times = np.arange(100, 1350, 50)  # [100, 150, ..., 1250]
    tolerance = 0.5  # Allow ±0.5 seconds tolerance
    target_times[0] = 101 #the simulation has a 1 second delay at the start
    
    sampled_indices = []
    for target_time in target_times:
        # Find the closest time within tolerance
        time_diffs = np.abs(times[:end_idx] - target_time)
        closest_idx = np.argmin(time_diffs)
        
        if time_diffs[closest_idx] <= tolerance:
            sampled_indices.append(closest_idx)
        else:
            print(f"Warning: No time found within {tolerance}s of target time {target_time}")
    
    print(f"Found {len(sampled_indices)} sample points every ~50 seconds")
    print(f"Sample times: {times[sampled_indices]}")
    
    filtered_data_list = []
    
    for i, data in enumerate(data_list):
        # Filter data to only include sampled indices (before 1300 seconds)
        if len(data) > max(sampled_indices):
            filtered_data = data[sampled_indices]
            filtered_data_list.append(filtered_data)
        else:
            # If data is shorter, take what we can
            valid_indices = [idx for idx in sampled_indices if idx < len(data)]
            if valid_indices:
                filtered_data = data[valid_indices]
                filtered_data_list.append(filtered_data)
    
    if not filtered_data_list:
        print("No filtered data available")
        return
    
    # Compute mean across runs, ignoring nan
    mean_path = np.nanmean(filtered_data_list, axis=0)
    mean_times = times[sampled_indices[:len(mean_path)]]

    plt.figure(figsize=(12, 8))
        
    # Plot individual runs with markers since we have fewer points
    for i, data in enumerate(filtered_data_list):
        sample_times = times[sampled_indices[:len(data)]]
        # Shift time axis to start from 0 (quantity = 0)
        quantity_axis = sample_times - sample_times[0]
        plt.plot(quantity_axis, data, color='lightblue', alpha=0.5, marker='o', markersize=3)
    
    # Plot mean with larger markers
    # Shift time axis to start from 0 (quantity = 0)
    quantity_axis_mean = mean_times - mean_times[0]
    plt.plot(quantity_axis_mean, mean_path, color='orange', linewidth=3, marker='o', markersize=6,
             label='Average (50s intervals)', alpha=0.8)
    
    # Remove NaN values for fitting
    valid_mask = ~np.isnan(mean_path)
    q_fit = quantity_axis_mean[valid_mask]  # Use shifted quantity axis instead of t_fit
    y_fit = mean_path[valid_mask]
    
    # Fit a square root function: y = a * sqrt(q) + b
    def sqrt_func(q, a, b):
        return a * np.sqrt(q + 1e-6) + b  # Add small epsilon to avoid sqrt(0)
    
    try:
        # Use only q > 0 to avoid sqrt(0) issues
        mask = q_fit >= 0 
        if np.sum(mask) > 2:  # Need at least 3 points for fitting
            popt, pcov = curve_fit(sqrt_func, q_fit[mask], y_fit[mask], p0=[0.001, 0])
            
            # Generate smooth curve for plotting
            q_smooth = np.linspace(min(q_fit[mask]), max(q_fit[mask]), 100)
            y_sqrt_fit_smooth = sqrt_func(q_smooth, *popt)
            
            plt.plot(q_smooth, y_sqrt_fit_smooth, color='red', linewidth=2, linestyle='--',
                     label=f'√ Fit: {popt[0]:.6f} * √q + {popt[1]:.6f}')
            
            # Calculate R-squared
            y_sqrt_fit_points = sqrt_func(q_fit[mask], *popt)
            ss_res = np.sum((y_fit[mask] - y_sqrt_fit_points) ** 2)
            ss_tot = np.sum((y_fit[mask] - np.mean(y_fit[mask])) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"Square Root Fit (50s intervals) - R²: {r_squared:.4f}")
            print(f"Square Root parameters: a={popt[0]:.6f}, b={popt[1]:.6f}")
            
            # Add R² to plot
            plt.text(0.02, 0.98, f'R² = {r_squared:.4f}\n50s sampling', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            print("Not enough valid data points for fitting")
            
    except Exception as e:
        print(f"Square root fit failed: {e}")
    
    plt.xlabel("Quantity (normalized to start at 0)")
    plt.ylabel("Price Impact (%)")
    plt.title("TWAP Price Impact Build-up - 50 Second Intervals (√q Fit)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/alirazajafree/30JuneCopy/Market Impact/Price path/TWAP/TWAP_predecay_50s_intervals.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return mean_times, mean_path, sampled_indices

# price_quantity_graph_predecay_mortised_starting_at_0()

# getSharpeNoEpisodeBoundaries(data)

aggregatePricePaths_decay_fitted_propogatormodel()