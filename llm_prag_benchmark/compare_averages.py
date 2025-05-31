#!/usr/bin/env python3
"""
Compare Average Performance Script

This script calculates the average performance across all scenarios for baseline vs. MIRROR
comparisons and creates visualizations with 95% confidence intervals.

Usage:
    python compare_averages.py [--results-dir DIR] [--baseline FILE] [--mirror FILE] [--output FILE]
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

def find_latest_results_dir():
    """Find the most recent results directory"""
    # First check for standard "results" directory
    if os.path.exists("results"):
        return "results"
        
    # Then look for timestamped directories
    result_dirs = glob.glob("results_*")
    if not result_dirs:
        return None
    
    # Sort by modification time (newest first)
    result_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return result_dirs[0]

def find_result_files(results_dir):
    """Find baseline and mirror binary result files in the specified directory"""
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return None, None
    
    # Find the latest binary results for baseline and mirror
    baseline_files = glob.glob(os.path.join(results_dir, "baseline", "*baseline-*binary*.xlsx"))
    mirror_files = glob.glob(os.path.join(results_dir, "mirror", "*mirror-*binary*.xlsx"))
    
    if not baseline_files:
        print(f"No baseline results found in {results_dir}/baseline/")
        return None, None
    
    if not mirror_files:
        print(f"No MIRROR results found in {results_dir}/mirror/")
        return None, None
    
    # Sort by modification time (newest first)
    baseline_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    mirror_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return baseline_files[0], mirror_files[0]

def simplify_scenario_names(df):
    """Simplify scenario names by extracting the main scenario part."""
    if df is None or df.empty or 'Scenario' not in df.columns:
        return df
        
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Extract main scenario names (before parentheses or any special characters)
    def get_main_scenario(scenario_str):
        if not isinstance(scenario_str, str):
            return str(scenario_str)
            
        # First look for pattern "Scenario X (details)" and extract "Scenario X"
        import re
        match = re.match(r'(Scenario\s+\d+)[\s\(].*', scenario_str, re.IGNORECASE)
        if match:
            return match.group(1)
            
        # Also match just "Scenario X" with no extras
        match = re.match(r'(Scenario\s+\d+)$', scenario_str, re.IGNORECASE)
        if match:
            return match.group(1)
            
        # Otherwise, take everything before the first opening parenthesis
        if '(' in scenario_str:
            part = scenario_str.split('(')[0].strip()
            # Check if this part matches the pattern "Scenario X"
            if re.match(r'Scenario\s+\d+$', part, re.IGNORECASE):
                return part
            
        # If no parenthesis or no match, try to find any "Scenario X" pattern in the string
        match = re.search(r'(Scenario\s+\d+)', scenario_str, re.IGNORECASE)
        if match:
            return match.group(1)
            
        # If no scenario pattern found, return as is
        return scenario_str
    
    # Apply the function to simplify scenario names
    df['Original Scenario'] = df['Scenario']  # Keep original for reference
    df['Scenario'] = df['Scenario'].apply(get_main_scenario)
    
    # Filter out scenarios that don't match the "Scenario X" pattern
    import re
    scenario_pattern = re.compile(r'Scenario\s+\d+', re.IGNORECASE)

    # Filter out rows with unknown scenarios
    df = df[df['Scenario'].apply(lambda s: scenario_pattern.match(str(s)) is not None)]
    
    
    return df

def extract_scenario_number(scenario_name):
    """Extract the number from a scenario name for sorting."""
    import re
    match = re.search(r'Scenario\s+(\d+)', str(scenario_name), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return float('inf')  # Return infinity for non-matching scenarios to sort them last

def load_results(file_path, model_name=None):
    """Load results from an Excel file and optionally rename the model"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        df = pd.read_excel(file_path)
        
        # Check if file has data
        if df.empty:
            print(f"No data found in {file_path}")
            return None
            
        # Check if 'Model' column exists
        if 'Model' not in df.columns:
            # Try to identify if there's another column that could be the model
            possible_model_cols = [col for col in df.columns if 'model' in col.lower()]
            
            if possible_model_cols:
                # Use the first found model-like column
                df['Model'] = df[possible_model_cols[0]]
                print(f"Using '{possible_model_cols[0]}' as model column")
            else:
                # If no model column, create one using the provided name or file basename
                if not model_name:
                    model_name = os.path.basename(file_path).split('_')[0]
                print(f"No model column found in {file_path}. Creating one with value: {model_name}")
                df['Model'] = model_name
        
        # If model_name is provided, rename the model
        if model_name and 'Model' in df.columns:
            df['Model'] = model_name
            
        # Ensure we have 'Evaluation Rating' column
        if 'Evaluation Rating' not in df.columns:
            possible_rating_cols = [col for col in df.columns if 'rating' in col.lower() or 'eval' in col.lower() or 'score' in col.lower()]
            if possible_rating_cols:
                # Use the first found rating-like column
                df['Evaluation Rating'] = df[possible_rating_cols[0]]
                print(f"Using '{possible_rating_cols[0]}' as evaluation rating column")
            else:
                print(f"Warning: No evaluation rating column found in {file_path}")
                return None
        
        # Ensure we have 'Scenario' column
        if 'Scenario' not in df.columns:
            possible_scenario_cols = [col for col in df.columns if 'scenario' in col.lower() or 'test' in col.lower()]
            if possible_scenario_cols:
                # Use the first found scenario-like column
                df['Scenario'] = df[possible_scenario_cols[0]]
                print(f"Using '{possible_scenario_cols[0]}' as scenario column")
            else:
                # If no scenario column, create a default one
                print(f"Warning: No scenario column found in {file_path}. Creating default.")
                df['Scenario'] = "All Scenarios"
        
        # Simplify scenario names
        df = simplify_scenario_names(df)
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for data"""
    if len(data) < 2:
        return 0
    
    # Calculate 95% confidence interval
    return stats.sem(data) * stats.t.ppf((1 + confidence) / 2, len(data) - 1)

def calculate_average_performance(df, model_column='Model', evaluation_column='Evaluation Rating'):
    """Calculate average performance and confidence intervals by model"""
    if df is None or df.empty:
        return None
    
    # Debug information
    print(f"\nCalculating average performance:")
    print(f"Input DataFrame has {len(df)} rows and columns: {df.columns.tolist()}")
    print(f"Using model column: '{model_column}' and evaluation column: '{evaluation_column}'")
    
    # Check if columns exist
    if model_column not in df.columns:
        print(f"Error: Model column '{model_column}' not found in DataFrame")
        print(f"Available columns: {df.columns.tolist()}")
        return None
        
    if evaluation_column not in df.columns:
        print(f"Error: Evaluation column '{evaluation_column}' not found in DataFrame")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Ensure evaluation ratings are numeric
    df[evaluation_column] = pd.to_numeric(df[evaluation_column], errors='coerce')
    df = df.dropna(subset=[evaluation_column])
    
    if df.empty:
        print("Error: No valid numeric data after conversion")
        return None
    
    # Calculate means for each model
    model_group = df.groupby(model_column)
    model_means = model_group[evaluation_column].mean()
    
    # Calculate confidence intervals
    confidence_intervals = {}
    for model in df[model_column].unique():
        model_data = df[df[model_column] == model][evaluation_column]
        if len(model_data) >= 2:  # Need at least 2 data points for confidence interval
            confidence_intervals[model] = calculate_confidence_interval(model_data)
        else:
            print(f"Warning: Not enough data points for confidence interval for {model}")
            confidence_intervals[model] = 0
    
    # Create a DataFrame with means and CIs
    result_df = pd.DataFrame({
        'Model': model_means.index,
        'Mean': model_means.values,
        'CI': [confidence_intervals[model] for model in model_means.index]
    })
    
    print("\nResult summary:")
    for _, row in result_df.iterrows():
        print(f"{row['Model']}: {row['Mean']:.3f} ± {row['CI']:.3f}")
    
    return result_df

def plot_average_comparison(baseline_df, mirror_df, output_file=None):
    """Create and save a plot comparing average performance with confidence intervals"""
    if baseline_df is None or mirror_df is None:
        print("Cannot create plot: Missing data")
        return
    
    # Combine the dataframes
    combined_df = pd.concat([baseline_df, mirror_df])
    
    # Ensure rating column is numeric
    if 'Evaluation Rating' in combined_df.columns:
        # Try to convert to numeric, coercing errors to NaN
        combined_df['Evaluation Rating'] = pd.to_numeric(combined_df['Evaluation Rating'], errors='coerce')
        # Drop NaN values
        combined_df = combined_df.dropna(subset=['Evaluation Rating'])
        
        if combined_df.empty:
            print("Warning: No valid numeric evaluation ratings found")
            return
    
    # Calculate performance metrics
    performance_df = calculate_average_performance(combined_df)
    if performance_df is None:
        print("Cannot create plot: Failed to calculate performance metrics")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Use simple matplotlib bar chart to ensure compatibility
    models = performance_df['Model']
    means = performance_df['Mean']
    cis = performance_df['CI']
    
    # Create bar chart
    plt.bar(range(len(models)), means, color=['#1f77b4', '#ff7f0e'])
    
    # Add error bars
    plt.errorbar(range(len(models)), means, yerr=cis, fmt='none', ecolor='black', capsize=5)
    
    # Add labels
    plt.xticks(range(len(models)), models)
    
    # Add text values
    for i, (mean, ci) in enumerate(zip(means, cis)):
        plt.text(i, mean + 0.02, f"{mean:.3f} ± {ci:.3f}", ha='center', fontweight='bold')
    
    # Set plot labels and title
    plt.title('Average Performance Comparison with 95% Confidence Intervals', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Mean Evaluation Rating', fontsize=14)
    plt.ylim(0, 1.1)  # Set y-axis limit from 0 to slightly above 1
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Make it pretty
    plt.tight_layout()
    
    # Save the plot if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    # Calculate statistical significance
    if len(models) >= 2:
        model1_data = combined_df[combined_df['Model'] == models[0]]['Evaluation Rating']
        model2_data = combined_df[combined_df['Model'] == models[1]]['Evaluation Rating']
        
        if len(model1_data) > 1 and len(model2_data) > 1:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(model1_data, model2_data, equal_var=False)
            
            print("\nStatistical Analysis:")
            print(f"Sample sizes: {models[0]}={len(model1_data)}, {models[1]}={len(model2_data)}")
            print(f"T-statistic: {t_stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print("The difference is statistically significant (p < 0.05)")
            else:
                print("The difference is not statistically significant (p >= 0.05)")
        else:
            print("\nWarning: Not enough data points for statistical analysis")
    
    return plt

def create_detailed_scenario_plot(baseline_df, mirror_df, output_file=None):
    """Create a plot showing performance by scenario with confidence intervals"""
    if baseline_df is None or mirror_df is None:
        print("Cannot create detailed plot: Missing data")
        return
    
    # Combine the dataframes
    combined_df = pd.concat([baseline_df, mirror_df])
    
    # Make sure ratings are numeric
    combined_df['Evaluation Rating'] = pd.to_numeric(combined_df['Evaluation Rating'], errors='coerce')
    combined_df = combined_df.dropna(subset=['Evaluation Rating'])
    
    # Group by Model and Scenario
    print("\nCalculating scenario-specific metrics...")
    
    # If we have original scenario names, print the sample sizes
    if 'Original Scenario' in combined_df.columns:
        print("\nDetailed scenario sample sizes:")
        for model in combined_df['Model'].unique():
            print(f"\n{model}:")
            model_df = combined_df[combined_df['Model'] == model]
            orig_scenarios = model_df.groupby('Original Scenario').size()
            for scenario, count in orig_scenarios.items():
                print(f"  - {scenario}: {count} samples")
    
    # Get unique models and scenarios, sorting scenarios by number
    models = combined_df['Model'].unique()
    
    # Sort scenarios by their numeric value
    all_scenarios = combined_df['Scenario'].unique()
    scenarios = sorted(all_scenarios, key=extract_scenario_number)
    
    # Create a dictionary to store metrics for the table
    scenario_stats = {}
    
    # Initialize figure
    plt.figure(figsize=(14, 8))
    
    # Create a grouped bar chart
    bar_width = 0.35
    index = np.arange(len(scenarios))
    
    # For each model, calculate means and CIs per scenario
    for i, model in enumerate(models):
        # Get data for this model
        model_data = combined_df[combined_df['Model'] == model]
        scenario_stats[model] = {}
        
        # Calculate mean and CI for each scenario
        scenario_means = []
        scenario_cis = []
        
        for scenario in scenarios:
            scenario_values = model_data[model_data['Scenario'] == scenario]['Evaluation Rating']
            sample_size = len(scenario_values)
            
            if sample_size > 0:
                mean = scenario_values.mean()
                # Calculate CI if enough data points
                if sample_size >= 2:
                    ci = calculate_confidence_interval(scenario_values)
                else:
                    ci = 0
                scenario_means.append(mean)
                scenario_cis.append(ci)
                
                # Store stats for printing
                scenario_stats[model][scenario] = {
                    'mean': mean,
                    'ci': ci,
                    'samples': sample_size
                }
            else:
                scenario_means.append(0)
                scenario_cis.append(0)
                scenario_stats[model][scenario] = {
                    'mean': 0,
                    'ci': 0,
                    'samples': 0
                }
        
        # Plot bars for this model
        x_positions = index + i * bar_width - bar_width/2
        bars = plt.bar(x_positions, scenario_means, bar_width, 
                       label=model, 
                       color='#1f77b4' if i == 0 else '#ff7f0e')
        
        # Add error bars
        plt.errorbar(x_positions, scenario_means, yerr=scenario_cis, 
                     fmt='none', ecolor='black', capsize=3)
        
        # Add value labels above bars
        for j, (x, y, ci) in enumerate(zip(x_positions, scenario_means, scenario_cis)):
            plt.text(x, y + 0.02, f"{y:.2f}", ha='center', fontsize=9)
    
    # Add labels and title
    plt.title('Performance by Scenario with 95% Confidence Intervals', fontsize=16)
    plt.xlabel('Scenario', fontsize=14)
    plt.ylabel('Mean Evaluation Rating', fontsize=14)
    plt.ylim(0, 1.1)
    
    # Set x-tick positions and labels
    plt.xticks(index, scenarios)
    
    # Add legend
    plt.legend(title='Model')
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Detailed scenario plot saved to {output_file}")
    
    # Print a table of the results
    print("\nScenario Performance Summary:")
    print("="*80)
    print(f"{'Scenario':<15} | {'Model':<20} | {'Mean':<8} | {'CI':<8} | {'Samples':<8}")
    print("-"*80)
    
    for scenario in scenarios:
        for model in models:
            stats = scenario_stats[model][scenario]
            print(f"{scenario:<15} | {model:<20} | {stats['mean']:.3f}   | ±{stats['ci']:.3f}  | {stats['samples']:<8}")
    
    print("="*80)
    
    return plt

def main():
    parser = argparse.ArgumentParser(description='Compare average performance between baseline and MIRROR models')
    parser.add_argument('--results-dir', type=str, default=None, 
                        help='Directory containing results (defaults to most recent results directory)')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Path to baseline results file (defaults to auto-detect)')
    parser.add_argument('--mirror', type=str, default=None,
                        help='Path to MIRROR results file (defaults to auto-detect)')
    parser.add_argument('--baseline-name', type=str, default="Baseline",
                        help='Display name for baseline model')
    parser.add_argument('--mirror-name', type=str, default="MIRROR",
                        help='Display name for MIRROR model')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file name for the plot (defaults to auto-generated)')

    args = parser.parse_args()
    
    # Find results directory if not specified
    results_dir = args.results_dir
    if not results_dir:
        results_dir = find_latest_results_dir()
        if not results_dir:
            print("No results directory found. Please specify one with --results-dir.")
            return 1
    
    print(f"Using results directory: {results_dir}")
    
    # Find result files if not specified
    baseline_file = args.baseline
    mirror_file = args.mirror
    
    if not baseline_file or not mirror_file:
        detected_baseline, detected_mirror = find_result_files(results_dir)
        
        if not baseline_file:
            baseline_file = detected_baseline
        
        if not mirror_file:
            mirror_file = detected_mirror
    
    if not baseline_file or not mirror_file:
        print("Could not find required result files. Please specify them directly.")
        return 1
    
    print(f"Baseline results: {baseline_file}")
    print(f"MIRROR results: {mirror_file}")
    
    # Load the results
    baseline_df = load_results(baseline_file, args.baseline_name)
    mirror_df = load_results(mirror_file, args.mirror_name)
    
    if baseline_df is None or mirror_df is None:
        print("Failed to load required result files.")
        return 1
    
    # Determine output file if not specified
    output_file = args.output
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{results_dir}/average_comparison_{timestamp}.png"
        detailed_output_file = f"{results_dir}/scenario_comparison_{timestamp}.png"
    else:
        base, ext = os.path.splitext(output_file)
        detailed_output_file = f"{base}_detailed{ext}"
    
    # Create and save the plot
    plot_average_comparison(baseline_df, mirror_df, output_file)
    create_detailed_scenario_plot(baseline_df, mirror_df, detailed_output_file)
    
    print("\nSummary:")
    print(f"- Baseline model: {args.baseline_name}")
    print(f"- MIRROR model: {args.mirror_name}")
    print(f"- Average comparison plot: {output_file}")
    print(f"- Detailed scenario plot: {detailed_output_file}")
    
    return 0

if __name__ == "__main__":
    main() 