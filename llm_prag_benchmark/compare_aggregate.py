#!/usr/bin/env python3
"""
Compare Aggregate Performance Script

This script calculates collective averages across multiple mirror/baseline model pairs,
showing performance across all scenarios with 95% confidence intervals.

Usage:
    python compare_aggregate.py --results-dir DIR1 [DIR2 DIR3...] --output output.png
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
import re

def find_result_files(results_dirs):
    """Find all baseline and mirror binary result files in the specified directories"""
    baseline_files = []
    mirror_files = []
    
    for results_dir in results_dirs:
        if not os.path.exists(results_dir):
            print(f"Results directory not found: {results_dir}")
            continue
        
        # Find all binary results for baseline and mirror
        dir_baseline_files = glob.glob(os.path.join(results_dir, "**", "*baseline-*binary*.xlsx"), recursive=True)
        dir_mirror_files = glob.glob(os.path.join(results_dir, "**", "*mirror-*binary*.xlsx"), recursive=True)
        
        if not dir_baseline_files:
            print(f"No baseline results found in {results_dir}")
        else:
            baseline_files.extend(dir_baseline_files)
            
        if not dir_mirror_files:
            print(f"No MIRROR results found in {results_dir}")
        else:
            mirror_files.extend(dir_mirror_files)
    
    # Sort by modification time (newest first) to ensure consistent ordering
    baseline_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    mirror_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return baseline_files, mirror_files

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
    scenario_pattern = re.compile(r'Scenario\s+\d+', re.IGNORECASE)
    df = df[df['Scenario'].apply(lambda s: scenario_pattern.match(str(s)) is not None)]
    
    return df

def extract_scenario_number(scenario_name):
    """Extract the number from a scenario name for sorting."""
    match = re.search(r'Scenario\s+(\d+)', str(scenario_name), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return float('inf')  # Return infinity for non-matching scenarios to sort them last

def extract_model_name(file_path):
    """Extract the model name from the file path for grouping."""
    basename = os.path.basename(file_path)
    match = re.search(r'(baseline|mirror)-([^_]+)', basename, re.IGNORECASE)
    if match:
        return match.group(2)
    return "unknown"

def load_results(file_path, model_type=None):
    """Load results from an Excel file and set the model type (baseline/mirror)"""
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
            # Extract model name from filename
            model_name = extract_model_name(file_path)
            print(f"No model column found in {file_path}. Creating one with value: {model_name}")
            df['Model'] = model_name
        
        # Add model type column (baseline or mirror)
        if model_type:
            df['Model Type'] = model_type
        elif 'baseline' in file_path.lower():
            df['Model Type'] = 'Baseline'
        elif 'mirror' in file_path.lower():
            df['Model Type'] = 'Mirror'
        else:
            df['Model Type'] = 'Unknown'
            
        # Ensure we have 'Evaluation Rating' column
        if 'Evaluation Rating' not in df.columns:
            possible_rating_cols = [col for col in df.columns if 'rating' in col.lower() or 'eval' in col.lower() or 'score' in col.lower()]
            if possible_rating_cols:
                df['Evaluation Rating'] = df[possible_rating_cols[0]]
                print(f"Using '{possible_rating_cols[0]}' as evaluation rating column")
            else:
                print(f"Warning: No evaluation rating column found in {file_path}")
                return None
        
        # Ensure we have 'Scenario' column
        if 'Scenario' not in df.columns:
            possible_scenario_cols = [col for col in df.columns if 'scenario' in col.lower() or 'test' in col.lower()]
            if possible_scenario_cols:
                df['Scenario'] = df[possible_scenario_cols[0]]
                print(f"Using '{possible_scenario_cols[0]}' as scenario column")
            else:
                print(f"Warning: No scenario column found in {file_path}")
                return None
        
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

def aggregate_results(dataframes, group_by='Model Type'):
    """Aggregate results from multiple dataframes, grouping by model type"""
    if not dataframes:
        print("No data to aggregate")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Ensure evaluation ratings are numeric
    combined_df['Evaluation Rating'] = pd.to_numeric(combined_df['Evaluation Rating'], errors='coerce')
    combined_df = combined_df.dropna(subset=['Evaluation Rating'])
    
    if combined_df.empty:
        print("No valid numeric data after conversion")
        return None
    
    # Extract scenario numbers for sorting
    combined_df['Scenario Number'] = combined_df['Scenario'].apply(extract_scenario_number)
    
    # Create a multi-level summary by Model Type and Scenario
    summary = combined_df.groupby([group_by, 'Scenario Number', 'Scenario'])['Evaluation Rating'].agg(['mean', 'count', 'std'])
    
    # Calculate confidence intervals
    summary['ci'] = summary.apply(
        lambda row: calculate_confidence_interval(
            combined_df[(combined_df[group_by] == row.name[0]) & 
                       (combined_df['Scenario'] == row.name[2])]['Evaluation Rating']
        ),
        axis=1
    )
    
    # Calculate overall averages across all scenarios
    overall = combined_df.groupby(group_by)['Evaluation Rating'].agg(['mean', 'count', 'std'])
    overall['ci'] = overall.apply(
        lambda row: calculate_confidence_interval(
            combined_df[combined_df[group_by] == row.name]['Evaluation Rating']
        ),
        axis=1
    )
    
    return summary, overall

def plot_overall_comparison(overall_df, output_file=None, baseline_label="Baseline", mirror_label="Mirror"):
    """Create a bar chart comparing overall performance of baseline vs mirror"""
    if overall_df is None or overall_df.empty:
        print("Cannot create plot: Missing data")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Get data for plot
    models = overall_df.index.tolist()
    means = overall_df['mean'].values
    cis = overall_df['ci'].values
    counts = overall_df['count'].values
    
    # Use custom labels
    plot_labels = []
    for model in models:
        if model.lower() == 'baseline':
            plot_labels.append(baseline_label)
        elif model.lower() == 'mirror':
            plot_labels.append(mirror_label)
        else:
            plot_labels.append(model)
    
    # Create a bar chart
    bars = plt.bar(plot_labels, means, color=['#1f77b4', '#ff7f0e'])
    
    # Add error bars
    plt.errorbar(
        plot_labels, means, yerr=cis, 
        fmt='none', ecolor='black', capsize=5
    )
    
    # Add labels with sample sizes
    for i, (mean, ci, count) in enumerate(zip(means, cis, counts)):
        plt.text(i, mean + 0.02, f"{mean:.3f} ± {ci:.3f}\n(n={count})", 
                 ha='center', fontweight='bold')
    
    # Set plot labels and title
    plt.title('Collective Performance: Mirror vs Baseline (95% CI)', fontsize=16)
    plt.xlabel('Model Type', fontsize=14)
    plt.ylabel('Average Evaluation Rating', fontsize=14)
    plt.ylim(0, 1.1)  # Set y-axis limit from 0 to slightly above 1
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Overall comparison plot saved to {output_file}")
    
    # Calculate statistical significance
    if len(models) >= 2:
        # Print statistics summary
        print("\nStatistical Analysis:")
        print(f"Sample sizes: {models[0]}={counts[0]}, {models[1]}={counts[1]}")
        print(f"Means: {models[0]}={means[0]:.4f} ± {cis[0]:.4f}, {models[1]}={means[1]:.4f} ± {cis[1]:.4f}")
        
        # Cannot do proper t-test without raw data, so just report the differences
        diff = abs(means[0] - means[1])
        combined_ci = (cis[0]**2 + cis[1]**2)**0.5  # Approximate combined CI
        
        print(f"Absolute difference: {diff:.4f}")
        print(f"Combined CI: {combined_ci:.4f}")
        
        if diff > combined_ci:
            print("The difference appears statistically significant (difference > combined CI)")
        else:
            print("The difference may not be statistically significant (difference <= combined CI)")
    
    return plt

def plot_scenario_comparison(summary_df, output_file=None, baseline_label="Baseline", mirror_label="Mirror"):
    """Create a grouped bar chart comparing performance across scenarios"""
    if summary_df is None or summary_df.empty:
        print("Cannot create scenario plot: Missing data")
        return
    
    # Reshape data for plotting
    # Create a pivot table with scenarios as rows and model types as columns
    pivot_df = summary_df['mean'].unstack(level=0)
    ci_df = summary_df['ci'].unstack(level=0)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Extract data for plotting
    scenarios = [f"Scenario {i}" for i in range(1, 6)]
    scenarios = [s for s in scenarios if s in pivot_df.index.get_level_values('Scenario').unique()]
    
    # Set up bar positions
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Get data for baseline and mirror models
    baseline_means = [pivot_df.loc[pivot_df.index.get_level_values('Scenario') == s, 'Baseline'].values[0] 
                    if s in pivot_df.index.get_level_values('Scenario') and 'Baseline' in pivot_df.columns else 0 
                    for s in scenarios]
    mirror_means = [pivot_df.loc[pivot_df.index.get_level_values('Scenario') == s, 'Mirror'].values[0] 
                   if s in pivot_df.index.get_level_values('Scenario') and 'Mirror' in pivot_df.columns else 0 
                   for s in scenarios]
    
    baseline_cis = [ci_df.loc[ci_df.index.get_level_values('Scenario') == s, 'Baseline'].values[0] 
                  if s in ci_df.index.get_level_values('Scenario') and 'Baseline' in ci_df.columns else 0 
                  for s in scenarios]
    mirror_cis = [ci_df.loc[ci_df.index.get_level_values('Scenario') == s, 'Mirror'].values[0] 
                 if s in ci_df.index.get_level_values('Scenario') and 'Mirror' in ci_df.columns else 0 
                 for s in scenarios]
    
    # Create the grouped bar chart
    baseline_bars = plt.bar(x - width/2, baseline_means, width, label=baseline_label, color='#1f77b4')
    mirror_bars = plt.bar(x + width/2, mirror_means, width, label=mirror_label, color='#ff7f0e')
    
    # Add error bars
    plt.errorbar(x - width/2, baseline_means, yerr=baseline_cis, fmt='none', ecolor='black', capsize=3)
    plt.errorbar(x + width/2, mirror_means, yerr=mirror_cis, fmt='none', ecolor='black', capsize=3)
    
    # Add labels
    for i, v in enumerate(baseline_means):
        plt.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(mirror_means):
        plt.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    
    # Add plot labels and title
    plt.title('Performance by Scenario: Mirror vs Baseline', fontsize=16)
    plt.xlabel('Scenario', fontsize=14)
    plt.ylabel('Average Evaluation Rating', fontsize=14)
    plt.ylim(0, 1.1)
    plt.xticks(x, scenarios)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Scenario comparison plot saved to {output_file}")
    
    return plt

def main():
    parser = argparse.ArgumentParser(description='Compare aggregate performance between baseline and mirror models')
    parser.add_argument('--results-dirs', nargs='+', required=True,
                        help='Directories containing results (multiple directories accepted)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file name for the overall plot (defaults to auto-generated)')
    parser.add_argument('--baseline-label', type=str, default="Baseline",
                        help='Display name for baseline models')
    parser.add_argument('--mirror-label', type=str, default="Mirror",
                        help='Display name for mirror models')

    args = parser.parse_args()
    
    # Find result files
    baseline_files, mirror_files = find_result_files(args.results_dirs)
    
    if not baseline_files:
        print("No baseline result files found.")
        return 1
    
    if not mirror_files:
        print("No mirror result files found.")
        return 1
    
    print(f"Found {len(baseline_files)} baseline files and {len(mirror_files)} mirror files")
    
    # Load all result files
    baseline_dfs = []
    mirror_dfs = []
    
    for file in baseline_files:
        df = load_results(file, "Baseline")
        if df is not None and not df.empty:
            baseline_dfs.append(df)
    
    for file in mirror_files:
        df = load_results(file, "Mirror")
        if df is not None and not df.empty:
            mirror_dfs.append(df)
    
    if not baseline_dfs:
        print("No valid data loaded from baseline files.")
        return 1
    
    if not mirror_dfs:
        print("No valid data loaded from mirror files.")
        return 1
    
    # Combine and aggregate results
    all_dfs = baseline_dfs + mirror_dfs
    summary, overall = aggregate_results(all_dfs)
    
    # Print summary statistics
    print("\nOverall Performance Summary:")
    print("="*80)
    print(f"{'Model Type':<15} | {'Mean':<8} | {'CI':<8} | {'Count':<8}")
    print("-"*80)
    
    for model_type, row in overall.iterrows():
        print(f"{model_type:<15} | {row['mean']:.3f}   | ±{row['ci']:.3f}  | {int(row['count']):<8}")
    
    print("="*80)
    
    print("\nScenario Performance Summary:")
    print("="*80)
    print(f"{'Scenario':<15} | {'Model Type':<15} | {'Mean':<8} | {'CI':<8} | {'Count':<8}")
    print("-"*80)
    
    for (model_type, _, scenario), row in summary.iterrows():
        print(f"{scenario:<15} | {model_type:<15} | {row['mean']:.3f}   | ±{row['ci']:.3f}  | {int(row['count']):<8}")
    
    print("="*80)
    
    # Determine output filenames
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.results_dirs[0], "aggregate")
        os.makedirs(output_dir, exist_ok=True)
        overall_output = os.path.join(output_dir, f"overall_comparison_{timestamp}.png")
        scenario_output = os.path.join(output_dir, f"scenario_comparison_{timestamp}.png")
    else:
        base, ext = os.path.splitext(args.output)
        overall_output = args.output
        scenario_output = f"{base}_scenarios{ext}"
    
    # Generate and save plots
    plot_overall_comparison(overall, overall_output, args.baseline_label, args.mirror_label)
    plot_scenario_comparison(summary, scenario_output, args.baseline_label, args.mirror_label)
    
    return 0

if __name__ == "__main__":
    main() 