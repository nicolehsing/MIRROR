import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse
import re
from datetime import datetime

def extract_scenario_number(scenario_name):
    """Extract the number from a scenario name for sorting."""
    match = re.search(r'Scenario\s+(\d+)', str(scenario_name), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return float('inf')  # Return infinity for non-matching scenarios to sort them last

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
    
    return df

def filter_unknown_scenarios(df):
    """Filter out scenarios that don't match the "Scenario X" pattern."""
    if df is None or df.empty or 'Scenario' not in df.columns:
        return df
        
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Define the scenario pattern
    scenario_pattern = re.compile(r'Scenario\s+\d+', re.IGNORECASE)
    
    # Find unknown scenarios
    unknown_scenarios = [s for s in df['Scenario'].unique() if not scenario_pattern.match(str(s))]
    
    if unknown_scenarios:
        print("\nRemoving unknown scenarios that don't match 'Scenario X' pattern:")
        for s in unknown_scenarios:
            print(f"  - {s}")
        
        # Count rows before filtering
        rows_before = len(df)
        
        # Filter out rows with unknown scenarios
        df = df[df['Scenario'].apply(lambda s: scenario_pattern.match(str(s)) is not None)]
        
        # Count rows after filtering
        rows_after = len(df)
        print(f"Removed {rows_before - rows_after} rows with unknown scenarios")
    
    return df

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compare performance between two LLM models")
    
    parser.add_argument("--base-model", type=str, required=True,
                        help="Name of the base model (e.g., mirror)")
    
    parser.add_argument("--comparison-model", type=str, required=True,
                        help="Name of the model to compare against (e.g., gpt-4o)")
    
    parser.add_argument("--base-file", type=str, default=None,
                        help="Path to the Excel file containing base model results")
    
    parser.add_argument("--comparison-file", type=str, default=None,
                        help="Path to the Excel file containing comparison model results")
    
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing result files (if individual files not specified)")
    
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save comparison results (default: base_vs_comparison_timestamp)")
    
    parser.add_argument("--use-neutral", action="store_true",
                        help="Use neutral results instead of binary")
                        

    return parser.parse_args()

def load_model_results(model_name, model_file=None, results_dir="results", use_neutral=False):
    """Load results for a specific model"""
    result_type = "neutral" if use_neutral else "binary"
    
    # If a specific file is provided, use it
    if model_file and os.path.exists(model_file):
        try:
            df = pd.read_excel(model_file)
            if 'Model' in df.columns:
                # Filter for the specific model
                model_df = df[df['Model'] == model_name]
                if not model_df.empty:
                    print(f"Loaded {len(model_df)} {model_name} results from {model_file}")
                    

                    model_df = simplify_scenario_names(model_df)
                    
                    model_df = filter_unknown_scenarios(model_df)
                        
                    return model_df
                else:
                    print(f"Warning: No {model_name} results found in {model_file}")
                    print(f"Available models: {df['Model'].unique()}")
            else:
                # If there's no Model column, assume all rows are for the specified model
                print(f"Loaded {len(df)} results from {model_file} (assuming all are {model_name})")
                df['Model'] = model_name  # Add model column
                
                # Process scenario names
                df = simplify_scenario_names(df)
                
                # Filter unknown scenarios
                df = filter_unknown_scenarios(df)
                    
                return df
        except Exception as e:
            print(f"Error loading file {model_file}: {e}")
    
    # Otherwise, search through results directory
    all_files = glob.glob(f'{results_dir}/eval_results_*_{result_type}_*.xlsx')
    if not all_files:
        all_files = glob.glob(f'{results_dir}/*_{result_type}_*.xlsx')  # Try alternative pattern
    
    all_model_results = []
    
    for file in all_files:
        try:
            df = pd.read_excel(file)
            if 'Model' in df.columns and model_name in df['Model'].unique():
                model_df = df[df['Model'] == model_name]
                all_model_results.append(model_df)
                print(f"Found {len(model_df)} {model_name} results in {os.path.basename(file)}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if all_model_results:
        combined_df = pd.concat(all_model_results, ignore_index=True)
        print(f"Combined {len(combined_df)} total {model_name} results")
        
        combined_df = simplify_scenario_names(combined_df)

        combined_df = filter_unknown_scenarios(combined_df)
            
        return combined_df
    else:
        print(f"No {model_name} results found")
        return None

def print_plain_english_comparison(base_df, comparison_df, base_model, comparison_model):
    """Print plain English comparison between two models"""
    if base_df is None or comparison_df is None:
        print("Error: Missing data for comparison")
        return
    
    # Add source column
    base_df = base_df.copy()
    comparison_df = comparison_df.copy()
    base_df['Source'] = base_model
    comparison_df['Source'] = comparison_model
    
    # Combine the dataframes
    combined_df = pd.concat([base_df, comparison_df], ignore_index=True)
    
    # Calculate overall performance
    base_overall = base_df['Evaluation Rating'].mean()
    comparison_overall = comparison_df['Evaluation Rating'].mean()
    diff_overall = base_overall - comparison_overall
    
    # Calculate mean pass rates by scenario and model
    mean_rates = combined_df.groupby(['Scenario', 'Source'])['Evaluation Rating'].mean().reset_index()
    
    # Create pivot table for easier comparison
    pivot_table = mean_rates.pivot(index='Scenario', columns='Source', values='Evaluation Rating')
    pivot_table['Difference'] = pivot_table[base_model] - pivot_table[comparison_model]
    
    # Calculate by category
    category_rates = combined_df.groupby(['Source', 'Category'])['Evaluation Rating'].mean().reset_index()
    category_pivot = category_rates.pivot(index='Category', columns='Source', values='Evaluation Rating')
    category_pivot['Difference'] = category_pivot[base_model] - category_pivot[comparison_model]
    
    # Print comparison in plain English
    print("\n" + "="*80)
    print(f"PLAIN ENGLISH COMPARISON: {base_model} vs {comparison_model}")
    print("="*80)
    
    # Overall comparison
    print("\nOVERALL PERFORMANCE:")
    print(f"- {base_model} overall success rate: {base_overall:.2f} ({base_overall*100:.1f}%)")
    print(f"- {comparison_model} overall success rate: {comparison_overall:.2f} ({comparison_overall*100:.1f}%)")
    
    if diff_overall > 0:
        print(f"- {base_model} outperforms {comparison_model} by {diff_overall:.2f} ({diff_overall*100:.1f}%)")
    elif diff_overall < 0:
        print(f"- {comparison_model} outperforms {base_model} by {abs(diff_overall):.2f} ({abs(diff_overall)*100:.1f}%)")
    else:
        print("- Both models have identical overall performance")
    
    # Scenario comparison
    print("\nPERFORMANCE BY SCENARIO:")
    for scenario in pivot_table.index:
        base_rate = pivot_table.loc[scenario, base_model]
        comparison_rate = pivot_table.loc[scenario, comparison_model]
        diff = pivot_table.loc[scenario, 'Difference']
        
        print(f"\n{scenario}:")
        print(f"- {base_model}: {base_rate:.2f} ({base_rate*100:.1f}%)")
        print(f"- {comparison_model}: {comparison_rate:.2f} ({comparison_rate*100:.1f}%)")
        
        if diff > 0:
            print(f"- {base_model} performs better by {diff:.2f} ({diff*100:.1f}%)")
        elif diff < 0:
            print(f"- {comparison_model} performs better by {abs(diff):.2f} ({abs(diff)*100:.1f}%)")
        else:
            print("- Both models perform identically")
    
    # Category comparison
    print("\nPERFORMANCE BY CATEGORY:")
    for category in category_pivot.index:
        base_rate = category_pivot.loc[category, base_model]
        comparison_rate = category_pivot.loc[category, comparison_model]
        diff = category_pivot.loc[category, 'Difference']
        
        print(f"\n{category}:")
        print(f"- {base_model}: {base_rate:.2f} ({base_rate*100:.1f}%)")
        print(f"- {comparison_model}: {comparison_rate:.2f} ({comparison_rate*100:.1f}%)")
        
        if diff > 0:
            print(f"- {base_model} performs better by {diff:.2f} ({diff*100:.1f}%)")
        elif diff < 0:
            print(f"- {comparison_model} performs better by {abs(diff):.2f} ({abs(diff)*100:.1f}%)")
        else:
            print("- Both models perform identically")
    
    # Summary of findings
    print("\nSUMMARY OF KEY FINDINGS:")
    
    # Find scenarios where one model significantly outperforms the other
    significant_diffs = pivot_table[abs(pivot_table['Difference']) > 0.1].sort_values('Difference', ascending=False)
    
    if not significant_diffs.empty:
        print("\nScenarios with significant performance differences (>10%):")
        for scenario in significant_diffs.index:
            diff = significant_diffs.loc[scenario, 'Difference']
            if diff > 0:
                print(f"- {scenario}: {base_model} outperforms {comparison_model} by {diff:.2f} ({diff*100:.1f}%)")
            else:
                print(f"- {scenario}: {comparison_model} outperforms {base_model} by {abs(diff):.2f} ({abs(diff)*100:.1f}%)")
    
    # Find categories where one model significantly outperforms the other
    significant_cat_diffs = category_pivot[abs(category_pivot['Difference']) > 0.1].sort_values('Difference', ascending=False)
    
    if not significant_cat_diffs.empty:
        print("\nCategories with significant performance differences (>10%):")
        for category in significant_cat_diffs.index:
            diff = significant_cat_diffs.loc[category, 'Difference']
            if diff > 0:
                print(f"- {category}: {base_model} outperforms {comparison_model} by {diff:.2f} ({diff*100:.1f}%)")
            else:
                print(f"- {category}: {comparison_model} outperforms {base_model} by {abs(diff):.2f} ({abs(diff)*100:.1f}%)")
    
    print("\n" + "="*80)

def create_comparison_charts(base_df, comparison_df, base_model, comparison_model, output_dir, use_neutral=False):
    """Create comparison visualizations between two models"""
    if base_df is None or comparison_df is None:
        print("Error: Missing data for comparison")
        return
    
    # Add source column
    base_df = base_df.copy()
    comparison_df = comparison_df.copy()
    base_df['Source'] = base_model
    comparison_df['Source'] = comparison_model
    
    # Combine the dataframes
    combined_df = pd.concat([base_df, comparison_df], ignore_index=True)
    
    # Get timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_type = "neutral" if use_neutral else "binary"
    
    # 1. Side-by-side bar chart with scenarios on x-axis
    # First, calculate mean pass rates by scenario and model
    mean_rates = combined_df.groupby(['Scenario', 'Source'])['Evaluation Rating'].mean().reset_index()
    
    # Order the scenarios properly by their numerical value
    all_scenarios = combined_df['Scenario'].unique()
    scenario_order = sorted(all_scenarios, key=extract_scenario_number)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    # Use seaborn for nicer grouped bars
    ax = sns.barplot(
        data=mean_rates, 
        x='Scenario', 
        y='Evaluation Rating', 
        hue='Source',
        order=scenario_order
    )
    
    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.2f}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            fontsize=9
        )
    
    plt.title(f'Success Rate by Scenario: {base_model} vs {comparison_model}', fontsize=14)
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Mean Success Rate', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    filename = f'{output_dir}/scenario_sidebyside_comparison_{result_type}_{timestamp}.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved comparison chart to {filename}")
    
    # 2. Alternative version with cleaner x-axis labels (if the above is too cluttered)
    # Create a simplified version with cleaner scenario names
    mean_rates_copy = mean_rates.copy()
    
    # Create a mapping of scenario names to simpler names
    scenario_map = {}
    for i, scenario in enumerate(scenario_order):
        if isinstance(scenario, str) and 'Scenario' in scenario:
            # Extract the scenario number if possible
            parts = scenario.split('(')[0].strip()
            simple_name = parts
        else:
            simple_name = f"Scenario {i+1}"
        scenario_map[scenario] = simple_name
    
    # Apply the mapping
    mean_rates_copy['Scenario'] = mean_rates_copy['Scenario'].map(lambda x: scenario_map.get(x, x))
    
    # Create a data table with the exact values
    pivot_table = mean_rates.pivot(index='Scenario', columns='Source', values='Evaluation Rating')
    pivot_table['Difference'] = pivot_table[base_model] - pivot_table[comparison_model]
    
    # Save the exact values to a separate Excel file
    filename = f'{output_dir}/scenario_comparison_values_{result_type}_{timestamp}.xlsx'
    pivot_table.to_excel(filename)
    print(f"Saved exact comparison values to {filename}")
    
    # 3. Performance by category
    if 'Category' in combined_df.columns:
        plt.figure(figsize=(14, 7))
        category_rates = combined_df.groupby(['Source', 'Category'])['Evaluation Rating'].mean().unstack()
        category_rates.plot(kind='bar')
        plt.title(f'Pass Rate by Category: {base_model} vs {comparison_model}')
        plt.xlabel('Model')
        plt.ylabel('Pass Rate')
        plt.ylim(0, 1)
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        filename = f'{output_dir}/category_comparison_{result_type}_{timestamp}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved category comparison chart to {filename}")
    
    # 4. Heatmaps for each scenario
    scenarios = sorted(combined_df['Scenario'].unique())
    
    for scenario in scenarios:
        scenario_df = combined_df[combined_df['Scenario'] == scenario]
        # Skip if either model doesn't have data for this scenario
        if base_model not in scenario_df['Source'].unique() or comparison_model not in scenario_df['Source'].unique():
            print(f"Skipping heatmap for {scenario}: missing data for one or both models")
            continue
        
        # Skip if Category column is missing
        if 'Category' not in scenario_df.columns:
            print(f"Skipping heatmap for {scenario}: missing Category information")
            continue
            
        pivot_df = scenario_df.pivot_table(
            values='Evaluation Rating',
            index='Category',
            columns='Source',
            aggfunc='mean'
        )
        
        # Add difference column
        if base_model in pivot_df.columns and comparison_model in pivot_df.columns:
            pivot_df['Difference'] = pivot_df[base_model] - pivot_df[comparison_model]
            
            plt.figure(figsize=(12, len(pivot_df) * 0.6))
            sns.heatmap(
                pivot_df,
                cmap='RdYlGn',
                annot=True,
                fmt='.2f',
                center=0,
                linewidths=0.5
            )
            plt.title(f'Performance Comparison for {scenario}')
            plt.tight_layout()
            scenario_name = str(scenario).lower().replace(' ', '_').replace('(', '').replace(')', '')
            filename = f'{output_dir}/{scenario_name}_heatmap_{result_type}_{timestamp}.png'
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"Saved heatmap for {scenario} to {filename}")
    
    # 5. Overall heatmap
    overall_pivot = combined_df.pivot_table(
        values='Evaluation Rating',
        index='Scenario',
        columns='Source',
        aggfunc='mean'
    )
    
    # Add difference column
    overall_pivot['Difference'] = overall_pivot[base_model] - overall_pivot[comparison_model]
    
    plt.figure(figsize=(10, len(overall_pivot) * 0.6))
    sns.heatmap(
        overall_pivot,
        cmap='RdYlGn',
        annot=True,
        fmt='.2f',
        center=0,
        linewidths=0.5
    )
    plt.title('Overall Performance Comparison by Scenario')
    plt.tight_layout()
    filename = f'{output_dir}/overall_heatmap_{result_type}_{timestamp}.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved overall heatmap to {filename}")
    
    # Save the combined data
    filename = f'{output_dir}/combined_data_{result_type}_{timestamp}.xlsx'
    combined_df.to_excel(filename, index=False)
    print(f"Saved combined data to {filename}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Get model names
    base_model = args.base_model
    comparison_model = args.comparison_model
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{base_model}_vs_{comparison_model}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}/")
    

    # Load base model results
    print(f"Loading {base_model} results...")
    base_df = load_model_results(
        base_model, 
        model_file=args.base_file, 
        results_dir=args.results_dir,
        use_neutral=args.use_neutral
    )
    
    if base_df is None or base_df.empty:
        print(f"Failed to load {base_model} results. Exiting.")
        return
    
    # Load comparison model results
    print(f"\nLoading {comparison_model} results...")
    comparison_df = load_model_results(
        comparison_model, 
        model_file=args.comparison_file, 
        results_dir=args.results_dir,
        use_neutral=args.use_neutral
    )
    
    if comparison_df is None or comparison_df.empty:
        print(f"Failed to load {comparison_model} results. Exiting.")
        return
    
    # Create visualizations
    print("\nCreating comparison visualizations...")
    create_comparison_charts(
        base_df, 
        comparison_df, 
        base_model, 
        comparison_model, 
        output_dir, 
        use_neutral=args.use_neutral
    )
    
    # Generate text comparison
    print("\nGenerating plain English comparison...")
    print_plain_english_comparison(
        base_df, 
        comparison_df, 
        base_model, 
        comparison_model
    )
    
    print("\nDone!")

if __name__ == "__main__":
    main() 