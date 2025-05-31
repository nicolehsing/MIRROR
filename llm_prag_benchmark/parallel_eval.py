"""
Scenario Parallel Runner for LLM Pragmatic Harms Eval

This script runs the evaluation scenarios in parallel processes, with each scenario
running independently to maximize CPU utilization while testing a single model.

Usage:
    python parallel_scenarios.py --mirror-model openai/gpt-4o --workers 6
"""

import os
import sys
import argparse
import concurrent.futures
import subprocess
from datetime import datetime

def run_scenario(scenario, mirror_model, results_dir, model_prefix="", extra_args=None):
    """Runs a single scenario using the main eval.py script"""
    print(f"Starting Scenario {scenario}...")
    
    # Create command to run eval.py with specific scenario
    cmd = [
        "python", "eval.py",
        "--mirror-model", mirror_model,
        "--scenario", str(scenario),
        "--results-dir", results_dir
    ]
    
    # Add model prefix if provided
    if model_prefix:
        cmd.extend(["--model-prefix", model_prefix])
    
    # Add any extra arguments
    if extra_args:
        cmd.extend(extra_args)
    
    # Print what we're running with clear label of model type
    model_type = "MIRROR" if model_prefix.startswith("mirror-") else "Standard"
    print(f"[Scenario {scenario}] Running {model_type} model: {mirror_model} with prefix: {model_prefix}")
    
    # Run as subprocess but don't capture output - let it stream to console
    try:
        print(f"[Scenario {scenario}] Command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            check=True,
            # Remove capture_output to let output stream to console
            text=True
        )
        print(f"[Scenario {scenario}] Completed successfully")
        return scenario, True, "Output streamed to console"
    except subprocess.CalledProcessError as e:
        print(f"[Scenario {scenario}] Failed with exit code: {e.returncode}")
        return scenario, False, f"Error exit code: {e.returncode}"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LLM Pragmatic Harms Eval scenarios in parallel")
    parser.add_argument(
        '--mirror-model',
        type=str,
        default="openai/gpt-4o",
        help='Model identifier for Mirror\'s internal LLM (usually an OpenRouter model).'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=6,
        help='Number of parallel workers (defaults to 6)'
    )
    parser.add_argument(
        '--scenarios',
        type=str,
        default="1,2,3,4,5",
        help='Comma-separated list of scenarios to run (default: all)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default="",
        help='Directory to save results'
    )
    parser.add_argument(
        '--model-prefix',
        type=str,
        default="",
        help='Prefix to add to result filenames (e.g., "baseline-" or "mirror-")'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        help='Maximum number of examples to process per scenario'
    )
    args, unknown_args = parser.parse_known_args()
    
    # Create results directory with timestamp if not provided
    if not args.results_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    # Parse scenarios to run
    scenarios_to_run = [int(s) for s in args.scenarios.split(",")]
    
    # Prepare extra arguments to pass to eval.py
    extra_args = []
    
    # Add max-examples if provided
    if args.max_examples is not None:
        extra_args.extend(["--max-examples", str(args.max_examples)])
    
    # Add any unknown args
    if unknown_args:
        extra_args.extend(unknown_args)

    # Track the total number of scenarios
    total_scenarios = len(scenarios_to_run)
    print(f"Running {total_scenarios} scenarios in parallel with {args.workers} workers")
    print(f"Results will be saved to: {results_dir}")
    if args.model_prefix:
        print(f"Using model prefix in filenames: {args.model_prefix}")
    if args.max_examples:
        print(f"Limiting to {args.max_examples} examples per scenario")
    
    # Run scenarios in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_scenario, 
                scenario, 
                args.mirror_model, 
                results_dir,
                args.model_prefix,
                extra_args
            ): scenario for scenario in scenarios_to_run
        }
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            scenario, success, output = future.result()
            completed += 1
            print(f"Progress: {completed}/{total_scenarios} scenarios completed")
    
    print(f"All scenarios completed. Results saved to {results_dir}/")
    
    # Combine results (optional - you may want to add this)
    # You could add code here to combine the individual scenario results
    
if __name__ == "__main__":
    main() 