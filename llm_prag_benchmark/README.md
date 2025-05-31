# LLM Pragmatic Harms Evaluation (CuRaTe)

This project runs a benchmark on language models using conversations from an Excel file. It evaluates the models' ability to account for relevant/sensitive personal information mentioned in conversations. 

It uses LLaMA 3.1 405B (Instruct) to conduct evaluations as it was found to be the current most reliable (and affordable)

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables using one of these methods:
   ```bash
   # Option 1: Create .env file from example
   cp .env.example .env
   nano .env  # Edit to add your API keys
   
   # Option 2: Set environment variables directly
   export OPENROUTER_API_KEY="your_openrouter_api_key"
   export OPENAI_API_KEY="your_openai_api_key" # optional
   export REPLICATE_API_TOKEN="your_replicate_token" # optional
   export GOOGLE_API_KEY="your_google_api_key" # optional
   export ANTHROPIC_API_KEY="your_anthropic_api_key" # optional
   ```

3. Run the standard evaluation script:
   ```bash
   python3 eval.py --mirror-model "openrouter/anthropic/claude-3.7-sonnet"
   ```

   Example with specific parameters:
   ```bash
   python3 eval.py --mirror-model "openrouter/anthropic/claude-3.7-sonnet" --model-prefix "baseline-" --results-dir "results_example/"
   ```

4. Alternatively, use the parallel execution script for faster processing:
   ```bash
   ./run_parallel.sh --mirror-model "openrouter/anthropic/claude-3.7-sonnet" --workers 6
   ```

## Complete Example Workflow

Here's a step-by-step example of running a complete benchmark and analyzing the results:

1. Run baseline evaluation with Claude 3.7 Sonnet (standard model):
   ```bash
   python3 eval.py --mirror-model "openrouter/anthropic/claude-3.7-sonnet" --model-prefix "baseline-" --results-dir "results_example/"
   ```
   This runs all scenarios and saves results to the "results_example" directory.

2. Run MIRROR evaluation with the same model:
   ```bash
   python3 eval.py --mirror-model "openrouter/anthropic/claude-3.7-sonnet" --model-prefix "mirror-" --results-dir "results_example/"
   ```
   This runs the same test using the MIRROR architecture.

3. Compare the results:
   ```bash
   python3 compare_models.py --base-model "MIRROR" --comparison-model "Baseline" --base-file "results_example/eval_results_binary_mirror-anthropic-claude-3.7-sonnet_*.xlsx" --comparison-file "results_example/eval_results_binary_baseline-anthropic-claude-3.7-sonnet-20240620_*.xlsx"
   ```
   This creates detailed comparison charts.

4. Run average comparison with confidence intervals:
   ```bash
   python3 compare_averages.py --results-dir "results_example/" --baseline "results_example/eval_results_binary_baseline-anthropic-claude-3.7-sonnet_*.xlsx" --mirror "results_example/eval_results_binary_mirror-anthropic-claude-3.7-sonnet_*.xlsx" --baseline-name "Claude 3.7" --mirror-name "MIRROR+Claude 3.7"
   ```
   This shows statistical significance of any improvements.

5. Or run multiple benchmarks using the parallel script (recommended for full evaluation):
   ```bash
   # Using named arguments (recommended)
   ./run_parallel.sh --mirror-model "openrouter/anthropic/claude-3.7-sonnet" --workers 8
   ```
   This automatically runs baseline and MIRROR evaluations and all comparisons in parallel.

6. After running multiple benchmark sets, analyze aggregate performance:
   ```bash
   python3 compare_aggregate.py --results-dirs "results_20240610_123456/" "results_20240615_789012/" --baseline-label "Baseline Models" --mirror-label "MIRROR Architecture" --output "aggregate_analysis.png"
   ```
   This compares performance across multiple result sets with confidence intervals.

Example output structure:
```
results_example/
├── eval_results_binary_baseline-anthropic-claude-3.7-sonnet_20240620_123456.xlsx
├── eval_results_neutral_baseline-anthropic-claude-3.7-sonnet_20240620_123456.xlsx
├── eval_results_binary_mirror-anthropic-claude-3.7-sonnet_20240620_123457.xlsx
├── eval_results_neutral_mirror-anthropic-claude-3.7-sonnet_20240620_123457.xlsx
├── model_comparison_20240620_123458.png
├── scenario_comparison_20240620_123458.png
└── average_comparison_20240620_123459.png
```

## Evaluation Command Arguments

The `eval.py` script supports the following command-line arguments:

```bash
python3 eval.py [OPTIONS]
```

Options:
- `--mirror-model MODEL`: Model identifier for the LLM to test. For MIRROR runs with prefix "mirror-", this becomes the internal model. For baseline runs, this is the model being tested directly. (default: "openai/gpt-4o")
- `--parallel`: Run evaluation in parallel mode
- `--workers N`: Number of parallel workers (defaults to CPU count)
- `--scenario N`: Run only a specific scenario (1-5)
- `--results-dir DIR`: Directory to save results (defaults to current directory)
- `--model-prefix PREFIX`: Prefix for result filenames. Use "baseline-" for standard model evaluation, "mirror-" for MIRROR architecture evaluation
- `--max-examples N`: Maximum number of examples to process per scenario

## Running with MIRROR Architecture

The benchmark supports evaluation using the MIRROR (Modular Internal Reasoning, Reflection, Orchestrated Response) Architecture:

1. To run with MIRROR, use the `--model-prefix "mirror-"` flag with `eval.py`
2. The `run_parallel.sh` script automatically:
   - First runs the benchmark with the baseline model (standard evaluation)
   - Then runs with the MIRROR architecture using the specified model
   - Finally compares results between the two approaches

## Parallel Evaluation Details

The `run_parallel.sh` script provides two argument styles:

### Named Arguments (Recommended)
```bash
./run_parallel.sh --mirror-model "MODEL_ID" [OPTIONS]
```
Options:
- `--mirror-model MODEL`: Model to use for both baseline and MIRROR evaluation
- `--workers N`: Number of worker processes (default: 6)
- `--scenarios LIST`: Comma-separated scenarios to run (default: "1,2,3,4,5")
- `--max-examples N`: Maximum examples per scenario
- `--skip-baseline`: Skip baseline evaluation (run only MIRROR)

### Positional Arguments (Backward Compatibility)
```bash
./run_parallel.sh [BASELINE_MODEL] [MIRROR_MODEL] [WORKERS]
```
- First argument: Model for baseline (currently both baseline and MIRROR use the same model)
- Second argument: Model for MIRROR (used as the model ID)
- Third argument: Number of workers

The script automatically:
1. Creates a timestamped results directory
2. Runs multiple scenarios in parallel across separate processes
3. Saves results to organized subdirectories
4. Runs comparison analysis between baseline and MIRROR results

## Comparison and Visualization

1. Basic visualizations are automatically generated when running eval.py
2. For more advanced comparisons between model runs, use the compare_models.py script:
```bash
python3 compare_models.py --base-model "Model1" --comparison-model "Model2" --base-file <path> --comparison-file <path>
```

3. The `run_parallel.sh` script automatically runs both comparison tools after evaluating both models:
   - Detailed model comparison using the `compare_models.py` script
   - Average performance comparison with confidence intervals using the `compare_averages.py` script

4. For aggregate comparisons across multiple result sets, use compare_aggregate.py:
```bash
python3 compare_aggregate.py --results-dirs DIR1 [DIR2 DIR3...] --baseline-label "Baseline" --mirror-label "MIRROR" --output output.png
```

5. Statistical Analysis:
   - The average comparison includes 95% confidence intervals to show reliability of results
   - T-tests are performed to determine statistical significance of differences
   - Both overall averages and per-scenario breakdowns are provided

Comparison tools command line options:
```bash
# Detailed comparison
python3 compare_models.py --base-model "MIRROR" --comparison-model "Baseline" --base-file <path> --comparison-file <path>

# Average comparison with confidence intervals
python3 compare_averages.py --results-dir <dir> --baseline <path> --mirror <path> --baseline-name "Baseline" --mirror-name "MIRROR"

# Aggregate comparison across multiple result sets
python3 compare_aggregate.py --results-dirs DIR1 [DIR2 DIR3...] --baseline-label "Baseline" --mirror-label "MIRROR" --output output.png
```

## Results

Results will be saved in a timestamped directory, with subdirectories for:
- baseline: Contains the baseline model evaluation results
- mirror: Contains the MIRROR architecture evaluation results  
- comparison: Contains comparison charts and analysis

Each model evaluation produces:
- `eval_results_binary.xlsx`: Clear pass/fail evaluations 
- `eval_results_neutral.xlsx`: Ambiguous evaluations requiring manual review

The comparison directory will contain:
- Detailed scenario comparisons from `compare_models.py`
- Average performance bar chart with 95% confidence intervals from `compare_averages.py`
- Scenario-by-scenario performance breakdown with confidence intervals
- Aggregate comparisons across multiple runs from `compare_aggregate.py`

## Note

This project requires API keys for various language models. Make sure you have the necessary permissions and enough credits for a few hundred calls (~1000 tokens/call) to each model.
