#!/bin/bash

# Run parallel scenarios for LLM evaluation with baseline and MIRROR models
# Usage: ./run_parallel.sh [BASELINE_MODEL] [MIRROR_MODEL] [WORKERS] [ADDITIONAL_ARGS...]
# OR: ./run_parallel.sh --mirror-model MODEL [--workers WORKERS] [--scenarios SCENARIOS] [--max-examples NUM]

# Default values
MODEL="openrouter/anthropic/claude-3.7-sonnet"
NUM_WORKERS=6
SCENARIOS=""
MAX_EXAMPLES=""
SKIP_BASELINE=false
ADDITIONAL_ARGS=""

# Check if first argument starts with -- (named arguments)
if [[ $1 == --* ]]; then
    # Parse named arguments
    while [[ $# -gt 0 ]]; do
      case $1 in
        --mirror-model)
          MODEL="$2"
          shift 2
          ;;
        --workers)
          NUM_WORKERS="$2"
          shift 2
          ;;
        --scenarios)
          SCENARIOS="$2"
          shift 2
          ;;
        --max-examples)
          MAX_EXAMPLES="$2"
          shift 2
          ;;
        --skip-baseline)
          SKIP_BASELINE=true
          shift
          ;;
        *)
          # Collect any other arguments for eval.py
          ADDITIONAL_ARGS="$ADDITIONAL_ARGS $1"
          shift
          ;;
      esac
    done
else
    # Handle old-style positional arguments for backward compatibility
    if [ $# -ge 1 ]; then
        MODEL="$1"
    fi
    if [ $# -ge 2 ]; then
        # For backward compatibility, use the second argument as the model (was separate baseline/mirror models)
        MODEL="$2"
    fi
    if [ $# -ge 3 ]; then
        NUM_WORKERS="$3"
    fi
    # Don't collect remaining arguments as ADDITIONAL_ARGS since they break eval.py
fi

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_$TIMESTAMP"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Build argument string for scenarios and max examples
ARGS=""
if [ -n "$SCENARIOS" ]; then
  ARGS="$ARGS --scenarios $SCENARIOS"
fi
if [ -n "$MAX_EXAMPLES" ]; then
  ARGS="$ARGS --max-examples $MAX_EXAMPLES"
fi
# Only add ADDITIONAL_ARGS if they were from named argument parsing
if [[ $1 == --* ]] && [ -n "$ADDITIONAL_ARGS" ]; then
  ARGS="$ARGS $ADDITIONAL_ARGS"
fi

# Print info
echo "Running parallel scenario evaluation:"
echo "Model: $MODEL"
echo "Number of workers: $NUM_WORKERS"
echo "Results directory: $RESULTS_DIR"
echo "Scenarios: ${SCENARIOS:-'all'}"
echo "Max Examples: ${MAX_EXAMPLES:-'all'}"
if [[ $1 == --* ]] && [ -n "$ADDITIONAL_ARGS" ]; then
    echo "Additional args: $ADDITIONAL_ARGS"
fi
echo "-------------------------"

# Function to handle evaluation runs
run_evaluation() {
    local model=$1
    local model_type=$2
    local output_prefix="$RESULTS_DIR/${model_type}"
    
    echo "Starting evaluation with $model_type model: $model"
    
    # Run the parallel script with arguments
    cmd="python parallel_eval.py --mirror-model \"$model\" --workers \"$NUM_WORKERS\" --results-dir \"$output_prefix\" --model-prefix \"${model_type}-\" $ARGS"
    echo "Running command: $cmd"
    eval $cmd
    
    # Return the exit status
    return $?
}

# Run baseline evaluation (only if not skipped)
if [ "$SKIP_BASELINE" = false ]; then
    echo "-------------------------"
    echo "Step 1/3: Running baseline model evaluation"
    run_evaluation "$MODEL" "baseline"

    if [ $? -ne 0 ]; then
        echo "Error: Baseline evaluation failed. Check the logs for details."
        exit 1
    fi

    echo "-------------------------"
    echo "Baseline evaluation complete. Results saved to $RESULTS_DIR/baseline/"
fi

# Run MIRROR evaluation
echo "-------------------------"
echo "Step 2/3: Running MIRROR model evaluation"
run_evaluation "$MODEL" "mirror"

if [ $? -ne 0 ]; then
    echo "Error: MIRROR evaluation failed. Check the logs for details."
    exit 1
fi

echo "-------------------------"
echo "MIRROR evaluation complete. Results saved to $RESULTS_DIR/mirror/"

# Run comparison (only if not skipped baseline)
if [ "$SKIP_BASELINE" = false ]; then
    echo "-------------------------"
    echo "Step 3/3: Running comparison analysis"

    # Find the latest binary results from each model (fixed pattern matching)
    BASELINE_BINARY=$(find "$RESULTS_DIR/baseline" -name "*binary*.xlsx" | sort -r | head -n 1)
    MIRROR_BINARY=$(find "$RESULTS_DIR/mirror" -name "*binary*.xlsx" | sort -r | head -n 1)
    
    echo "Found baseline binary results: $BASELINE_BINARY"
    echo "Found mirror binary results: $MIRROR_BINARY"

    # Create comparison directory
    mkdir -p "$RESULTS_DIR/comparison"

    # First, run the standard comparison using compare_models.py
    if [ -f "compare_models.py" ] && [ -n "$BASELINE_BINARY" ] && [ -n "$MIRROR_BINARY" ]; then
        echo "Running detailed comparison:"
        echo "- Baseline: $BASELINE_BINARY"
        echo "- MIRROR: $MIRROR_BINARY"
        
        # Extract model names from filenames for better labeling
        BASELINE_MODEL_NAME=$(basename "$MODEL" | tr '/' '-')
        MIRROR_MODEL_NAME="MIRROR (with $(basename "$MODEL" | tr '/' '-'))"
        
        # Run comparison with specific files using compare_models.py (removed unsupported flag)
        python compare_models.py \
            --base-model "$MIRROR_MODEL_NAME" \
            --comparison-model "$BASELINE_MODEL_NAME" \
            --base-file "$MIRROR_BINARY" \
            --comparison-file "$BASELINE_BINARY" \
            --output-dir "$RESULTS_DIR/comparison"
        
        if [ $? -eq 0 ]; then
            echo "Detailed comparison complete. Visualization and analysis files saved to $RESULTS_DIR/comparison/"
        else
            echo "Warning: Detailed comparison analysis failed. Check the logs for details."
        fi
    else
        echo "Warning: Could not run detailed comparison."
        echo "Reasons might include:"
        echo "- compare_models.py not found"
        echo "- No binary results files found in $RESULTS_DIR/baseline"
        echo "- No binary results files found in $RESULTS_DIR/mirror"
    fi

    # Next, run the average comparison with confidence intervals
    if [ -f "compare_averages.py" ] && [ -n "$BASELINE_BINARY" ] && [ -n "$MIRROR_BINARY" ]; then
        echo "-------------------------"
        echo "Running average comparison with confidence intervals:"
        
        # Run the average comparison script
        python compare_averages.py \
            --results-dir "$RESULTS_DIR" \
            --baseline "$BASELINE_BINARY" \
            --mirror "$MIRROR_BINARY" \
            --baseline-name "$BASELINE_MODEL_NAME" \
            --mirror-name "$MIRROR_MODEL_NAME" \
            --output "$RESULTS_DIR/comparison/average_comparison.png"
        
        if [ $? -eq 0 ]; then
            echo "Average comparison complete. Plots saved to $RESULTS_DIR/comparison/"
        else
            echo "Warning: Average comparison analysis failed. Check the logs for details."
        fi
    else
        echo "Warning: Could not run average comparison."
        echo "Reasons might include:"
        echo "- compare_averages.py not found"
        echo "- No binary results files found"
    fi
fi

echo "-------------------------"
echo "Evaluation process complete. All results saved to $RESULTS_DIR/" 