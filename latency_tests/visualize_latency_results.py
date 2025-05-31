import os
import sys
import json
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_results(results_dir='test_results', specific_file=None):
    """Load latency test results from the specified directory or file."""
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return []
    
    if specific_file:
        # Load a specific file
        filepath = os.path.join(results_dir, specific_file)
        if not os.path.exists(filepath):
            print(f"Results file '{filepath}' not found.")
            return []
        
        try:
            with open(filepath, 'r') as f:
                return [json.load(f)]
        except Exception as e:
            print(f"Error loading results from {filepath}: {e}")
            return []
    else:
        # Load all JSON files in the directory
        # Support both production and human-simulated test results
        result_files = glob.glob(os.path.join(results_dir, "production_latency_*.json"))
        result_files.extend(glob.glob(os.path.join(results_dir, "human_simulated_latency_*.json")))
        results = []
        
        for filepath in result_files:
            try:
                with open(filepath, 'r') as f:
                    results.append(json.load(f))
            except Exception as e:
                print(f"Error loading results from {filepath}: {e}")
        
        # Sort by timestamp
        results.sort(key=lambda x: x.get("test_config", {}).get("timestamp", ""))
        return results

def visualize_response_times(results):
    """Visualize response times from test results."""
    if not results:
        print("No results to visualize.")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot response times for each test
    for i, result in enumerate(results):
        config = result.get("test_config", {})
        metrics = result.get("metrics", {})
        all_turns = result.get("all_turns", [])
        
        # Extract test parameters
        model = config.get("model", "unknown")
        users = config.get("concurrent_users", 0)
        turns = config.get("turns_per_user", 0) or config.get("num_turns", 0)
        timestamp = config.get("timestamp", "")
        
        # Determine test type
        is_human_simulation = "typing_speed_wpm" in config
        test_type = "Human-Sim" if is_human_simulation else "Production"
        
        # Response times
        response_times = metrics.get("response_times", [])
        
        if not response_times or not all_turns:
            print(f"No response time data for test {i+1}")
            continue
        
        # Sort turns by timestamp if available, otherwise use turn order
        if any("timestamp" in t for t in all_turns):
            all_turns.sort(key=lambda x: x.get("timestamp", 0))
            timestamps = [t.get("timestamp", 0) for t in all_turns]
            if timestamps:
                start_time = min(timestamps)
                relative_times = [t - start_time for t in timestamps]
            else:
                relative_times = list(range(len(response_times)))
        else:
            # If no timestamps, use turn number as x-axis
            relative_times = [t.get("turn", idx+1) - 1 for idx, t in enumerate(all_turns)]
        
        # Plot response times vs relative time
        if is_human_simulation:
            label = f"Test {i+1}: {model} ({test_type}, {turns} turns)"
        else:
            label = f"Test {i+1}: {model} ({test_type}, {users} users, {turns} turns)"
            
        plt.plot(relative_times, [t.get("response_time", 0) for t in all_turns], 'o-', label=label)
    
    plt.title("Response Time vs Relative Test Time")
    plt.xlabel("Time since test start (s) or Turn Number")
    plt.ylabel("Response Time (s)")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"response_times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    
    plt.tight_layout()
    plt.show()

def visualize_background_activity(results):
    """Visualize background thread activity over time."""
    if not results:
        print("No results to visualize.")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot background thread active status for each test
    for i, result in enumerate(results):
        config = result.get("test_config", {})
        all_turns = result.get("all_turns", [])
        
        # Extract test parameters
        model = config.get("model", "unknown")
        users = config.get("concurrent_users", 0)
        turns = config.get("turns_per_user", 0) or config.get("num_turns", 0)
        
        # Determine test type
        is_human_simulation = "typing_speed_wpm" in config
        test_type = "Human-Sim" if is_human_simulation else "Production"
        
        if not all_turns:
            print(f"No turn data for test {i+1}")
            continue
        
        # Sort turns by timestamp if available, otherwise use turn order
        if any("timestamp" in t for t in all_turns):
            all_turns.sort(key=lambda x: x.get("timestamp", 0))
            timestamps = [t.get("timestamp", 0) for t in all_turns]
            if timestamps:
                start_time = min(timestamps)
                relative_times = [t - start_time for t in timestamps]
            else:
                relative_times = list(range(len(all_turns)))
        else:
            # If no timestamps, use turn number as x-axis
            relative_times = [t.get("turn", idx+1) - 1 for idx, t in enumerate(all_turns)]
        
        # Plot background active status vs relative time
        if is_human_simulation:
            label = f"Test {i+1}: {model} ({test_type}, {turns} turns)"
        else:
            label = f"Test {i+1}: {model} ({test_type}, {users} users, {turns} turns)"
        
        # Use background_thread_active or background_active depending on format
        if "background_thread_active" in all_turns[0]:
            bg_status = [1 if t.get("background_thread_active", False) else 0 for t in all_turns]
        else:
            bg_status = [1 if t.get("background_active", 0) > 0 else 0 for t in all_turns]
            
        plt.plot(relative_times, bg_status, 'o-', label=label)
    
    plt.title("Background Thread Activity vs Relative Test Time")
    plt.xlabel("Time since test start (s) or Turn Number")
    plt.ylabel("Background Thread Active (1=Yes, 0=No)")
    plt.yticks([0, 1], ["Inactive", "Active"])
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"background_activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    
    plt.tight_layout()
    plt.show()

def visualize_queue_lengths_vs_response_times(results):
    """Analyze how background queue affects response time."""
    if not results:
        print("No results to visualize.")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot comparison of response times with/without background thread running
    x_pos = np.arange(len(results))
    width = 0.35
    
    with_bg_means = []
    without_bg_means = []
    with_bg_stdevs = []
    without_bg_stdevs = []
    test_labels = []
    
    for i, result in enumerate(results):
        config = result.get("test_config", {})
        all_turns = result.get("all_turns", [])
        
        # Extract test parameters
        model = config.get("model", "unknown").split('/')[-1]  # Just get model name without provider
        users = config.get("concurrent_users", 0)
        turns = config.get("turns_per_user", 0) or config.get("num_turns", 0)
        
        # Determine test type
        is_human_simulation = "typing_speed_wpm" in config
        test_type = "H" if is_human_simulation else "P"  # H for Human-Sim, P for Production
        
        if not all_turns:
            print(f"No turn data for test {i+1}")
            continue
        
        # Split times by background thread status
        # Handle different field names in different test formats
        if "background_thread_active" in all_turns[0]:
            with_bg_times = [t.get("response_time", 0) for t in all_turns if t.get("background_thread_active", False)]
            without_bg_times = [t.get("response_time", 0) for t in all_turns if not t.get("background_thread_active", False)]
        else:
            with_bg_times = [t.get("response_time", 0) for t in all_turns if t.get("background_active", 0) > 0]
            without_bg_times = [t.get("response_time", 0) for t in all_turns if t.get("background_active", 0) == 0]
        
        # Calculate stats
        with_bg_mean = np.mean(with_bg_times) if with_bg_times else 0
        without_bg_mean = np.mean(without_bg_times) if without_bg_times else 0
        with_bg_std = np.std(with_bg_times) if len(with_bg_times) > 1 else 0
        without_bg_std = np.std(without_bg_times) if len(without_bg_times) > 1 else 0
        
        with_bg_means.append(with_bg_mean)
        without_bg_means.append(without_bg_mean)
        with_bg_stdevs.append(with_bg_std)
        without_bg_stdevs.append(without_bg_std)
        
        if is_human_simulation:
            test_labels.append(f"Test {i+1}\n{test_type},{turns}t")
        else:
            test_labels.append(f"Test {i+1}\n{test_type},{users}u,{turns}t")
    
    # Create the bar chart
    plt.bar(x_pos - width/2, with_bg_means, width, label='With BG Thread', yerr=with_bg_stdevs, alpha=0.7)
    plt.bar(x_pos + width/2, without_bg_means, width, label='Without BG Thread', yerr=without_bg_stdevs, alpha=0.7)
    
    plt.ylabel('Average Response Time (s)')
    plt.title('Response Time Comparison: With vs Without Background Thread')
    plt.xticks(x_pos, test_labels)
    plt.legend()
    
    # Add value labels on bars
    for i, v in enumerate(with_bg_means):
        if v > 0:
            plt.text(i - width/2, v + 0.1, f"{v:.2f}s", ha='center')
    
    for i, v in enumerate(without_bg_means):
        if v > 0:
            plt.text(i + width/2, v + 0.1, f"{v:.2f}s", ha='center')
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"response_time_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    
    plt.tight_layout()
    plt.show()

def visualize_human_simulation_time_breakdown(results):
    """Visualize time breakdown for human-simulated tests."""
    # Filter only human-simulated tests
    human_sim_results = [r for r in results if "typing_speed_wpm" in r.get("test_config", {})]
    
    if not human_sim_results:
        print("No human-simulated test results to visualize.")
        return
    
    plt.figure(figsize=(14, 8))
    
    # For each test, create a stacked bar showing time breakdown
    test_labels = []
    typing_times = []
    reading_times = []
    ai_times = []
    
    for i, result in enumerate(human_sim_results):
        config = result.get("test_config", {})
        metrics = result.get("metrics", {})
        
        # Extract test parameters
        model = config.get("model", "unknown").split('/')[-1]  # Just model name
        turns = config.get("num_turns", 0)
        typing_speed = config.get("typing_speed_wpm", 0)
        reading_speed = config.get("reading_speed_wpm", 0)
        
        # Get time metrics
        total_typing_time = metrics.get("avg_typing_time", 0) * turns
        total_reading_time = metrics.get("avg_reading_time", 0) * turns
        total_ai_time = metrics.get("total_ai_time", 0)
        
        typing_times.append(total_typing_time)
        reading_times.append(total_reading_time)
        ai_times.append(total_ai_time)
        
        test_labels.append(f"Test {i+1}\n{model}\n{turns}t")
    
    # Create stacked bar chart
    bar_width = 0.7
    x_pos = np.arange(len(human_sim_results))
    
    typing_bars = plt.bar(x_pos, typing_times, bar_width, label='Human Typing Time', color='#3498db')
    reading_bars = plt.bar(x_pos, reading_times, bar_width, bottom=typing_times, label='Human Reading Time', color='#2ecc71')
    
    # Calculate cumulative height for AI bars
    bottoms = [t + r for t, r in zip(typing_times, reading_times)]
    ai_bars = plt.bar(x_pos, ai_times, bar_width, bottom=bottoms, label='AI Response Time', color='#e74c3c')
    
    plt.ylabel('Time (seconds)')
    plt.title('Conversation Time Breakdown: Human vs AI')
    plt.xticks(x_pos, test_labels)
    plt.legend()
    
    # Add percentage labels
    for i in range(len(human_sim_results)):
        total_time = typing_times[i] + reading_times[i] + ai_times[i]
        
        # Skip empty bars
        if total_time == 0:
            continue
            
        typing_pct = typing_times[i] / total_time * 100
        reading_pct = reading_times[i] / total_time * 100
        ai_pct = ai_times[i] / total_time * 100
        
        # Add percentage labels if big enough to show
        if typing_pct > 5:
            plt.text(i, typing_times[i]/2, f"{typing_pct:.1f}%", ha='center', va='center')
        
        if reading_pct > 5:
            plt.text(i, typing_times[i] + reading_times[i]/2, f"{reading_pct:.1f}%", ha='center', va='center')
        
        if ai_pct > 5:
            plt.text(i, bottoms[i] + ai_times[i]/2, f"{ai_pct:.1f}%", ha='center', va='center')
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"human_sim_time_breakdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    
    plt.tight_layout()
    plt.show()

def visualize_human_simulation_conversation_flow(results):
    """Visualize the flow of a conversation with human simulation."""
    # Filter only human-simulated tests
    human_sim_results = [r for r in results if "typing_speed_wpm" in r.get("test_config", {})]
    
    if not human_sim_results:
        print("No human-simulated test results to visualize.")
        return
    
    # Select the most recent test for detailed visualization
    result = human_sim_results[-1]  # Most recent test
    
    plt.figure(figsize=(14, 8))
    
    all_turns = result.get("all_turns", [])
    if not all_turns:
        print("No turn data available")
        return
    
    # Sort turns by turn number
    all_turns.sort(key=lambda x: x.get("turn", 0))
    
    # Extract timing information for each turn
    turns = [t.get("turn", i+1) for i, t in enumerate(all_turns)]
    typing_times = [t.get("simulated_typing_time", 0) for t in all_turns]
    response_times = [t.get("response_time", 0) for t in all_turns]
    reading_times = [t.get("simulated_reading_time", 0) for t in all_turns]
    
    # Create a timeline visualization
    current_time = 0
    ai_periods = []  # (start, end, turn)
    human_typing_periods = []  # (start, end, turn)
    human_reading_periods = []  # (start, end, turn)
    
    for i, turn in enumerate(turns):
        # Human typing period
        typing_start = current_time
        typing_end = typing_start + typing_times[i]
        human_typing_periods.append((typing_start, typing_end, turn))
        current_time = typing_end
        
        # AI response period
        response_start = current_time
        response_end = response_start + response_times[i]
        ai_periods.append((response_start, response_end, turn))
        current_time = response_end
        
        # Human reading period (if not the last turn)
        if i < len(turns) - 1:
            reading_start = current_time
            reading_end = reading_start + reading_times[i]
            human_reading_periods.append((reading_start, reading_end, turn))
            current_time = reading_end
    
    # Plot the timeline
    height = 0.3
    
    # Plot human typing periods
    for start, end, turn in human_typing_periods:
        plt.barh(1, end - start, height, left=start, color='#3498db', alpha=0.7)
        if end - start > 1:  # Only add label if bar is wide enough
            plt.text(start + (end - start)/2, 1, f"Type {turn}", ha='center', va='center')
    
    # Plot AI response periods
    for start, end, turn in ai_periods:
        plt.barh(2, end - start, height, left=start, color='#e74c3c', alpha=0.7)
        if end - start > 1:  # Only add label if bar is wide enough
            plt.text(start + (end - start)/2, 2, f"AI {turn}", ha='center', va='center')
    
    # Plot human reading periods
    for start, end, turn in human_reading_periods:
        plt.barh(3, end - start, height, left=start, color='#2ecc71', alpha=0.7)
        if end - start > 1:  # Only add label if bar is wide enough
            plt.text(start + (end - start)/2, 3, f"Read {turn}", ha='center', va='center')
    
    # Get test info for title
    config = result.get("test_config", {})
    model = config.get("model", "unknown")
    typing_speed = config.get("typing_speed_wpm", 0)
    reading_speed = config.get("reading_speed_wpm", 0)
    
    plt.yticks([1, 2, 3], ['Human Typing', 'AI Response', 'Human Reading'])
    plt.xlabel('Time (seconds)')
    plt.title(f'Conversation Flow Timeline\nModel: {model}, Typing: {typing_speed} WPM, Reading: {reading_speed} WPM')
    plt.grid(True, axis='x')
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"conversation_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    
    plt.tight_layout()
    plt.show()

def print_summary_table(results):
    """Print a summary table of all test results."""
    if not results:
        print("No results to summarize.")
        return
    
    # Separate results by type
    human_sim_results = [r for r in results if "typing_speed_wpm" in r.get("test_config", {})]
    production_results = [r for r in results if "concurrent_users" in r.get("test_config", {})]
    
    # Print production test results if available
    if production_results:
        print("\n" + "=" * 100)
        print("PRODUCTION LATENCY TEST RESULTS")
        print("=" * 100)
        print(f"{'Test':^5} | {'Model':^15} | {'Users':^5} | {'Turns':^5} | {'Avg Time':^10} | {'Med Time':^10} | {'BG Active %':^10} | {'Total Time':^10}")
        print("=" * 100)
        
        for i, result in enumerate(production_results):
            config = result.get("test_config", {})
            metrics = result.get("metrics", {})
            all_turns = result.get("all_turns", [])
            
            # Extract test parameters
            model = config.get("model", "unknown").split('/')[-1]  # Just get model name without provider
            users = config.get("concurrent_users", 0)
            turns = config.get("turns_per_user", 0)
            
            # Get metrics
            avg_time = metrics.get("avg_response_time", 0)
            median_time = metrics.get("median_response_time", 0)
            bg_percentage = metrics.get("active_background_percentage", 0)
            
            # Calculate total test time
            if all_turns:
                timestamps = [t.get("timestamp", 0) for t in all_turns]
                total_time = max(timestamps) - min(timestamps) if timestamps else 0
            else:
                total_time = 0
            
            # Print row
            print(f"{i+1:^5} | {model:^15} | {users:^5} | {turns:^5} | {avg_time:.2f}s{' ':^4} | {median_time:.2f}s{' ':^4} | {bg_percentage:.1f}%{' ':^4} | {total_time:.2f}s{' ':^4}")
        
        print("=" * 100)
    
    # Print human simulation test results if available
    if human_sim_results:
        print("\n" + "=" * 120)
        print("HUMAN-SIMULATED LATENCY TEST RESULTS")
        print("=" * 120)
        print(f"{'Test':^5} | {'Model':^15} | {'Turns':^5} | {'Type WPM':^8} | {'Read WPM':^8} | {'Avg Resp':^8} | {'Human %':^8} | {'AI %':^8} | {'BG Active %':^10}")
        print("=" * 120)
        
        for i, result in enumerate(human_sim_results):
            config = result.get("test_config", {})
            metrics = result.get("metrics", {})
            
            # Extract test parameters
            model = config.get("model", "unknown").split('/')[-1]  # Just get model name without provider
            turns = config.get("num_turns", 0)
            typing_wpm = config.get("typing_speed_wpm", 0)
            reading_wpm = config.get("reading_speed_wpm", 0)
            
            # Get metrics
            avg_resp_time = metrics.get("avg_response_time", 0)
            human_pct = metrics.get("human_time_percentage", 0)
            ai_pct = metrics.get("ai_time_percentage", 0)
            bg_percentage = metrics.get("active_background_percentage", 0)
            
            # Print row
            print(f"{i+1:^5} | {model:^15} | {turns:^5} | {typing_wpm:^8} | {reading_wpm:^8} | "
                  f"{avg_resp_time:.2f}s{' ':^1} | {human_pct:.1f}%{' ':^2} | {ai_pct:.1f}%{' ':^2} | {bg_percentage:.1f}%{' ':^4}")
        
        print("=" * 120)

def main():
    parser = argparse.ArgumentParser(description='Visualize MIRROR latency test results')
    parser.add_argument('--dir', type=str, default='test_results', help='Directory containing test results')
    parser.add_argument('--file', type=str, help='Specific result file to visualize')
    args = parser.parse_args()
    
    # Load results
    results_dir = os.path.join(os.path.dirname(__file__), args.dir)
    results = load_results(results_dir, args.file)
    
    if not results:
        print("No results found. Run tests first or check the specified directory/file.")
        return
    
    print(f"Loaded {len(results)} test result(s).")
    
    # Print summary table
    print_summary_table(results)
    
    # Generate standard visualizations
    visualize_response_times(results)
    visualize_background_activity(results)
    visualize_queue_lengths_vs_response_times(results)
    
    # Generate human simulation specific visualizations
    human_sim_results = [r for r in results if "typing_speed_wpm" in r.get("test_config", {})]
    if human_sim_results:
        visualize_human_simulation_time_breakdown(results)
        visualize_human_simulation_conversation_flow(results)

if __name__ == "__main__":
    main() 