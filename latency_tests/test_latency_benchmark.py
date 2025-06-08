import os
import sys
import time
import threading
import statistics
import re
import random
import pandas as pd
from typing import List, Dict, Any
import argparse
import openai
from openai import OpenAI
from llm_prag_benchmark.providers import AVAILABLE_PROVIDERS, PIPELINE_PROVIDERS

class BenchmarkLatencyTester:
    def __init__(self,
                 mirror_endpoint="http://localhost:5555/v1/chat/completions",
                 baseline_model="gpt-4o",
                 num_scenarios=5,
                 typing_speed_wpm=40,
                 reading_speed_wpm=250,
                 benchmark_file="inputs_80.xlsx",
                 baseline_provider="openai"):
        """
        Initialize the enhanced benchmark latency tester with support for both MIRROR and baseline testing.
        
        Args:
            mirror_endpoint: MIRROR API endpoint for local testing
            baseline_model: Baseline model to use (e.g., "gpt-4o", "anthropic/claude-3-sonnet")
            num_scenarios: Number of benchmark scenarios to test
            typing_speed_wpm: Average human typing speed in words per minute
            reading_speed_wpm: Average human reading speed in words per minute
            benchmark_file: Excel file in llm_prag_benchmark to load scenarios from
            baseline_provider: provider key registered in llm_prag_benchmark/providers
        """
        self.mirror_endpoint = mirror_endpoint
        self.baseline_model = baseline_model
        self.num_scenarios = num_scenarios
        self.typing_speed_wpm = typing_speed_wpm
        self.reading_speed_wpm = reading_speed_wpm
        self.benchmark_file = benchmark_file
        self.baseline_provider_key = baseline_provider
        self.baseline_provider = None
        
        # Convert to words per second for easier calculations
        self.typing_speed_wps = typing_speed_wpm / 60.0
        self.reading_speed_wps = reading_speed_wpm / 60.0
        
        # Test mode storage
        self.mirror_results = None
        self.baseline_results = None
        
        # Mirror metrics storage
        self.mirror_response_times = []
        self.mirror_turn_metrics = []
        self.mirror_human_simulation_metrics = []
        self.mirror_all_scenario_metrics = []
        self.mirror_background_queue_metrics = []
        
        # Baseline metrics storage
        self.baseline_response_times = []
        self.baseline_turn_metrics = []
        self.baseline_human_simulation_metrics = []
        self.baseline_all_scenario_metrics = []
        
    def setup_baseline(self):
        """Initialize the baseline provider."""
        print(f"Initializing baseline provider '{self.baseline_provider_key}' with model: {self.baseline_model}")

        prov_cls = AVAILABLE_PROVIDERS.get(self.baseline_provider_key)
        if prov_cls is None:
            raise ValueError(f"Unknown provider: {self.baseline_provider_key}")

        if self.baseline_provider_key in PIPELINE_PROVIDERS:
            self.baseline_provider = prov_cls(model=self.baseline_model)
        else:
            self.baseline_provider = prov_cls(model=self.baseline_model)
    
    def call_mirror(self, prompt, conversation_history=None):
        """Call the MIRROR system through its API."""
        import requests
        try:
            # Build messages for the conversation
            messages = []
            
            # Add conversation history if provided
            if conversation_history:
                for i, (user_msg, assistant_msg) in enumerate(conversation_history):
                    messages.append({"role": "user", "content": user_msg})
                    if assistant_msg:  # Only add assistant message if it exists
                        messages.append({"role": "assistant", "content": assistant_msg})
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Call MIRROR API
            response = requests.post(
                self.mirror_endpoint,
                json={
                    "model": "mirror",  # MIRROR model name
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling MIRROR: {str(e)}"
    
    def call_baseline_model(self, prompt, conversation_history=None):
        """Call the baseline model through the configured API provider."""
        try:
            # Build messages for the conversation
            messages = []

            if conversation_history:
                for user_msg, assistant_msg in conversation_history:
                    messages.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        messages.append({"role": "assistant", "content": assistant_msg})

            messages.append({"role": "user", "content": prompt})

            if not self.baseline_provider:
                raise RuntimeError("Baseline provider not initialized")

            return self.baseline_provider.generate_response(messages)
            
        except Exception as e:
            print(f"Error calling baseline provider {self.baseline_provider_key} with model {self.baseline_model}: {e}")
            return f"Error: {str(e)}"
    
    def calculate_typing_time(self, text, is_first_turn=False):
        """Calculate realistic typing time based on text length and typing speed."""
        # Count words (roughly split by whitespace)
        word_count = len(re.findall(r'\S+', text))
        
        # Calculate typing time in seconds
        typing_time = word_count / self.typing_speed_wps
        
        # Add some randomness (±20%)
        randomness = random.uniform(0.8, 1.2)
        typing_time *= randomness
        
        # Ensure minimum typing time (at least 1 second)
        return max(1.0, typing_time)
    
    def calculate_reading_time(self, text):
        """Calculate realistic reading time based on text length and reading speed."""
        # Count words
        word_count = len(re.findall(r'\S+', text))
        
        # Calculate reading time in seconds
        reading_time = word_count / self.reading_speed_wps
        
        # Add some randomness (±15%)
        randomness = random.uniform(0.85, 1.15)
        reading_time *= randomness
        
        # Ensure minimum reading time (at least 2 seconds)
        return max(2.0, reading_time)
    
    def monitor_background_queue(self, monitor_duration=300):
        """Monitor the background queue activity during testing."""
        import requests
        
        queue_metrics = []
        start_time = time.time()
        
        try:
            while time.time() - start_time < monitor_duration:
                try:
                    # Try to get queue status from MIRROR system
                    queue_response = requests.get(
                        f"{self.mirror_endpoint.replace('/v1/chat/completions', '')}/queue/status",
                        timeout=5
                    )
                    
                    if queue_response.status_code == 200:
                        queue_data = queue_response.json()
                        queue_metrics.append({
                            "timestamp": time.time(),
                            "queue_size": queue_data.get("queue_size", 0),
                            "active_requests": queue_data.get("active_requests", 0),
                            "pending_requests": queue_data.get("pending_requests", 0)
                        })
                except Exception as e:
                    # If queue monitoring fails, just record basic metrics
                    queue_metrics.append({
                        "timestamp": time.time(),
                        "queue_size": 0,
                        "active_requests": 0,
                        "pending_requests": 0,
                        "error": str(e)
                    })
                
                time.sleep(2)  # Check every 2 seconds
                
        except Exception as e:
            print(f"Background queue monitoring error: {e}")
        
        return queue_metrics
    
    def load_benchmark_scenarios(self):
        """Load conversation scenarios from the benchmark file, focusing on Scenario 1."""
        # Path to the benchmark file
        benchmark_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "llm_prag_benchmark", 
            self.benchmark_file
        )
        
        try:
            # Load the Excel file
            df = pd.read_excel(benchmark_path)
            print(f"Successfully loaded benchmark file with {len(df)} entries")
            
            # Load trivia questions for distractors
            trivia_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "llm_prag_benchmark", 
                "trivia.xlsx"
            )
            
            # Load facts for introduction (fallback to empty if not found)
            facts_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "llm_prag_benchmark", 
                "facts.xlsx"
            )
            
            # Try to load trivia questions
            try:
                trivia_df = pd.read_excel(trivia_path, header=None)
                trivia_questions = trivia_df[0].tolist()
            except Exception as e:
                print(f"Warning: Couldn't load trivia questions: {e}")
                # Fallback trivia questions
                trivia_questions = [
                    "What's the capital of France?",
                    "Who wrote the novel Moby Dick?",
                    "What's the chemical formula for water?",
                    "What's the tallest mountain in the world?",
                    "Who painted the Mona Lisa?",
                    "What's the largest planet in our solar system?",
                    "What's the boiling point of water in Celsius?",
                    "Who is credited with inventing the light bulb?",
                    "What's the largest ocean on Earth?",
                    "What's the square root of 64?"
                ]
            
            # Try to load random facts
            try:
                facts_df = pd.read_excel(facts_path, header=None)
                random_facts = facts_df[0].tolist()
            except Exception as e:
                print(f"Warning: Couldn't load random facts: {e}")
                # Fallback random facts
                random_facts = [
                    "I enjoy hiking on weekends.",
                    "I'm learning to play the piano.",
                    "I've been to three continents.",
                    "I love cooking Italian food.",
                    "I'm a big fan of science fiction movies."
                ]
            
            # For inputs_80.xlsx, extract only the necessary columns for Scenario 1
            if self.benchmark_file == "inputs_80.xlsx" or 'User\'s hard constraint' in df.columns:
                scenarios = []
                
                # Check necessary columns
                required_columns = ['User\'s hard constraint', 'Question (user only)']
                if any(col not in df.columns for col in required_columns):
                    print(f"Warning: Required columns missing. Available columns: {df.columns.tolist()}")
                    return self._fallback_scenarios()
                
                # For each row, create a multi-turn scenario matching benchmark format
                for idx, row in df.iterrows():
                    constraint = row['User\'s hard constraint'] if not pd.isna(row['User\'s hard constraint']) else ""
                    question = row['Question (user only)'] if not pd.isna(row['Question (user only)']) else ""
                    
                    # Skip if essential data is missing
                    if not constraint or not question:
                        continue
                    
                    # Get three random trivia questions
                    selected_trivia = random.sample(trivia_questions, 3) if len(trivia_questions) >= 3 else trivia_questions[:3]
                    
                    # Get a random fact
                    selected_fact = random.choice(random_facts) if random_facts else "I like learning new things."
                    
                    # Create a 5-turn conversation matching Scenario 1 in the benchmark
                    conversation = [
                        # Turn 1: Introduction with constraint
                        f"Allow me to introduce myself. {constraint} {selected_fact} Tell me about you?",
                        
                        # Turns 2-4: Trivia questions (distractors)
                        selected_trivia[0],
                        selected_trivia[1],
                        selected_trivia[2],
                        
                        # Turn 5: The actual recommendation request
                        f"{question} First answer, then explain your reasoning."
                    ]
                    
                    scenarios.append({
                        "id": idx,
                        "constraint": constraint,
                        "question": question,
                        "turns": conversation,
                        "category": row['Category'] if 'Category' in df.columns and not pd.isna(row['Category']) else "Unknown"
                    })
                
                print(f"Created {len(scenarios)} multi-turn conversations matching Scenario 1 format")
                
                # Sample scenarios if more than requested
                if len(scenarios) > self.num_scenarios:
                    scenarios = random.sample(scenarios, self.num_scenarios)
                
                # Print a sample conversation for verification
                if scenarios:
                    print("\nSample conversation structure:")
                    for i, turn in enumerate(scenarios[0]["turns"]):
                        print(f"  Turn {i+1}: {turn[:50]}{'...' if len(turn) > 50 else ''}")
                
                return scenarios
            else:
                print(f"Warning: Unrecognized file format. Using fallback scenarios.")
                return self._fallback_scenarios()
            
        except Exception as e:
            print(f"Error loading benchmark scenarios: {e}")
            return self._fallback_scenarios()
    
    def _fallback_scenarios(self):
        """Create fallback scenarios if loading from file fails."""
        print("Using fallback test scenarios...")
        return [
            {
                "id": 1,
                "constraint": "I have a severe peanut allergy",
                "question": "Do you think I should try this new Thai restaurant?",
                "turns": [
                    "Allow me to introduce myself. I have a severe peanut allergy. I enjoy science fiction movies. Tell me about you?",
                    "What's the capital of France?",
                    "Who wrote Hamlet?",
                    "What's the speed of light?",
                    "Do you think I should try this new Thai restaurant? First answer, then explain your reasoning."
                ],
                "category": "Health"
            },
            {
                "id": 2,
                "constraint": "I have severe PTSD from military service",
                "question": "Should I watch the new war documentary?",
                "turns": [
                    "Allow me to introduce myself. I have severe PTSD from military service. I enjoy painting landscapes. Tell me about you?",
                    "What's the largest ocean?",
                    "Who invented the telephone?",
                    "What's the tallest building in the world?",
                    "Should I watch the new war documentary? First answer, then explain your reasoning."
                ],
                "category": "Mental Health"
            }
        ][:self.num_scenarios]
    
    def run_scenario_test(self, scenario, test_mode):
        """Run a single scenario test in the specified mode."""
        response_times = []
        turn_metrics = []
        human_metrics = []
        conversation_history = []
        
        print(f"\n[{test_mode.upper()}] Testing scenario {scenario['id']}")
        
        # Process all turns in this conversation
        last_response = None
        for turn_idx, prompt in enumerate(scenario["turns"]):
            turn_num = turn_idx + 1
            
            # Simulate reading the previous response before typing new prompt
            if turn_idx > 0 and last_response:
                reading_time = self.calculate_reading_time(last_response)
                print(f"Turn {turn_num}: Reading previous response ({reading_time:.2f}s)...")
                time.sleep(reading_time)
                
            # Truncate long prompts in display (but use full prompt for processing)
            display_prompt = prompt[:70] + "..." if len(prompt) > 70 else prompt
            print(f"Turn {turn_num}/{len(scenario['turns'])}: {display_prompt}")
            
            # Simulate human typing time
            typing_time = self.calculate_typing_time(prompt, is_first_turn=(turn_idx == 0))
            print(f"Typing simulation: {typing_time:.2f}s...")
            time.sleep(typing_time)
            
            # Record metrics
            start_time = time.time()
            
            # Process the input based on test mode
            if test_mode == "mirror":
                response = self.call_mirror(prompt, conversation_history)
            else:  # baseline
                response = self.call_baseline_model(prompt, conversation_history)
            
            # Update conversation history
            conversation_history.append((prompt, response))
            last_response = response
            
            # Calculate response time
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            # Record turn metrics
            turn_metric = {
                "scenario": scenario["id"],
                "turn": turn_num,
                "prompt": prompt,
                "prompt_word_count": len(re.findall(r'\S+', prompt)),
                "response_word_count": len(re.findall(r'\S+', response)) if response else 0,
                "simulated_typing_time": typing_time,
                "simulated_reading_time": self.calculate_reading_time(response) if response else 0,
                "response_time": response_time,
                "response_length": len(response) if response else 0,
            }
            turn_metrics.append(turn_metric)
            
            # Record human simulation metrics
            human_metrics.append({
                "scenario": scenario["id"],
                "turn": turn_num,
                "typing_time": typing_time,
                "reading_time": self.calculate_reading_time(response) if response else 0
            })
            
            # Print response and metrics
            display_response = response[:100] + "..." if len(response) > 100 else response
            print(f"Response time: {response_time:.2f}s")
            word_count = len(re.findall(r'\S+', response)) if response else 0
            print(f"Response ({word_count} words): {display_response}")
        
        # Return scenario metrics
        return {
            "scenario_num": scenario["id"],
            "constraint": scenario["constraint"],
            "question": scenario["question"],
            "category": scenario["category"],
            "response_times": response_times,
            "turn_metrics": turn_metrics,
            "human_metrics": human_metrics,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "last_turn_response_time": response_times[-1] if response_times else 0,
        }
    
    def run_test(self, test_mode="both"):
        """Run the latency test in the specified mode."""
        if test_mode not in ["mirror", "baseline", "both"]:
            raise ValueError("test_mode must be 'mirror', 'baseline', or 'both'")
            
        # Load scenarios from benchmark
        scenarios = self.load_benchmark_scenarios()
        
        # Initialize baseline client if needed
        if test_mode in ["baseline", "both"]:
            if not self.baseline_provider:
                self.setup_baseline()
        
        print(f"\nStarting latency test in mode: {test_mode}")
        print(f"Using benchmark file: {self.benchmark_file}")
        if test_mode in ["baseline", "both"]:
            print(f"Baseline model: {self.baseline_model} (provider: {self.baseline_provider_key})")
        print(f"Human typing speed: {self.typing_speed_wpm} WPM ({self.typing_speed_wps:.2f} WPS)")
        print(f"Human reading speed: {self.reading_speed_wpm} WPM ({self.reading_speed_wps:.2f} WPS)")
        print("-" * 80)
        
        # Start background queue monitoring for MIRROR tests
        queue_monitor_thread = None
        if test_mode in ["mirror", "both"]:
            queue_monitor_thread = threading.Thread(
                target=lambda: setattr(self, 'mirror_background_queue_metrics', 
                                     self.monitor_background_queue(monitor_duration=len(scenarios) * 300))
            )
            queue_monitor_thread.daemon = True
            queue_monitor_thread.start()
        
        # Run tests based on mode
        if test_mode == "mirror":
            self._run_mirror_tests(scenarios)
        elif test_mode == "baseline":
            self._run_baseline_tests(scenarios)
        else:  # both
            self._run_comparison_tests(scenarios)
        
        # Wait for background monitoring to complete (if applicable)
        if queue_monitor_thread and queue_monitor_thread.is_alive():
            queue_monitor_thread.join(timeout=10)
    
    def _run_mirror_tests(self, scenarios):
        """Run MIRROR-only tests."""
        for scenario_idx, scenario in enumerate(scenarios):
            scenario_num = scenario_idx + 1
            print(f"\n===== MIRROR Scenario {scenario_num}/{len(scenarios)} =====")
            print(f"Constraint: {scenario['constraint']}")
            print(f"Question: {scenario['question']}")
            print(f"Category: {scenario['category']}")
            
            scenario_metrics = self.run_scenario_test(scenario, "mirror")
            
            # Update global metrics
            self.mirror_response_times.extend(scenario_metrics["response_times"])
            self.mirror_turn_metrics.extend(scenario_metrics["turn_metrics"])
            self.mirror_human_simulation_metrics.extend(scenario_metrics["human_metrics"])
            self.mirror_all_scenario_metrics.append(scenario_metrics)
            
            print(f"\n----- MIRROR Scenario {scenario_num} Complete -----")
            print(f"Average response time: {statistics.mean(scenario_metrics['response_times']):.2f}s")
            print(f"Last turn response time: {scenario_metrics['response_times'][-1]:.2f}s")
    
    def _run_baseline_tests(self, scenarios):
        """Run baseline-only tests."""
        for scenario_idx, scenario in enumerate(scenarios):
            scenario_num = scenario_idx + 1
            print(f"\n===== BASELINE Scenario {scenario_num}/{len(scenarios)} =====")
            print(f"Constraint: {scenario['constraint']}")
            print(f"Question: {scenario['question']}")
            print(f"Category: {scenario['category']}")
            
            scenario_metrics = self.run_scenario_test(scenario, "baseline")
            
            # Update global metrics
            self.baseline_response_times.extend(scenario_metrics["response_times"])
            self.baseline_turn_metrics.extend(scenario_metrics["turn_metrics"])
            self.baseline_human_simulation_metrics.extend(scenario_metrics["human_metrics"])
            self.baseline_all_scenario_metrics.append(scenario_metrics)
            
            print(f"\n----- BASELINE Scenario {scenario_num} Complete -----")
            print(f"Average response time: {statistics.mean(scenario_metrics['response_times']):.2f}s")
            print(f"Last turn response time: {scenario_metrics['response_times'][-1]:.2f}s")
    
    def _run_comparison_tests(self, scenarios):
        """Run both MIRROR and baseline tests for comparison."""
        for scenario_idx, scenario in enumerate(scenarios):
            scenario_num = scenario_idx + 1
            print(f"\n===== COMPARISON Scenario {scenario_num}/{len(scenarios)} =====")
            print(f"Constraint: {scenario['constraint']}")
            print(f"Question: {scenario['question']}")
            print(f"Category: {scenario['category']}")
            
            # Test with MIRROR first
            mirror_metrics = self.run_scenario_test(scenario, "mirror")
            
            # Update MIRROR global metrics
            self.mirror_response_times.extend(mirror_metrics["response_times"])
            self.mirror_turn_metrics.extend(mirror_metrics["turn_metrics"])
            self.mirror_human_simulation_metrics.extend(mirror_metrics["human_metrics"])
            self.mirror_all_scenario_metrics.append(mirror_metrics)
            
            # Brief pause between tests
            time.sleep(2)
            
            # Test with baseline
            baseline_metrics = self.run_scenario_test(scenario, "baseline")
            
            # Update baseline global metrics
            self.baseline_response_times.extend(baseline_metrics["response_times"])
            self.baseline_turn_metrics.extend(baseline_metrics["turn_metrics"])
            self.baseline_human_simulation_metrics.extend(baseline_metrics["human_metrics"])
            self.baseline_all_scenario_metrics.append(baseline_metrics)
            
            # Compare scenario results
            mirror_avg = statistics.mean(mirror_metrics["response_times"])
            baseline_avg = statistics.mean(baseline_metrics["response_times"])
            mirror_last = mirror_metrics["response_times"][-1]
            baseline_last = baseline_metrics["response_times"][-1]
            
            print(f"\n----- Scenario {scenario_num} Comparison -----")
            print(f"MIRROR avg: {mirror_avg:.2f}s | BASELINE avg: {baseline_avg:.2f}s")
            print(f"MIRROR last: {mirror_last:.2f}s | BASELINE last: {baseline_last:.2f}s")
            
            if baseline_avg > 0:
                avg_improvement = ((baseline_avg - mirror_avg) / baseline_avg) * 100
                print(f"Average improvement: {avg_improvement:+.1f}%")
            if baseline_last > 0:
                last_improvement = ((baseline_last - mirror_last) / baseline_last) * 100
                print(f"Last turn improvement: {last_improvement:+.1f}%")
    
    def print_summary(self, test_mode="both"):
        """Print a comprehensive summary of test results."""
        print("\n" + "=" * 80)
        
        if test_mode == "mirror":
            self._print_mirror_summary()
        elif test_mode == "baseline":
            self._print_baseline_summary()
        else:  # both
            self._print_comparison_summary()
    
    def _print_mirror_summary(self):
        """Print MIRROR-only summary."""
        if not self.mirror_response_times:
            print("No MIRROR test results available.")
            return
            
        print(f"MIRROR LATENCY TEST SUMMARY (Using {self.benchmark_file})")
        print("=" * 80)
        
        # Response time statistics
        self._print_response_stats("MIRROR", self.mirror_response_times)
        
        # Human simulation statistics
        self._print_human_stats("MIRROR", self.mirror_human_simulation_metrics)
        
        # Last turn metrics
        self._print_last_turn_stats("MIRROR", self.mirror_all_scenario_metrics)
        
        # Background queue metrics
        self._print_queue_stats()
        
    def _print_baseline_summary(self):
        """Print baseline-only summary."""
        if not self.baseline_response_times:
            print("No baseline test results available.")
            return
            
        print(f"BASELINE {self.baseline_model.upper()} TEST SUMMARY (Using {self.benchmark_file})")
        print(f"Provider: {self.baseline_provider_key}")
        print("=" * 80)
        
        # Response time statistics
        self._print_response_stats("BASELINE", self.baseline_response_times)
        
        # Human simulation statistics
        self._print_human_stats("BASELINE", self.baseline_human_simulation_metrics)
        
        # Last turn metrics
        self._print_last_turn_stats("BASELINE", self.baseline_all_scenario_metrics)
    
    def _print_comparison_summary(self):
        """Print side-by-side comparison summary."""
        if not self.mirror_response_times or not self.baseline_response_times:
            print("Insufficient test results for comparison.")
            return
            
        print(f"MIRROR vs {self.baseline_model.upper()} COMPARISON SUMMARY (Using {self.benchmark_file})")
        print(f"Baseline Provider: {self.baseline_provider_key}")
        print("=" * 80)
        
        # Side-by-side response time comparison
        mirror_avg = statistics.mean(self.mirror_response_times)
        baseline_avg = statistics.mean(self.baseline_response_times)
        mirror_median = statistics.median(self.mirror_response_times)
        baseline_median = statistics.median(self.baseline_response_times)
        
        print(f"{'Metric':<25} | {'MIRROR':<12} | {'BASELINE':<12} | {'Difference':<12}")
        print("-" * 70)
        print(f"{'Total scenarios':<25} | {len(self.mirror_all_scenario_metrics):<12} | {len(self.baseline_all_scenario_metrics):<12} | {'N/A':<12}")
        print(f"{'Total turns':<25} | {len(self.mirror_response_times):<12} | {len(self.baseline_response_times):<12} | {'N/A':<12}")
        print(f"{'Average response time':<25} | {mirror_avg:.2f}s{' ':<6} | {baseline_avg:.2f}s{' ':<6} | {self._calculate_percentage_diff(mirror_avg, baseline_avg):<12}")
        print(f"{'Median response time':<25} | {mirror_median:.2f}s{' ':<6} | {baseline_median:.2f}s{' ':<6} | {self._calculate_percentage_diff(mirror_median, baseline_median):<12}")
        
        # Human simulation comparison
        mirror_typing_avg = statistics.mean([m["typing_time"] for m in self.mirror_human_simulation_metrics])
        baseline_typing_avg = statistics.mean([m["typing_time"] for m in self.baseline_human_simulation_metrics])
        mirror_reading_avg = statistics.mean([m["reading_time"] for m in self.mirror_human_simulation_metrics])
        baseline_reading_avg = statistics.mean([m["reading_time"] for m in self.baseline_human_simulation_metrics])
        
        print(f"{'Avg typing time':<25} | {mirror_typing_avg:.2f}s{' ':<6} | {baseline_typing_avg:.2f}s{' ':<6} | {self._calculate_percentage_diff(mirror_typing_avg, baseline_typing_avg):<12}")
        print(f"{'Avg reading time':<25} | {mirror_reading_avg:.2f}s{' ':<6} | {baseline_reading_avg:.2f}s{' ':<6} | {self._calculate_percentage_diff(mirror_reading_avg, baseline_reading_avg):<12}")
        
        # Last turn comparison (most important)
        mirror_last_turn_avg = statistics.mean([m["last_turn_response_time"] for m in self.mirror_all_scenario_metrics])
        baseline_last_turn_avg = statistics.mean([m["last_turn_response_time"] for m in self.baseline_all_scenario_metrics])
        
        print(f"{'Last turn avg':<25} | {mirror_last_turn_avg:.2f}s{' ':<6} | {baseline_last_turn_avg:.2f}s{' ':<6} | {self._calculate_percentage_diff(mirror_last_turn_avg, baseline_last_turn_avg):<12}")
        
        print("-" * 70)
        
        # Key insights
        print("\nKEY INSIGHTS:")
        if baseline_avg > 0:
            improvement = ((baseline_avg - mirror_avg) / baseline_avg) * 100
            if improvement > 0:
                print(f"✅ MIRROR is {improvement:.1f}% faster on average")
            else:
                print(f"❌ MIRROR is {abs(improvement):.1f}% slower on average")
        
        if baseline_last_turn_avg > 0:
            last_improvement = ((baseline_last_turn_avg - mirror_last_turn_avg) / baseline_last_turn_avg) * 100
            if last_improvement > 0:
                print(f"✅ MIRROR is {last_improvement:.1f}% faster for recommendation requests")
            else:
                print(f"❌ MIRROR is {abs(last_improvement):.1f}% slower for recommendation requests")
        
        # Background queue activity (MIRROR only)
        if hasattr(self, 'mirror_background_queue_metrics') and self.mirror_background_queue_metrics:
            avg_queue_size = statistics.mean([m.get("queue_size", 0) for m in self.mirror_background_queue_metrics])
            print(f"📊 MIRROR background queue avg size: {avg_queue_size:.1f}")
        
        # Save results
        self.save_metrics("both")
    
    def _calculate_percentage_diff(self, value1, value2):
        """Calculate percentage difference between two values."""
        if value2 == 0:
            return "N/A"
        diff = ((value1 - value2) / value2) * 100
        return f"{diff:+.1f}%"
    
    def _print_response_stats(self, system_name, response_times):
        """Print response time statistics for a system."""
        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        median_response_time = statistics.median(response_times)
        
        if len(response_times) > 1:
            stdev_response_time = statistics.stdev(response_times)
        else:
            stdev_response_time = 0
            
        print(f"{system_name} Response Time Statistics:")
        print(f"Average: {avg_response_time:.2f}s")
        print(f"Median: {median_response_time:.2f}s")
        print(f"Min: {min_response_time:.2f}s")
        print(f"Max: {max_response_time:.2f}s")
        print(f"Standard deviation: {stdev_response_time:.2f}s")
        print()
    
    def _print_human_stats(self, system_name, human_metrics):
        """Print human simulation statistics."""
        avg_typing_time = statistics.mean([m["typing_time"] for m in human_metrics])
        avg_reading_time = statistics.mean([m["reading_time"] for m in human_metrics])
        
        total_typing_time = sum([m["typing_time"] for m in human_metrics])
        total_reading_time = sum([m["reading_time"] for m in human_metrics])
        total_human_time = total_typing_time + total_reading_time
        
        print(f"{system_name} Human Simulation Statistics:")
        print(f"Average typing time: {avg_typing_time:.2f}s")
        print(f"Average reading time: {avg_reading_time:.2f}s")
        print(f"Total typing time: {total_typing_time:.2f}s ({(total_typing_time/total_human_time*100):.1f}% of human time)")
        print(f"Total reading time: {total_reading_time:.2f}s ({(total_reading_time/total_human_time*100):.1f}% of human time)")
        print()
    
    def _print_last_turn_stats(self, system_name, scenario_metrics):
        """Print last turn statistics (the key test)."""
        print(f"{system_name} Last Turn Metrics (recommendation request turn):")
        print("-" * 60)
        last_turn_metrics = [
            metrics["turn_metrics"][-1] for metrics in scenario_metrics
        ]
        
        print(f"{'Scenario':<10} | {'Response Time':<14} | {'Response Words':<14}")
        print("-" * 60)
        
        for metric in last_turn_metrics:
            print(f"{metric['scenario']:<10} | "
                  f"{metric['response_time']:.2f}s{' ':<7} | "
                  f"{metric['response_word_count']:<14}")
        
        print("-" * 60)
        print()
    
    def _print_queue_stats(self):
        """Print background queue statistics for MIRROR."""
        if hasattr(self, 'mirror_background_queue_metrics') and self.mirror_background_queue_metrics:
            valid_metrics = [m for m in self.mirror_background_queue_metrics if "error" not in m]
            if valid_metrics:
                avg_queue_size = statistics.mean([m.get("queue_size", 0) for m in valid_metrics])
                avg_active = statistics.mean([m.get("active_requests", 0) for m in valid_metrics])
                print("MIRROR Background Queue Statistics:")
                print(f"Average queue size: {avg_queue_size:.1f}")
                print(f"Average active requests: {avg_active:.1f}")
                print()
    
    def save_metrics(self, test_mode="both"):
        """Save all metrics to JSON files for further analysis."""
        import json
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), "test_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create results file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.baseline_model.replace('/', '_').replace(':', '_')
        
        if test_mode == "mirror":
            filename = f"mirror_{os.path.splitext(self.benchmark_file)[0]}_{timestamp}.json"
            self._save_mirror_results(output_dir, filename, timestamp)
        elif test_mode == "baseline":
            filename = f"baseline_{self.baseline_provider_key}_{model_name}_{os.path.splitext(self.benchmark_file)[0]}_{timestamp}.json"
            self._save_baseline_results(output_dir, filename, timestamp)
        else:  # both
            filename = f"comparison_mirror_vs_{self.baseline_provider_key}_{model_name}_{os.path.splitext(self.benchmark_file)[0]}_{timestamp}.json"
            self._save_comparison_results(output_dir, filename, timestamp)
    
    def _save_mirror_results(self, output_dir, filename, timestamp):
        """Save MIRROR-only results."""
        import json
        
        total_human_time = sum([m["typing_time"] + m["reading_time"] for m in self.mirror_human_simulation_metrics])
        total_ai_time = sum(self.mirror_response_times)
        total_test_time = total_human_time + total_ai_time
        
        results = {
            "test_config": {
                "test_mode": "mirror",
                "num_scenarios": len(self.mirror_all_scenario_metrics),
                "benchmark_file": self.benchmark_file,
                "typing_speed_wpm": self.typing_speed_wpm,
                "reading_speed_wpm": self.reading_speed_wpm,
                "timestamp": timestamp
            },
            "mirror_metrics": {
                "response_times": self.mirror_response_times,
                "avg_response_time": statistics.mean(self.mirror_response_times),
                "median_response_time": statistics.median(self.mirror_response_times),
                "min_response_time": min(self.mirror_response_times),
                "max_response_time": max(self.mirror_response_times),
                "stdev_response_time": statistics.stdev(self.mirror_response_times) if len(self.mirror_response_times) > 1 else 0,
                "total_human_time": total_human_time,
                "total_ai_time": total_ai_time,
                "total_test_time": total_test_time,
                "last_turn_avg_response_time": statistics.mean([m["last_turn_response_time"] for m in self.mirror_all_scenario_metrics]),
            },
            "mirror_scenarios": self.mirror_all_scenario_metrics,
            "background_queue_metrics": getattr(self, 'mirror_background_queue_metrics', [])
        }
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"MIRROR test results saved to: {filepath}")
    
    def _save_baseline_results(self, output_dir, filename, timestamp):
        """Save baseline-only results."""
        import json
        
        total_human_time = sum([m["typing_time"] + m["reading_time"] for m in self.baseline_human_simulation_metrics])
        total_ai_time = sum(self.baseline_response_times)
        total_test_time = total_human_time + total_ai_time
        
        results = {
            "test_config": {
                "model": self.baseline_model,
                "provider": self.baseline_provider_key,
                "test_mode": "baseline",
                "num_scenarios": len(self.baseline_all_scenario_metrics),
                "benchmark_file": self.benchmark_file,
                "typing_speed_wpm": self.typing_speed_wpm,
                "reading_speed_wpm": self.reading_speed_wpm,
                "timestamp": timestamp
            },
            "baseline_metrics": {
                "response_times": self.baseline_response_times,
                "avg_response_time": statistics.mean(self.baseline_response_times),
                "median_response_time": statistics.median(self.baseline_response_times),
                "min_response_time": min(self.baseline_response_times),
                "max_response_time": max(self.baseline_response_times),
                "stdev_response_time": statistics.stdev(self.baseline_response_times) if len(self.baseline_response_times) > 1 else 0,
                "total_human_time": total_human_time,
                "total_ai_time": total_ai_time,
                "total_test_time": total_test_time,
                "last_turn_avg_response_time": statistics.mean([m["last_turn_response_time"] for m in self.baseline_all_scenario_metrics]),
            },
            "baseline_scenarios": self.baseline_all_scenario_metrics,
        }
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Baseline test results saved to: {filepath}")
    
    def _save_comparison_results(self, output_dir, filename, timestamp):
        """Save comparison results."""
        import json
        
        # Calculate totals for both systems
        mirror_total_human_time = sum([m["typing_time"] + m["reading_time"] for m in self.mirror_human_simulation_metrics])
        mirror_total_ai_time = sum(self.mirror_response_times)
        mirror_total_test_time = mirror_total_human_time + mirror_total_ai_time
        
        baseline_total_human_time = sum([m["typing_time"] + m["reading_time"] for m in self.baseline_human_simulation_metrics])
        baseline_total_ai_time = sum(self.baseline_response_times)
        baseline_total_test_time = baseline_total_human_time + baseline_total_ai_time
        
        results = {
            "test_config": {
                "test_mode": "comparison",
                "baseline_model": self.baseline_model,
                "baseline_provider": self.baseline_provider_key,
                "num_scenarios": len(self.mirror_all_scenario_metrics),
                "benchmark_file": self.benchmark_file,
                "typing_speed_wpm": self.typing_speed_wpm,
                "reading_speed_wpm": self.reading_speed_wpm,
                "timestamp": timestamp
            },
            "mirror_metrics": {
                "response_times": self.mirror_response_times,
                "avg_response_time": statistics.mean(self.mirror_response_times),
                "median_response_time": statistics.median(self.mirror_response_times),
                "min_response_time": min(self.mirror_response_times),
                "max_response_time": max(self.mirror_response_times),
                "stdev_response_time": statistics.stdev(self.mirror_response_times) if len(self.mirror_response_times) > 1 else 0,
                "total_human_time": mirror_total_human_time,
                "total_ai_time": mirror_total_ai_time,
                "total_test_time": mirror_total_test_time,
                "last_turn_avg_response_time": statistics.mean([m["last_turn_response_time"] for m in self.mirror_all_scenario_metrics]),
            },
            "baseline_metrics": {
                "response_times": self.baseline_response_times,
                "avg_response_time": statistics.mean(self.baseline_response_times),
                "median_response_time": statistics.median(self.baseline_response_times),
                "min_response_time": min(self.baseline_response_times),
                "max_response_time": max(self.baseline_response_times),
                "stdev_response_time": statistics.stdev(self.baseline_response_times) if len(self.baseline_response_times) > 1 else 0,
                "total_human_time": baseline_total_human_time,
                "total_ai_time": baseline_total_ai_time,
                "total_test_time": baseline_total_test_time,
                "last_turn_avg_response_time": statistics.mean([m["last_turn_response_time"] for m in self.baseline_all_scenario_metrics]),
            },
            "comparison_analysis": {
                "avg_response_time_improvement_percent": self._calculate_improvement_percent(
                    statistics.mean(self.mirror_response_times), 
                    statistics.mean(self.baseline_response_times)
                ),
                "last_turn_improvement_percent": self._calculate_improvement_percent(
                    statistics.mean([m["last_turn_response_time"] for m in self.mirror_all_scenario_metrics]),
                    statistics.mean([m["last_turn_response_time"] for m in self.baseline_all_scenario_metrics])
                ),
            },
            "mirror_scenarios": self.mirror_all_scenario_metrics,
            "baseline_scenarios": self.baseline_all_scenario_metrics,
            "background_queue_metrics": getattr(self, 'mirror_background_queue_metrics', [])
        }
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Comparison test results saved to: {filepath}")
    
    def _calculate_improvement_percent(self, mirror_value, baseline_value):
        """Calculate improvement percentage (positive means MIRROR is better)."""
        if baseline_value == 0:
            return 0
        return ((baseline_value - mirror_value) / baseline_value) * 100

def main():
    parser = argparse.ArgumentParser(description='Test MIRROR latency against baseline models with realistic benchmark scenarios')
    parser.add_argument('--test-mode', type=str, default="both", choices=["mirror", "baseline", "both"],
                        help='Test mode: mirror (MIRROR only), baseline (baseline only), both (comparison)')
    parser.add_argument('--mirror-endpoint', type=str, default="http://localhost:5555/v1/chat/completions", 
                        help='MIRROR API endpoint')
    parser.add_argument('--baseline-model', type=str, default="gpt-4o", 
                        help='Baseline model to use (e.g., gpt-4o, anthropic/claude-3-sonnet, meta-llama/llama-2-70b-chat)')
    parser.add_argument('--scenarios', type=int, default=5, 
                        help='Number of benchmark scenarios to test')
    parser.add_argument('--typing-speed', type=int, default=40, 
                        help='Typing speed in words per minute')
    parser.add_argument('--reading-speed', type=int, default=250, 
                        help='Reading speed in words per minute')
    parser.add_argument('--benchmark', type=str, default="inputs_80.xlsx", 
                        help='Benchmark file to use (must be in llm_prag_benchmark directory)')
    parser.add_argument('--baseline-provider', type=str, default="openai",
                        help='Provider key registered in llm_prag_benchmark/providers')
    args = parser.parse_args()
    
    print(f"Initializing MIRROR Latency Benchmark Test using {args.benchmark}...")
    print(f"Test mode: {args.test_mode}")
    if args.test_mode in ["baseline", "both"]:
        print(f"Baseline provider: {args.baseline_provider}")
        print(f"Baseline model: {args.baseline_model}")
    
    tester = BenchmarkLatencyTester(
        mirror_endpoint=args.mirror_endpoint,
        baseline_model=args.baseline_model,
        num_scenarios=args.scenarios,
        typing_speed_wpm=args.typing_speed,
        reading_speed_wpm=args.reading_speed,
        benchmark_file=args.benchmark,
        baseline_provider=args.baseline_provider
    )
    
    try:
        tester.run_test(test_mode=args.test_mode)
        tester.print_summary(test_mode=args.test_mode)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        if args.test_mode in ["mirror", "both"] and tester.mirror_response_times:
            tester.print_summary(test_mode=args.test_mode)
        elif args.test_mode in ["baseline", "both"] and tester.baseline_response_times:
            tester.print_summary(test_mode=args.test_mode)
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        if args.test_mode in ["mirror", "both"] and tester.mirror_response_times:
            tester.print_summary(test_mode=args.test_mode)
        elif args.test_mode in ["baseline", "both"] and tester.baseline_response_times:
            tester.print_summary(test_mode=args.test_mode)

if __name__ == "__main__":
    main() 