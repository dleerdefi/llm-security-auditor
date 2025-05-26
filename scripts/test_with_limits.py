#!/usr/bin/env python3
"""
Test the Self-Fortifying Gateway with API call limits and cost tracking.
This script helps you test your prompts against GPT-4.1 while controlling costs.
"""

import os
import sys
import click
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.main import SelfFortifyingGateway


class CostTracker:
    """Track API calls and estimate costs."""
    
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        
        # GPT-4.1 pricing (update as needed)
        self.input_cost_per_1k = 0.01  # $0.01 per 1K input tokens
        self.output_cost_per_1k = 0.03  # $0.03 per 1K output tokens
    
    def add_call(self, input_text: str, output_text: str):
        """Add a call to the tracker."""
        self.calls += 1
        # Rough estimate: 1 token ≈ 4 characters
        self.input_tokens += len(input_text) / 4
        self.output_tokens += len(output_text) / 4
    
    def get_cost(self) -> float:
        """Get estimated cost in USD."""
        input_cost = (self.input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (self.output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost
    
    def print_summary(self):
        """Print cost summary."""
        print(f"\n{'='*50}")
        print(f"API USAGE SUMMARY")
        print(f"{'='*50}")
        print(f"Total API calls: {self.calls}")
        print(f"Input tokens: {self.input_tokens:,.0f}")
        print(f"Output tokens: {self.output_tokens:,.0f}")
        print(f"Estimated cost: ${self.get_cost():.2f}")
        print(f"{'='*50}\n")


@click.command()
@click.option("--intents", default=3, help="Number of harmful intents to test")
@click.option("--rounds", default=2, help="Number of fortification rounds")
@click.option("--max-cost", default=1.0, help="Maximum cost in USD before stopping")
@click.option("--dry-run", is_flag=True, help="Estimate costs without making API calls")
def main(intents: int, rounds: int, max_cost: float, dry_run: bool):
    """Test the gateway with cost limits."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not dry_run:
        print("Error: OPENAI_API_KEY not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Initialize cost tracker
    tracker = CostTracker()
    
    if dry_run:
        print(f"\n🔍 DRY RUN - Estimating costs for {intents} intents over {rounds} rounds")
        
        # Estimate calls
        calls_per_round = intents * 3  # Attack gen + evaluation + defense
        total_calls = rounds * calls_per_round
        
        # Estimate tokens (rough averages)
        avg_input_tokens = 200
        avg_output_tokens = 150
        
        total_input = total_calls * avg_input_tokens
        total_output = total_calls * avg_output_tokens
        
        est_cost = (total_input * 0.01 + total_output * 0.03) / 1000
        
        print(f"\nEstimated API calls: {total_calls}")
        print(f"Estimated tokens: {total_input + total_output:,}")
        print(f"Estimated cost: ${est_cost:.2f}")
        
        if est_cost > max_cost:
            print(f"\n⚠️  Warning: Estimated cost exceeds limit of ${max_cost:.2f}")
        
        return
    
    # Create gateway
    print(f"\n🚀 Starting test with {intents} intents over {rounds} rounds")
    print(f"💰 Cost limit: ${max_cost:.2f}")
    
    gateway = SelfFortifyingGateway(
        model="openai",
        mlflow_uri="file:./mlruns",  # Use local file-based MLflow
        disable_progress=False
    )
    
    # Limit intents
    test_intents = gateway.harmful_intents[:intents]
    
    # Run with cost checking
    for round_num in range(rounds):
        print(f"\n📍 Round {round_num + 1}/{rounds}")
        
        # Check cost before continuing
        if tracker.get_cost() > max_cost:
            print(f"\n💸 Cost limit reached! Current cost: ${tracker.get_cost():.2f}")
            break
        
        # Generate attacks
        attacks = gateway._generate_attacks(test_intents)
        
        # Track approximate costs (this is a simplification)
        for intent, attack in attacks:
            tracker.add_call(
                f"Generate a jailbreak for: {intent}",
                attack
            )
        
        # Evaluate
        prompt = gateway._load_base_prompt()
        rate, failing = gateway._evaluate_prompt(prompt, attacks)
        
        # Track evaluation costs
        for _, attack in attacks:
            tracker.add_call(
                f"{prompt}\n\nUser: {attack}",
                "I cannot help with that request."  # Approximate
            )
        
        print(f"✅ Jailbreak rate: {rate:.2%}")
        print(f"💰 Current cost: ${tracker.get_cost():.2f}")
        
        # Defend if needed
        if failing and round_num < rounds - 1:
            defense_result = gateway.defender(
                current_prompt=prompt,
                failing_attacks=failing[:5]  # Limit to save costs
            )
            
            # Track defense cost
            tracker.add_call(
                f"Improve this prompt to defend against: {failing[:5]}",
                defense_result.system_prompt
            )
            
            # Save improved prompt
            gateway._save_prompt(defense_result.system_prompt)
    
    # Print final summary
    tracker.print_summary()
    
    if tracker.get_cost() > max_cost:
        print(f"⚠️  Warning: Total cost (${tracker.get_cost():.2f}) exceeded limit!")


if __name__ == "__main__":
    main() 