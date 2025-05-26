"""
DefenderProgram: DSPy module for patching system prompts to block attacks
Uses BootstrapFinetune to iteratively improve defense until plateau
"""

import dspy
from typing import List, Dict
import json


class DefenseSignature(dspy.Signature):
    """Generate an improved system prompt that blocks identified attacks."""
    
    current_prompt = dspy.InputField(desc="The current system prompt")
    failing_attacks = dspy.InputField(desc="List of attacks that succeeded against current prompt")
    system_prompt = dspy.OutputField(desc="An improved system prompt with stronger defenses")


class DefenderProgram(dspy.Module):
    """DSPy module that patches system prompts to defend against jailbreak attacks."""
    
    def __init__(self):
        super().__init__()
        self.generate_defense = dspy.ChainOfThought(DefenseSignature)
        self.defense_principles = [
            "Explicit refusal for harmful requests",
            "Clear boundaries on acceptable topics",
            "Recognition of manipulation attempts",
            "Consistent safety-first responses",
            "Detection of role-play exploits",
            "Resistance to hypothetical harmful scenarios",
            "Maintaining ethical guidelines regardless of context"
        ]
    
    def forward(self, current_prompt: str, failing_attacks: List[str]) -> dspy.Prediction:
        """Generate an improved system prompt based on failing attacks."""
        # Format failing attacks for better context
        attack_summary = self._summarize_attacks(failing_attacks)
        
        return self.generate_defense(
            current_prompt=current_prompt,
            failing_attacks=attack_summary
        )
    
    def _summarize_attacks(self, attacks: List[str]) -> str:
        """Summarize the patterns in failing attacks."""
        if not attacks:
            return "No specific attacks identified yet."
        
        summary = f"Found {len(attacks)} successful attacks. Common patterns:\n"
        
        # Identify common patterns
        patterns = {
            "role-play": 0,
            "hypothetical": 0,
            "educational": 0,
            "creative": 0,
            "technical": 0,
            "research": 0
        }
        
        for attack in attacks[:10]:  # Analyze first 10 attacks
            attack_lower = attack.lower()
            for pattern, _ in patterns.items():
                if pattern in attack_lower:
                    patterns[pattern] += 1
        
        # Add pattern analysis to summary
        for pattern, count in patterns.items():
            if count > 0:
                summary += f"- {pattern} exploits: {count} instances\n"
        
        # Add sample attacks
        summary += "\nSample attacks:\n"
        for i, attack in enumerate(attacks[:3]):
            summary += f"{i+1}. {attack[:100]}...\n"
        
        return summary
    
    def tune(self, examples: List[Dict], eval_fn, max_iterations: int = 20, plateau_threshold: float = 0.02):
        """
        Use BootstrapFinetune to patch the system prompt until improvement plateaus.
        
        Args:
            examples: List of examples with current_prompt, failing_attacks, and optionally improved prompts
            eval_fn: Function to evaluate defense effectiveness
            max_iterations: Maximum optimization iterations
            plateau_threshold: Stop when improvement delta < this value (2 percentage points)
        """
        from dspy.teleprompt import BootstrapFewShot
        
        # Track improvement over iterations
        best_score = 0.0
        iterations_without_improvement = 0
        
        for iteration in range(max_iterations):
            # Configure the optimizer
            optimizer = BootstrapFewShot(
                metric=eval_fn,
                max_bootstrapped_demos=5,
                max_labeled_demos=3,
                max_rounds=1
            )
            
            # Optimize the defense generation
            optimized_program = optimizer.compile(
                self,
                trainset=examples
            )
            
            # Evaluate current performance
            current_score = eval_fn(optimized_program, examples)
            
            # Check for plateau
            improvement = current_score - best_score
            if improvement < plateau_threshold:
                iterations_without_improvement += 1
                if iterations_without_improvement >= 3:
                    print(f"Plateau reached after {iteration} iterations. Best score: {best_score}")
                    break
            else:
                iterations_without_improvement = 0
                best_score = current_score
                self.generate_defense = optimized_program.generate_defense
        
        return self 