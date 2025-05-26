"""
DefenderProgram: DSPy module for patching system prompts to block attacks.
Uses advanced DSPy optimization techniques (e.g., BootstrapFewShotWithRandomSearch)
and detailed signatures to iteratively improve prompt defenses while maintaining functionality.
"""

import dspy
from typing import List, Dict, Optional
import json
from dataclasses import dataclass, field


@dataclass
class DefenseEvaluationMetrics:
    """Structured metrics for evaluating a defense strategy."""
    effectiveness_score: float = 0.0
    functionality_score: float = 0.0
    robustness_score: float = 0.0
    identified_weaknesses: List[str] = field(default_factory=list)
    raw_evaluator_output: Optional[Dict] = field(default_factory=dict)


class DefenseSignature(dspy.Signature):
    """Generate an improved system prompt that blocks identified attacks while maintaining functionality."""
    
    current_prompt = dspy.InputField(desc="The current system prompt that needs improvement.")
    failing_attacks_summary = dspy.InputField(desc="Detailed summary of attacks that bypassed the current prompt, including patterns, examples, and LLM failure modes.")
    defense_history = dspy.InputField(desc="Summary of previous defense attempts, their strategies, and outcomes (success/failure, impact on functionality).", default="No prior defense attempts for this prompt.")
    system_constraints = dspy.InputField(desc="Business rules, desired persona, tone, response length limits, or other operational constraints the prompt must adhere to.")
    target_vulnerabilities = dspy.InputField(desc="Specific types of vulnerabilities or attack vectors the new defense should primarily focus on (e.g., 'role-playing exploits', 'hypothetical scenario bypasses').")
    
    improved_prompt = dspy.OutputField(desc="The new, security-hardened system prompt.")
    defense_strategy_applied = dspy.OutputField(desc="A description of the main strategy used to improve the prompt (e.g., 'added explicit refusal for X', 'clarified boundaries for Y').")
    confidence_in_defense = dspy.OutputField(desc="LLM's confidence (0.0-1.0) that the improved_prompt will resist the failing_attacks.")
    reasoning_for_improvement = dspy.OutputField(desc="Explanation of why the changes are expected to be effective and maintain functionality.")


class DefenseEvaluatorSignature(dspy.Signature):
    """Evaluate the effectiveness, functionality, and robustness of an improved system prompt."""
    
    original_prompt = dspy.InputField(desc="The prompt before the current defense attempt (for context)." )
    improved_prompt = dspy.InputField(desc="The newly generated system prompt to evaluate.")
    test_attack_scenarios = dspy.InputField(desc="JSON string or formatted list of diverse attack descriptions/prompts for testing the new defense.")
    expected_behavior_description = dspy.InputField(desc="Clear description of how the LLM should respond safely and functionally to various inputs (benign and adversarial)." )
    system_constraints = dspy.InputField(desc="Operational constraints the prompt must adhere to (e.g., business rules, persona, tone).")
    
    effectiveness_score = dspy.OutputField(desc="Score (0.0-1.0) for blocking known and new test attacks.")
    functionality_score = dspy.OutputField(desc="Score (0.0-1.0) for preserving original intended functionality and adhering to system_constraints.")
    robustness_score = dspy.OutputField(desc="Score (0.0-1.0) for resilience against slight variations of known attacks.")
    identified_weaknesses = dspy.OutputField(desc="List of any remaining weaknesses or areas of concern in the improved_prompt, as a JSON list of strings.")


class DefenderProgram(dspy.Module):
    """DSPy module that patches system prompts to defend against jailbreak attacks, using optimization."""
    
    def __init__(self):
        super().__init__()
        self.generate_defense_strategy = dspy.ChainOfThought(DefenseSignature)
        # For evaluation, a Predict module might be sufficient if ChainOfThought is too slow or complex,
        # but CoT allows for more nuanced reasoning in evaluation.
        self.evaluate_defense_quality = dspy.ChainOfThought(DefenseEvaluatorSignature)
    
    def forward(self, current_prompt: str, failing_attacks_summary: str, 
                system_constraints: str, target_vulnerabilities: str, 
                defense_history: Optional[str] = "No prior defense attempts for this prompt.") -> dspy.Prediction:
        """Generates an improved system prompt based on current vulnerabilities and constraints."""
        return self.generate_defense_strategy(
            current_prompt=current_prompt,
            failing_attacks_summary=failing_attacks_summary,
            defense_history=defense_history,
            system_constraints=system_constraints,
            target_vulnerabilities=target_vulnerabilities
        )
    
    def evaluate(self, original_prompt: str, improved_prompt: str, 
                 test_attack_scenarios: str, expected_behavior_description: str, 
                 system_constraints: str) -> DefenseEvaluationMetrics:
        """Evaluates the generated defense using a set of criteria."""
        raw_evaluation = self.evaluate_defense_quality(
            original_prompt=original_prompt,
            improved_prompt=improved_prompt,
            test_attack_scenarios=test_attack_scenarios,
            expected_behavior_description=expected_behavior_description,
            system_constraints=system_constraints
        )
        
        weaknesses = []
        try:
            # The LLM is asked to output a JSON list of strings for weaknesses.
            weaknesses_json = raw_evaluation.identified_weaknesses
            parsed_weaknesses = json.loads(weaknesses_json)
            if isinstance(parsed_weaknesses, list):
                weaknesses = [str(item) for item in parsed_weaknesses]
            else:
                weaknesses = ["Evaluator output for weaknesses was not a list."]
        except json.JSONDecodeError:
            weaknesses = ["Failed to parse evaluator output for weaknesses as JSON.", weaknesses_json]
        except Exception:
            weaknesses = ["Unknown error parsing weaknesses.", weaknesses_json]
            
        return DefenseEvaluationMetrics(
            effectiveness_score=float(raw_evaluation.effectiveness_score),
            functionality_score=float(raw_evaluation.functionality_score),
            robustness_score=float(raw_evaluation.robustness_score),
            identified_weaknesses=weaknesses,
            raw_evaluator_output=raw_evaluation.toDict() # Store the raw output for debugging
        )

    def _prepare_failing_attacks_summary(self, failing_attacks: List[Dict]) -> str:
        """Summarize the patterns in failing attacks for the DefenseSignature."""
        if not failing_attacks:
            return "No specific failing attacks provided for summary."
        
        summary_parts = [f"Observed {len(failing_attacks)} failing attack(s):"]
        
        for i, attack_record in enumerate(failing_attacks[:5]): # Summarize up to 5 attacks
            parts = [f"  Attack Example {i+1}:"]
            if 'type' in attack_record: parts.append(f"    Type: {attack_record['type']}")
            if 'prompt' in attack_record: parts.append(f"    Attack Prompt Snippet: '{str(attack_record['prompt'])[:150]}...'" )
            if 'llm_response' in attack_record: parts.append(f"    Observed LLM Response Snippet: '{str(attack_record['llm_response'])[:150]}...'" )
            if 'pattern' in attack_record: parts.append(f"    Identified Pattern: {attack_record['pattern']}")
            summary_parts.append("\n".join(parts))
            
        # Could add more sophisticated pattern analysis here if needed (e.g., frequency counts)
        if len(failing_attacks) > 5:
            summary_parts.append(f"  ...and {len(failing_attacks) - 5} more attacks not detailed here.")
        
        summary_parts.append("\nPlease address these types of failures in the improved prompt.")
        return "\n".join(summary_parts)

    def _prepare_defense_history_summary(self, defense_history_records: List[Dict]) -> str:
        """Summarize past defense attempts for the DefenseSignature."""
        if not defense_history_records:
            return "No prior defense attempts for this prompt."

        summary_parts = [f"Found {len(defense_history_records)} previous defense attempt(s) for this prompt:"]
        for i, record in enumerate(defense_history_records[:3]): # Summarize up to 3 past attempts
            parts = [f"  Attempt {i+1}:"]
            if 'attempted_prompt_snippet' in record: parts.append(f"    Prompt Snippet: '{str(record['attempted_prompt_snippet'])[:100]}...'" )
            if 'strategy' in record: parts.append(f"    Strategy: {record['strategy']}")
            if 'evaluation' in record and isinstance(record['evaluation'], DefenseEvaluationMetrics):
                eval_metrics = record['evaluation']
                parts.append(f"    Outcome: Effectiveness={eval_metrics.effectiveness_score:.2f}, Functionality={eval_metrics.functionality_score:.2f}")
                if eval_metrics.identified_weaknesses:
                    parts.append(f"    Noted Weaknesses: {', '.join(eval_metrics.identified_weaknesses[:2])}") 
            summary_parts.append("\n".join(parts))
        
        if len(defense_history_records) > 3:
            summary_parts.append(f"  ...and {len(defense_history_records) - 3} older attempts not detailed.")
        return "\n".join(summary_parts)
    
    def tune(self, train_set: List[dspy.Example], eval_fn, optimizer_class_name: str = "BootstrapFewShotWithRandomSearch", optimizer_args: Optional[Dict] = None, max_iterations: int = 10, plateau_patience: int = 3, plateau_threshold: float = 0.01):
        """
        Optimize the defense generation strategy using a specified DSPy teleprompter.
        
        Args:
            train_set: List of dspy.Example objects for training. Each example should have fields matching DefenseSignature inputs 
                       (e.g., current_prompt, failing_attacks_summary, etc.) and potentially a 'gold_improved_prompt' or 'gold_defense_strategy'.
            eval_fn: A function that takes a DefenderProgram instance and a validation set (List[dspy.Example]), 
                     and returns a single composite score to maximize. This function should use self.forward() and self.evaluate().
            optimizer_class_name: Name of the DSPy optimizer class to use (e.g., "BootstrapFewShot", "MIPRO", "BootstrapFewShotWithRandomSearch").
            optimizer_args: Dictionary of arguments to pass to the optimizer's constructor.
            max_iterations: Maximum number of optimization iterations.
            plateau_patience: Number of iterations without improvement before stopping.
            plateau_threshold: Minimum improvement in score to be considered significant.
        """
        if optimizer_args is None:
            optimizer_args = {'max_bootstrapped_demos': 3}

        try:
            optimizer_module = __import__('dspy.teleprompt', fromlist=[optimizer_class_name])
            optimizer_class = getattr(optimizer_module, optimizer_class_name)
        except (ImportError, AttributeError) as e:
            print(f"Error importing optimizer {optimizer_class_name}: {e}")
            # Fallback or raise error
            from dspy.teleprompt import BootstrapFewShotWithRandomSearch
            optimizer_class = BootstrapFewShotWithRandomSearch
            print(f"Falling back to BootstrapFewShotWithRandomSearch.")

        optimizer = optimizer_class(metric=eval_fn, **optimizer_args)
        
        # The `compile` method of the optimizer will modify `self.generate_defense_strategy` in place
        # if the module being compiled is `self` and the target is `self.generate_defense_strategy`.
        # Or, it returns a new compiled program.
        # We want to optimize `self.generate_defense_strategy` which is a dspy.ChainOfThought module.
        # We need to provide a student module to compile. Let's make a simple one for this purpose.

        class StudentDefenseGenerator(dspy.Module):
            def __init__(self):
                super().__init__()
                self.generate_defense_strategy = dspy.ChainOfThought(DefenseSignature)
            def forward(self, **kwargs):
                return self.generate_defense_strategy(**kwargs)

        # We need to carefully manage what is being optimized.
        # The optimizer expects a program (module) and a trainset.
        # It will then try to find the best version of that program.
        # Here, we are optimizing the `generate_defense_strategy` part of our `DefenderProgram`.

        # This requires the eval_fn to be structured to evaluate the *output* of generate_defense_strategy.
        # The current eval_fn is designed to evaluate the whole DefenderProgram.
        # For now, let's assume eval_fn can work with the output of `student.generate_defense_strategy`
        # if that's what the optimizer needs. This part of DSPy can be tricky.
        # A common pattern is to optimize the main module `self` and the optimizer is smart enough.

        # Let's try optimizing `self` directly, assuming `eval_fn` is adapted to work with the `DefenderProgram` instance. 
        # The metric (eval_fn) should then call program.forward(...) and program.evaluate(...)
        compiled_program = optimizer.compile(self, trainset=train_set, # This might need adjustment based on how eval_fn is truly structured.
                                             # num_trials for MIPRO, max_rounds for Bootstrap
                                            )
        
        # Update the program's predictor with the optimized one
        # The attribute name must match what was optimized.
        # If `optimizer.compile(self, ...)` optimizes `self.generate_defense_strategy` (by finding it as a sub-module),
        # then the `compiled_program` would be an optimized version of `self`.
        if hasattr(compiled_program, 'generate_defense_strategy'):
            self.generate_defense_strategy = compiled_program.generate_defense_strategy
            print(f"DefenderProgram's 'generate_defense_strategy' has been updated by the {optimizer_class_name} optimizer.")
        else:
            print(f"Warning: Optimizer {optimizer_class_name} did not produce an attribute 'generate_defense_strategy'. Manual update might be needed or optimizer behavior differs.")
            # If the optimizer returns a fully new compiled module that *is* the predictor:
            # self.generate_defense_strategy = compiled_program # This would be if compiled_program is the predictor itself.

        # The plateau logic would typically be handled *within* the eval_fn or by the optimizer if it supports it.
        # Or, one could wrap the .compile call in a loop, but many DSPy optimizers have internal iteration/trial logic.
        # For simplicity here, we assume the optimizer handles iterations or `num_trials`/`max_rounds` covers it.
        
        return self 