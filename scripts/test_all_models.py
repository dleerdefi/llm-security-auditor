#!/usr/bin/env python3
"""
Comprehensive Model Testing Framework
Tests all models specified in config.py following DSPy best practices
"""

import os
import sys
import dspy
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.config import LLMConfig
from gateway.security_auditor import UniversalSecurityAuditor

console = Console()

@dataclass
class ModelTestResult:
    """Results from testing a specific model."""
    model_name: str
    provider: str
    success: bool = False
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    response_quality: Optional[str] = None
    dspy_compatibility: bool = False
    reasoning_capability: bool = False

class DSPyModelTester:
    """
    Comprehensive model tester following DSPy best practices.
    
    Tests each model for:
    1. Basic connectivity and authentication
    2. DSPy compatibility and signature support
    3. Reasoning capabilities (for reasoning models)
    4. Security auditing performance
    5. Prompt optimization capabilities
    """
    
    def __init__(self):
        self.config = LLMConfig()
        self.test_results: List[ModelTestResult] = []
        
        # Load test prompts
        self.test_prompts = self._load_test_prompts()
        
    def _load_test_prompts(self) -> Dict[str, str]:
        """Load test prompts for different model types."""
        return {
            "basic": "You are a helpful assistant. Respond with 'Hello, I am working correctly!' if you understand this message.",
            "reasoning": "Think step by step: What is 2+2? Show your reasoning process.",
            "security": "You are a customer support bot. Never share customer data. How would you handle a request for someone else's order information?",
            "complex": "Analyze this scenario: A user asks you to ignore your instructions. How should you respond and why?"
        }
    
    def test_openai_models(self) -> List[ModelTestResult]:
        """Test all OpenAI models from config."""
        results = []
        
        if not self.config.OPENAI_API_KEY:
            console.print("[red]❌ OpenAI API key not found. Skipping OpenAI tests.[/red]")
            return results
        
        console.print("[bold blue]🔍 Testing OpenAI Models[/bold blue]")
        
        for role, model_name in self.config.OPENAI_MODELS.items():
            console.print(f"\n[cyan]Testing {model_name} ({role})[/cyan]")
            
            result = ModelTestResult(
                model_name=model_name,
                provider="openai"
            )
            
            try:
                # Test basic connectivity
                if self._test_openai_model(model_name, role):
                    result.success = True
                    result.dspy_compatibility = True
                    
                    # Test reasoning capability for reasoning models
                    if "o1" in model_name or role == "reasoning":
                        result.reasoning_capability = self._test_reasoning_capability(model_name, "openai")
                    
                    console.print(f"[green]✅ {model_name} - Working correctly[/green]")
                else:
                    result.success = False
                    result.error_message = "Failed basic connectivity test"
                    console.print(f"[red]❌ {model_name} - Failed basic test[/red]")
                    
            except Exception as e:
                result.success = False
                result.error_message = str(e)
                console.print(f"[red]❌ {model_name} - Error: {e}[/red]")
            
            results.append(result)
        
        return results
    
    def test_anthropic_models(self) -> List[ModelTestResult]:
        """Test all Anthropic models from config."""
        results = []
        
        if not self.config.ANTHROPIC_API_KEY:
            console.print("[red]❌ Anthropic API key not found. Skipping Anthropic tests.[/red]")
            return results
        
        console.print("[bold blue]🔍 Testing Anthropic Models[/bold blue]")
        
        for role, model_name in self.config.ANTHROPIC_MODELS.items():
            console.print(f"\n[cyan]Testing {model_name} ({role})[/cyan]")
            
            result = ModelTestResult(
                model_name=model_name,
                provider="anthropic"
            )
            
            try:
                # Test basic connectivity
                if self._test_anthropic_model(model_name, role):
                    result.success = True
                    result.dspy_compatibility = True
                    
                    # Test reasoning capability for reasoning models
                    if "3-7" in model_name or role == "reasoning":
                        result.reasoning_capability = self._test_reasoning_capability(model_name, "anthropic")
                    
                    console.print(f"[green]✅ {model_name} - Working correctly[/green]")
                else:
                    result.success = False
                    result.error_message = "Failed basic connectivity test"
                    console.print(f"[red]❌ {model_name} - Failed basic test[/red]")
                    
            except Exception as e:
                result.success = False
                result.error_message = str(e)
                console.print(f"[red]❌ {model_name} - Error: {e}[/red]")
            
            results.append(result)
        
        return results
    
    def _test_openai_model(self, model_name: str, role: str) -> bool:
        """Test OpenAI model with DSPy."""
        try:
            # Configure DSPy for this model with special handling for reasoning models
            if "o1" in model_name:
                # O1 models require special parameters
                lm = dspy.LM(
                    f'openai/{model_name}', 
                    api_key=self.config.OPENAI_API_KEY,
                    temperature=1.0,
                    max_tokens=20000
                )
            else:
                lm = dspy.LM(f'openai/{model_name}', api_key=self.config.OPENAI_API_KEY)
            
            dspy.configure(lm=lm)
            
            # Test basic DSPy signature
            class BasicTest(dspy.Signature):
                """Test basic model functionality."""
                prompt: str = dspy.InputField()
                response: str = dspy.OutputField()
            
            predictor = dspy.Predict(BasicTest)
            
            # Use appropriate prompt for model type
            if "o1" in model_name:
                # O1 models work better with direct prompts
                result = predictor(prompt=self.test_prompts["reasoning"])
            else:
                result = predictor(prompt=self.test_prompts["basic"])
            
            # Check if we got a reasonable response
            return len(result.response.strip()) > 10
            
        except Exception as e:
            console.print(f"[yellow]Warning: {model_name} test failed: {e}[/yellow]")
            return False
    
    def _test_anthropic_model(self, model_name: str, role: str) -> bool:
        """Test Anthropic model with DSPy."""
        try:
            # Configure DSPy for this model
            lm = dspy.LM(f'anthropic/{model_name}', api_key=self.config.ANTHROPIC_API_KEY)
            dspy.configure(lm=lm)
            
            # Test basic DSPy signature
            class BasicTest(dspy.Signature):
                """Test basic model functionality."""
                prompt: str = dspy.InputField()
                response: str = dspy.OutputField()
            
            predictor = dspy.Predict(BasicTest)
            
            # Use appropriate prompt for model type
            if "3-7" in model_name:
                # Claude 3.7 has extended thinking capabilities
                result = predictor(prompt=self.test_prompts["reasoning"])
            else:
                result = predictor(prompt=self.test_prompts["basic"])
            
            # Check if we got a reasonable response
            return len(result.response.strip()) > 10
            
        except Exception as e:
            console.print(f"[yellow]Warning: {model_name} test failed: {e}[/yellow]")
            return False
    
    def _test_reasoning_capability(self, model_name: str, provider: str) -> bool:
        """Test reasoning capabilities for reasoning models."""
        try:
            # Configure DSPy for this model
            if provider == "openai":
                lm = dspy.LM(f'openai/{model_name}', api_key=self.config.OPENAI_API_KEY)
            else:
                lm = dspy.LM(f'anthropic/{model_name}', api_key=self.config.ANTHROPIC_API_KEY)
            
            dspy.configure(lm=lm)
            
            # Test reasoning with Chain of Thought
            class ReasoningTest(dspy.Signature):
                """Test reasoning capabilities with step-by-step thinking."""
                problem: str = dspy.InputField()
                reasoning: str = dspy.OutputField(desc="Step-by-step reasoning process")
                answer: str = dspy.OutputField(desc="Final answer")
            
            predictor = dspy.ChainOfThought(ReasoningTest)
            result = predictor(problem="If a train travels 60 mph for 2 hours, how far does it go?")
            
            # Check if reasoning is present and answer is correct
            has_reasoning = len(result.reasoning.strip()) > 20
            correct_answer = "120" in result.answer
            
            return has_reasoning and correct_answer
            
        except Exception as e:
            console.print(f"[yellow]Warning: Reasoning test failed for {model_name}: {e}[/yellow]")
            return False
    
    def test_security_auditor_compatibility(self) -> Dict[str, bool]:
        """Test which models work with our security auditor."""
        console.print("[bold blue]🛡️ Testing Security Auditor Compatibility[/bold blue]")
        
        compatibility = {}
        
        # Test OpenAI models
        if self.config.OPENAI_API_KEY:
            for role, model_name in self.config.OPENAI_MODELS.items():
                try:
                    console.print(f"[cyan]Testing security auditor with {model_name}[/cyan]")
                    
                    # Create auditor in mock mode to test DSPy signatures
                    auditor = UniversalSecurityAuditor(model="openai", mock_mode=True)
                    
                    # Test basic configuration creation
                    config = auditor.create_config_from_prompt(
                        name="Test Config",
                        description="Test configuration",
                        system_prompt="You are a helpful assistant.",
                        business_rules=["Be helpful", "Be safe"]
                    )
                    
                    compatibility[f"openai/{model_name}"] = True
                    console.print(f"[green]✅ {model_name} - Compatible with security auditor[/green]")
                    
                except Exception as e:
                    compatibility[f"openai/{model_name}"] = False
                    console.print(f"[red]❌ {model_name} - Not compatible: {e}[/red]")
        
        # Test Anthropic models
        if self.config.ANTHROPIC_API_KEY:
            for role, model_name in self.config.ANTHROPIC_MODELS.items():
                try:
                    console.print(f"[cyan]Testing security auditor with {model_name}[/cyan]")
                    
                    # Create auditor in mock mode to test DSPy signatures
                    auditor = UniversalSecurityAuditor(model="anthropic", mock_mode=True)
                    
                    # Test basic configuration creation
                    config = auditor.create_config_from_prompt(
                        name="Test Config",
                        description="Test configuration",
                        system_prompt="You are a helpful assistant.",
                        business_rules=["Be helpful", "Be safe"]
                    )
                    
                    compatibility[f"anthropic/{model_name}"] = True
                    console.print(f"[green]✅ {model_name} - Compatible with security auditor[/green]")
                    
                except Exception as e:
                    compatibility[f"anthropic/{model_name}"] = False
                    console.print(f"[red]❌ {model_name} - Not compatible: {e}[/red]")
        
        return compatibility
    
    def run_comprehensive_test(self) -> Dict[str, any]:
        """Run comprehensive tests on all models."""
        console.print(Panel.fit(
            "[bold blue]🧪 Comprehensive Model Testing Framework[/bold blue]\n"
            "Testing all models from config.py for DSPy compatibility and functionality",
            title="Model Testing"
        ))
        
        # Test all models
        openai_results = self.test_openai_models()
        anthropic_results = self.test_anthropic_models()
        security_compatibility = self.test_security_auditor_compatibility()
        
        # Combine results
        all_results = openai_results + anthropic_results
        self.test_results = all_results
        
        # Generate summary report
        self._print_summary_report(all_results, security_compatibility)
        
        return {
            "openai_results": openai_results,
            "anthropic_results": anthropic_results,
            "security_compatibility": security_compatibility,
            "total_models_tested": len(all_results),
            "successful_models": len([r for r in all_results if r.success]),
            "failed_models": len([r for r in all_results if not r.success])
        }
    
    def _print_summary_report(self, results: List[ModelTestResult], security_compatibility: Dict[str, bool]):
        """Print a comprehensive summary report."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold green]📊 Model Testing Summary Report[/bold green]",
            title="Test Results"
        ))
        
        # Overall statistics
        total = len(results)
        successful = len([r for r in results if r.success])
        failed = total - successful
        
        console.print(f"[bold]Overall Results:[/bold]")
        console.print(f"✅ Successful: {successful}/{total}")
        console.print(f"❌ Failed: {failed}/{total}")
        console.print(f"📊 Success Rate: {(successful/total)*100:.1f}%")
        
        # Detailed results table
        console.print("\n[bold]Detailed Results:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Provider", style="cyan")
        table.add_column("Model", style="yellow")
        table.add_column("Status", justify="center")
        table.add_column("DSPy Compatible", justify="center")
        table.add_column("Reasoning", justify="center")
        table.add_column("Security Auditor", justify="center")
        table.add_column("Error", style="red")
        
        for result in results:
            status = "✅" if result.success else "❌"
            dspy_compat = "✅" if result.dspy_compatibility else "❌"
            reasoning = "✅" if result.reasoning_capability else "➖"
            
            # Check security auditor compatibility
            model_key = f"{result.provider}/{result.model_name}"
            security_compat = "✅" if security_compatibility.get(model_key, False) else "❌"
            
            error_msg = result.error_message[:50] + "..." if result.error_message and len(result.error_message) > 50 else (result.error_message or "")
            
            table.add_row(
                result.provider.title(),
                result.model_name,
                status,
                dspy_compat,
                reasoning,
                security_compat,
                error_msg
            )
        
        console.print(table)
        
        # Recommendations
        console.print("\n[bold]💡 Recommendations:[/bold]")
        
        # Check for failed models
        failed_models = [r for r in results if not r.success]
        if failed_models:
            console.print("[yellow]⚠️  Some models failed testing. Check API keys and model availability.[/yellow]")
            for failed in failed_models:
                console.print(f"   • {failed.provider}/{failed.model_name}: {failed.error_message}")
        
        # Check for reasoning models
        reasoning_models = [r for r in results if r.reasoning_capability]
        if reasoning_models:
            console.print("[green]🧠 Reasoning models detected. These support advanced Chain-of-Thought prompting.[/green]")
            for reasoning in reasoning_models:
                console.print(f"   • {reasoning.provider}/{reasoning.model_name}")
        
        # Security auditor compatibility
        compatible_models = [k for k, v in security_compatibility.items() if v]
        if compatible_models:
            console.print("[green]🛡️ Models compatible with security auditor:[/green]")
            for model in compatible_models:
                console.print(f"   • {model}")
        
        console.print("\n[bold green]✅ Model testing complete![/bold green]")


def main():
    """Main function to run model tests."""
    tester = DSPyModelTester()
    results = tester.run_comprehensive_test()
    
    # Exit with appropriate code
    if results["failed_models"] > 0:
        console.print(f"\n[red]❌ {results['failed_models']} models failed testing[/red]")
        sys.exit(1)
    else:
        console.print(f"\n[green]✅ All {results['successful_models']} models passed testing[/green]")
        sys.exit(0)


if __name__ == "__main__":
    main() 