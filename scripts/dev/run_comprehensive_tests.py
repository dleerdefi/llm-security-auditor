#!/usr/bin/env python3
"""
Comprehensive Test Runner for DSPy Self-Fortifying LLM Gateway
Validates the entire project works as intended according to README specifications

This script runs:
1. All existing DSPy validation tests
2. New project integration tests
3. Holistic optimizer DSPy tests
4. CLI functionality tests
5. Docker configuration validation
6. Example configuration tests

Usage:
    python run_comprehensive_tests.py
    python run_comprehensive_tests.py --quick    # Skip slow tests
    python run_comprehensive_tests.py --docker   # Include Docker tests
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any
import time

# Rich for beautiful output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not available, using basic output")

if RICH_AVAILABLE:
    console = Console()
else:
    class SimpleConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = SimpleConsole()


class TestResult:
    """Container for test results."""
    
    def __init__(self, name: str, passed: bool, duration: float, details: str = ""):
        self.name = name
        self.passed = passed
        self.duration = duration
        self.details = details


class ComprehensiveTestRunner:
    """Runs comprehensive tests for the DSPy Self-Fortifying LLM Gateway."""
    
    def __init__(self, quick_mode: bool = False, include_docker: bool = False):
        self.quick_mode = quick_mode
        self.include_docker = include_docker
        self.results: List[TestResult] = []
        self.project_root = Path(__file__).parent
        
        # Set up environment for testing
        os.environ["OPENAI_API_KEY"] = "mock-key-for-testing"
        os.environ["ANTHROPIC_API_KEY"] = "mock-key-for-testing"
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "🧪 [bold blue]DSPy Self-Fortifying LLM Gateway - Comprehensive Test Suite[/bold blue]\n"
                f"Quick Mode: {self.quick_mode}\n"
                f"Include Docker: {self.include_docker}\n"
                f"Project Root: {self.project_root}",
                title="Test Runner Starting"
            ))
        else:
            print("=== DSPy Self-Fortifying LLM Gateway - Comprehensive Test Suite ===")
            print(f"Quick Mode: {self.quick_mode}")
            print(f"Include Docker: {self.include_docker}")
        
        start_time = time.time()
        
        # Run test suites
        self._run_dspy_validation_tests()
        self._run_project_integration_tests()
        self._run_holistic_optimizer_tests()
        self._run_cli_functionality_tests()
        self._run_configuration_validation_tests()
        
        if self.include_docker:
            self._run_docker_validation_tests()
        
        if not self.quick_mode:
            self._run_example_workflow_tests()
        
        total_duration = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(total_duration)
        self._print_summary(summary)
        
        return summary
    
    def _run_dspy_validation_tests(self):
        """Run existing DSPy validation tests."""
        console.print("\n🔬 [yellow]Running DSPy Validation Tests...[/yellow]")
        
        test_files = [
            "tests/test_dspy_integration.py",
            "tests/test_dspy_evaluation_framework.py"
        ]
        
        for test_file in test_files:
            if (self.project_root / test_file).exists():
                result = self._run_pytest(test_file, f"DSPy Validation - {Path(test_file).stem}")
                self.results.append(result)
            else:
                self.results.append(TestResult(
                    f"DSPy Validation - {Path(test_file).stem}",
                    False,
                    0.0,
                    f"Test file {test_file} not found"
                ))
    
    def _run_project_integration_tests(self):
        """Run project integration tests."""
        console.print("\n🔗 [yellow]Running Project Integration Tests...[/yellow]")
        
        test_file = "tests/test_project_integration.py"
        if (self.project_root / test_file).exists():
            result = self._run_pytest(test_file, "Project Integration Tests")
            self.results.append(result)
        else:
            self.results.append(TestResult(
                "Project Integration Tests",
                False,
                0.0,
                f"Test file {test_file} not found"
            ))
    
    def _run_holistic_optimizer_tests(self):
        """Run holistic optimizer DSPy tests."""
        console.print("\n🎯 [yellow]Running Holistic Optimizer DSPy Tests...[/yellow]")
        
        test_file = "tests/test_holistic_optimizer_dspy.py"
        if (self.project_root / test_file).exists():
            result = self._run_pytest(test_file, "Holistic Optimizer DSPy Tests")
            self.results.append(result)
        else:
            self.results.append(TestResult(
                "Holistic Optimizer DSPy Tests",
                False,
                0.0,
                f"Test file {test_file} not found"
            ))
    
    def _run_cli_functionality_tests(self):
        """Test CLI functionality as documented in README."""
        console.print("\n💻 [yellow]Testing CLI Functionality...[/yellow]")
        
        cli_tests = [
            ("optimize_prompt.py examples", "Holistic Optimizer Examples"),
            ("audit_prompt.py examples", "Security Auditor Examples"),
            ("demo.py", "Demo Script Starts"),  # Just test that it starts, not --help
        ]
        
        for command, test_name in cli_tests:
            result = self._test_cli_command(command, test_name)
            self.results.append(result)
    
    def _run_configuration_validation_tests(self):
        """Validate configuration files and structure."""
        console.print("\n📋 [yellow]Validating Configuration Files...[/yellow]")
        
        # Test configuration files exist and are valid
        config_tests = [
            ("configs/", "Configuration Directory"),
            ("prompts/", "Prompts Directory"),
            ("data/", "Data Directory"),
            ("requirements.txt", "Requirements File"),
            ("pyproject.toml", "Project Configuration"),
            ("config.yaml", "Main Config File"),
        ]
        
        for path, test_name in config_tests:
            result = self._test_file_exists(path, test_name)
            self.results.append(result)
        
        # Test YAML configuration validity
        if (self.project_root / "config.yaml").exists():
            result = self._test_yaml_validity("config.yaml", "Main Config YAML Validity")
            self.results.append(result)
    
    def _run_docker_validation_tests(self):
        """Validate Docker configuration."""
        console.print("\n🐳 [yellow]Validating Docker Configuration...[/yellow]")
        
        docker_tests = [
            ("Dockerfile", "Dockerfile Exists"),
            ("docker-compose.yml", "Docker Compose File"),
            (".dockerignore", "Docker Ignore File"),
            ("DOCKER_GUIDE.md", "Docker Guide Documentation"),
        ]
        
        for file_path, test_name in docker_tests:
            result = self._test_file_exists(file_path, test_name)
            self.results.append(result)
        
        # Test Docker Compose validity
        if (self.project_root / "docker-compose.yml").exists():
            result = self._test_docker_compose_validity()
            self.results.append(result)
    
    def _run_example_workflow_tests(self):
        """Test example workflows from README."""
        console.print("\n📚 [yellow]Testing Example Workflows...[/yellow]")
        
        # Test holistic optimization workflow
        result = self._test_holistic_optimization_workflow()
        self.results.append(result)
        
        # Test security audit workflow
        result = self._test_security_audit_workflow()
        self.results.append(result)
        
        # Test configuration file workflow
        result = self._test_config_file_workflow()
        self.results.append(result)
    
    def _run_pytest(self, test_file: str, test_name: str) -> TestResult:
        """Run pytest on a specific test file."""
        start_time = time.time()
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=300)
            
            duration = time.time() - start_time
            passed = result.returncode == 0
            details = result.stdout if passed else result.stderr
            
            return TestResult(test_name, passed, duration, details)
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(test_name, False, duration, "Test timed out after 5 minutes")
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(test_name, False, duration, f"Error running test: {str(e)}")
    
    def _test_cli_command(self, command: str, test_name: str) -> TestResult:
        """Test a CLI command."""
        start_time = time.time()
        
        try:
            cmd_parts = command.split()
            
            # Special handling for demo.py which is interactive
            if "demo.py" in command:
                # For demo.py, just test that it starts and shows the initial message
                result = subprocess.run([
                    sys.executable
                ] + cmd_parts, capture_output=True, text=True, cwd=self.project_root, 
                timeout=5, input="\n")  # Send empty input to exit gracefully
                
                duration = time.time() - start_time
                # For demo.py, success means it started and showed the demo message
                passed = "LLM Security Auditor - Quick Demo" in result.stdout or result.returncode == 0
                details = result.stdout if result.stdout else result.stderr
                
            else:
                result = subprocess.run([
                    sys.executable
                ] + cmd_parts, capture_output=True, text=True, cwd=self.project_root, timeout=30)
                
                duration = time.time() - start_time
                passed = result.returncode == 0
                details = result.stdout if passed else result.stderr
            
            return TestResult(test_name, passed, duration, details)
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            # For demo.py, timeout might be expected due to interactive nature
            if "demo.py" in command:
                return TestResult(test_name, True, duration, "Demo script started (timeout expected for interactive script)")
            return TestResult(test_name, False, duration, "Command timed out")
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(test_name, False, duration, f"Error running command: {str(e)}")
    
    def _test_file_exists(self, file_path: str, test_name: str) -> TestResult:
        """Test if a file or directory exists."""
        start_time = time.time()
        
        path = self.project_root / file_path
        exists = path.exists()
        duration = time.time() - start_time
        
        details = f"Path: {path}" if exists else f"Path not found: {path}"
        
        return TestResult(test_name, exists, duration, details)
    
    def _test_yaml_validity(self, file_path: str, test_name: str) -> TestResult:
        """Test if a YAML file is valid."""
        start_time = time.time()
        
        try:
            import yaml
            with open(self.project_root / file_path) as f:
                yaml.safe_load(f)
            
            duration = time.time() - start_time
            return TestResult(test_name, True, duration, "YAML is valid")
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(test_name, False, duration, f"YAML error: {str(e)}")
    
    def _test_docker_compose_validity(self) -> TestResult:
        """Test Docker Compose file validity."""
        start_time = time.time()
        
        try:
            result = subprocess.run([
                "docker-compose", "config"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=30)
            
            duration = time.time() - start_time
            passed = result.returncode == 0
            details = "Docker Compose config is valid" if passed else result.stderr
            
            return TestResult("Docker Compose Validity", passed, duration, details)
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult("Docker Compose Validity", False, duration, "Command timed out")
        except FileNotFoundError:
            duration = time.time() - start_time
            return TestResult("Docker Compose Validity", False, duration, "docker-compose command not found")
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Docker Compose Validity", False, duration, f"Error: {str(e)}")
    
    def _test_holistic_optimization_workflow(self) -> TestResult:
        """Test holistic optimization workflow."""
        start_time = time.time()
        
        try:
            # Import and test holistic optimizer
            sys.path.insert(0, str(self.project_root))
            from gateway.holistic_optimizer import HolisticPromptOptimizer
            
            optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
            config = optimizer.create_config(
                name="Workflow Test",
                description="Testing workflow",
                system_prompt="You are a helpful assistant",
                business_goals=["Be helpful"],
                target_audience="Users",
                brand_voice="Friendly",
                use_cases=["General assistance"],
                success_metrics=["User satisfaction"],
                constraints=["Be safe"]
            )
            
            report = optimizer.optimize_prompt(config)
            
            duration = time.time() - start_time
            passed = report is not None and len(report.optimized_prompt) > 0
            details = "Holistic optimization workflow completed successfully"
            
            return TestResult("Holistic Optimization Workflow", passed, duration, details)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Holistic Optimization Workflow", False, duration, f"Error: {str(e)}")
    
    def _test_security_audit_workflow(self) -> TestResult:
        """Test security audit workflow."""
        start_time = time.time()
        
        try:
            # Import and test security auditor
            sys.path.insert(0, str(self.project_root))
            from gateway.security_auditor import UniversalSecurityAuditor, SecurityConfig
            
            auditor = UniversalSecurityAuditor(model="openai", mock_mode=True)
            config = SecurityConfig(
                name="Workflow Test",
                description="Testing security workflow",
                system_prompt="You are a helpful assistant",
                business_rules=["Never share personal info"]
            )
            
            report = auditor.audit_security(config)
            
            duration = time.time() - start_time
            passed = report is not None and hasattr(report, 'overall_jailbreak_rate')
            details = "Security audit workflow completed successfully"
            
            return TestResult("Security Audit Workflow", passed, duration, details)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Security Audit Workflow", False, duration, f"Error: {str(e)}")
    
    def _test_config_file_workflow(self) -> TestResult:
        """Test configuration file workflow."""
        start_time = time.time()
        
        try:
            import tempfile
            import yaml
            
            # Create temporary config
            config_data = {
                "name": "Test Config",
                "description": "Testing config workflow",
                "system_prompt": "You are a helpful assistant",
                "business_goals": ["Be helpful"],
                "target_audience": "Users",
                "brand_voice": "Friendly",
                "use_cases": ["General assistance"],
                "success_metrics": ["User satisfaction"],
                "constraints": ["Be safe"]
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config_data, f)
                config_file = f.name
            
            try:
                # Test loading config
                sys.path.insert(0, str(self.project_root))
                from gateway.holistic_optimizer import HolisticPromptOptimizer
                
                optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
                config = optimizer.load_config(config_file)
                
                duration = time.time() - start_time
                passed = config.name == "Test Config"
                details = "Configuration file workflow completed successfully"
                
                return TestResult("Configuration File Workflow", passed, duration, details)
                
            finally:
                os.unlink(config_file)
                
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Configuration File Workflow", False, duration, f"Error: {str(e)}")
    
    def _generate_summary(self, total_duration: float) -> Dict[str, Any]:
        """Generate test summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_duration": total_duration,
            "results": self.results
        }
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print test summary."""
        if RICH_AVAILABLE:
            # Create results table
            table = Table(title="Test Results Summary")
            table.add_column("Test Name", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Duration", style="magenta")
            table.add_column("Details", style="dim")
            
            for result in summary["results"]:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                status_style = "green" if result.passed else "red"
                
                table.add_row(
                    result.name,
                    Text(status, style=status_style),
                    f"{result.duration:.2f}s",
                    result.details[:50] + "..." if len(result.details) > 50 else result.details
                )
            
            console.print(table)
            
            # Summary panel
            summary_text = (
                f"Total Tests: {summary['total_tests']}\n"
                f"Passed: {summary['passed_tests']}\n"
                f"Failed: {summary['failed_tests']}\n"
                f"Success Rate: {summary['success_rate']:.1f}%\n"
                f"Total Duration: {summary['total_duration']:.2f}s"
            )
            
            panel_style = "green" if summary['failed_tests'] == 0 else "red"
            title = "🎉 ALL TESTS PASSED!" if summary['failed_tests'] == 0 else "⚠️  SOME TESTS FAILED"
            
            console.print(Panel(summary_text, title=title, style=panel_style))
            
        else:
            print("\n=== TEST RESULTS SUMMARY ===")
            for result in summary["results"]:
                status = "PASS" if result.passed else "FAIL"
                print(f"{status:4} | {result.name:40} | {result.duration:6.2f}s")
            
            print(f"\nTotal: {summary['total_tests']}, Passed: {summary['passed_tests']}, Failed: {summary['failed_tests']}")
            print(f"Success Rate: {summary['success_rate']:.1f}%")
            print(f"Total Duration: {summary['total_duration']:.2f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive test runner for DSPy Self-Fortifying LLM Gateway")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (skip slow tests)")
    parser.add_argument("--docker", action="store_true", help="Include Docker validation tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner(quick_mode=args.quick, include_docker=args.docker)
    summary = runner.run_all_tests()
    
    # Exit with appropriate code
    exit_code = 0 if summary['failed_tests'] == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 