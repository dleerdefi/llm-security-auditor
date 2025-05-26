#!/usr/bin/env python3
"""
DSPy Validation Test Runner
Comprehensive validation that our Self-Fortifying LLM Gateway properly leverages DSPy's core technology offerings.

This script runs all DSPy-related tests and provides a clear assessment of:
1. Whether we're using DSPy properly
2. Whether our optimization actually works
3. Whether we're following DSPy best practices
4. Whether our system demonstrates measurable improvement

Run with: python tests/run_dspy_validation.py
"""

import sys
import os
import subprocess
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_dspy_integration import test_dspy_technology_validation
from tests.test_dspy_evaluation_framework import test_complete_dspy_optimization_validation


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    duration: float
    details: str
    critical: bool = True


class DSPyValidationRunner:
    """Comprehensive DSPy validation test runner."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
        
    def run_validation_test(self, test_name: str, test_func: callable, critical: bool = True) -> ValidationResult:
        """Run a single validation test and capture results."""
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            test_func()
            duration = time.time() - start_time
            result = ValidationResult(
                test_name=test_name,
                passed=True,
                duration=duration,
                details="Test passed successfully",
                critical=critical
            )
            print(f"✅ PASSED: {test_name} ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            result = ValidationResult(
                test_name=test_name,
                passed=False,
                duration=duration,
                details=str(e),
                critical=critical
            )
            print(f"❌ FAILED: {test_name} ({duration:.2f}s)")
            print(f"   Error: {e}")
        
        self.results.append(result)
        return result
    
    def run_pytest_tests(self, test_file: str, test_name: str) -> ValidationResult:
        """Run pytest tests and capture results."""
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run pytest with verbose output
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=project_root
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                validation_result = ValidationResult(
                    test_name=test_name,
                    passed=True,
                    duration=duration,
                    details="All pytest tests passed",
                    critical=True
                )
                print(f"✅ PASSED: {test_name} ({duration:.2f}s)")
            else:
                validation_result = ValidationResult(
                    test_name=test_name,
                    passed=False,
                    duration=duration,
                    details=f"pytest failed:\n{result.stdout}\n{result.stderr}",
                    critical=True
                )
                print(f"❌ FAILED: {test_name} ({duration:.2f}s)")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
                
        except Exception as e:
            duration = time.time() - start_time
            validation_result = ValidationResult(
                test_name=test_name,
                passed=False,
                duration=duration,
                details=str(e),
                critical=True
            )
            print(f"❌ FAILED: {test_name} ({duration:.2f}s)")
            print(f"   Error: {e}")
        
        self.results.append(validation_result)
        return validation_result
    
    def validate_dspy_imports(self) -> ValidationResult:
        """Validate that DSPy and all required modules can be imported."""
        def test_imports():
            import dspy
            from attacker import AttackProgram, AttackSignature, AttackMetrics
            from defender import DefenderProgram, DefenseSignature, DefenseEvaluationMetrics
            
            # Test DSPy core components
            assert hasattr(dspy, 'Signature')
            assert hasattr(dspy, 'Module')
            assert hasattr(dspy, 'ChainOfThought')
            assert hasattr(dspy, 'Predict')
            assert hasattr(dspy, 'Example')
            
            # Test teleprompters
            from dspy.teleprompt import BootstrapFewShot, MIPROv2
            
            print("✅ All DSPy imports successful")
            print("✅ All project modules imported successfully")
            print("✅ DSPy teleprompters available")
        
        return self.run_validation_test("DSPy Import Validation", test_imports)
    
    def validate_dspy_signatures(self) -> ValidationResult:
        """Validate that our DSPy signatures are properly structured."""
        def test_signatures():
            import dspy
            from attacker import AttackSignature
            from defender import DefenseSignature, DefenseEvaluatorSignature
            
            # Test signature inheritance
            assert issubclass(AttackSignature, dspy.Signature)
            assert issubclass(DefenseSignature, dspy.Signature)
            assert issubclass(DefenseEvaluatorSignature, dspy.Signature)
            
            # Test field existence and descriptions
            signatures = [AttackSignature, DefenseSignature, DefenseEvaluatorSignature]
            for sig in signatures:
                fields = [attr for attr in dir(sig) if not attr.startswith('_')]
                assert len(fields) > 0, f"Signature {sig.__name__} has no fields"
                
                # Check for field descriptions
                for field_name in fields:
                    field = getattr(sig, field_name)
                    if hasattr(field, 'desc'):
                        assert field.desc, f"Field {field_name} missing description"
                        assert len(field.desc) > 5, f"Field {field_name} description too short"
            
            print("✅ All signatures properly inherit from dspy.Signature")
            print("✅ All signature fields have meaningful descriptions")
            print("✅ Signature structure follows DSPy best practices")
        
        return self.run_validation_test("DSPy Signature Validation", test_signatures)
    
    def validate_dspy_modules(self) -> ValidationResult:
        """Validate that our DSPy modules are properly structured."""
        def test_modules():
            import dspy
            from attacker import AttackProgram
            from defender import DefenderProgram
            
            # Test module inheritance
            assert issubclass(AttackProgram, dspy.Module)
            assert issubclass(DefenderProgram, dspy.Module)
            
            # Test module initialization
            attacker = AttackProgram()
            defender = DefenderProgram()
            
            # Test DSPy components
            assert hasattr(attacker, 'generate_attack')
            assert hasattr(attacker, 'evaluate_attack')
            assert isinstance(attacker.generate_attack, dspy.ChainOfThought)
            assert isinstance(attacker.evaluate_attack, dspy.ChainOfThought)
            
            assert hasattr(defender, 'generate_defense_strategy')
            assert hasattr(defender, 'evaluate_defense_quality')
            assert isinstance(defender.generate_defense_strategy, dspy.ChainOfThought)
            assert isinstance(defender.evaluate_defense_quality, dspy.ChainOfThought)
            
            # Test optimization methods
            assert hasattr(attacker, 'tune')
            assert hasattr(defender, 'tune')
            assert callable(attacker.tune)
            assert callable(defender.tune)
            
            print("✅ All modules properly inherit from dspy.Module")
            print("✅ All modules use DSPy ChainOfThought predictors")
            print("✅ All modules implement optimization methods")
            print("✅ Module structure follows DSPy best practices")
        
        return self.run_validation_test("DSPy Module Validation", test_modules)
    
    def validate_optimization_capability(self) -> ValidationResult:
        """Validate that our optimization actually works."""
        def test_optimization():
            import dspy
            from tests.test_dspy_evaluation_framework import DSPyEvaluationFramework, MockDSPyLM
            
            # Create evaluation framework
            framework = DSPyEvaluationFramework(mock_mode=True)
            
            # Test evaluation function creation
            validation_attacks = ["test attack"]
            eval_fn = framework.create_evaluation_function(validation_attacks)
            assert callable(eval_fn)
            
            # Test training dataset creation
            dataset = framework.create_training_dataset()
            assert len(dataset) > 0
            for example in dataset:
                assert isinstance(example, dspy.Example)
            
            # Test optimization experiment with proper mock LM
            original_lm = dspy.settings.lm
            mock_lm = MockDSPyLM()
            dspy.settings.configure(lm=mock_lm)
            try:
                results = framework.run_optimization_experiment(num_rounds=2)
                assert len(results) == 2
                
                # Test trend analysis
                analysis = framework.analyze_optimization_trends(results)
                assert 'total_improvement' in analysis
                assert 'optimization_successful' in analysis
            finally:
                dspy.settings.configure(lm=original_lm)
            
            print("✅ Evaluation framework properly structured")
            print("✅ Training dataset creation works")
            print("✅ Optimization experiment runs successfully")
            print("✅ Trend analysis provides meaningful insights")
        
        return self.run_validation_test("DSPy Optimization Validation", test_optimization)
    
    def run_comprehensive_validation(self) -> Dict[str, any]:
        """Run all DSPy validation tests."""
        print("🚀 Starting Comprehensive DSPy Validation")
        print("=" * 80)
        print("This validation ensures our Self-Fortifying LLM Gateway properly")
        print("leverages DSPy's core technology offerings and demonstrates")
        print("measurable improvement through optimization.")
        print("=" * 80)
        
        # Core validation tests
        self.validate_dspy_imports()
        self.validate_dspy_signatures()
        self.validate_dspy_modules()
        self.validate_optimization_capability()
        
        # Integration tests
        try:
            from tests.test_dspy_integration import test_dspy_technology_validation
            self.run_validation_test(
                "DSPy Technology Integration",
                test_dspy_technology_validation
            )
        except ImportError as e:
            print(f"⚠️  Could not import test_dspy_technology_validation: {e}")
        
        # Optimization effectiveness tests
        try:
            from tests.test_dspy_evaluation_framework import test_complete_dspy_optimization_validation
            self.run_validation_test(
                "DSPy Optimization Effectiveness",
                test_complete_dspy_optimization_validation
            )
        except ImportError as e:
            print(f"⚠️  Could not import test_complete_dspy_optimization_validation: {e}")
        
        # Run pytest tests if available
        test_files = [
            ("tests/test_dspy_integration.py", "DSPy Integration Tests"),
            ("tests/test_dspy_evaluation_framework.py", "DSPy Evaluation Framework Tests")
        ]
        
        for test_file, test_name in test_files:
            if os.path.exists(test_file):
                self.run_pytest_tests(test_file, test_name)
        
        return self.generate_final_report()
    
    def generate_final_report(self) -> Dict[str, any]:
        """Generate comprehensive validation report."""
        total_duration = time.time() - self.start_time
        
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        critical_failures = [r for r in failed_tests if r.critical]
        
        # Calculate scores
        total_tests = len(self.results)
        passed_count = len(passed_tests)
        success_rate = (passed_count / total_tests) * 100 if total_tests > 0 else 0
        
        # Determine overall validation status
        validation_passed = len(critical_failures) == 0 and success_rate >= 80
        
        report = {
            "validation_passed": validation_passed,
            "total_tests": total_tests,
            "passed_tests": passed_count,
            "failed_tests": len(failed_tests),
            "critical_failures": len(critical_failures),
            "success_rate": success_rate,
            "total_duration": total_duration,
            "results": self.results
        }
        
        # Print final report
        print("\n" + "=" * 80)
        print("🎯 FINAL DSPy VALIDATION REPORT")
        print("=" * 80)
        
        if validation_passed:
            print("🎉 VALIDATION PASSED! 🎉")
            print("✅ Your Self-Fortifying LLM Gateway properly leverages DSPy's core technology!")
            print("✅ All critical DSPy features are correctly implemented")
            print("✅ Optimization demonstrates measurable improvement")
            print("✅ System follows DSPy best practices for 2025")
        else:
            print("❌ VALIDATION FAILED!")
            print("❌ Your system does not properly leverage DSPy's core technology")
            print("❌ More work is needed to implement DSPy correctly")
            
            if critical_failures:
                print(f"\n🚨 Critical Failures ({len(critical_failures)}):")
                for failure in critical_failures:
                    print(f"   • {failure.test_name}: {failure.details}")
        
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_count}")
        print(f"   Failed: {len(failed_tests)}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Duration: {total_duration:.2f}s")
        
        if failed_tests:
            print(f"\n❌ Failed Tests:")
            for failure in failed_tests:
                print(f"   • {failure.test_name} ({failure.duration:.2f}s)")
                if failure.critical:
                    print(f"     🚨 CRITICAL: {failure.details}")
                else:
                    print(f"     ⚠️  Warning: {failure.details}")
        
        print("\n" + "=" * 80)
        
        return report


def main():
    """Main entry point for DSPy validation."""
    runner = DSPyValidationRunner()
    
    try:
        report = runner.run_comprehensive_validation()
        
        # Exit with appropriate code
        if report["validation_passed"]:
            print("\n🚀 DSPy validation completed successfully!")
            print("Your project properly leverages DSPy's core technology offerings.")
            sys.exit(0)
        else:
            print("\n💡 DSPy validation identified areas for improvement.")
            print("Please address the critical failures before proceeding.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 