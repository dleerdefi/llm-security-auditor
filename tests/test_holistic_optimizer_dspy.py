#!/usr/bin/env python3
"""
Holistic Optimizer DSPy Implementation Tests
Verifies that the holistic optimizer properly uses DSPy patterns and best practices

This test suite specifically validates:
1. DSPy signatures are properly defined
2. DSPy modules are correctly implemented
3. Mock mode works for testing
4. All optimization dimensions are covered
5. Integration with existing security auditor
"""

import os
import sys
import pytest
import dspy
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gateway.holistic_optimizer import (
    HolisticPromptOptimizer, 
    OptimizationConfig,
    EffectivenessAnalyzer,
    UserExperienceAnalyzer,
    BusinessAlignmentAnalyzer,
    CostEfficiencyAnalyzer,
    ComplianceAnalyzer,
    HolisticOptimizer
)


class TestDSPySignatures:
    """Test that all DSPy signatures are properly defined."""
    
    def test_effectiveness_analyzer_signature(self):
        """Test EffectivenessAnalyzer signature structure."""
        # Check that it's a proper DSPy signature
        assert issubclass(EffectivenessAnalyzer, dspy.Signature)
        
        # Verify field types and descriptions
        fields = EffectivenessAnalyzer.model_fields
        
        # Check input fields
        assert 'system_prompt' in fields
        assert 'use_cases' in fields
        assert 'target_audience' in fields
        assert 'success_metrics' in fields
        
        # Check output fields
        assert 'effectiveness_score' in fields
        assert 'strengths' in fields
        assert 'weaknesses' in fields
        assert 'task_completion_quality' in fields
        assert 'clarity_score' in fields
        
        # Check that fields have descriptions (DSPy stores them in json_schema_extra)
        assert 'desc' in fields['system_prompt'].json_schema_extra
        assert 'desc' in fields['effectiveness_score'].json_schema_extra
    
    def test_user_experience_analyzer_signature(self):
        """Test UserExperienceAnalyzer signature structure."""
        assert issubclass(UserExperienceAnalyzer, dspy.Signature)
        
        # Check key fields exist
        fields = UserExperienceAnalyzer.model_fields
        assert 'system_prompt' in fields
        assert 'target_audience' in fields
        assert 'brand_voice' in fields
        assert 'ux_score' in fields
        assert 'tone_alignment' in fields
        assert 'ux_improvements' in fields
    
    def test_business_alignment_analyzer_signature(self):
        """Test BusinessAlignmentAnalyzer signature structure."""
        assert issubclass(BusinessAlignmentAnalyzer, dspy.Signature)
        
        fields = BusinessAlignmentAnalyzer.model_fields
        assert 'system_prompt' in fields
        assert 'business_goals' in fields
        assert 'alignment_score' in fields
        assert 'goal_support' in fields
        assert 'business_improvements' in fields
    
    def test_cost_efficiency_analyzer_signature(self):
        """Test CostEfficiencyAnalyzer signature structure."""
        assert issubclass(CostEfficiencyAnalyzer, dspy.Signature)
        
        fields = CostEfficiencyAnalyzer.model_fields
        assert 'system_prompt' in fields
        assert 'use_cases' in fields
        assert 'cost_budget' in fields
        assert 'efficiency_score' in fields
        assert 'token_efficiency' in fields
        assert 'cost_improvements' in fields
    
    def test_compliance_analyzer_signature(self):
        """Test ComplianceAnalyzer signature structure."""
        assert issubclass(ComplianceAnalyzer, dspy.Signature)
        
        fields = ComplianceAnalyzer.model_fields
        assert 'system_prompt' in fields
        assert 'compliance_requirements' in fields
        assert 'compliance_score' in fields
        assert 'ethical_alignment' in fields
        assert 'compliance_improvements' in fields
    
    def test_holistic_optimizer_signature(self):
        """Test HolisticOptimizer signature structure."""
        assert issubclass(HolisticOptimizer, dspy.Signature)
        
        fields = HolisticOptimizer.model_fields
        assert 'original_prompt' in fields
        assert 'effectiveness_analysis' in fields
        assert 'ux_analysis' in fields
        assert 'business_analysis' in fields
        assert 'cost_analysis' in fields
        assert 'compliance_analysis' in fields
        assert 'security_analysis' in fields
        assert 'optimization_priorities' in fields
        assert 'optimized_prompt' in fields
        assert 'optimization_strategy' in fields
        assert 'improvement_summary' in fields


class TestHolisticOptimizerImplementation:
    """Test the HolisticPromptOptimizer class implementation."""
    
    def setup_method(self):
        """Setup for each test."""
        os.environ["OPENAI_API_KEY"] = "mock-key-for-testing"
        os.environ["ANTHROPIC_API_KEY"] = "mock-key-for-testing"
    
    def test_optimizer_initialization_mock_mode(self):
        """Test optimizer initializes correctly in mock mode."""
        optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
        
        assert optimizer.model == "openai"
        assert optimizer.mock_mode is True
        
        # In mock mode, DSPy modules should be None
        assert optimizer.effectiveness_analyzer is None
        assert optimizer.ux_analyzer is None
        assert optimizer.business_analyzer is None
        assert optimizer.cost_analyzer is None
        assert optimizer.compliance_analyzer is None
        assert optimizer.holistic_optimizer is None
    
    def test_optimizer_initialization_real_mode(self):
        """Test optimizer initializes correctly in real mode."""
        # Mock DSPy configuration to avoid actual API calls
        with patch('dspy.configure') as mock_configure, \
             patch('dspy.LM') as mock_lm:
            
            optimizer = HolisticPromptOptimizer(model="openai", mock_mode=False)
            
            assert optimizer.model == "openai"
            assert optimizer.mock_mode is False
            
            # DSPy modules should be initialized
            assert optimizer.effectiveness_analyzer is not None
            assert optimizer.ux_analyzer is not None
            assert optimizer.business_analyzer is not None
            assert optimizer.cost_analyzer is not None
            assert optimizer.compliance_analyzer is not None
            assert optimizer.holistic_optimizer is not None
            
            # Verify DSPy was configured
            mock_configure.assert_called_once()
    
    def test_config_creation(self):
        """Test configuration creation."""
        optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
        
        config = optimizer.create_config(
            name="Test Config",
            description="Test description",
            system_prompt="Test prompt",
            business_goals=["Goal 1", "Goal 2"],
            target_audience="Test audience",
            brand_voice="Test voice",
            use_cases=["Use case 1"],
            success_metrics=["Metric 1"],
            constraints=["Constraint 1"],
            compliance_requirements=["Requirement 1"],
            cost_budget="Test budget"
        )
        
        assert isinstance(config, OptimizationConfig)
        assert config.name == "Test Config"
        assert config.description == "Test description"
        assert config.system_prompt == "Test prompt"
        assert len(config.business_goals) == 2
        assert config.target_audience == "Test audience"
        assert config.brand_voice == "Test voice"
        assert len(config.use_cases) == 1
        assert len(config.success_metrics) == 1
        assert len(config.constraints) == 1
        assert len(config.compliance_requirements) == 1
        assert config.cost_budget == "Test budget"
    
    def test_optimization_mock_mode(self):
        """Test optimization runs in mock mode."""
        optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
        
        config = optimizer.create_config(
            name="Mock Test",
            description="Testing mock optimization",
            system_prompt="You are a helpful assistant",
            business_goals=["Be helpful"],
            target_audience="Users",
            brand_voice="Friendly",
            use_cases=["General assistance"],
            success_metrics=["User satisfaction"],
            constraints=["Be safe"]
        )
        
        # Run optimization
        report = optimizer.optimize_prompt(config)
        
        # Verify report structure
        assert report is not None
        assert report.config_name == "Mock Test"
        assert report.original_prompt == "You are a helpful assistant"
        assert len(report.optimized_prompt) > 0
        assert report.overall_score > 0.0
        assert report.overall_score <= 1.0
        assert len(report.recommendations) > 0
        assert isinstance(report.improvements, dict)
        assert isinstance(report.cost_analysis, dict)
        
        # Verify all dimensions are covered
        expected_dimensions = ['effectiveness', 'ux', 'business', 'cost', 'compliance', 'security']
        for dimension in expected_dimensions:
            assert dimension in report.improvements
    
    def test_custom_optimization_priorities(self):
        """Test optimization with custom priorities."""
        optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
        
        config = optimizer.create_config(
            name="Priority Test",
            description="Testing custom priorities",
            system_prompt="Test prompt",
            business_goals=["Test goal"],
            target_audience="Test users",
            brand_voice="Test voice",
            use_cases=["Test case"],
            success_metrics=["Test metric"],
            constraints=["Test constraint"]
        )
        
        # Custom priorities favoring security
        custom_priorities = {
            "security": 0.5,
            "effectiveness": 0.2,
            "ux": 0.1,
            "business": 0.1,
            "cost": 0.05,
            "compliance": 0.05
        }
        
        report = optimizer.optimize_prompt(config, custom_priorities)
        
        assert report is not None
        assert report.overall_score > 0.0
        # In mock mode, we can't verify priority weighting, but optimization should complete
    
    def test_security_integration(self):
        """Test integration with existing security auditor."""
        optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
        
        config = optimizer.create_config(
            name="Security Integration Test",
            description="Testing security integration",
            system_prompt="You are a helpful assistant",
            business_goals=["Maintain security"],
            target_audience="All users",
            brand_voice="Secure",
            use_cases=["General assistance"],
            success_metrics=["Security score"],
            constraints=["No vulnerabilities"]
        )
        
        report = optimizer.optimize_prompt(config)
        
        # Verify security analysis is included
        assert "security" in report.improvements
        security_analysis = report.improvements["security"]
        assert "jailbreak_rate" in security_analysis
        assert "vulnerabilities" in security_analysis
        assert "recommendations" in security_analysis
    
    def test_config_save_and_load(self):
        """Test configuration save and load functionality."""
        import tempfile
        import yaml
        
        optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
        
        config = optimizer.create_config(
            name="Save/Load Test",
            description="Testing save and load",
            system_prompt="Test prompt",
            business_goals=["Test goal"],
            target_audience="Test users",
            brand_voice="Test voice",
            use_cases=["Test case"],
            success_metrics=["Test metric"],
            constraints=["Test constraint"]
        )
        
        # Save configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            optimizer.save_config(config, f.name)
            config_file = f.name
        
        try:
            # Load configuration
            loaded_config = optimizer.load_config(config_file)
            
            # Verify loaded config matches original
            assert loaded_config.name == config.name
            assert loaded_config.description == config.description
            assert loaded_config.system_prompt == config.system_prompt
            assert loaded_config.business_goals == config.business_goals
            assert loaded_config.target_audience == config.target_audience
            assert loaded_config.brand_voice == config.brand_voice
            
        finally:
            os.unlink(config_file)


class TestMockAnalysisMethods:
    """Test the mock analysis methods work correctly."""
    
    def test_mock_effectiveness_analysis(self):
        """Test mock effectiveness analysis."""
        optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
        
        result = optimizer._mock_effectiveness_analysis()
        
        assert isinstance(result, dict)
        assert 'effectiveness_score' in result
        assert 'strengths' in result
        assert 'weaknesses' in result
        assert 'task_completion_quality' in result
        assert 'clarity_score' in result
        
        # Verify score ranges
        assert 0.0 <= result['effectiveness_score'] <= 1.0
        assert 0.0 <= result['task_completion_quality'] <= 1.0
        assert 0.0 <= result['clarity_score'] <= 1.0
        
        # Verify lists
        assert isinstance(result['strengths'], list)
        assert isinstance(result['weaknesses'], list)
        assert len(result['strengths']) > 0
        assert len(result['weaknesses']) > 0
    
    def test_mock_ux_analysis(self):
        """Test mock UX analysis."""
        optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
        
        result = optimizer._mock_ux_analysis()
        
        assert isinstance(result, dict)
        assert 'ux_score' in result
        assert 'tone_alignment' in result
        assert 'helpfulness' in result
        assert 'accessibility' in result
        assert 'engagement_quality' in result
        assert 'ux_improvements' in result
        
        # Verify score ranges
        for score_field in ['ux_score', 'tone_alignment', 'helpfulness', 'accessibility', 'engagement_quality']:
            assert 0.0 <= result[score_field] <= 1.0
        
        assert isinstance(result['ux_improvements'], list)
        assert len(result['ux_improvements']) > 0
    
    def test_mock_business_analysis(self):
        """Test mock business analysis."""
        optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
        
        result = optimizer._mock_business_analysis()
        
        assert isinstance(result, dict)
        assert 'alignment_score' in result
        assert 'goal_support' in result
        assert 'brand_consistency' in result
        assert 'constraint_compliance' in result
        assert 'business_improvements' in result
        
        # Verify score ranges
        assert 0.0 <= result['alignment_score'] <= 1.0
        assert 0.0 <= result['brand_consistency'] <= 1.0
        assert 0.0 <= result['constraint_compliance'] <= 1.0
        
        assert isinstance(result['goal_support'], dict)
        assert isinstance(result['business_improvements'], list)
    
    def test_mock_cost_analysis(self):
        """Test mock cost analysis."""
        optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
        
        result = optimizer._mock_cost_analysis()
        
        assert isinstance(result, dict)
        assert 'efficiency_score' in result
        assert 'token_efficiency' in result
        assert 'response_optimization' in result
        assert 'cost_improvements' in result
        assert 'estimated_savings' in result
        
        # Verify score ranges
        assert 0.0 <= result['efficiency_score'] <= 1.0
        assert 0.0 <= result['token_efficiency'] <= 1.0
        assert 0.0 <= result['response_optimization'] <= 1.0
        
        assert isinstance(result['cost_improvements'], list)
        assert isinstance(result['estimated_savings'], str)
    
    def test_mock_compliance_analysis(self):
        """Test mock compliance analysis."""
        optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
        
        result = optimizer._mock_compliance_analysis()
        
        assert isinstance(result, dict)
        assert 'compliance_score' in result
        assert 'ethical_alignment' in result
        assert 'regulatory_compliance' in result
        assert 'risk_areas' in result
        assert 'compliance_improvements' in result
        
        # Verify score ranges
        assert 0.0 <= result['compliance_score'] <= 1.0
        assert 0.0 <= result['ethical_alignment'] <= 1.0
        assert 0.0 <= result['regulatory_compliance'] <= 1.0
        
        assert isinstance(result['risk_areas'], list)
        assert isinstance(result['compliance_improvements'], list)
    
    def test_mock_security_analysis(self):
        """Test mock security analysis."""
        optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
        
        result = optimizer._mock_security_analysis()
        
        assert isinstance(result, dict)
        assert 'jailbreak_rate' in result
        assert 'vulnerabilities' in result
        assert 'recommendations' in result
        
        # Verify jailbreak rate range
        assert 0.0 <= result['jailbreak_rate'] <= 1.0
        
        assert isinstance(result['vulnerabilities'], list)
        assert isinstance(result['recommendations'], list)


class TestOptimizationReport:
    """Test optimization report generation and formatting."""
    
    def test_report_generation(self):
        """Test that optimization generates a complete report."""
        optimizer = HolisticPromptOptimizer(model="openai", mock_mode=True)
        
        config = optimizer.create_config(
            name="Report Test",
            description="Testing report generation",
            system_prompt="You are a helpful assistant",
            business_goals=["Be helpful"],
            target_audience="Users",
            brand_voice="Friendly",
            use_cases=["General assistance"],
            success_metrics=["User satisfaction"],
            constraints=["Be safe"]
        )
        
        report = optimizer.optimize_prompt(config)
        
        # Verify report has all required fields
        assert hasattr(report, 'config_name')
        assert hasattr(report, 'original_prompt')
        assert hasattr(report, 'optimized_prompt')
        assert hasattr(report, 'improvements')
        assert hasattr(report, 'overall_score')
        assert hasattr(report, 'recommendations')
        assert hasattr(report, 'cost_analysis')
        assert hasattr(report, 'timestamp')
        
        # Verify field types and content
        assert isinstance(report.config_name, str)
        assert isinstance(report.original_prompt, str)
        assert isinstance(report.optimized_prompt, str)
        assert isinstance(report.improvements, dict)
        assert isinstance(report.overall_score, float)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.cost_analysis, dict)
        
        # Verify content quality
        assert len(report.config_name) > 0
        assert len(report.original_prompt) > 0
        assert len(report.optimized_prompt) > 0
        assert len(report.improvements) > 0
        assert len(report.recommendations) > 0
        assert 0.0 <= report.overall_score <= 1.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"]) 