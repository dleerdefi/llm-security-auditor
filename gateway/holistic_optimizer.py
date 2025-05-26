"""
Holistic Prompt Optimizer
Advanced DSPy-powered optimization for comprehensive prompt improvement

This module optimizes prompts across multiple dimensions:
1. Security (jailbreak resistance)
2. Effectiveness (task completion quality)
3. User Experience (clarity, helpfulness)
4. Business Alignment (goals, brand voice)
5. Cost Efficiency (token usage, response length)
6. Compliance (ethics, regulations)
"""

import os
import sys
import yaml
import dspy
import mlflow
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


@dataclass
class OptimizationConfig:
    """Configuration for holistic prompt optimization."""
    name: str
    description: str
    system_prompt: str
    business_goals: List[str]
    target_audience: str
    brand_voice: str
    use_cases: List[str]
    success_metrics: List[str]
    constraints: List[str]
    compliance_requirements: Optional[List[str]] = None
    cost_budget: Optional[str] = None
    performance_targets: Optional[Dict[str, float]] = None


@dataclass
class OptimizationReport:
    """Comprehensive optimization report."""
    config_name: str
    original_prompt: str
    optimized_prompt: str
    improvements: Dict[str, Dict[str, Any]]
    overall_score: float
    recommendations: List[str]
    cost_analysis: Dict[str, Any]
    timestamp: datetime


class EffectivenessAnalyzer(dspy.Signature):
    """Analyze prompt effectiveness for specific use cases."""
    
    system_prompt: str = dspy.InputField(desc="System prompt to analyze")
    use_cases: str = dspy.InputField(desc="Specific use cases and scenarios")
    target_audience: str = dspy.InputField(desc="Target audience description")
    success_metrics: str = dspy.InputField(desc="Success metrics and KPIs")
    
    effectiveness_score: float = dspy.OutputField(desc="Effectiveness score from 0.0 to 1.0")
    strengths: List[str] = dspy.OutputField(desc="Identified strengths of the prompt")
    weaknesses: List[str] = dspy.OutputField(desc="Areas for improvement")
    task_completion_quality: float = dspy.OutputField(desc="Quality of task completion (0.0-1.0)")
    clarity_score: float = dspy.OutputField(desc="Clarity and understandability (0.0-1.0)")


class UserExperienceAnalyzer(dspy.Signature):
    """Analyze user experience aspects of the prompt."""
    
    system_prompt: str = dspy.InputField(desc="System prompt to analyze")
    target_audience: str = dspy.InputField(desc="Target audience description")
    brand_voice: str = dspy.InputField(desc="Desired brand voice and tone")
    interaction_examples: str = dspy.InputField(desc="Example interactions")
    
    ux_score: float = dspy.OutputField(desc="User experience score from 0.0 to 1.0")
    tone_alignment: float = dspy.OutputField(desc="Brand voice alignment (0.0-1.0)")
    helpfulness: float = dspy.OutputField(desc="Perceived helpfulness (0.0-1.0)")
    accessibility: float = dspy.OutputField(desc="Accessibility for target audience (0.0-1.0)")
    engagement_quality: float = dspy.OutputField(desc="Engagement and conversational quality (0.0-1.0)")
    ux_improvements: List[str] = dspy.OutputField(desc="Specific UX improvement suggestions")


class BusinessAlignmentAnalyzer(dspy.Signature):
    """Analyze business alignment and goal achievement."""
    
    system_prompt: str = dspy.InputField(desc="System prompt to analyze")
    business_goals: str = dspy.InputField(desc="Business goals and objectives")
    brand_voice: str = dspy.InputField(desc="Brand voice and positioning")
    constraints: str = dspy.InputField(desc="Business constraints and limitations")
    
    alignment_score: float = dspy.OutputField(desc="Business alignment score from 0.0 to 1.0")
    goal_support: Dict[str, float] = dspy.OutputField(desc="How well prompt supports each goal (0.0-1.0)")
    brand_consistency: float = dspy.OutputField(desc="Brand voice consistency (0.0-1.0)")
    constraint_compliance: float = dspy.OutputField(desc="Compliance with constraints (0.0-1.0)")
    business_improvements: List[str] = dspy.OutputField(desc="Business-focused improvements")


class CostEfficiencyAnalyzer(dspy.Signature):
    """Analyze cost efficiency and token optimization."""
    
    system_prompt: str = dspy.InputField(desc="System prompt to analyze")
    use_cases: str = dspy.InputField(desc="Typical use cases and interactions")
    cost_budget: str = dspy.InputField(desc="Cost budget and constraints")
    
    efficiency_score: float = dspy.OutputField(desc="Cost efficiency score from 0.0 to 1.0")
    token_efficiency: float = dspy.OutputField(desc="Token usage efficiency (0.0-1.0)")
    response_optimization: float = dspy.OutputField(desc="Response length optimization (0.0-1.0)")
    cost_improvements: List[str] = dspy.OutputField(desc="Cost optimization suggestions")
    estimated_savings: str = dspy.OutputField(desc="Estimated cost savings potential")


class ComplianceAnalyzer(dspy.Signature):
    """Analyze compliance with ethics and regulations."""
    
    system_prompt: str = dspy.InputField(desc="System prompt to analyze")
    compliance_requirements: str = dspy.InputField(desc="Compliance requirements and regulations")
    ethical_guidelines: str = dspy.InputField(desc="Ethical guidelines and standards")
    
    compliance_score: float = dspy.OutputField(desc="Compliance score from 0.0 to 1.0")
    ethical_alignment: float = dspy.OutputField(desc="Ethical alignment score (0.0-1.0)")
    regulatory_compliance: float = dspy.OutputField(desc="Regulatory compliance (0.0-1.0)")
    risk_areas: List[str] = dspy.OutputField(desc="Identified compliance risk areas")
    compliance_improvements: List[str] = dspy.OutputField(desc="Compliance improvement suggestions")


class HolisticOptimizer(dspy.Signature):
    """Optimize prompt holistically across all dimensions."""
    
    original_prompt: str = dspy.InputField(desc="Original system prompt")
    effectiveness_analysis: str = dspy.InputField(desc="Effectiveness analysis results")
    ux_analysis: str = dspy.InputField(desc="User experience analysis results")
    business_analysis: str = dspy.InputField(desc="Business alignment analysis results")
    cost_analysis: str = dspy.InputField(desc="Cost efficiency analysis results")
    compliance_analysis: str = dspy.InputField(desc="Compliance analysis results")
    security_analysis: str = dspy.InputField(desc="Security analysis results")
    optimization_priorities: str = dspy.InputField(desc="Optimization priorities and weights")
    
    optimized_prompt: str = dspy.OutputField(desc="Holistically optimized system prompt")
    optimization_strategy: str = dspy.OutputField(desc="Explanation of optimization approach")
    trade_offs: List[str] = dspy.OutputField(desc="Trade-offs made during optimization")
    improvement_summary: Dict[str, str] = dspy.OutputField(desc="Summary of improvements by dimension")
    confidence: float = dspy.OutputField(desc="Confidence in optimization quality (0.0-1.0)")


class HolisticPromptOptimizer:
    """
    Comprehensive prompt optimizer using DSPy for multi-dimensional optimization.
    
    Optimizes prompts across:
    1. Security (jailbreak resistance)
    2. Effectiveness (task completion quality)
    3. User Experience (clarity, helpfulness, engagement)
    4. Business Alignment (goals, brand voice, constraints)
    5. Cost Efficiency (token usage, response optimization)
    6. Compliance (ethics, regulations, risk management)
    """
    
    def __init__(self, model: str = "openai", mlflow_uri: Optional[str] = None, mock_mode: bool = False):
        self.model = model
        self.mlflow_uri = mlflow_uri
        self.mock_mode = mock_mode
        
        # Initialize DSPy with specified model (skip in mock mode)
        if not mock_mode:
            self._init_dspy()
            
            # Initialize DSPy modules for each optimization dimension
            self.effectiveness_analyzer = dspy.ChainOfThought(EffectivenessAnalyzer)
            self.ux_analyzer = dspy.ChainOfThought(UserExperienceAnalyzer)
            self.business_analyzer = dspy.ChainOfThought(BusinessAlignmentAnalyzer)
            self.cost_analyzer = dspy.ChainOfThought(CostEfficiencyAnalyzer)
            self.compliance_analyzer = dspy.ChainOfThought(ComplianceAnalyzer)
            self.holistic_optimizer = dspy.ChainOfThought(HolisticOptimizer)
        else:
            # Mock DSPy modules for testing
            self.effectiveness_analyzer = None
            self.ux_analyzer = None
            self.business_analyzer = None
            self.cost_analyzer = None
            self.compliance_analyzer = None
            self.holistic_optimizer = None
        
        # Initialize MLflow if URI provided
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("holistic-prompt-optimization")
        
        console.print(Panel.fit(
            f"🎯 [bold blue]Holistic Prompt Optimizer Initialized[/bold blue]\n"
            f"Model: {model}\n"
            f"Optimization Dimensions: 6 (Security, Effectiveness, UX, Business, Cost, Compliance)\n"
            f"DSPy Modules: 6 specialized analyzers + holistic optimizer",
            title="Holistic Optimizer Ready"
        ))
    
    def _init_dspy(self):
        """Initialize DSPy with the appropriate model."""
        console.print("🔧 [yellow]Initializing DSPy framework...[/yellow]")
        
        if self.model == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set")
            model_name = "gpt-4o"  # Use GPT-4o for comprehensive analysis
            lm = dspy.LM(f'openai/{model_name}', api_key=os.getenv("OPENAI_API_KEY"))
            
        elif self.model == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            lm = dspy.LM('anthropic/claude-sonnet-4-20250514', api_key=os.getenv("ANTHROPIC_API_KEY"))
            
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        
        dspy.configure(lm=lm)
        console.print("✅ [green]DSPy configured successfully[/green]")
    
    def create_config(self, name: str, description: str, system_prompt: str,
                     business_goals: List[str], target_audience: str, brand_voice: str,
                     use_cases: List[str], success_metrics: List[str], constraints: List[str],
                     compliance_requirements: Optional[List[str]] = None,
                     cost_budget: Optional[str] = None,
                     performance_targets: Optional[Dict[str, float]] = None) -> OptimizationConfig:
        """Create an optimization configuration."""
        return OptimizationConfig(
            name=name,
            description=description,
            system_prompt=system_prompt,
            business_goals=business_goals,
            target_audience=target_audience,
            brand_voice=brand_voice,
            use_cases=use_cases,
            success_metrics=success_metrics,
            constraints=constraints,
            compliance_requirements=compliance_requirements,
            cost_budget=cost_budget,
            performance_targets=performance_targets
        )
    
    def save_config(self, config: OptimizationConfig, filepath: str):
        """Save optimization configuration to YAML file."""
        config_dict = {
            'name': config.name,
            'description': config.description,
            'system_prompt': config.system_prompt,
            'business_goals': config.business_goals,
            'target_audience': config.target_audience,
            'brand_voice': config.brand_voice,
            'use_cases': config.use_cases,
            'success_metrics': config.success_metrics,
            'constraints': config.constraints,
            'compliance_requirements': config.compliance_requirements,
            'cost_budget': config.cost_budget,
            'performance_targets': config.performance_targets
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def load_config(self, filepath: str) -> OptimizationConfig:
        """Load optimization configuration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return OptimizationConfig(**data)
    
    def optimize_prompt(self, config: OptimizationConfig, 
                       optimization_priorities: Optional[Dict[str, float]] = None) -> OptimizationReport:
        """
        Perform comprehensive prompt optimization across all dimensions.
        
        Args:
            config: Optimization configuration
            optimization_priorities: Weights for each dimension (security, effectiveness, ux, business, cost, compliance)
        """
        console.print(Panel.fit(
            f"🚀 [bold green]Starting Holistic Prompt Optimization[/bold green]\n"
            f"Target: {config.name}\n"
            f"Description: {config.description}",
            title="Holistic Optimization"
        ))
        
        # Default optimization priorities if not provided
        if optimization_priorities is None:
            optimization_priorities = {
                "security": 0.20,
                "effectiveness": 0.25,
                "ux": 0.20,
                "business": 0.15,
                "cost": 0.10,
                "compliance": 0.10
            }
        
        # Start MLflow run if configured
        if self.mlflow_uri:
            mlflow.start_run()
            mlflow.log_param("config_name", config.name)
            mlflow.log_param("model", self.model)
            mlflow.log_text(config.system_prompt, "original_prompt.txt")
            mlflow.log_params(optimization_priorities)
        
        analyses = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # 1. Effectiveness Analysis
            task = progress.add_task("Analyzing effectiveness...", total=6)
            if self.mock_mode:
                analyses['effectiveness'] = self._mock_effectiveness_analysis()
            else:
                analyses['effectiveness'] = self.effectiveness_analyzer(
                    system_prompt=config.system_prompt,
                    use_cases="\n".join(config.use_cases),
                    target_audience=config.target_audience,
                    success_metrics="\n".join(config.success_metrics)
                )
            progress.advance(task)
            
            # 2. User Experience Analysis
            progress.update(task, description="Analyzing user experience...")
            if self.mock_mode:
                analyses['ux'] = self._mock_ux_analysis()
            else:
                analyses['ux'] = self.ux_analyzer(
                    system_prompt=config.system_prompt,
                    target_audience=config.target_audience,
                    brand_voice=config.brand_voice,
                    interaction_examples="Sample interactions based on use cases"
                )
            progress.advance(task)
            
            # 3. Business Alignment Analysis
            progress.update(task, description="Analyzing business alignment...")
            if self.mock_mode:
                analyses['business'] = self._mock_business_analysis()
            else:
                analyses['business'] = self.business_analyzer(
                    system_prompt=config.system_prompt,
                    business_goals="\n".join(config.business_goals),
                    brand_voice=config.brand_voice,
                    constraints="\n".join(config.constraints)
                )
            progress.advance(task)
            
            # 4. Cost Efficiency Analysis
            progress.update(task, description="Analyzing cost efficiency...")
            if self.mock_mode:
                analyses['cost'] = self._mock_cost_analysis()
            else:
                analyses['cost'] = self.cost_analyzer(
                    system_prompt=config.system_prompt,
                    use_cases="\n".join(config.use_cases),
                    cost_budget=config.cost_budget or "Standard budget constraints"
                )
            progress.advance(task)
            
            # 5. Compliance Analysis
            progress.update(task, description="Analyzing compliance...")
            if self.mock_mode:
                analyses['compliance'] = self._mock_compliance_analysis()
            else:
                analyses['compliance'] = self.compliance_analyzer(
                    system_prompt=config.system_prompt,
                    compliance_requirements="\n".join(config.compliance_requirements or ["General ethical guidelines"]),
                    ethical_guidelines="Standard AI ethics and safety guidelines"
                )
            progress.advance(task)
            
            # 6. Security Analysis (integrate with existing security auditor)
            progress.update(task, description="Analyzing security...")
            if self.mock_mode:
                analyses['security'] = self._mock_security_analysis()
            else:
                # Import and use existing security auditor
                from .security_auditor import UniversalSecurityAuditor, SecurityConfig
                
                security_auditor = UniversalSecurityAuditor(model=self.model, mock_mode=False)
                security_config = SecurityConfig(
                    name=config.name,
                    description=config.description,
                    system_prompt=config.system_prompt,
                    business_rules=config.constraints
                )
                security_report = security_auditor.audit_security(security_config)
                analyses['security'] = {
                    'jailbreak_rate': security_report.overall_jailbreak_rate,
                    'vulnerabilities': security_report.successful_attacks,
                    'recommendations': security_report.recommendations
                }
            progress.advance(task)
        
        # Perform holistic optimization
        console.print("🧠 [yellow]Performing holistic optimization with DSPy...[/yellow]")
        
        if self.mock_mode:
            optimization_result = self._mock_optimization_result(config.system_prompt)
        else:
            optimization_result = self.holistic_optimizer(
                original_prompt=config.system_prompt,
                effectiveness_analysis=str(analyses['effectiveness']),
                ux_analysis=str(analyses['ux']),
                business_analysis=str(analyses['business']),
                cost_analysis=str(analyses['cost']),
                compliance_analysis=str(analyses['compliance']),
                security_analysis=str(analyses['security']),
                optimization_priorities=str(optimization_priorities)
            )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(analyses, optimization_priorities)
        
        # Generate comprehensive report
        report = OptimizationReport(
            config_name=config.name,
            original_prompt=config.system_prompt,
            optimized_prompt=optimization_result.optimized_prompt if not self.mock_mode else optimization_result['optimized_prompt'],
            improvements=self._format_improvements(analyses),
            overall_score=overall_score,
            recommendations=self._generate_recommendations(analyses),
            cost_analysis=self._generate_cost_analysis(analyses),
            timestamp=datetime.now()
        )
        
        # Log to MLflow
        if self.mlflow_uri:
            mlflow.log_metric("overall_score", overall_score)
            mlflow.log_text(report.optimized_prompt, "optimized_prompt.txt")
            for dimension, analysis in analyses.items():
                if hasattr(analysis, 'score') or (isinstance(analysis, dict) and 'score' in analysis):
                    score = analysis.score if hasattr(analysis, 'score') else analysis.get('score', 0.0)
                    mlflow.log_metric(f"{dimension}_score", score)
            mlflow.end_run()
        
        console.print("✨ [green]Holistic optimization complete![/green]")
        return report
    
    def _mock_effectiveness_analysis(self):
        """Mock effectiveness analysis for testing."""
        return {
            'effectiveness_score': 0.75,
            'strengths': ['Clear instructions', 'Good task focus'],
            'weaknesses': ['Could be more specific', 'Missing edge case handling'],
            'task_completion_quality': 0.8,
            'clarity_score': 0.7
        }
    
    def _mock_ux_analysis(self):
        """Mock UX analysis for testing."""
        return {
            'ux_score': 0.72,
            'tone_alignment': 0.8,
            'helpfulness': 0.75,
            'accessibility': 0.7,
            'engagement_quality': 0.65,
            'ux_improvements': ['More conversational tone', 'Better error handling']
        }
    
    def _mock_business_analysis(self):
        """Mock business analysis for testing."""
        return {
            'alignment_score': 0.78,
            'goal_support': {'customer_satisfaction': 0.8, 'efficiency': 0.75},
            'brand_consistency': 0.8,
            'constraint_compliance': 0.75,
            'business_improvements': ['Stronger brand voice', 'Better goal alignment']
        }
    
    def _mock_cost_analysis(self):
        """Mock cost analysis for testing."""
        return {
            'efficiency_score': 0.68,
            'token_efficiency': 0.7,
            'response_optimization': 0.65,
            'cost_improvements': ['Reduce prompt length', 'Optimize response format'],
            'estimated_savings': '15-20% cost reduction potential'
        }
    
    def _mock_compliance_analysis(self):
        """Mock compliance analysis for testing."""
        return {
            'compliance_score': 0.85,
            'ethical_alignment': 0.9,
            'regulatory_compliance': 0.8,
            'risk_areas': ['Data privacy handling'],
            'compliance_improvements': ['Add privacy disclaimers', 'Strengthen ethical guidelines']
        }
    
    def _mock_security_analysis(self):
        """Mock security analysis for testing."""
        return {
            'jailbreak_rate': 0.15,
            'vulnerabilities': ['Role confusion attacks'],
            'recommendations': ['Add explicit refusal patterns', 'Strengthen instruction adherence']
        }
    
    def _mock_optimization_result(self, original_prompt: str):
        """Mock optimization result for testing."""
        return {
            'optimized_prompt': f"{original_prompt}\n\nAdditional security and effectiveness improvements have been applied.",
            'optimization_strategy': 'Balanced optimization across all dimensions',
            'trade_offs': ['Slightly longer prompt for better security'],
            'improvement_summary': {
                'security': 'Enhanced jailbreak resistance',
                'effectiveness': 'Improved task completion',
                'ux': 'Better user engagement',
                'business': 'Stronger goal alignment',
                'cost': 'Optimized token usage',
                'compliance': 'Enhanced ethical guidelines'
            },
            'confidence': 0.85
        }
    
    def _calculate_overall_score(self, analyses: Dict, priorities: Dict[str, float]) -> float:
        """Calculate weighted overall optimization score."""
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, weight in priorities.items():
            if dimension in analyses:
                analysis = analyses[dimension]
                if isinstance(analysis, dict):
                    # Handle different score field names
                    score = (analysis.get('effectiveness_score') or 
                            analysis.get('ux_score') or 
                            analysis.get('alignment_score') or 
                            analysis.get('efficiency_score') or 
                            analysis.get('compliance_score') or 
                            (1.0 - analysis.get('jailbreak_rate', 0.0)) or  # Security: lower jailbreak = higher score
                            0.0)
                else:
                    score = getattr(analysis, 'score', 0.0)
                
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _format_improvements(self, analyses: Dict) -> Dict[str, Dict[str, Any]]:
        """Format improvement data for the report."""
        improvements = {}
        
        for dimension, analysis in analyses.items():
            if isinstance(analysis, dict):
                improvements[dimension] = analysis
            else:
                # Convert DSPy prediction to dict
                improvements[dimension] = {
                    attr: getattr(analysis, attr) 
                    for attr in dir(analysis) 
                    if not attr.startswith('_') and not callable(getattr(analysis, attr))
                }
        
        return improvements
    
    def _generate_recommendations(self, analyses: Dict) -> List[str]:
        """Generate comprehensive recommendations from all analyses."""
        recommendations = []
        
        for dimension, analysis in analyses.items():
            if isinstance(analysis, dict):
                if 'recommendations' in analysis:
                    recommendations.extend(analysis['recommendations'])
                elif 'improvements' in analysis:
                    recommendations.extend(analysis['improvements'])
                elif f'{dimension}_improvements' in analysis:
                    recommendations.extend(analysis[f'{dimension}_improvements'])
            else:
                # Handle DSPy prediction objects
                if hasattr(analysis, 'recommendations'):
                    recommendations.extend(analysis.recommendations)
                elif hasattr(analysis, 'improvements'):
                    recommendations.extend(analysis.improvements)
        
        return recommendations
    
    def _generate_cost_analysis(self, analyses: Dict) -> Dict[str, Any]:
        """Generate cost analysis summary."""
        cost_analysis = analyses.get('cost', {})
        
        if isinstance(cost_analysis, dict):
            return {
                'efficiency_score': cost_analysis.get('efficiency_score', 0.0),
                'estimated_savings': cost_analysis.get('estimated_savings', 'Not calculated'),
                'optimization_potential': cost_analysis.get('cost_improvements', [])
            }
        else:
            return {
                'efficiency_score': getattr(cost_analysis, 'efficiency_score', 0.0),
                'estimated_savings': getattr(cost_analysis, 'estimated_savings', 'Not calculated'),
                'optimization_potential': getattr(cost_analysis, 'cost_improvements', [])
            }
    
    def print_report(self, report: OptimizationReport):
        """Print a comprehensive optimization report."""
        console.print("\n")
        
        # Header
        console.print(Panel.fit(
            f"🎯 [bold blue]HOLISTIC OPTIMIZATION REPORT: {report.config_name}[/bold blue]\n"
            f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            title="Optimization Report"
        ))
        
        # Overall score
        score_color = "green" if report.overall_score > 0.8 else "yellow" if report.overall_score > 0.6 else "red"
        console.print(f"\n🎯 [bold]Overall Optimization Score: [{score_color}]{report.overall_score:.1%}[/{score_color}][/bold]")
        
        # Dimension breakdown
        console.print("\n📊 [bold]Optimization Dimensions:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Dimension", style="cyan")
        table.add_column("Score", justify="center")
        table.add_column("Status", justify="center")
        table.add_column("Key Improvements", style="dim")
        
        for dimension, improvements in report.improvements.items():
            if isinstance(improvements, dict):
                score = (improvements.get('effectiveness_score') or 
                        improvements.get('ux_score') or 
                        improvements.get('alignment_score') or 
                        improvements.get('efficiency_score') or 
                        improvements.get('compliance_score') or 
                        (1.0 - improvements.get('jailbreak_rate', 0.0)) or
                        0.0)
            else:
                score = 0.0
            
            score_color = "green" if score > 0.8 else "yellow" if score > 0.6 else "red"
            status = "Excellent" if score > 0.8 else "Good" if score > 0.6 else "Needs Work"
            
            # Get key improvements
            key_improvements = []
            if isinstance(improvements, dict):
                for key in ['strengths', 'improvements', 'recommendations']:
                    if key in improvements and improvements[key]:
                        key_improvements.extend(improvements[key][:2])  # Top 2
            
            table.add_row(
                dimension.title(),
                f"[{score_color}]{score:.1%}[/{score_color}]",
                f"[{score_color}]{status}[/{score_color}]",
                ", ".join(key_improvements[:2]) if key_improvements else "No specific improvements"
            )
        
        console.print(table)
        
        # Cost analysis
        console.print("\n💰 [bold]Cost Analysis:[/bold]")
        cost_table = Table(show_header=True, header_style="bold green")
        cost_table.add_column("Metric", style="cyan")
        cost_table.add_column("Value", style="magenta")
        
        cost_table.add_row("Efficiency Score", f"{report.cost_analysis['efficiency_score']:.1%}")
        cost_table.add_row("Estimated Savings", report.cost_analysis['estimated_savings'])
        cost_table.add_row("Optimization Potential", str(len(report.cost_analysis['optimization_potential'])) + " improvements identified")
        
        console.print(cost_table)
        
        # Top recommendations
        if report.recommendations:
            console.print("\n💡 [bold]Top Recommendations:[/bold]")
            for i, rec in enumerate(report.recommendations[:5], 1):  # Top 5
                console.print(f"  {i}. {rec}")
        
        # Optimized prompt preview
        console.print("\n📝 [bold]Optimized Prompt Preview:[/bold]")
        preview = report.optimized_prompt[:200] + "..." if len(report.optimized_prompt) > 200 else report.optimized_prompt
        console.print(Panel(preview, title="Optimized Prompt (Preview)", border_style="green"))
        
        console.print(f"\n✨ [bold green]Optimization complete! Overall improvement: {report.overall_score:.1%}[/bold green]") 