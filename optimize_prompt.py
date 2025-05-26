#!/usr/bin/env python3
"""
Holistic Prompt Optimizer CLI
Comprehensive prompt optimization across multiple dimensions

Usage:
    python optimize_prompt.py --prompt "Your system prompt" --goals "Goal 1" "Goal 2" --audience "Target audience"
    python optimize_prompt.py --config path/to/config.yaml
    python optimize_prompt.py --interactive
"""

import os
import sys
import click
import yaml
from pathlib import Path
from typing import List, Dict

# Add gateway to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gateway.holistic_optimizer import HolisticPromptOptimizer, OptimizationConfig


@click.group()
def cli():
    """Holistic Prompt Optimizer - Optimize your prompts across multiple dimensions."""
    pass


@cli.command()
@click.option("--prompt", required=True, help="Your system prompt to optimize")
@click.option("--goals", multiple=True, help="Business goals (can specify multiple)")
@click.option("--audience", required=True, help="Target audience description")
@click.option("--voice", required=True, help="Brand voice and tone")
@click.option("--use-cases", multiple=True, help="Use cases and scenarios")
@click.option("--metrics", multiple=True, help="Success metrics and KPIs")
@click.option("--constraints", multiple=True, help="Business constraints")
@click.option("--compliance", multiple=True, help="Compliance requirements")
@click.option("--budget", help="Cost budget constraints")
@click.option("--name", default="Custom Prompt", help="Name for this optimization")
@click.option("--description", default="Custom prompt optimization", help="Description")
@click.option("--save", help="Save configuration to file")
@click.option("--model", default="openai", help="Model to use (openai, anthropic)")
@click.option("--mock", is_flag=True, help="Run in mock mode (no API calls)")
@click.option("--priorities", help="JSON string of optimization priorities (e.g., '{\"security\": 0.3, \"effectiveness\": 0.4}')")
def optimize(prompt: str, goals: tuple, audience: str, voice: str, use_cases: tuple,
            metrics: tuple, constraints: tuple, compliance: tuple, budget: str,
            name: str, description: str, save: str, model: str, mock: bool, priorities: str):
    """Optimize a prompt holistically across multiple dimensions."""
    
    # Check for API key (skip in mock mode)
    if not mock:
        if model == "openai" and not os.getenv("OPENAI_API_KEY"):
            click.echo("❌ Error: OPENAI_API_KEY not set")
            click.echo("Set your OpenAI API key in environment or .env file")
            sys.exit(1)
        
        if model == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            click.echo("❌ Error: ANTHROPIC_API_KEY not set") 
            click.echo("Set your Anthropic API key in environment or .env file")
            sys.exit(1)
    
    # Create optimizer
    if mock:
        click.echo("🎭 Running in MOCK mode - no API calls will be made")
        # Set dummy API key for mock mode
        os.environ["OPENAI_API_KEY"] = "mock-key"
    
    optimizer = HolisticPromptOptimizer(model=model, mock_mode=mock)
    
    # Parse optimization priorities if provided
    optimization_priorities = None
    if priorities:
        try:
            import json
            optimization_priorities = json.loads(priorities)
        except json.JSONDecodeError:
            click.echo("❌ Error: Invalid JSON format for priorities")
            sys.exit(1)
    
    # Create configuration
    config = optimizer.create_config(
        name=name,
        description=description,
        system_prompt=prompt,
        business_goals=list(goals) if goals else ["Improve user satisfaction"],
        target_audience=audience,
        brand_voice=voice,
        use_cases=list(use_cases) if use_cases else ["General assistance"],
        success_metrics=list(metrics) if metrics else ["User satisfaction", "Task completion"],
        constraints=list(constraints) if constraints else ["Maintain safety"],
        compliance_requirements=list(compliance) if compliance else None,
        cost_budget=budget
    )
    
    # Save config if requested
    if save:
        optimizer.save_config(config, save)
        click.echo(f"💾 Configuration saved to {save}")
    
    # Run optimization
    click.echo("🚀 Starting holistic prompt optimization...")
    report = optimizer.optimize_prompt(config, optimization_priorities)
    
    # Print report
    optimizer.print_report(report)
    
    # Save optimized prompt
    optimized_file = f"prompts/{name.lower().replace(' ', '_')}_optimized.txt"
    Path(optimized_file).parent.mkdir(exist_ok=True)
    Path(optimized_file).write_text(report.optimized_prompt)
    click.echo(f"💾 Optimized prompt saved to {optimized_file}")
    
    # Save full report
    report_file = f"results/{name.lower().replace(' ', '_')}_optimization_report.yaml"
    Path(report_file).parent.mkdir(exist_ok=True)
    report_data = {
        'config_name': report.config_name,
        'timestamp': report.timestamp.isoformat(),
        'overall_score': report.overall_score,
        'original_prompt': report.original_prompt,
        'optimized_prompt': report.optimized_prompt,
        'improvements': report.improvements,
        'recommendations': report.recommendations,
        'cost_analysis': report.cost_analysis
    }
    with open(report_file, 'w') as f:
        yaml.dump(report_data, f, default_flow_style=False)
    click.echo(f"📊 Full report saved to {report_file}")


@cli.command()
@click.option("--config", required=True, help="Path to configuration YAML file")
@click.option("--model", default="openai", help="Model to use (openai, anthropic)")
@click.option("--mock", is_flag=True, help="Run in mock mode (no API calls)")
@click.option("--priorities", help="JSON string of optimization priorities")
def optimize_config(config: str, model: str, mock: bool, priorities: str):
    """Optimize using a saved configuration file."""
    
    if not Path(config).exists():
        click.echo(f"❌ Error: Configuration file {config} not found")
        sys.exit(1)
    
    # Create optimizer
    if mock:
        click.echo("🎭 Running in MOCK mode - no API calls will be made")
        # Set dummy API key for mock mode
        os.environ["OPENAI_API_KEY"] = "mock-key"
    
    optimizer = HolisticPromptOptimizer(model=model, mock_mode=mock)
    
    # Parse optimization priorities if provided
    optimization_priorities = None
    if priorities:
        try:
            import json
            optimization_priorities = json.loads(priorities)
        except json.JSONDecodeError:
            click.echo("❌ Error: Invalid JSON format for priorities")
            sys.exit(1)
    
    # Load configuration
    optimization_config = optimizer.load_config(config)
    
    # Run optimization
    click.echo("🚀 Starting holistic prompt optimization...")
    report = optimizer.optimize_prompt(optimization_config, optimization_priorities)
    
    # Print report
    optimizer.print_report(report)
    
    # Save results
    config_name = optimization_config.name.lower().replace(' ', '_')
    
    # Save optimized prompt
    optimized_file = f"prompts/{config_name}_optimized.txt"
    Path(optimized_file).parent.mkdir(exist_ok=True)
    Path(optimized_file).write_text(report.optimized_prompt)
    click.echo(f"💾 Optimized prompt saved to {optimized_file}")


@cli.command()
@click.option("--model", default="openai", help="Model to use (openai, anthropic)")
def interactive(model: str):
    """Interactive mode - guided holistic prompt optimization."""
    
    click.echo("🎯 Welcome to the Interactive Holistic Prompt Optimizer!")
    click.echo("=" * 70)
    
    # Get basic info
    name = click.prompt("📝 What's the name of your application/prompt?")
    description = click.prompt("📄 Brief description")
    
    # Get system prompt
    click.echo("\n📋 Enter your system prompt (press Ctrl+D when done):")
    prompt_lines = []
    try:
        while True:
            line = input()
            prompt_lines.append(line)
    except EOFError:
        pass
    
    system_prompt = "\n".join(prompt_lines)
    
    if not system_prompt.strip():
        click.echo("❌ Error: System prompt cannot be empty")
        sys.exit(1)
    
    # Get business goals
    click.echo("\n🎯 Enter business goals (one per line, empty line to finish):")
    business_goals = []
    while True:
        goal = click.prompt("Goal", default="", show_default=False)
        if not goal:
            break
        business_goals.append(goal)
    
    if not business_goals:
        business_goals = ["Improve user satisfaction"]
    
    # Get target audience
    target_audience = click.prompt("\n👥 Describe your target audience")
    
    # Get brand voice
    brand_voice = click.prompt("🎨 Describe your brand voice and tone")
    
    # Get use cases
    click.echo("\n💼 Enter use cases/scenarios (one per line, empty line to finish):")
    use_cases = []
    while True:
        use_case = click.prompt("Use case", default="", show_default=False)
        if not use_case:
            break
        use_cases.append(use_case)
    
    if not use_cases:
        use_cases = ["General assistance"]
    
    # Get success metrics
    click.echo("\n📊 Enter success metrics/KPIs (one per line, empty line to finish):")
    success_metrics = []
    while True:
        metric = click.prompt("Metric", default="", show_default=False)
        if not metric:
            break
        success_metrics.append(metric)
    
    if not success_metrics:
        success_metrics = ["User satisfaction", "Task completion"]
    
    # Get constraints
    click.echo("\n🔒 Enter business constraints (one per line, empty line to finish):")
    constraints = []
    while True:
        constraint = click.prompt("Constraint", default="", show_default=False)
        if not constraint:
            break
        constraints.append(constraint)
    
    if not constraints:
        constraints = ["Maintain safety and ethics"]
    
    # Get compliance requirements
    click.echo("\n⚖️ Enter compliance requirements (one per line, empty line to finish):")
    compliance_requirements = []
    while True:
        requirement = click.prompt("Requirement", default="", show_default=False)
        if not requirement:
            break
        compliance_requirements.append(requirement)
    
    # Get cost budget
    cost_budget = click.prompt("\n💰 Cost budget constraints (optional)", default="", show_default=False)
    
    # Get optimization priorities
    click.echo("\n⚖️ Set optimization priorities (0.0-1.0, must sum to 1.0):")
    click.echo("Default: Security=0.2, Effectiveness=0.25, UX=0.2, Business=0.15, Cost=0.1, Compliance=0.1")
    
    use_custom_priorities = click.confirm("Use custom priorities?")
    optimization_priorities = None
    
    if use_custom_priorities:
        priorities = {}
        dimensions = ["security", "effectiveness", "ux", "business", "cost", "compliance"]
        
        for dim in dimensions:
            while True:
                try:
                    weight = float(click.prompt(f"{dim.title()} priority (0.0-1.0)"))
                    if 0.0 <= weight <= 1.0:
                        priorities[dim] = weight
                        break
                    else:
                        click.echo("Please enter a value between 0.0 and 1.0")
                except ValueError:
                    click.echo("Please enter a valid number")
        
        # Normalize to sum to 1.0
        total = sum(priorities.values())
        if total > 0:
            optimization_priorities = {k: v/total for k, v in priorities.items()}
            click.echo(f"Normalized priorities: {optimization_priorities}")
    
    # Create optimizer and config
    optimizer = HolisticPromptOptimizer(model=model)
    config = optimizer.create_config(
        name=name,
        description=description,
        system_prompt=system_prompt,
        business_goals=business_goals,
        target_audience=target_audience,
        brand_voice=brand_voice,
        use_cases=use_cases,
        success_metrics=success_metrics,
        constraints=constraints,
        compliance_requirements=compliance_requirements if compliance_requirements else None,
        cost_budget=cost_budget if cost_budget else None
    )
    
    # Ask to save config
    if click.confirm("\n💾 Save this configuration for future use?"):
        filename = f"configs/holistic_{name.lower().replace(' ', '_')}.yaml"
        optimizer.save_config(config, filename)
        click.echo(f"✅ Configuration saved to {filename}")
    
    # Run optimization
    click.echo("\n🚀 Starting holistic prompt optimization...")
    report = optimizer.optimize_prompt(config, optimization_priorities)
    
    # Print report
    optimizer.print_report(report)
    
    # Save results
    config_name = name.lower().replace(' ', '_')
    
    # Save optimized prompt
    optimized_file = f"prompts/{config_name}_optimized.txt"
    Path(optimized_file).parent.mkdir(exist_ok=True)
    Path(optimized_file).write_text(report.optimized_prompt)
    click.echo(f"💾 Optimized prompt saved to {optimized_file}")
    
    # Save full report
    report_file = f"results/{config_name}_optimization_report.yaml"
    Path(report_file).parent.mkdir(exist_ok=True)
    report_data = {
        'config_name': report.config_name,
        'timestamp': report.timestamp.isoformat(),
        'overall_score': report.overall_score,
        'original_prompt': report.original_prompt,
        'optimized_prompt': report.optimized_prompt,
        'improvements': report.improvements,
        'recommendations': report.recommendations,
        'cost_analysis': report.cost_analysis
    }
    with open(report_file, 'w') as f:
        yaml.dump(report_data, f, default_flow_style=False)
    click.echo(f"📊 Full report saved to {report_file}")


@cli.command()
def examples():
    """Show example usage and configurations."""
    
    click.echo("🎯 Example Usage:")
    click.echo("=" * 50)
    
    click.echo("\n1. Quick optimization:")
    click.echo('   python optimize_prompt.py optimize \\')
    click.echo('     --prompt "You are a helpful assistant" \\')
    click.echo('     --goals "Increase user satisfaction" "Reduce support tickets" \\')
    click.echo('     --audience "Small business owners" \\')
    click.echo('     --voice "Professional but friendly"')
    
    click.echo("\n2. Interactive mode:")
    click.echo("   python optimize_prompt.py interactive")
    
    click.echo("\n3. Use saved configuration:")
    click.echo("   python optimize_prompt.py optimize-config --config configs/my_app.yaml")
    
    click.echo("\n4. Custom optimization priorities:")
    click.echo('   python optimize_prompt.py optimize \\')
    click.echo('     --prompt "..." \\')
    click.echo('     --priorities \'{"security": 0.4, "effectiveness": 0.3, "ux": 0.3}\'')
    
    click.echo("\n📋 Example Configuration File (YAML):")
    click.echo("-" * 30)
    
    example_config = {
        "name": "Customer Support Bot",
        "description": "AI assistant for customer support",
        "system_prompt": "You are a helpful customer support assistant. Provide accurate information about our products and services.",
        "business_goals": [
            "Increase customer satisfaction",
            "Reduce support ticket volume",
            "Improve first-contact resolution"
        ],
        "target_audience": "Existing customers seeking support",
        "brand_voice": "Professional, empathetic, and solution-focused",
        "use_cases": [
            "Product information requests",
            "Troubleshooting assistance",
            "Account management help"
        ],
        "success_metrics": [
            "Customer satisfaction score > 4.5/5",
            "First-contact resolution rate > 80%",
            "Average response time < 30 seconds"
        ],
        "constraints": [
            "Never share customer personal information",
            "Only discuss company products and services",
            "Escalate complex issues to human agents"
        ],
        "compliance_requirements": [
            "GDPR compliance for EU customers",
            "PCI DSS for payment-related queries"
        ],
        "cost_budget": "Optimize for cost efficiency while maintaining quality"
    }
    
    click.echo(yaml.dump(example_config, default_flow_style=False))
    
    click.echo("\n🎯 Optimization Dimensions:")
    click.echo("-" * 30)
    click.echo("• Security: Jailbreak resistance, safety measures")
    click.echo("• Effectiveness: Task completion quality, accuracy")
    click.echo("• User Experience: Clarity, helpfulness, engagement")
    click.echo("• Business Alignment: Goal support, brand consistency")
    click.echo("• Cost Efficiency: Token usage, response optimization")
    click.echo("• Compliance: Ethics, regulations, risk management")


@cli.command()
@click.option("--original", required=True, help="Path to original prompt file")
@click.option("--optimized", required=True, help="Path to optimized prompt file")
def compare(original: str, optimized: str):
    """Compare original and optimized prompts."""
    
    if not Path(original).exists():
        click.echo(f"❌ Error: Original prompt file {original} not found")
        sys.exit(1)
    
    if not Path(optimized).exists():
        click.echo(f"❌ Error: Optimized prompt file {optimized} not found")
        sys.exit(1)
    
    original_text = Path(original).read_text()
    optimized_text = Path(optimized).read_text()
    
    click.echo("📊 Prompt Comparison:")
    click.echo("=" * 50)
    
    click.echo(f"\n📄 Original ({len(original_text)} chars):")
    click.echo("-" * 30)
    click.echo(original_text[:200] + "..." if len(original_text) > 200 else original_text)
    
    click.echo(f"\n✨ Optimized ({len(optimized_text)} chars):")
    click.echo("-" * 30)
    click.echo(optimized_text[:200] + "..." if len(optimized_text) > 200 else optimized_text)
    
    # Basic metrics
    length_change = len(optimized_text) - len(original_text)
    length_change_pct = (length_change / len(original_text)) * 100 if len(original_text) > 0 else 0
    
    click.echo(f"\n📈 Changes:")
    click.echo(f"• Length change: {length_change:+d} chars ({length_change_pct:+.1f}%)")
    
    # Word count
    original_words = len(original_text.split())
    optimized_words = len(optimized_text.split())
    word_change = optimized_words - original_words
    word_change_pct = (word_change / original_words) * 100 if original_words > 0 else 0
    
    click.echo(f"• Word count change: {word_change:+d} words ({word_change_pct:+.1f}%)")


if __name__ == "__main__":
    cli() 