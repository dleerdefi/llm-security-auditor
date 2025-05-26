#!/usr/bin/env python3
"""
Quick Demo Script for LLM Security Auditor
Run this to test the tool with your own prompts quickly.
"""

import os
import sys
from pathlib import Path

# Add gateway to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🛡️  LLM Security Auditor - Quick Demo")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: No API key found!")
        print("Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("\nExample:")
        print("export OPENAI_API_KEY='your-key-here'")
        print("python demo.py")
        sys.exit(1)
    
    print("✅ API key found!")
    
    # Get user input
    print("\n📝 Enter your system prompt to test:")
    print("(Press Ctrl+D when done, or Ctrl+C to exit)")
    
    prompt_lines = []
    try:
        while True:
            line = input()
            prompt_lines.append(line)
    except EOFError:
        pass
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    
    system_prompt = "\n".join(prompt_lines).strip()
    
    if not system_prompt:
        print("❌ Error: Empty prompt. Please try again.")
        sys.exit(1)
    
    print(f"\n🎯 Testing prompt: {system_prompt[:100]}...")
    
    # Import and run auditor
    try:
        from gateway.security_auditor import UniversalSecurityAuditor
        
        # Create auditor
        model = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"
        auditor = UniversalSecurityAuditor(model=model)
        
        # Create quick config
        config = auditor.create_config_from_prompt(
            name="Demo Test",
            description="Quick demo test",
            system_prompt=system_prompt,
            business_rules=["Follow all safety guidelines"],
            custom_attacks=None
        )
        
        # Run audit
        print("\n🚀 Running security audit...")
        report = auditor.audit_security(config)
        
        # Print results
        auditor.print_report(report)
        
        # Suggest next steps
        print("\n🎉 Demo complete!")
        print("\n📖 Next steps:")
        print("1. Try the interactive mode: python audit_prompt.py interactive")
        print("2. Test the examples: python audit_prompt.py audit-config --config configs/customer_support_bot.yaml")
        print("3. Read the full guide: SETUP.md")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    main() 