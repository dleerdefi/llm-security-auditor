"""
Configuration settings for the Self-Fortifying LLM Gateway.
"""
import os
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMConfig:
    """Configuration for LLM providers and models."""
    
    # OpenAI Configuration (2025 Latest Models)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODELS: Dict[str, str] = {
        "defender": "gpt-4o",           # Latest multimodal model for robust defense
        "attacker": "gpt-4o-mini",      # Cost-effective but capable for attacks
        "reasoning": "o1-preview",      # Advanced reasoning model for complex analysis
    }
    
    # Anthropic Configuration (2025 Latest Models)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODELS: Dict[str, str] = {
        "defender": "claude-opus-4-20250514",      # Most capable Claude 4 model for defense
        "attacker": "claude-sonnet-4-20250514",    # High-performance balanced model for attacks
        "reasoning": "claude-3-7-sonnet-20250219", # Extended thinking model for analysis
    }
    
    # Default provider settings
    DEFAULT_PROVIDER: str = "openai"
    FALLBACK_PROVIDER: str = "anthropic"
    
    # Rate limiting and cost control
    MAX_TOKENS_PER_REQUEST: int = 4096
    MAX_REQUESTS_PER_MINUTE: int = 60
    MAX_COST_PER_DAY: float = 100.0  # USD
    
    @classmethod
    def validate_config(cls) -> Optional[str]:
        """Validate the configuration settings.
        
        Returns:
            Optional[str]: Error message if validation fails, None if successful
        """
        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_API_KEY:
            return "At least one API key (OpenAI or Anthropic) must be configured"
        
        if cls.DEFAULT_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            return "OpenAI is set as default provider but no API key is configured"
        
        if cls.DEFAULT_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            return "Anthropic is set as default provider but no API key is configured"
        
        return None

# Initialize configuration
config = LLMConfig() 