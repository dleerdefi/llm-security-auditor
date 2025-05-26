"""
LLM client implementations for OpenAI and Anthropic APIs.
"""
import time
from typing import Dict, Optional, List, Any
import openai
import anthropic
from .config import LLMConfig

class BaseLLMClient:
    """Base class for LLM clients."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.request_timestamps: List[float] = []
        self.daily_cost: float = 0.0
    
    def _check_rate_limit(self) -> bool:
        """Check if the current request would exceed rate limits."""
        current_time = time.time()
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                 if current_time - ts < 60]
        return len(self.request_timestamps) < self.config.MAX_REQUESTS_PER_MINUTE
    
    def _check_cost_limit(self, estimated_cost: float) -> bool:
        """Check if the current request would exceed cost limits."""
        return self.daily_cost + estimated_cost <= self.config.MAX_COST_PER_DAY
    
    def _update_cost(self, cost: float):
        """Update the daily cost tracking."""
        self.daily_cost += cost
        # Reset daily cost at midnight
        if time.localtime().tm_hour == 0 and time.localtime().tm_min == 0:
            self.daily_cost = 0.0

class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API interactions."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        openai.api_key = config.OPENAI_API_KEY
    
    async def generate_response(
        self,
        prompt: str,
        role: str = "defender",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate a response using OpenAI's API.
        
        Args:
            prompt: The input prompt
            role: Either "defender" or "attacker"
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict containing the response and metadata
        """
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        model = self.config.OPENAI_MODELS[role]
        max_tokens = max_tokens or self.config.MAX_TOKENS_PER_REQUEST
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Update rate limiting
            self.request_timestamps.append(time.time())
            
            # Estimate and update cost (simplified)
            estimated_cost = (len(prompt.split()) + max_tokens) * 0.0001
            if not self._check_cost_limit(estimated_cost):
                raise Exception("Cost limit exceeded")
            self._update_cost(estimated_cost)
            
            return {
                "text": response.choices[0].message.content,
                "model": model,
                "usage": response.usage,
                "cost": estimated_cost
            }
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

class AnthropicClient(BaseLLMClient):
    """Client for Anthropic API interactions."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    
    async def generate_response(
        self,
        prompt: str,
        role: str = "defender",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate a response using Anthropic's API.
        
        Args:
            prompt: The input prompt
            role: Either "defender" or "attacker"
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict containing the response and metadata
        """
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        model = self.config.ANTHROPIC_MODELS[role]
        max_tokens = max_tokens or self.config.MAX_TOKENS_PER_REQUEST
        
        try:
            response = await self.client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Update rate limiting
            self.request_timestamps.append(time.time())
            
            # Estimate and update cost (simplified)
            estimated_cost = (len(prompt.split()) + max_tokens) * 0.0001
            if not self._check_cost_limit(estimated_cost):
                raise Exception("Cost limit exceeded")
            self._update_cost(estimated_cost)
            
            return {
                "text": response.content[0].text,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                "cost": estimated_cost
            }
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")

class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    @staticmethod
    def create_client(provider: str, config: LLMConfig) -> BaseLLMClient:
        """Create an LLM client for the specified provider.
        
        Args:
            provider: Either "openai" or "anthropic"
            config: LLMConfig instance
            
        Returns:
            An instance of the appropriate LLM client
        """
        if provider == "openai":
            return OpenAIClient(config)
        elif provider == "anthropic":
            return AnthropicClient(config)
        else:
            raise ValueError(f"Unknown provider: {provider}") 