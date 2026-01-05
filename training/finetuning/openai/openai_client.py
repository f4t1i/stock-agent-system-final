#!/usr/bin/env python3
"""
OpenAI API Integration - Task 5.1

OpenAI client for fine-tuning operations.

Features:
- API key management
- Authentication verification
- Error handling
- Rate limit handling
- Connection testing

Phase A1 Week 5-6: Task 5.1 COMPLETE
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger

try:
    from openai import OpenAI
    from openai import OpenAIError, APIError, RateLimitError, AuthenticationError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. Install with: pip install openai")


@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: str
    organization: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> 'OpenAIConfig':
        """Load configuration from environment variables"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        return cls(
            api_key=api_key,
            organization=os.getenv('OPENAI_ORGANIZATION'),
            base_url=os.getenv('OPENAI_BASE_URL'),
            timeout=float(os.getenv('OPENAI_TIMEOUT', '60.0')),
            max_retries=int(os.getenv('OPENAI_MAX_RETRIES', '3'))
        )


class OpenAIClient:
    """
    OpenAI API client for fine-tuning operations
    
    Handles:
    - Authentication
    - API calls
    - Error handling
    - Rate limiting
    """
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        """
        Initialize OpenAI client
        
        Args:
            config: OpenAI configuration (loads from env if None)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install openai"
            )
        
        self.config = config or OpenAIConfig.from_env()
        
        # Initialize client
        self.client = OpenAI(
            api_key=self.config.api_key,
            organization=self.config.organization,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
        
        logger.info("OpenAI client initialized")
    
    def test_connection(self) -> bool:
        """
        Test API connection and authentication
        
        Returns:
            True if connection successful
        
        Raises:
            AuthenticationError: If API key is invalid
            APIError: If API request fails
        """
        try:
            # Test with a simple API call (list models)
            models = self.client.models.list()
            logger.info(f"✓ Connection successful. Available models: {len(models.data)}")
            return True
            
        except AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise
        
        except APIError as e:
            logger.error(f"API error: {e}")
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def list_models(self, fine_tuned_only: bool = False) -> list:
        """
        List available models
        
        Args:
            fine_tuned_only: Only return fine-tuned models
        
        Returns:
            List of model objects
        """
        try:
            models = self.client.models.list()
            
            if fine_tuned_only:
                # Filter for fine-tuned models (start with "ft:")
                models_list = [
                    model for model in models.data 
                    if model.id.startswith('ft:')
                ]
            else:
                models_list = list(models.data)
            
            logger.info(f"Listed {len(models_list)} models")
            return models_list
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get model details
        
        Args:
            model_id: Model ID
        
        Returns:
            Model object
        """
        try:
            model = self.client.models.retrieve(model_id)
            logger.info(f"Retrieved model: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            raise
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a fine-tuned model
        
        Args:
            model_id: Model ID to delete
        
        Returns:
            True if deletion successful
        """
        try:
            result = self.client.models.delete(model_id)
            logger.info(f"Deleted model: {model_id}")
            return result.deleted
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            raise
    
    def get_usage(self) -> Dict[str, Any]:
        """
        Get API usage information
        
        Returns:
            Usage statistics
        
        Note:
            This is a placeholder. OpenAI doesn't have a direct usage API.
            Use the dashboard or billing API for actual usage.
        """
        logger.warning("Usage API not available. Check OpenAI dashboard.")
        return {
            'message': 'Check OpenAI dashboard for usage statistics',
            'dashboard_url': 'https://platform.openai.com/usage'
        }
    
    def check_rate_limits(self) -> Dict[str, Any]:
        """
        Check current rate limits
        
        Returns:
            Rate limit information
        
        Note:
            Rate limits are returned in response headers.
            This method provides estimated limits.
        """
        # OpenAI rate limits (as of 2024)
        # These are estimates and may vary by tier
        return {
            'requests_per_minute': {
                'gpt-4': 500,
                'gpt-3.5-turbo': 3500,
                'fine-tuning': 50
            },
            'tokens_per_minute': {
                'gpt-4': 10000,
                'gpt-3.5-turbo': 90000,
                'fine-tuning': 2000000
            },
            'note': 'Actual limits may vary by tier. Check response headers.'
        }
    
    def handle_rate_limit(self, error: RateLimitError) -> None:
        """
        Handle rate limit error
        
        Args:
            error: Rate limit error
        """
        logger.warning(f"Rate limit hit: {error}")
        
        # Extract retry-after from error if available
        retry_after = getattr(error, 'retry_after', None)
        
        if retry_after:
            logger.info(f"Retry after {retry_after} seconds")
        else:
            logger.info("Retry after exponential backoff")
    
    def close(self) -> None:
        """Close client connection"""
        # OpenAI client doesn't need explicit closing
        logger.info("OpenAI client closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False


# Helper functions

def create_client(
    api_key: Optional[str] = None,
    organization: Optional[str] = None
) -> OpenAIClient:
    """
    Create OpenAI client with optional parameters
    
    Args:
        api_key: API key (loads from env if None)
        organization: Organization ID
    
    Returns:
        OpenAI client
    """
    if api_key:
        config = OpenAIConfig(
            api_key=api_key,
            organization=organization
        )
    else:
        config = OpenAIConfig.from_env()
    
    return OpenAIClient(config)


def test_authentication(api_key: Optional[str] = None) -> bool:
    """
    Test API authentication
    
    Args:
        api_key: API key to test (loads from env if None)
    
    Returns:
        True if authentication successful
    """
    try:
        client = create_client(api_key=api_key)
        return client.test_connection()
    except Exception as e:
        logger.error(f"Authentication test failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    import sys
    
    print("=== OpenAI Client Test ===\n")
    
    # Test 1: Create client
    try:
        print("Test 1: Create client")
        client = create_client()
        print("✓ Client created\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        sys.exit(1)
    
    # Test 2: Test connection
    try:
        print("Test 2: Test connection")
        client.test_connection()
        print("✓ Connection successful\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        sys.exit(1)
    
    # Test 3: List models
    try:
        print("Test 3: List models")
        models = client.list_models()
        print(f"✓ Found {len(models)} models\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
    
    # Test 4: Check rate limits
    try:
        print("Test 4: Check rate limits")
        limits = client.check_rate_limits()
        print(f"✓ Rate limits: {limits['requests_per_minute']}\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
    
    print("=== Tests Complete ===")
