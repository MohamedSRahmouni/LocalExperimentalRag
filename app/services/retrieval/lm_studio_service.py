"""
LM Studio Service
Handles communication with LM Studio local API
"""

import logging
import requests
from typing import Optional, Dict, Any, Iterator
import json

logger = logging.getLogger(__name__)


class LMStudioConfig:
    """Configuration for LM Studio"""
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        model: str = "qwen2.5-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout


class LMStudioService:
    """
    Service to interact with LM Studio local API
    Compatible with OpenAI-style API
    """
    
    def __init__(self, config: Optional[LMStudioConfig] = None):
        """
        Initialize LM Studio service
        
        Args:
            config: LM Studio configuration
        """
        self.config = config or LMStudioConfig()
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to LM Studio"""
        try:
            response = requests.get(
                f"{self.config.base_url}/models",
                timeout=5
            )
            
            if response.status_code == 200:
                models = response.json()
                logger.info("✅ LM Studio connected successfully")
                logger.info(f"   Base URL: {self.config.base_url}")
                
                if 'data' in models and len(models['data']) > 0:
                    available_models = [m['id'] for m in models['data']]
                    logger.info(f"   Available models: {available_models}")
                    
                    # Auto-select first model if using default
                    if self.config.model == "local-model" and available_models:
                        self.config.model = available_models[0]
                        logger.info(f"   Auto-selected model: {self.config.model}")
                
                return True
            else:
                logger.warning(f"⚠️  LM Studio responded with status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to LM Studio")
            logger.error(f"   Make sure LM Studio is running on {self.config.base_url}")
            logger.error("   In LM Studio: Go to 'Developer' tab and start the server")
            return False
        except Exception as e:
            logger.error(f"❌ Error connecting to LM Studio: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if LM Studio is available"""
        return self._test_connection()
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate a response from LM Studio
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Temperature override
            max_tokens: Max tokens override
            
        Returns:
            Generated text or None if failed
        """
        try:
            # Build messages
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Request payload
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "stream": False
            }
            
            logger.info(f"🤖 Generating response with {self.config.model}...")
            logger.debug(f"Prompt length: {len(prompt)} chars")
            
            # Call API
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract generated text
                if 'choices' in result and len(result['choices']) > 0:
                    generated_text = result['choices'][0]['message']['content']
                    
                    # Log token usage if available
                    if 'usage' in result:
                        usage = result['usage']
                        logger.info(f"✅ Generated successfully")
                        logger.info(f"   Tokens: {usage.get('total_tokens', 'N/A')}")
                        logger.info(f"   Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                        logger.info(f"   Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                    
                    return generated_text
                else:
                    logger.error("❌ Unexpected response format from LM Studio")
                    return None
            else:
                logger.error(f"❌ LM Studio API error: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("❌ Request to LM Studio timed out")
            return None
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        """
        Generate a streaming response from LM Studio
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Temperature override
            max_tokens: Max tokens override
            
        Yields:
            Text chunks as they're generated
        """
        try:
            # Build messages
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Request payload
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "stream": True  # Enable streaming
            }
            
            logger.info(f"🤖 Starting streaming generation with {self.config.model}...")
            
            # Call API with streaming
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                stream=True,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        
                        # Skip empty lines and comments
                        if not line_text.strip() or line_text.startswith(':'):
                            continue
                        
                        # Remove "data: " prefix
                        if line_text.startswith('data: '):
                            line_text = line_text[6:]
                        
                        # Check for end of stream
                        if line_text.strip() == '[DONE]':
                            break
                        
                        try:
                            # Parse JSON
                            chunk = json.loads(line_text)
                            
                            # Extract content
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
                
                logger.info("✅ Streaming completed")
            else:
                logger.error(f"❌ Streaming error: {response.status_code}")
                yield f"Error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"❌ Error in streaming: {e}")
            yield f"Error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            response = requests.get(
                f"{self.config.base_url}/models",
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}