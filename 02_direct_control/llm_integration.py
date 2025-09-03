"""
Real LLM Integration for Direct Control

This module provides interfaces for connecting to real LLMs:
1. VLLMController - Connect to local vLLM server
2. OpenAIController - Connect to OpenAI API (backup)

Usage:
    controller = VLLMController(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    action = controller.get_action(observation, target)
"""

import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  requests not available. Install with: pip install requests")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  openai not available. Install with: pip install openai")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLMController(ABC):
    """Base class for all LLM controllers"""
    
    def __init__(self, max_retries: int = 3, timeout: float = 30.0):
        self.max_retries = max_retries
        self.timeout = timeout
        self.step_count = 0
        
    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt and return response"""
        pass
    
    def _create_observation_prompt(self, position: float, velocity: float, 
                                 target_pos: float, target_vel: float) -> str:
        """Create natural language observation for LLM"""
        pos_error = target_pos - position
        vel_error = target_vel - velocity
        distance = abs(pos_error)
        
        # Determine urgency and direction
        if distance > 0.5:
            urgency = "high"
        elif distance > 0.1:
            urgency = "medium"
        else:
            urgency = "low"
            
        direction = "right" if pos_error > 0 else "left"
        velocity_desc = "moving right" if velocity > 0.05 else "moving left" if velocity < -0.05 else "nearly stationary"
        
        prompt = f"""You are controlling a spacecraft in 1D space. Your goal is to reach the target position with zero velocity.

CURRENT STATE:
- Position: {position:.3f} m
- Velocity: {velocity:.3f} m/s
- Status: {velocity_desc}

TARGET STATE:
- Position: {target_pos:.3f} m  
- Velocity: {target_vel:.3f} m/s

ANALYSIS:
- Position error: {pos_error:.3f} m (need to move {direction})
- Velocity error: {vel_error:.3f} m/s
- Distance to target: {distance:.3f} m
- Urgency level: {urgency}

PHYSICS REMINDER:
- Your control input is FORCE (acceleration)
- Positive force accelerates right (+), negative force accelerates left (-)
- Position changes based on velocity: next_position = position + velocity * dt
- Velocity changes based on your force: next_velocity = velocity + force * dt
- dt = 0.1 seconds per step

CONTROL STRATEGY:
1. If far from target: apply force toward target
2. If moving too fast toward target: apply opposite force to brake
3. If close to target: use gentle force for fine positioning
4. Consider both position AND velocity errors

Respond with a JSON object containing:
- "force": float between -1.0 and 1.0 (your control action)
- "confidence": float between 0.0 and 1.0 (how confident you are)  
- "reasoning": string explaining your decision

Example response:
{{"force": -0.3, "confidence": 0.8, "reasoning": "Need to move left and slow down"}}"""

        return prompt
    
    def _parse_llm_response(self, response: str) -> Tuple[float, float, str]:
        """Parse LLM response and extract action, confidence, reasoning"""
        try:
            # Try to find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
                
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            force = float(data.get('force', 0.0))
            confidence = float(data.get('confidence', 0.5))
            reasoning = str(data.get('reasoning', 'No reasoning provided'))
            
            # Clamp force to valid range
            force = np.clip(force, -1.0, 1.0)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            return force, confidence, reasoning
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.warning(f"Response was: {response}")
            
            # Fallback: try to extract numbers
            try:
                # Look for force-like numbers
                import re
                force_matches = re.findall(r'[-+]?\d*\.?\d+', response)
                if force_matches:
                    force = float(force_matches[0])
                    force = np.clip(force, -1.0, 1.0)
                    return force, 0.3, "Fallback parsing"
            except:
                pass
                
            # Last resort: return safe default
            return 0.0, 0.1, "Parse error - using safe default"
    
    def get_action(self, position: float, velocity: float, 
                   target_pos: float, target_vel: float) -> Dict:
        """Get control action from LLM"""
        self.step_count += 1
        
        prompt = self._create_observation_prompt(position, velocity, target_pos, target_vel)
        
        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(prompt)
                force, confidence, reasoning = self._parse_llm_response(response)
                
                return {
                    'action': force,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'step': self.step_count,
                    'raw_response': response[:200] + '...' if len(response) > 200 else response
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return {
                        'action': 0.0,
                        'confidence': 0.0,
                        'reasoning': f"All attempts failed: {e}",
                        'step': self.step_count,
                        'raw_response': ''
                    }
                time.sleep(1.0)  # Brief pause before retry


class VLLMController(BaseLLMController):
    """Controller that connects to local vLLM server"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", 
                 port: int = 8000, host: str = "localhost", **kwargs):
        super().__init__(**kwargs)
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package required for vLLM. Install with: pip install requests")
        
        self.model_name = model_name
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/v1/completions"
        
        # Test connection
        self._test_connection()
        
    def _test_connection(self):
        """Test if vLLM server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5.0)
            if response.status_code == 200:
                logger.info(f"✅ Connected to vLLM server at {self.base_url}")
            else:
                logger.warning(f"⚠️  vLLM server responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Could not connect to vLLM server: {e}")
            logger.error(f"Make sure vLLM is running: vllm serve {self.model_name} --port {self.base_url.split(':')[-1]}")
            raise ConnectionError(f"vLLM server not available at {self.base_url}")
    
    def _call_llm(self, prompt: str) -> str:
        """Call vLLM API with prompt"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop": ["\n\n"],
        }
        
        response = requests.post(
            self.api_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"vLLM API error {response.status_code}: {response.text}")
        
        result = response.json()
        
        if 'choices' not in result or len(result['choices']) == 0:
            raise RuntimeError(f"Invalid vLLM response format: {result}")
        
        return result['choices'][0]['text'].strip()


class OpenAIController(BaseLLMController):
    """Controller that connects to OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(**kwargs)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        
        # Test connection
        self._test_connection()
        
    def _test_connection(self):
        """Test OpenAI API connection"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                timeout=5.0
            )
            logger.info(f"✅ Connected to OpenAI API with model {self.model}")
        except Exception as e:
            logger.error(f"❌ Could not connect to OpenAI API: {e}")
            raise ConnectionError(f"OpenAI API not available: {e}")
    
    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API with prompt"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a spacecraft control system. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1,
            timeout=self.timeout
        )
        
        return response.choices[0].message.content.strip()


def test_controllers():
    """Test both controller types if available"""
    print("🧪 Testing LLM Controllers...")
    print("=" * 50)
    
    # Test scenario
    position, velocity = 0.5, 0.0
    target_pos, target_vel = 0.0, 0.0
    
    controllers_to_test = []
    
    # Test vLLM if requests available
    if REQUESTS_AVAILABLE:
        try:
            vllm_controller = VLLMController()
            controllers_to_test.append(("vLLM", vllm_controller))
        except Exception as e:
            print(f"⚠️  vLLM controller not available: {e}")
    
    # Test OpenAI if available
    if OPENAI_AVAILABLE:
        try:
            openai_controller = OpenAIController()
            controllers_to_test.append(("OpenAI", openai_controller))
        except Exception as e:
            print(f"⚠️  OpenAI controller not available: {e}")
    
    # Test controllers
    for name, controller in controllers_to_test:
        print(f"\n🤖 Testing {name} Controller:")
        print("-" * 30)
        
        start_time = time.time()
        result = controller.get_action(position, velocity, target_pos, target_vel)
        end_time = time.time()
        
        print(f"Action: {result['action']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Response time: {end_time - start_time:.2f}s")
        print(f"✅ {name} controller working!")
    
    if not controllers_to_test:
        print("❌ No controllers available.")
        print("Setup instructions:")
        print("1. For vLLM: pip install requests && vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000")
        print("2. For OpenAI: pip install openai && export OPENAI_API_KEY=your_key")


if __name__ == "__main__":
    test_controllers()