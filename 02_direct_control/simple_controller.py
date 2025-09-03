#!/usr/bin/env python3
"""
Simple Direct LLM Controller
============================

Implement a basic LLM-based controller that directly outputs control actions
based on natural language observations of the system state.

This demonstrates the "direct control" approach where the LLM must learn
control policy without any tools or physics knowledge.
"""

import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import time

# Add parent directory to path to import double integrator
sys.path.append('../01_basic_physics')
from double_integrator import DoubleIntegrator, EASY_SCENARIOS, MEDIUM_SCENARIOS


@dataclass
class ControlResponse:
    """Structured response from LLM controller"""
    action: float
    reasoning: str
    confidence: float
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ControlResponse':
        """Parse JSON response from LLM"""
        try:
            data = json.loads(json_str.strip())
            return cls(
                action=float(data.get('action', 0.0)),
                reasoning=str(data.get('reasoning', 'No reasoning provided')),
                confidence=float(data.get('confidence', 0.5))
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback for malformed responses
            return cls(action=0.0, reasoning=f"Parse error: {e}", confidence=0.0)


class MockLLMController:
    """
    Mock LLM controller for testing without actual LLM inference.
    
    This mimics what an LLM might do - tries to learn PD control
    but with some noise and imperfection.
    """
    
    def __init__(self, kp: float = 0.8, kd: float = 1.5, noise_level: float = 0.1):
        self.kp = kp  # Less optimal than true PD controller
        self.kd = kd  # to simulate LLM imperfection
        self.noise_level = noise_level
        self.call_count = 0
        
    def get_system_prompt(self) -> str:
        """System prompt for the LLM controller"""
        return """You are a control system agent for a double integrator system.

SYSTEM DYNAMICS:
- The system follows: acceleration = control_force
- State: [position, velocity] 
- Control: force/acceleration input (limited to ±1.0)
- Goal: Reach target position with zero velocity

CONTROL TASK:
You will observe the current state and target, then output a control action.

OUTPUT FORMAT (JSON only):
{
  "action": <control_force_number>,
  "reasoning": "<your_reasoning_about_the_control_decision>", 
  "confidence": <confidence_0_to_1>
}

CONTROL PRINCIPLES:
- If position error is large, apply force toward target
- If velocity is high, apply damping force
- Balance position correction with velocity damping
- Stay within force limits ±1.0
- Minimize overshoot and oscillation"""

    def generate_response(self, observation: str) -> ControlResponse:
        """Generate control response (mock LLM behavior)"""
        self.call_count += 1
        
        # Parse observation to extract state
        # Example: "Position: 0.500 m, Velocity: 0.200 m/s, Target: 0.000 m"
        try:
            # Simple parsing (in real implementation, LLM would do this)
            parts = observation.split(',')
            pos_str = parts[0].split(':')[1].strip().split()[0]
            vel_str = parts[1].split(':')[1].strip().split()[0]
            target_str = parts[2].split(':')[1].strip().split()[0]
            
            position = float(pos_str)
            velocity = float(vel_str)
            target = float(target_str)
            
        except (IndexError, ValueError):
            # Fallback if parsing fails
            return ControlResponse(
                action=0.0,
                reasoning="Failed to parse observation",
                confidence=0.0
            )
        
        # Mock "LLM reasoning" - imperfect PD control with noise
        pos_error = target - position
        vel_error = 0.0 - velocity  # Target velocity is 0
        
        # Base PD control with noise
        base_control = self.kp * pos_error + self.kd * vel_error
        noise = np.random.normal(0, self.noise_level)
        control = base_control + noise
        
        # Clip to limits
        control = np.clip(control, -1.0, 1.0)
        
        # Generate reasoning (simulate what LLM might think)
        reasoning_templates = [
            f"Position error is {pos_error:.3f}, velocity is {velocity:.3f}. Need {'positive' if control > 0 else 'negative'} force.",
            f"System is {'far from' if abs(pos_error) > 0.5 else 'close to'} target. Applying {'strong' if abs(control) > 0.5 else 'gentle'} control.",
            f"Velocity {'opposes' if pos_error * velocity < 0 else 'aids'} position correction. Adjusting control accordingly.",
            f"Large {'position' if abs(pos_error) > abs(velocity) else 'velocity'} error dominates. Focus on {'positioning' if abs(pos_error) > abs(velocity) else 'damping'}."
        ]
        
        reasoning = np.random.choice(reasoning_templates)
        
        # Confidence based on error magnitude (smaller error = higher confidence)
        total_error = abs(pos_error) + abs(vel_error)
        confidence = max(0.1, 1.0 - total_error)
        
        return ControlResponse(
            action=control,
            reasoning=reasoning,
            confidence=confidence
        )


class DirectLLMController:
    """
    Direct LLM controller interface.
    
    In a real implementation, this would connect to vLLM, OpenAI API, 
    or local model inference. For now, uses mock responses.
    """
    
    def __init__(self, use_mock: bool = True, model_name: str = "mock"):
        self.use_mock = use_mock
        self.model_name = model_name
        
        if use_mock:
            self.llm = MockLLMController()
            print(f"🤖 Using mock LLM controller (simulates imperfect learning)")
        else:
            print(f"🚫 Real LLM integration not implemented yet")
            print("   Use use_mock=True for testing")
            raise NotImplementedError("Real LLM integration coming in next phase")
    
    def control_step(self, position: float, velocity: float, target_position: float, target_velocity: float = 0.0) -> ControlResponse:
        """Execute one control step"""
        
        # Create natural language observation
        observation = f"Position: {position:.3f} m, Velocity: {velocity:.3f} m/s, Target: {target_position:.3f} m"
        
        # Get response from LLM
        response = self.llm.generate_response(observation)
        
        return response
    
    def run_episode(self, env: DoubleIntegrator, scenario: Dict, max_steps: int = 50, verbose: bool = False) -> Dict:
        """Run a complete control episode"""
        
        # Reset environment
        initial_pos, initial_vel = scenario['initial_state']
        target_pos, target_vel = scenario['target_state']
        
        env.reset(initial_pos, initial_vel)
        
        if verbose:
            print(f"\n🎯 Starting episode: {scenario['description']}")
            print(f"Initial: {env.get_state_string()}")
            print(f"Target: Position: {target_pos:.3f} m, Velocity: {target_vel:.3f} m/s")
        
        # Episode history
        responses = []
        success = False
        
        for step in range(max_steps):
            # Get current state
            pos, vel = env.get_state()
            
            # Get control action from LLM
            response = self.control_step(pos, vel, target_pos, target_vel)
            responses.append(response)
            
            if verbose and step % 10 == 0:
                print(f"Step {step}: {env.get_state_string()}")
                print(f"  LLM Action: {response.action:.3f}, Confidence: {response.confidence:.3f}")
                print(f"  Reasoning: {response.reasoning}")
            
            # Apply control to environment
            env.step(response.action)
            
            # Check for success
            if env.is_at_target(target_pos, target_vel, pos_tol=0.1, vel_tol=0.1):
                success = True
                if verbose:
                    print(f"✅ Target reached in {step+1} steps!")
                break
        
        # Calculate performance metrics
        final_pos, final_vel = env.get_state()
        pos_error = abs(final_pos - target_pos)
        vel_error = abs(final_vel - target_vel)
        control_effort = sum(abs(r.action) for r in responses)
        avg_confidence = np.mean([r.confidence for r in responses])
        
        results = {
            'success': success,
            'steps': len(responses),
            'final_pos_error': pos_error,
            'final_vel_error': vel_error,
            'control_effort': control_effort,
            'avg_confidence': avg_confidence,
            'responses': responses,
            'trajectory': {
                'time': env.history['time'].copy(),
                'position': env.history['position'].copy(),
                'velocity': env.history['velocity'].copy(),
                'control': env.history['control'].copy()
            }
        }
        
        if verbose:
            print(f"Final: {env.get_state_string()}")
            print(f"Success: {success}, Steps: {len(responses)}")
            print(f"Pos Error: {pos_error:.3f}, Vel Error: {vel_error:.3f}")
            print(f"Control Effort: {control_effort:.2f}, Avg Confidence: {avg_confidence:.3f}")
        
        return results


def test_mock_controller():
    """Test the mock LLM controller"""
    print("🧪 Testing Mock LLM Controller...")
    
    controller = DirectLLMController(use_mock=True)
    env = DoubleIntegrator(max_force=1.0, dt=0.1)
    
    # Test on easy scenarios
    scenarios = EASY_SCENARIOS[:3]  # Test first 3 scenarios
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {i+1} ---")
        result = controller.run_episode(env, scenario, verbose=True)
        results.append(result)
    
    # Summary statistics
    success_rate = sum(r['success'] for r in results) / len(results)
    avg_steps = np.mean([r['steps'] for r in results])
    avg_effort = np.mean([r['control_effort'] for r in results])
    avg_confidence = np.mean([r['avg_confidence'] for r in results])
    
    print(f"\n📊 Mock LLM Results Summary:")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Control Effort: {avg_effort:.2f}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    return results


def compare_with_pd_baseline():
    """Compare mock LLM performance with PD controller baseline"""
    print("\n⚡ Comparing Mock LLM vs PD Controller...")
    
    # Test scenarios
    test_scenarios = EASY_SCENARIOS + MEDIUM_SCENARIOS[:2]
    
    # Test mock LLM
    llm_controller = DirectLLMController(use_mock=True)
    llm_results = []
    
    for scenario in test_scenarios:
        env = DoubleIntegrator(max_force=1.0, dt=0.1)
        result = llm_controller.run_episode(env, scenario, verbose=False)
        llm_results.append(result)
    
    # Test PD controller
    pd_results = []
    
    for scenario in test_scenarios:
        env = DoubleIntegrator(max_force=1.0, dt=0.1)
        
        # Reset environment
        initial_pos, initial_vel = scenario['initial_state']
        target_pos, target_vel = scenario['target_state']
        env.reset(initial_pos, initial_vel)
        
        # Run PD controller
        for step in range(50):
            control = env.pd_controller(target_pos, target_vel, kp=1.0, kd=2.0)
            env.step(control)
            
            if env.is_at_target(target_pos, target_vel, pos_tol=0.1, vel_tol=0.1):
                break
        
        # Calculate metrics
        final_pos, final_vel = env.get_state()
        pos_error = abs(final_pos - target_pos)
        vel_error = abs(final_vel - target_vel)
        control_effort = sum(abs(u) for u in env.history['control'])
        
        pd_results.append({
            'success': pos_error < 0.1 and vel_error < 0.1,
            'steps': len(env.history['control']),
            'final_pos_error': pos_error,
            'final_vel_error': vel_error,
            'control_effort': control_effort
        })
    
    # Compare results
    print("\n📈 Performance Comparison:")
    print("=" * 50)
    
    llm_success = sum(r['success'] for r in llm_results) / len(llm_results)
    pd_success = sum(r['success'] for r in pd_results) / len(pd_results)
    
    llm_steps = np.mean([r['steps'] for r in llm_results])
    pd_steps = np.mean([r['steps'] for r in pd_results])
    
    llm_effort = np.mean([r['control_effort'] for r in llm_results])
    pd_effort = np.mean([r['control_effort'] for r in pd_results])
    
    print(f"Success Rate:")
    print(f"  Mock LLM:     {llm_success:.1%}")
    print(f"  PD Control:   {pd_success:.1%}")
    print(f"  Winner:       {'Mock LLM' if llm_success > pd_success else 'PD Control'}")
    
    print(f"\nAverage Steps to Target:")
    print(f"  Mock LLM:     {llm_steps:.1f}")
    print(f"  PD Control:   {pd_steps:.1f}")
    print(f"  Winner:       {'Mock LLM' if llm_steps < pd_steps else 'PD Control'}")
    
    print(f"\nAverage Control Effort:")
    print(f"  Mock LLM:     {llm_effort:.2f}")
    print(f"  PD Control:   {pd_effort:.2f}")
    print(f"  Winner:       {'Mock LLM' if llm_effort < pd_effort else 'PD Control'}")
    
    return llm_results, pd_results


def main():
    """Main testing function"""
    print("🚀 Testing Direct LLM Controller")
    print("=" * 50)
    
    try:
        # Test mock controller
        mock_results = test_mock_controller()
        
        # Compare with baseline
        llm_results, pd_results = compare_with_pd_baseline()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📋 Key Findings:")
        print("1. Mock LLM controller works but is imperfect compared to PD")
        print("2. LLM adds noise and suboptimal control decisions")
        print("3. Natural language interface enables interpretable control")
        print("4. Ready to implement real LLM integration")
        
        print("\n📋 Next Steps:")
        print("1. Connect to real LLM (vLLM, OpenAI API)")
        print("2. Test with different model sizes and prompts")
        print("3. Move to 03_langgraph_tools/ for tool-augmented control")
        
        return mock_results, llm_results, pd_results
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    results = main()