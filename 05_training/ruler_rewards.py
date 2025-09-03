#!/usr/bin/env python3
"""
RULER (Relative Universal LLM-Elicited Rewards) Integration for Control Systems
=============================================================================

Implements automatic reward scoring using LLM-as-judge for control trajectories,
eliminating the need for hand-crafted reward functions.

Based on ART RULER framework: https://art.openpipe.ai/fundamentals/ruler
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import asdict

# ART RULER imports
try:
    from art.rewards import ruler_score_trajectory, ruler_score_group
    RULER_AVAILABLE = True
    print("✅ ART RULER imported successfully")
except ImportError:
    print("⚠️ ART RULER not available - using fallback manual rewards")
    RULER_AVAILABLE = False

# Our imports
from art_integration import ControlScenario
from art import Trajectory, TrajectoryGroup


class ControlRulerJudge:
    """LLM judge for control system trajectories using RULER"""
    
    def __init__(self, judge_model: str = "openai/o1"):
        self.judge_model = judge_model
        self.judge_prompt = self._create_control_judge_prompt()
        
    def _create_control_judge_prompt(self) -> str:
        """Create the system prompt for the control system judge"""
        return """
You are an expert control systems judge evaluating spacecraft control performance.

TASK: Rate the quality of a control agent's performance on a double integrator system (ẍ = u).

EVALUATION CRITERIA:
1. **Success** (40 points): Did the agent reach the target position (±0.1m) with low velocity (±0.1 m/s)?
2. **Efficiency** (25 points): How quickly did the agent achieve the goal? Fewer steps = better.
3. **Control Quality** (20 points): How smooth and reasonable were the control actions? Avoid excessive oscillation.
4. **Understanding** (15 points): Does the agent demonstrate understanding of physics and control principles?

PHYSICS CONTEXT:
- System: ẍ = u (acceleration = control force)  
- Control bounds: -1.0 ≤ u ≤ 1.0 N
- Target: Reach (0, 0) position and velocity
- Success criteria: |position| ≤ 0.1m AND |velocity| ≤ 0.1m/s

SCORING:
- Excellent (90-100): Perfect control with optimal efficiency
- Good (70-89): Successful with minor inefficiencies  
- Fair (50-69): Eventually successful but with significant issues
- Poor (30-49): Unsuccessful but shows some control understanding
- Failed (0-29): No meaningful control or completely wrong approach

Provide your score (0-100) and brief reasoning focusing on control quality and physics understanding.
"""

    async def judge_trajectory(self, trajectory: Trajectory, scenario: ControlScenario) -> Dict[str, Any]:
        """Judge a single trajectory using RULER"""
        
        if not RULER_AVAILABLE:
            # Fallback to manual scoring
            return self._fallback_manual_score(trajectory, scenario)
        
        try:
            # Create judgment context
            judgment_context = self._create_judgment_context(trajectory, scenario)
            
            # Use RULER to score the trajectory
            scored_trajectory = await ruler_score_trajectory(
                trajectory, 
                judge_model=self.judge_model,
                system_prompt=self.judge_prompt,
                context=judgment_context
            )
            
            # Extract RULER score and reasoning
            ruler_score = getattr(scored_trajectory, 'ruler_score', 0.0)
            ruler_reasoning = getattr(scored_trajectory, 'ruler_reasoning', 'No reasoning provided')
            
            # Convert RULER score (0-100) to reward scale (-10 to +10)
            reward = self._convert_ruler_score_to_reward(ruler_score)
            
            return {
                'reward': reward,
                'ruler_score': ruler_score,
                'ruler_reasoning': ruler_reasoning,
                'judgment_method': 'RULER',
                'judge_model': self.judge_model
            }
            
        except Exception as e:
            print(f"⚠️ RULER judgment failed: {e}, using fallback")
            return self._fallback_manual_score(trajectory, scenario)
    
    async def judge_trajectory_group(self, group: TrajectoryGroup, scenarios: List[ControlScenario]) -> TrajectoryGroup:
        """Judge an entire trajectory group using RULER"""
        
        if not RULER_AVAILABLE:
            # Apply fallback scoring to each trajectory
            for i, trajectory in enumerate(group.trajectories):
                scenario = scenarios[i] if i < len(scenarios) else scenarios[0]
                judgment = self._fallback_manual_score(trajectory, scenario)
                trajectory.reward = judgment['reward']
            return group
        
        try:
            # Create group context
            group_context = {
                'scenarios': [asdict(s) for s in scenarios],
                'task_description': 'Double integrator spacecraft control',
                'group_size': len(group.trajectories)
            }
            
            # Use RULER to score the entire group
            scored_group = await ruler_score_group(
                group,
                judge_model=self.judge_model,
                swallow_exceptions=True  # Continue if some trajectories fail
            )
            
            return scored_group if scored_group else group
            
        except Exception as e:
            print(f"⚠️ RULER group judgment failed: {e}")
            return group
    
    def _create_judgment_context(self, trajectory: Trajectory, scenario: ControlScenario) -> Dict[str, Any]:
        """Create rich context for LLM judge"""
        
        # Extract trajectory information
        messages = trajectory.messages_and_choices
        metadata = trajectory.metadata
        
        # Find the agent's control decisions
        control_decisions = []
        reasoning_steps = []
        
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if content and '{' in content:  # Likely JSON response
                    try:
                        response = json.loads(content)
                        if 'action' in response:
                            control_decisions.append({
                                'action': response.get('action'),
                                'reasoning': response.get('reasoning', ''),
                                'confidence': response.get('confidence', 0)
                            })
                            if response.get('reasoning'):
                                reasoning_steps.append(response['reasoning'])
                    except:
                        pass
        
        return {
            'scenario': asdict(scenario),
            'performance_metrics': {
                'success': metadata.get('success', False),
                'final_position': metadata.get('final_position', 0),
                'final_velocity': metadata.get('final_velocity', 0), 
                'steps_taken': metadata.get('steps', 0),
                'control_effort': metadata.get('control_effort', 0),
                'final_pos_error': metadata.get('final_pos_error', 0),
                'final_vel_error': metadata.get('final_vel_error', 0)
            },
            'control_decisions': control_decisions,
            'reasoning_quality': reasoning_steps,
            'approach': metadata.get('approach', 'unknown')
        }
    
    def _convert_ruler_score_to_reward(self, ruler_score: float) -> float:
        """Convert RULER score (0-100) to reward scale"""
        # RULER score of 50 = neutral (0 reward)
        # RULER score of 100 = excellent (+10 reward)
        # RULER score of 0 = failed (-10 reward)
        return ((ruler_score - 50) / 50) * 10
    
    def _fallback_manual_score(self, trajectory: Trajectory, scenario: ControlScenario) -> Dict[str, Any]:
        """Fallback manual scoring when RULER is not available"""
        metadata = trajectory.metadata
        
        # Physics-based reward calculation
        success = metadata.get('success', False)
        final_pos_error = abs(metadata.get('final_position', 0))
        final_vel_error = abs(metadata.get('final_velocity', 0))
        steps = metadata.get('steps', 0)
        control_effort = metadata.get('control_effort', 0)
        
        reward = 0.0
        reasoning_parts = []
        
        # Success component (most important)
        if success:
            reward += 6.0
            reasoning_parts.append("Successfully reached target")
        else:
            reward -= 3.0
            reasoning_parts.append("Failed to reach target")
        
        # Position accuracy
        pos_penalty = min(5.0, final_pos_error * 10)
        reward -= pos_penalty
        reasoning_parts.append(f"Position error penalty: -{pos_penalty:.2f}")
        
        # Velocity accuracy  
        vel_penalty = min(3.0, final_vel_error * 6)
        reward -= vel_penalty
        reasoning_parts.append(f"Velocity error penalty: -{vel_penalty:.2f}")
        
        # Efficiency bonus
        if steps > 0:
            efficiency = max(0, scenario.max_steps - steps) / scenario.max_steps
            efficiency_bonus = efficiency * 2.0
            reward += efficiency_bonus
            reasoning_parts.append(f"Efficiency bonus: +{efficiency_bonus:.2f}")
        
        # Control effort penalty
        effort_penalty = min(1.0, control_effort * 0.1)
        reward -= effort_penalty
        reasoning_parts.append(f"Control effort penalty: -{effort_penalty:.2f}")
        
        # Difficulty bonus
        difficulty_bonuses = {"easy": 0.0, "medium": 1.0, "hard": 2.0}
        diff_bonus = difficulty_bonuses.get(scenario.difficulty, 0.0)
        reward += diff_bonus
        reasoning_parts.append(f"Difficulty bonus: +{diff_bonus:.2f}")
        
        return {
            'reward': reward,
            'ruler_score': max(0, min(100, (reward + 10) * 5)),  # Convert to 0-100 scale
            'ruler_reasoning': '; '.join(reasoning_parts),
            'judgment_method': 'Manual Fallback',
            'judge_model': 'physics-based'
        }


class ControlRewardManager:
    """Manages reward assignment for control training"""
    
    def __init__(self, 
                 use_ruler: bool = True,
                 judge_model: str = "openai/gpt-4",  # More economical than o1
                 fallback_to_manual: bool = True):
        
        self.use_ruler = use_ruler and RULER_AVAILABLE
        self.fallback_to_manual = fallback_to_manual
        
        if self.use_ruler:
            self.judge = ControlRulerJudge(judge_model)
            print(f"✅ Using RULER rewards with {judge_model}")
        else:
            self.judge = ControlRulerJudge()  # Will use manual fallback
            print("✅ Using manual physics-based rewards")
    
    async def assign_rewards_to_trajectory(self, trajectory: Trajectory, scenario: ControlScenario) -> Trajectory:
        """Assign reward to a single trajectory"""
        
        # Judge the trajectory
        judgment = await self.judge.judge_trajectory(trajectory, scenario)
        
        # Update trajectory with judged reward
        trajectory.reward = judgment['reward']
        
        # Add judgment metadata
        trajectory.metadata.update({
            'ruler_score': judgment.get('ruler_score', 0),
            'ruler_reasoning': judgment.get('ruler_reasoning', ''),
            'judgment_method': judgment.get('judgment_method', 'unknown'),
            'judge_model': judgment.get('judge_model', 'unknown')
        })
        
        return trajectory
    
    async def assign_rewards_to_group(self, group: TrajectoryGroup, scenarios: List[ControlScenario]) -> TrajectoryGroup:
        """Assign rewards to all trajectories in a group"""
        
        if self.use_ruler and len(scenarios) > 0:
            # Use RULER group scoring for better consistency
            scored_group = await self.judge.judge_trajectory_group(group, scenarios)
            return scored_group
        else:
            # Score trajectories individually
            for i, trajectory in enumerate(group.trajectories):
                scenario = scenarios[i] if i < len(scenarios) else scenarios[0]
                await self.assign_rewards_to_trajectory(trajectory, scenario)
            
            return group


# Quick test function
async def test_ruler_integration():
    """Test RULER integration with sample trajectories"""
    print("🧪 Testing RULER Integration")
    print("=" * 50)
    
    # Create mock trajectory and scenario for testing
    from art_integration import ControlScenario
    
    scenario = ControlScenario(
        name="test_scenario",
        initial_position=0.3,
        initial_velocity=0.1,
        difficulty="medium"
    )
    
    # Mock trajectory messages
    messages = [
        {"role": "system", "content": "You are a control agent..."},
        {"role": "user", "content": "Control the system from position 0.3, velocity 0.1"},
        {"role": "assistant", "content": '{"action": -0.5, "reasoning": "Need to decelerate and move toward target", "confidence": 0.8}'},
        {"role": "user", "content": "Position: 0.25, Velocity: 0.05"},
        {"role": "assistant", "content": '{"action": -0.3, "reasoning": "Continue gentle deceleration", "confidence": 0.9}'},
    ]
    
    # Mock metadata
    metadata = {
        'success': True,
        'final_position': 0.05,
        'final_velocity': 0.02,
        'steps': 8,
        'control_effort': 2.1,
        'approach': 'direct_control'
    }
    
    # Create mock trajectory
    mock_trajectory = Trajectory(
        messages_and_choices=messages,
        reward=0.0,  # Will be assigned by RULER
        metadata=metadata
    )
    
    # Test reward manager
    reward_manager = ControlRewardManager(use_ruler=True)
    
    # Test single trajectory scoring
    scored_trajectory = await reward_manager.assign_rewards_to_trajectory(mock_trajectory, scenario)
    
    print(f"📊 RULER Test Results:")
    print(f"   Reward: {scored_trajectory.reward:.3f}")
    print(f"   RULER Score: {scored_trajectory.metadata.get('ruler_score', 0):.1f}/100")
    print(f"   Method: {scored_trajectory.metadata.get('judgment_method', 'unknown')}")
    print(f"   Reasoning: {scored_trajectory.metadata.get('ruler_reasoning', 'none')[:100]}...")
    
    print("✅ RULER integration test completed")


if __name__ == "__main__":
    asyncio.run(test_ruler_integration())