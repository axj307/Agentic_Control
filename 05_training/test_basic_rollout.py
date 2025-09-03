#!/usr/bin/env python3
"""
Test Basic ART Rollout
=====================

Simple test to verify our ART integration works for basic trajectory generation
without full training pipeline.
"""

import asyncio
import sys
import os

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_basic_physics'))

from art_integration import ARTControlTrainer, ControlScenario, direct_control_rollout


async def test_basic_rollout():
    """Test basic rollout functionality"""
    print("🧪 Testing Basic ART Rollout")
    print("=" * 50)
    
    # Create trainer and initialize model
    trainer = ARTControlTrainer("test-control-agent")
    
    print("🔄 Initializing ART model...")
    await trainer.initialize_model()
    
    # Create simple test scenario
    test_scenario = ControlScenario(
        name="simple_test",
        initial_position=0.3,
        initial_velocity=0.1,
        target_position=0.0,
        target_velocity=0.0,
        max_steps=20,
        difficulty="easy"
    )
    
    print(f"🎯 Testing scenario: {test_scenario}")
    
    # Run single rollout
    try:
        print("🔄 Running single trajectory...")
        trajectory = await direct_control_rollout(trainer.model, test_scenario)
        
        # Display results
        print("\n📊 Rollout Results:")
        print(f"   Reward: {trajectory.reward:.3f}")
        print(f"   Success: {trajectory.metadata.get('success', False)}")
        print(f"   Steps: {trajectory.metadata.get('steps', 0)}")
        print(f"   Final position: {trajectory.metadata.get('final_position', 0):.3f}")
        print(f"   Final velocity: {trajectory.metadata.get('final_velocity', 0):.3f}")
        print(f"   Control effort: {trajectory.metadata.get('control_effort', 0):.3f}")
        
        # Check trajectory structure
        print(f"   Message pairs: {len(trajectory.messages_and_choices) // 2}")
        print(f"   Metadata keys: {list(trajectory.metadata.keys())}")
        
        # Sample a few messages
        print("\n💬 Sample Conversation:")
        for i, msg in enumerate(trajectory.messages_and_choices[:6]):  # First 3 pairs
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:100] + "..." if len(msg.get('content', '')) > 100 else msg.get('content', '')
            print(f"   {i+1}. {role.upper()}: {content}")
        
        print("\n✅ Basic rollout test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Rollout test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_scenario_variety():
    """Test different scenario types"""
    print("\n🔄 Testing Multiple Scenarios")
    print("=" * 50)
    
    trainer = ARTControlTrainer("test-variety-agent")
    await trainer.initialize_model()
    
    # Test scenarios of different difficulties
    scenarios = [
        ControlScenario("easy_close", 0.1, 0.05, difficulty="easy", max_steps=15),
        ControlScenario("med_distance", 0.5, 0.2, difficulty="medium", max_steps=25),
        ControlScenario("hard_far", 0.8, -0.4, difficulty="hard", max_steps=35),
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n🎯 Testing: {scenario}")
        try:
            trajectory = await direct_control_rollout(trainer.model, scenario)
            results.append({
                'scenario': scenario.name,
                'success': trajectory.metadata.get('success', False),
                'reward': trajectory.reward,
                'steps': trajectory.metadata.get('steps', 0),
                'final_error': abs(trajectory.metadata.get('final_position', 0))
            })
            print(f"   ✅ Success: {trajectory.metadata.get('success', False)}, "
                  f"Reward: {trajectory.reward:.3f}, "
                  f"Steps: {trajectory.metadata.get('steps', 0)}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'scenario': scenario.name,
                'success': False,
                'reward': -10.0,
                'error': str(e)
            })
    
    # Summary
    print("\n📊 Variety Test Summary:")
    successful = sum(1 for r in results if r.get('success', False))
    print(f"   Successful trajectories: {successful}/{len(scenarios)}")
    avg_reward = sum(r.get('reward', 0) for r in results) / len(results)
    print(f"   Average reward: {avg_reward:.3f}")
    
    return successful > 0  # At least one should work


if __name__ == "__main__":
    async def main():
        print("🚀 ART Basic Rollout Test Suite")
        print("=" * 60)
        
        # Test 1: Basic rollout
        success1 = await test_basic_rollout()
        
        # Test 2: Multiple scenarios
        success2 = await test_scenario_variety()
        
        # Final result
        if success1 and success2:
            print("\n🎉 All basic rollout tests PASSED!")
            print("✅ Ready to proceed to training pipeline implementation")
        else:
            print("\n❌ Some tests FAILED - check configuration")
            
    asyncio.run(main())