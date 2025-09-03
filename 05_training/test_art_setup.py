#!/usr/bin/env python3
"""
Test ART Integration Setup
=========================

Quick test to verify ART integration is working before running full training.
"""

import asyncio
import sys
import os

# Test ART import
print("🔍 Testing ART Integration Setup")
print("=" * 40)

try:
    from art_integration import ARTControlTrainer, ART_AVAILABLE, ControlScenario
    print("✅ art_integration module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import art_integration: {e}")
    sys.exit(1)

# Test ART library
if ART_AVAILABLE:
    print("✅ ART library is available")
    try:
        import art
        print(f"✅ ART version: {getattr(art, '__version__', 'unknown')}")
    except Exception as e:
        print(f"⚠️ ART import issue: {e}")
else:
    print("⚠️ ART library not available - will use simulation mode")

# Test double integrator import
try:
    sys.path.append('../01_basic_physics')
    from double_integrator import DoubleIntegrator
    print("✅ DoubleIntegrator imported successfully")
except ImportError as e:
    print(f"❌ Failed to import DoubleIntegrator: {e}")

async def test_trainer_initialization():
    """Test trainer initialization"""
    print("\n🧪 Testing Trainer Initialization")
    print("-" * 40)
    
    try:
        trainer = ARTControlTrainer("test-model")
        print("✅ ARTControlTrainer created")
        
        await trainer.initialize_model()
        print("✅ Model initialized")
        
        scenarios = trainer.create_training_scenarios()
        print(f"✅ Created {len(scenarios)} training scenarios")
        
        # Show first few scenarios
        print("\n📋 Sample Scenarios:")
        for i, scenario in enumerate(scenarios[:3]):
            print(f"  {i+1}. {scenario}")
        
        return True
    except Exception as e:
        print(f"❌ Trainer test failed: {e}")
        return False

async def test_rollout_functions():
    """Test rollout functions"""
    print("\n🎮 Testing Rollout Functions")
    print("-" * 40)
    
    try:
        from art_integration import direct_control_rollout, tool_augmented_rollout
        print("✅ Rollout functions imported")
        
        # Create test trainer and scenario
        trainer = ARTControlTrainer("test-rollout")
        await trainer.initialize_model()
        
        test_scenario = ControlScenario(
            name="test_scenario",
            initial_position=0.5,
            initial_velocity=0.0,
            difficulty="easy"
        )
        
        print(f"🎯 Testing with scenario: {test_scenario}")
        
        # Test direct control rollout
        print("  Testing direct control rollout...", end=" ")
        direct_trajectory = await direct_control_rollout(trainer.model, test_scenario)
        print(f"✅ Reward: {direct_trajectory.reward:.2f}")
        
        # Test tool-augmented rollout
        print("  Testing tool-augmented rollout...", end=" ")
        tool_trajectory = await tool_augmented_rollout(trainer.model, test_scenario)
        print(f"✅ Reward: {tool_trajectory.reward:.2f}")
        
        # Compare results
        print(f"\n📊 Quick Comparison:")
        print(f"  Direct Control:     Reward={direct_trajectory.reward:.2f}, Success={direct_trajectory.metadata['success']}")
        print(f"  Tool-Augmented:     Reward={tool_trajectory.reward:.2f}, Success={tool_trajectory.metadata['success']}")
        
        return True
    except Exception as e:
        print(f"❌ Rollout test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🚀 ART Integration Test Suite")
    print("=" * 40)
    
    all_passed = True
    
    # Test 1: Trainer initialization
    trainer_ok = await test_trainer_initialization()
    all_passed = all_passed and trainer_ok
    
    # Test 2: Rollout functions
    rollout_ok = await test_rollout_functions()
    all_passed = all_passed and rollout_ok
    
    # Summary
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Ready to run full training.")
        print("\nNext steps:")
        print("1. Start vLLM server: vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000")
        print("2. Run training: python run_training.py")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("Make sure all dependencies are installed and paths are correct.")

if __name__ == "__main__":
    asyncio.run(main())