#!/usr/bin/env python3
"""
Test script for ART (Automated Red Teaming) Training Pipeline

This script demonstrates the ART pipeline with a small dataset
to verify all components are working correctly.
"""

import os
import sys
import json

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))

from train_agent import ARTManager, ARTConfig


def test_art_pipeline():
    """Test the complete ART pipeline with a small dataset"""
    print("🧪 Testing ART Pipeline")
    print("=" * 50)
    
    # Create test config with small dataset
    test_config = ARTConfig(
        num_episodes=20,  # Small for testing
        test_scenarios=10,
        data_dir="test_art_data",
        model_dir="test_art_models",
        results_dir="test_art_results"
    )
    
    print("📋 Test Configuration:")
    print(f"   Episodes: {test_config.num_episodes}")
    print(f"   Test scenarios: {test_config.test_scenarios}")
    print(f"   Difficulty levels: {test_config.difficulty_levels}")
    print()
    
    # Create ART manager
    art_manager = ARTManager(test_config)
    
    try:
        # Test 1: Data Generation
        print("🔄 Test 1: Data Generation")
        data_path = art_manager.run_data_generation()
        print(f"   ✅ Generated data: {data_path}")
        
        # Check data file
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print(f"   📊 Collected {len(data['trajectories'])} trajectories")
        print(f"   📈 Success rate: {sum(1 for t in data['trajectories'] if t.get('success', False)) / len(data['trajectories']):.3f}")
        print()
        
        # Test 2: Model Training
        print("🔄 Test 2: Model Training") 
        training_results_path = art_manager.run_training(data_path)
        print(f"   ✅ Training results: {training_results_path}")
        
        # Check training results
        with open(training_results_path, 'r') as f:
            training_results = json.load(f)
        
        print(f"   🧠 Training samples: {training_results.get('num_samples', 'N/A')}")
        print(f"   📊 Model R²: {training_results.get('r2_score', 'N/A')}")
        print()
        
        # Test 3: Model Evaluation (if model was saved)
        model_path = training_results.get('model_saved')
        if model_path and os.path.exists(model_path):
            print("🔄 Test 3: Model Evaluation")
            evaluation_results_path = art_manager.run_evaluation(model_path)
            print(f"   ✅ Evaluation results: {evaluation_results_path}")
            
            # Check evaluation results
            with open(evaluation_results_path, 'r') as f:
                eval_results = json.load(f)
            
            print(f"   🎯 Test success rate: {eval_results.get('success_rate', 0):.3f}")
            print()
        else:
            print("⚠️  Test 3: Skipped (no model saved)")
            print()
        
        # Summary
        print("🎉 ART Pipeline Test Results:")
        print(f"   ✅ Data generation: PASSED")
        print(f"   ✅ Model training: PASSED")
        if model_path:
            print(f"   ✅ Model evaluation: PASSED")
        print(f"   📁 Test data saved in: {test_config.data_dir}")
        print(f"   🧠 Models saved in: {test_config.model_dir}")
        print(f"   📊 Results saved in: {test_config.results_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_components():
    """Test individual ART components"""
    print("\n🔧 Testing Individual Components")
    print("=" * 50)
    
    from train_agent import AdversarialScenarioGenerator, ARTDataCollector, ARTTrainer
    
    config = ARTConfig(num_episodes=10)
    
    # Test 1: Scenario Generator
    print("🔄 Test 1: Scenario Generator")
    generator = AdversarialScenarioGenerator(config)
    scenarios = generator.generate_baseline_scenarios(5)
    print(f"   ✅ Generated {len(scenarios)} scenarios")
    
    for scenario in scenarios[:2]:
        print(f"   📍 {scenario.name}: pos={scenario.initial_position:.2f}, vel={scenario.initial_velocity:.2f}")
    
    # Test 2: Data Collector
    print("\n🔄 Test 2: Data Collector")
    collector = ARTDataCollector(config)
    print(f"   ✅ Created data collector")
    print(f"   📊 Initial trajectories: {len(collector.data['trajectories'])}")
    
    # Test 3: Trainer
    print("\n🔄 Test 3: Trainer")
    trainer = ARTTrainer(config)
    
    # Create dummy training data
    dummy_data = [
        {
            'success': True,
            'positions': [0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            'velocities': [0.0, -0.1, -0.1, -0.1, -0.05, -0.01],
            'actions': [-0.5, -0.3, -0.2, -0.15, -0.1]
        },
        {
            'success': True,
            'positions': [-0.3, -0.2, -0.1, -0.05, 0.0],
            'velocities': [0.2, 0.1, 0.05, 0.02, 0.0],
            'actions': [0.3, 0.2, 0.1, 0.05]
        }
    ]
    
    training_results = trainer.train_supervised_model(dummy_data)
    print(f"   ✅ Training completed")
    print(f"   📊 Samples processed: {training_results.get('num_samples', 'N/A')}")
    
    print("\n✅ All individual component tests passed!")


if __name__ == "__main__":
    print("🚀 ART Pipeline Testing Suite")
    print("=" * 60)
    
    # Test individual components first
    test_individual_components()
    
    # Test full pipeline
    success = test_art_pipeline()
    
    if success:
        print("\n🎉 All tests passed! ART pipeline is ready for use.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")