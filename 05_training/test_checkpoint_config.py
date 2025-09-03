#!/usr/bin/env python3
"""
Test Checkpoint Configuration
============================

Simple test to verify that checkpoint configuration is working correctly.
"""

def test_checkpoint_config():
    """Test that checkpoint directory is configured correctly."""
    print("🧪 Testing checkpoint configuration...")
    
    # Import checkpoint config
    from checkpoint_config import setup_checkpoint_directory, get_checkpoint_dir, print_checkpoint_status
    
    # Setup directory
    checkpoint_dir = setup_checkpoint_directory()
    
    # Verify transformers configuration
    try:
        import transformers.trainer_utils as trainer_utils
        print(f"✅ Transformers checkpoint prefix: {trainer_utils.PREFIX_CHECKPOINT_DIR}")
        
        # Check if it points to scratch
        if "/scratch/" in trainer_utils.PREFIX_CHECKPOINT_DIR:
            print("✅ Checkpoint prefix correctly points to /scratch")
        else:
            print("❌ Checkpoint prefix does not point to /scratch")
            
    except ImportError:
        print("⚠️  Transformers not available for testing")
    
    # Show status
    print_checkpoint_status()
    
    print("\n🎉 Checkpoint configuration test completed!")

if __name__ == "__main__":
    test_checkpoint_config()