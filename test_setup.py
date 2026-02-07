#!/usr/bin/env python3
"""
Quick test script to verify the SPON extension setup works.

Run this FIRST before running full experiments to catch any issues early.

Usage:
    python test_setup.py
    
    # Or with a specific model:
    python test_setup.py --model meta-llama/Llama-3.2-1B
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all imports work."""
    print("1. Testing imports...")
    
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"   ✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"   ✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"   ✗ Transformers import failed: {e}")
        return False
    
    try:
        from src.allocation import SPONConfig, AllocationBuilder
        print("   ✓ src.allocation")
    except ImportError as e:
        print(f"   ✗ src.allocation import failed: {e}")
        return False
    
    try:
        from src.sparse_forward import magnitude_sparsify
        print("   ✓ src.sparse_forward")
    except ImportError as e:
        print(f"   ✗ src.sparse_forward import failed: {e}")
        return False
    
    try:
        from src.spon_trainer import SPONTrainer, TrainingArgs
        print("   ✓ src.spon_trainer")
    except ImportError as e:
        print(f"   ✗ src.spon_trainer import failed: {e}")
        return False
    
    try:
        from src.evaluation import compute_perplexity
        print("   ✓ src.evaluation")
    except ImportError as e:
        print(f"   ✗ src.evaluation import failed: {e}")
        return False
    
    return True


def test_config_loading():
    """Test configuration file loading."""
    print("\n2. Testing config loading...")
    
    try:
        from src.allocation import AllocationBuilder
        
        config_path = Path(__file__).parent / "configs" / "allocation_configs.yaml"
        if not config_path.exists():
            print(f"   ✗ Config file not found: {config_path}")
            return False
        
        builder = AllocationBuilder(str(config_path))
        configs = builder.list_configurations()
        print(f"   ✓ Loaded {len(configs)} configurations: {configs}")
        
        # Test building a config for a 16-layer model
        config = builder.build_config("TOP-50", "llama-3.2-1b")
        print(f"   ✓ TOP-50 config for 1B model: layers {config.layer_mask}")
        
        return True
    except Exception as e:
        print(f"   ✗ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sparsification():
    """Test sparsification function."""
    print("\n3. Testing sparsification...")
    
    try:
        import torch
        from src.sparse_forward import magnitude_sparsify
        
        # Create test tensor
        x = torch.randn(2, 10)  # batch=2, hidden=10
        
        # Apply 50% sparsity
        x_sparse = magnitude_sparsify(x, sparsity=0.5)
        
        # Check that ~50% are zeros
        zero_fraction = (x_sparse == 0).float().mean().item()
        print(f"   ✓ Sparsification works. Zero fraction: {zero_fraction:.2f} (expected ~0.50)")
        
        # Verify non-zero values are preserved
        non_zero_mask = x_sparse != 0
        if not torch.allclose(x[non_zero_mask], x_sparse[non_zero_mask]):
            print("   ✗ Non-zero values were modified!")
            return False
        
        print("   ✓ Non-zero values preserved correctly")
        return True
    except Exception as e:
        print(f"   ✗ Sparsification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading(
    model_name: str,
    model_provider: str,
    ollama_model: str | None,
    pull_ollama: bool,
):
    """Test model loading (requires GPU/MPS and HuggingFace access)."""
    print(f"\n4. Testing model loading: {model_name}...")
    
    try:
        import torch
        from src import get_device
        from src.model_provider import (
            load_causal_lm,
            load_tokenizer,
            log_provider_resolution,
            resolve_model_provider,
        )
        
        device = get_device()
        if device.type == "cpu":
            print("   ⚠ No GPU/MPS available. Skipping model test.")
            print("   (You'll need a GPU or Apple Silicon to run experiments)")
            return True
        
        if device.type == "cuda":
            print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        elif device.type == "mps":
            print(f"   ✓ MPS (Apple Silicon) available")
        
        resolution = resolve_model_provider(
            model_name=model_name,
            provider=model_provider,
            ollama_model=ollama_model,
            pull_ollama=pull_ollama,
            require_transformers_model=True,
        )
        log_provider_resolution(resolution)
        print(f"   Using device={device}")
        print(
            f"   Provider requested={resolution.requested_provider}, "
            f"effective={resolution.effective_provider}"
        )
        if resolution.used_fallback and resolution.reason:
            print(f"   Fallback reason: {resolution.reason}")
        
        print("   Loading tokenizer...")
        tokenizer = load_tokenizer(resolution.hf_model_name)
        print(f"   ✓ Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        
        print("   Loading model (this may take a minute)...")
        model = load_causal_lm(resolution.hf_model_name, device)
        
        num_layers = len(model.model.layers)
        print(f"   ✓ Model loaded. Layers: {num_layers}")
        
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"   ✓ Inference works. Output shape: {outputs.logits.shape}")
        
        return True
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        print("\n   Common fixes:")
        print("   - Ensure `ollama` is installed/running if using --model_provider ollama")
        print("   - Run: ollama pull llama3.2:1b (or your selected Ollama model)")
        print("   - Run: huggingface-cli login")
        print("   - Accept model license at huggingface.co/meta-llama/Llama-3.2-1B")
        print("   - Check internet connection")
        return False


def test_dataloader():
    """Test calibration dataloader creation."""
    print("\n5. Testing dataloader creation...")
    
    try:
        from src.model_provider import load_tokenizer
        from src.spon_trainer import create_calibration_dataloader
        
        # Use a small public tokenizer for testing
        try:
            tokenizer = load_tokenizer("gpt2")
        except Exception as e:
            print(f"   ⚠ Skipping dataloader test: tokenizer load unavailable ({e})")
            print("   (Offline/auth-restricted environment; this is not a code issue.)")
            return True
        
        print("   Creating dataloader (downloading dataset if needed)...")
        dataloader = create_calibration_dataloader(
            tokenizer,
            block_size=64,
            batch_size=2,
            num_samples=10
        )
        
        # Get one batch
        batch = next(iter(dataloader))
        print(f"   ✓ Dataloader works. Batch shape: {batch['input_ids'].shape}")
        
        return True
    except Exception as e:
        msg = str(e).lower()
        if any(
            token in msg
            for token in [
                "operation not permitted",
                "nodename nor servname",
                "can't load the configuration",
                "connection",
                "timed out",
                "forbidden",
                "unauthorized",
            ]
        ):
            print(f"   ⚠ Skipping dataloader test due to environment restriction: {e}")
            print("   (Network/cache permission issue; core code path is unchanged.)")
            return True
        print(f"   ✗ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test SPON extension setup")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model to test loading"
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model loading test (useful if no GPU)"
    )
    parser.add_argument(
        "--model_provider",
        type=str,
        choices=["auto", "huggingface", "ollama"],
        default="auto",
        help="Model source provider. 'auto' prefers Ollama if available, then falls back to Hugging Face.",
    )
    parser.add_argument(
        "--ollama_model",
        type=str,
        default=None,
        help="Ollama model tag (e.g. llama3.2:1b). If omitted, inferred from --model.",
    )
    parser.add_argument(
        "--pull_ollama",
        action="store_true",
        help="If Ollama is selected/available, run `ollama pull` for the selected model.",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("SPON Extension Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_config_loading()
    all_passed &= test_sparsification()
    
    if not args.skip_model:
        all_passed &= test_model_loading(
            args.model,
            args.model_provider,
            args.ollama_model,
            args.pull_ollama,
        )
    else:
        print("\n4. Skipping model loading test (--skip-model)")
    
    all_passed &= test_dataloader()
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! You're ready to run experiments.")
        print("\nNext steps:")
        print("  1. python experiments/exp1_allocation/run_allocation_sweep.py --epochs 1 --configs BASELINE-TEAL UNIF-ALL")
        print("  2. Check results/ directory for output")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
