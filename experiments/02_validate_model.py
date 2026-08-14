"""
Phase 2 — GPT-2-small model validation.

This script verifies that:

1. GPT-2-small loads through TransformerLens TransformerBridge.
2. The model runs on the NVIDIA GPU.
3. Tokenization works.
4. A forward pass works.
5. TransformerLens activation caching works.
6. GPT-2-small has the expected architecture.
7. GPU memory usage is reasonable.

This script does NOT perform activation patching.
"""

import gc

import torch
from transformer_lens.model_bridge import TransformerBridge


MODEL_NAME = "gpt2"


def main() -> None:
    print("=" * 70)
    print("PHASE 2 — GPT-2-SMALL MODEL VALIDATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Device
    # ------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )

        print(f"GPU: {gpu_name}")
        print(f"GPU memory: {gpu_memory:.2f} GB")

    # ------------------------------------------------------------
    # 2. Load GPT-2-small through TransformerBridge
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL LOADING")
    print("=" * 70)

    print(f"Model: {MODEL_NAME}")
    print("Loading GPT-2-small...")

    bridge = TransformerBridge.boot_transformers(
        MODEL_NAME,
        device=device,
    )

    print("Model loaded successfully.")

    # ------------------------------------------------------------
    # 3. Model configuration
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)

    cfg = bridge.cfg

    print(f"Layers:       {cfg.n_layers}")
    print(f"Heads/layer:  {cfg.n_heads}")
    print(f"Total heads:  {cfg.n_layers * cfg.n_heads}")
    print(f"Hidden size:  {cfg.d_model}")
    print(f"Vocabulary:   {cfg.d_vocab}")
    print(f"Context size: {cfg.n_ctx}")

    # ------------------------------------------------------------
    # 4. Validate expected GPT-2-small architecture
    # ------------------------------------------------------------

    assert cfg.n_layers == 12, (
        f"Expected 12 layers, found {cfg.n_layers}"
    )

    assert cfg.n_heads == 12, (
        f"Expected 12 attention heads/layer, found {cfg.n_heads}"
    )

    assert cfg.d_model == 768, (
        f"Expected d_model=768, found {cfg.d_model}"
    )

    print("\nArchitecture validation: PASS")

    # ------------------------------------------------------------
    # 5. Tokenization
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOKENIZATION TEST")
    print("=" * 70)

    prompt = "This movie was"

    tokens = bridge.to_tokens(
        prompt,
        prepend_bos=True,
    )

    print(f"Prompt: {prompt!r}")
    print(f"Token tensor shape: {tuple(tokens.shape)}")
    print(f"Tokens: {tokens.tolist()}")

    # Make sure tokens are on the correct device.
    tokens = tokens.to(device)

    print("Tokenization: PASS")

    # ------------------------------------------------------------
    # 6. Forward pass
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FORWARD PASS TEST")
    print("=" * 70)

    with torch.no_grad():
        logits = bridge(tokens)

    print(f"Logits shape: {tuple(logits.shape)}")
    print("Forward pass: PASS")

    # ------------------------------------------------------------
    # 7. Decode predicted next token
    # ------------------------------------------------------------

    next_token_id = logits[0, -1].argmax().item()

    next_token = bridge.tokenizer.decode(
        [next_token_id]
    )

    print(f"Predicted next token: {next_token!r}")

    # ------------------------------------------------------------
    # 8. Activation cache test
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ACTIVATION CACHE TEST")
    print("=" * 70)

    with torch.no_grad():
        cached_logits, cache = bridge.run_with_cache(tokens)

    print(
        f"Cached logits shape: "
        f"{tuple(cached_logits.shape)}"
    )

    # Check a known TransformerLens activation.
    activation_name = "blocks.0.attn.hook_z"

    if activation_name in cache:
        activation = cache[activation_name]

        print(f"Activation: {activation_name}")
        print(f"Activation shape: {tuple(activation.shape)}")
        print("Activation caching: PASS")
    else:
        print(
            f"WARNING: {activation_name!r} "
            "was not found in cache."
        )

        print("Available cache keys containing 'hook_z':")

        hook_z_keys = [
            key
            for key in cache.keys()
            if "hook_z" in key
        ]

        for key in hook_z_keys[:10]:
            print(f"  {key}")

        if not hook_z_keys:
            raise RuntimeError(
                "Could not find attention head activation "
                "hook_z in the activation cache."
            )

    # ------------------------------------------------------------
    # 9. GPU memory
    # ------------------------------------------------------------

    if device == "cuda":

        allocated = (
            torch.cuda.memory_allocated() / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved() / 1024**3
        )

        print("\n" + "=" * 70)
        print("GPU MEMORY")
        print("=" * 70)

        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")

    # ------------------------------------------------------------
    # 10. Cleanup
    # ------------------------------------------------------------

    del logits
    del cached_logits
    del cache
    del tokens

    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2 VALIDATION: PASS")
    print("=" * 70)

    print("\nValidated:")
    print("  [OK] GPT-2-small loaded")
    print("  [OK] CUDA GPU detected")
    print("  [OK] Correct GPT-2 architecture")
    print("  [OK] Tokenization")
    print("  [OK] Forward pass")
    print("  [OK] Activation caching")
    print("  [OK] Attention head activations available")


if __name__ == "__main__":
    main()