#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_embed_entry_point.py — Verify embedding dequant against fixture.

This is the test that was MISSING from per_op_gates: it calls
bpd_embed_lookup_q8_0_cpu with the actual GGUF weight bytes
and compares against the fixture embedding.

Per-op gates test each kernel with FIXTURE inputs. But the embedding
is the FIRST op — its input is raw GGUF bytes, not fixture data.
If the dequant is wrong, per-op gates can't catch it because they
never test the dequant against the raw weights.

This test catches the 4.5x scale error in bpd_dequant_q8_0_cpu.
"""
import sys, os, numpy as np, ctypes

sys.path.insert(0, "bench")
sys.path.insert(0, "tests/correctness")

c_float_p = ctypes.POINTER(ctypes.c_float)
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
c_int32_p = ctypes.POINTER(ctypes.c_int)


def test_embed_entry_point(so_path, gguf_path, fixture_dir):
    """Call bpd_embed_lookup_q8_0_cpu with GGUF weights, compare against fixture."""

    from llama_fixture_loader import load_manifest

    lib = ctypes.CDLL(so_path)
    lib.bpd_embed_lookup_q8_0_cpu.argtypes = [
        c_uint8_p, c_int32_p, c_float_p, ctypes.c_int, ctypes.c_int]
    lib.bpd_embed_lookup_q8_0_cpu.restype = None

    # Load fixture embedding (the ground truth from ggml)
    tensors = load_manifest(fixture_dir)
    by_idx = {t.idx: t for t in tensors}

    # idx 3 = second GET_ROWS = the real embedding lookup
    fix_embd = np.ascontiguousarray(by_idx[3].as_numpy(), dtype=np.float32).flatten()
    n_tokens = by_idx[3].ne[1]  # ne = [embed_dim, n_tokens, 1, 1]
    embed_dim = by_idx[3].ne[0]

    print(f"Fixture embedding: {n_tokens} tokens × {embed_dim} dims")
    print(f"  range: [{fix_embd.min():.4f}, {fix_embd.max():.4f}]")

    # Get the token IDs from the fixture (idx 1 = leaf with token IDs)
    tok_tensor = by_idx[1]
    tok_data = np.frombuffer(tok_tensor.data, dtype=np.int32)
    print(f"  tokens: {tok_data}")

    # Load the token_embd weight from the GGUF
    # Use the MmapWeightLoader from the orchestrator
    from bpd_llamatov_infer import GgufWeightLoader

    # Query tensor offset via Prolog
    import tempfile, subprocess
    uid = os.getuid()
    script_path = f"/tmp/_query_embd_{uid}.pl"
    with open(script_path, "w") as f:
        f.write(f"""
:- use_module('../lib/llamatov_loader').
main :-
    gguf_load_metadata('{gguf_path}', Meta),
    member(tensor_info('token_embd.weight', Offset, Size, _Type, _Dims), Meta),
    format('~w ~w~n', [Offset, Size]),
    halt.
:- main.
""")

    result = subprocess.run(
        ["swipl", "-q", "-g", f"consult('{script_path}'), main"],
        capture_output=True, text=True, timeout=10, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        # Fallback: load via the orchestrator's build_model
        print("  Prolog query failed, using orchestrator loader...")
        sys.argv = ["", "--gguf", gguf_path, "--so", so_path, "--tokens", "4,1", "--n-generate", "0"]
        from bpd_llamatov_infer import build_model, BpdLlamaConfig
        cfg, weights, loader = build_model(gguf_path)

        # Call embedding lookup
        our_embd = np.zeros(n_tokens * embed_dim, dtype=np.float32)
        lib.bpd_embed_lookup_q8_0_cpu(
            weights.token_embd,
            tok_data.ctypes.data_as(c_int32_p),
            our_embd.ctypes.data_as(c_float_p),
            n_tokens, embed_dim)

        loader.close()
    else:
        print("  Prolog query succeeded")
        # TODO: load weight bytes directly
        our_embd = np.zeros(1)  # placeholder

    print(f"\nOur embedding:")
    print(f"  range: [{our_embd.min():.4f}, {our_embd.max():.4f}]")

    # Compare
    if our_embd.shape == fix_embd.shape:
        max_diff = np.abs(our_embd - fix_embd).max()
        ratio = fix_embd.max() / our_embd.max() if our_embd.max() != 0 else float('inf')

        our_bits = our_embd.view(np.int32).astype(np.int64)
        fix_bits = fix_embd.view(np.int32).astype(np.int64)
        diffs = np.abs(our_bits - fix_bits)
        max_ulp = int(diffs.max())
        n_diffs = int((diffs > 0).sum())

        print(f"\n=== EMBEDDING ENTRY POINT TEST ===")
        print(f"  max_abs_diff: {max_diff:.4e}")
        print(f"  max_ulp:      {max_ulp}")
        print(f"  n_diffs:      {n_diffs}/{len(fix_embd)}")
        print(f"  scale_ratio:  {ratio:.2f}x")

        if max_ulp == 0:
            print(f"  ✅ PASS — embedding matches fixture at 0 ULP")
            return True
        elif max_diff < 1e-6:
            print(f"  ✅ PASS — embedding matches fixture within 1e-6")
            return True
        else:
            print(f"  ❌ FAIL — embedding diverges from fixture")
            print(f"         Our dequant produces values {ratio:.1f}x smaller than ggml's")
            print(f"         This is the ROOT CAUSE of end-to-end divergence.")
            print(f"         Every downstream op inherits this error.")
            print()
            print(f"  First 5 values:")
            print(f"    ours:    {our_embd[:5]}")
            print(f"    fixture: {fix_embd[:5]}")
            print(f"    ratio:   {fix_embd[:5] / (our_embd[:5] + 1e-30)}")
            return False
    else:
        print(f"  ❌ Shape mismatch: ours={our_embd.shape} fixture={fix_embd.shape}")
        return False


if __name__ == "__main__":
    so = sys.argv[1] if len(sys.argv) > 1 else "build/bpd_cpu.so"
    gguf = sys.argv[2] if len(sys.argv) > 2 else "/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
    fixtures = sys.argv[3] if len(sys.argv) > 3 else "/tmp/llama_dump_hello_8_v2"

    passed = test_embed_entry_point(so, gguf, fixtures)
    sys.exit(0 if passed else 1)
