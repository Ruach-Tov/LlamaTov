# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
ieee754_hex.py — Canonical IEEE 754 hex float representation.

All four backends use this format for human-readable float output.
Eliminates printf-style representation ambiguity entirely.

Format: 8 hex chars for F32, 16 hex chars for F64.
Example: 0.1f → "3dcccccd"
"""
import struct


def f32_to_hex(val: float) -> str:
    """F32 float → 8-char canonical hex string."""
    return struct.pack('<f', val).hex()


def hex_to_f32(h: str) -> float:
    """8-char hex string → F32 float."""
    return struct.unpack('<f', bytes.fromhex(h))[0]


def f64_to_hex(val: float) -> str:
    """F64 double → 16-char canonical hex string."""
    return struct.pack('<d', val).hex()


def hex_to_f64(h: str) -> float:
    """16-char hex string → F64 double."""
    return struct.unpack('<d', bytes.fromhex(h))[0]


def f32_array_to_hex(arr) -> list:
    """NumPy F32 array → list of hex strings."""
    import numpy as np
    flat = arr.astype(np.float32).flatten()
    return [struct.pack('<f', v).hex() for v in flat]


def hex_to_f32_array(hexes: list):
    """List of hex strings → NumPy F32 array."""
    import numpy as np
    return np.array([struct.unpack('<f', bytes.fromhex(h))[0] for h in hexes], dtype=np.float32)


def diff_hex(a_hex: str, b_hex: str) -> dict:
    """Compare two hex floats, report bit-level difference."""
    a_bits = int(a_hex, 16)
    b_bits = int(b_hex, 16)
    xor = a_bits ^ b_bits
    a_val = hex_to_f32(a_hex)
    b_val = hex_to_f32(b_hex)
    return {
        'a_hex': a_hex, 'b_hex': b_hex,
        'a_val': a_val, 'b_val': b_val,
        'xor': f"{xor:08x}",
        'bits_differ': bin(xor).count('1'),
        'abs_diff': abs(a_val - b_val),
    }


def dump_comparison(name, gpu_arr, cpu_arr, max_show=5):
    """Print canonical hex comparison of two F32 arrays."""
    import numpy as np
    gpu = gpu_arr.astype(np.float32).flatten()
    cpu = cpu_arr.astype(np.float32).flatten()
    diff = np.abs(gpu - cpu)
    worst_indices = np.argsort(-diff)[:max_show]
    
    print(f"  {name}: {len(gpu)} elements, max_diff={diff.max():.2e}")
    if diff.max() > 0:
        for i in worst_indices:
            g_hex = struct.pack('<f', float(gpu[i])).hex()
            c_hex = struct.pack('<f', float(cpu[i])).hex()
            xor = int(g_hex, 16) ^ int(c_hex, 16)
            print(f"    [{i:4d}] gpu={g_hex} cpu={c_hex} xor={xor:08x} "
                  f"({float(gpu[i]):+.8e} vs {float(cpu[i]):+.8e})")
    else:
        print(f"    BIT-IDENTICAL")


# C equivalent (paste into .cu or .c files):
C_HEADER = """
// ieee754_hex.h — Canonical hex float output
#include <stdint.h>
#include <string.h>
#include <stdio.h>

static inline void f32_to_hex(float val, char out[9]) {
    uint32_t bits;
    memcpy(&bits, &val, 4);
    snprintf(out, 9, "%08x", bits);
}

static inline float hex_to_f32(const char hex[9]) {
    uint32_t bits;
    sscanf(hex, "%08x", &bits);
    float val;
    memcpy(&val, &bits, 4);
    return val;
}

static inline void dump_f32_hex(const char *name, const float *arr, int n, int max_show) {
    printf("  %s: %d elements\\n", name, n);
    for (int i = 0; i < n && i < max_show; i++) {
        char hex[9];
        f32_to_hex(arr[i], hex);
        printf("    [%4d] %s (%+.8e)\\n", i, hex, arr[i]);
    }
}
"""

# Rust equivalent:
RUST_SNIPPET = """
// ieee754_hex.rs — Canonical hex float output
fn f32_to_hex(val: f32) -> String {
    format!("{:08x}", val.to_bits())
}

fn hex_to_f32(hex: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(hex, 16).unwrap())
}
"""

# Prolog equivalent:
PROLOG_SNIPPET = """
%% ieee754_hex.pl — Canonical hex float output
:- use_module(library(format)).

f32_to_hex(Val, Hex) :-
    Val32 is float(Val),
    float_to_bits(Val32, Bits),
    format(atom(Hex), '~`0t~16r~8|', [Bits]).

%% float_to_bits/2 requires library(ieee754) or foreign predicate
"""


if __name__ == "__main__":
    # Demo
    import numpy as np
    
    print("=== IEEE 754 Canonical Hex Demo ===\n")
    
    test_vals = [0.0, 1.0, -1.0, 0.1, 3.14159, float('inf'), float('-inf')]
    print("  Value            F32 Hex     Round-trip")
    for v in test_vals:
        h = f32_to_hex(v)
        rt = hex_to_f32(h)
        match = "✓" if abs(v - rt) < 1e-7 or v == rt or (v != v and rt != rt) else "~"  # ~ = F64→F32 precision
        print(f"  {v:>15.8f}  {h}    {rt:>15.8f}  {match}")
    
    print("\n=== Cross-backend Comparison Demo ===\n")
    gpu = np.array([0.10000001, 3.1415925, -0.5000001], dtype=np.float32)
    cpu = np.array([0.1, 3.14159265, -0.5], dtype=np.float32)
    dump_comparison("silu_output", gpu, cpu)
    
    print("\n=== C header (paste into .cu) ===")
    print(C_HEADER)
    
    print("=== Rust snippet ===")
    print(RUST_SNIPPET)
