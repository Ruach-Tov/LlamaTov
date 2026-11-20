#!/usr/bin/env python3
"""auto_pipeline.py — the fully-automatic lift -> fuse -> codegen pipe.

Closes the last manual hop: previously a human ran lift_chain.py, copied
its stdout, and pasted the chain-fact into a .pl file that the bridge
consumed. This driver wires lifter-output -> file -> bridge with NO human
in the loop.

Usage:
    python3 lib/auto_pipeline.py <problem.py> <pid> [--emit-only]

Pipeline (zero hand-work):
    1. LIFT:    lib/lift_chain.py <problem.py> <pid> -> chain(pid, [...]).
    2. WIRE:    write the chain-fact to a generated .pl file (the pipe that
                was manual).
    3. FUSE+CODEGEN: swipl consults the generated chain-fact + the bridge's
                spell_stmt vocabulary + composition rules -> emits CUDA.

The bridge's spell_stmt facts and composition logic are UNCHANGED — only
the chain-fact source moves from hand-pasted-literal to lifter-generated-file.
"""
import re
import subprocess
import re
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def lift(problem_path, pid):
    """Run the AST lifter; return its stdout chain-fact (or raise)."""
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, 'lift_chain.py'), problem_path, pid],
        capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        raise RuntimeError(f'lift failed: {r.stderr.strip()}')
    fact = r.stdout.strip()
    if not fact.startswith('chain(') or 'UNSUPPORTED_' in fact:
        # honest failure: the lifter hit a vocabulary-gap. Name it, don't fake it.
        raise RuntimeError(f'lift produced an unsupported/empty chain: {fact}')
    return fact


import re as _re

PARAM_OP_RE = _re.compile(
    r"(add_param|mul_param|sub_param|add_param_scalar|mul_param_scalar|"
    r"sub_param_scalar|add_param_chan|mul_param_chan|sub_param_chan|"
    r"min_param_scalar|max_param_scalar)"
    r"\('([^']+)'\)")


def normalize_params(fact):
    """The multi-param aliasing fix (Doresh's 6th catch): rename each
    DISTINCT param name to its own buffer atom p0/p1/... by first
    appearance, returning (fact', manifest). manifest = list of
    (buffer_atom, original_name, kind) where kind in row|scalar|chan.
    Distinct params MUST get distinct buffers — never one shared p."""
    manifest = []
    name_to_atom = {}

    def sub(m):
        op, name = m.group(1), m.group(2)
        if name not in name_to_atom:
            atom = f'p{len(name_to_atom)}'
            name_to_atom[name] = atom
            kind = ('scalar' if op.endswith('_scalar')
                    else 'chan' if op.endswith('_chan') else 'row')
            manifest.append((atom, name, kind))
        return f'{op}({name_to_atom[name]})'

    fact2 = PARAM_OP_RE.sub(sub, fact)
    return fact2, manifest


def insert_param_args(cuda, manifest):
    """Insert per-param buffer args into the emitted kernel signature.
    For each (atom, name, kind): const float* atom, plus
    long long plen_atom (row) | nothing extra (scalar) |
    long long hw_atom + chn_atom (chan). Inserted after 'v' and
    before 'out' so callers bind positionally: v, p0.., out, n, lens."""
    if not manifest:
        return cuda
    bufs = []
    lens = []
    for atom, _name, kind in manifest:
        bufs.append(f'const float* __restrict__ {atom}')
        if kind == 'row':
            lens.append(f'long long plen_{atom}')
        elif kind == 'chan':
            lens.append(f'long long hw_{atom}')
            lens.append(f'long long chn_{atom}')
    # GENERAL param plumbing (the #91/#75 fix — one mechanism, all
    # templates): every template's k_auto signature starts with
    # 'v, out' then its own shape args. Param BUFFERS insert after v
    # (callers bind v, p0.., out, ...); LENGTH args append at the
    # signature END (after the template's own args). Anchored on the
    # k_auto signature via regex up to the closing paren — refuses
    # if the prefix is absent (never mis-signs).
    import re
    if re.search(r"__restrict__ p0\b", cuda):
        return cuda  # already inserted (the reduction routes insert
                     # before the final pass — IDEMPOTENT, not drift)
    m = re.search(
        r"__global__ void k_auto\(const float\* __restrict__ v, "
        r"float\* __restrict__ out,([^)]*)\)", cuda)
    if not m:
        raise RuntimeError('param-arg insertion: k_auto v/out prefix not '
                           'found (template drift — refuse, never mis-sign).')
    rest = m.group(1)
    new_sig = ('__global__ void k_auto(const float* __restrict__ v, ' +
               ', '.join(bufs) + ', float* __restrict__ out,' + rest +
               (', ' + ', '.join(lens) if lens else '') + ')')
    return cuda[:m.start()] + new_sig + cuda[m.end():]


def _capacity_ok(width):
    """Reg-path capacity: reg[8] x 1024 threads = 8192. PROVE-at-the-
    lifter: width must be KNOWN and <= 8192 (unknown refuses — a
    width we can't prove could silently half-cover)."""
    return width is not None and width <= 8192


# Pascal/Tesla-P4 default shared-memory-per-block budget (bytes). Read
# directly from the RUNNING torch (2.7.0) source tree on the enclave
# (/nix/store/0msk7ql7dp1qvwikav5j433v63mfxxzv-pytorch/aten/src/ATen/
# native/cuda/SoftMax.cu — the version-check Bocher asked for before
# wiring: GitHub main's dispatch arithmetic was confirmed IDENTICAL to
# 2.7's for the potential_reg_cnt formula and the block-size formula,
# but the can_use_smem threshold's exact byte-accounting (below) was
# transcribed from 2.7 directly, not assumed from main). NOTE: this is
# a per-hardware constant (torch reads it from
# cudaDeviceProp::sharedMemPerBlock at runtime), not a universal one —
# a GPU with a larger smem budget (Volta+ opt-in up to 227KB) would
# route more widths through torch's cunn_SoftMaxForwardSmem instead of
# cunn_SoftMaxForward, and BPD does not yet have a smem-cache template
# for that middle regime (see _softmax_regime below).
_P4_SMEM_BUDGET_BYTES = 49152


def _softmax_regime(width):
    """Which torch kernel regime a given reduce-width dispatches to
    (SoftMax.cu's own three-way split, transcribed here from the
    RUNNING 2.7.0 tree, not just GitHub main) — 'reg', 'smem_unbuilt',
    or 'gridstride'. Returns None for an unproven/unknown width (caller
    must refuse, never guess).

    potential_reg_cnt = ceil(width / block_size), block_size =
    round-up-to-32(min(width, 1024)) — torch's own formula
    (potential_register_count + SoftMaxForward_getBlockSize).
      < 10                        -> 'reg' (cunn_SoftMaxForwardReg;
                                      BPD's SOFTMAX_TEMPLATE/LSE_TEMPLATE,
                                      the reg[8]x1024=8192 certified path).
      >= 10, row fits smem budget -> 'smem_unbuilt' (cunn_SoftMaxForwardSmem;
                                      BPD has NO template for this regime
                                      yet — refuse honestly rather than
                                      emit the wrong kernel for the width).
      >= 10, row does NOT fit smem -> 'gridstride' (cunn_SoftMaxForward,
                                      ILP=4 vectorized; BPD's
                                      GRIDSTRIDE_SOFTMAX_TEMPLATE/
                                      GRIDSTRIDE_LSE_TEMPLATE — verified
                                      0-ULP against real torch at
                                      width=16384, #66, two random seeds).

    can_use_smem's exact formula (SoftMax.cu, transcribed precisely,
    not approximated): smem_reduction_sz = (block_size/32)*4 bytes (one
    float per warp, for the block-reduce scratch); max_elements_per_smem
    = (sharedMemPerBlock - smem_reduction_sz) / 4; can_use_smem = width
    < max_elements_per_smem (STRICT less-than). Earlier drafts of this
    function used a flat `width*4 < budget` approximation that omitted
    the reduction-scratch subtraction — narrow but real: it mis-refused
    widths 12256-12287 as 'smem_unbuilt' (over-conservative refusal,
    not a wrong-kernel risk, but imprecise) that torch's own arithmetic
    actually routes to 'gridstride', which BPD DOES support. Fixed to
    match torch's byte-accounting exactly. Torch's can_use_smem also
    checks input/output pointer 16-byte alignment and `width % ILP==0`
    (ILP=4) — both are RUNTIME conditions on the actual tensor's memory
    address, not staticaly determinable at emission time; omitted here
    deliberately (the smem regime isn't built regardless, so this
    function's job is only to correctly identify 'not reg, not
    gridstride' — a possibly-imprecise 'smem_unbuilt' classification in
    the alignment-dependent edge case is the SAFE direction, since it
    refuses rather than emits).
    """
    if width is None:
        return None
    max_threads = 1024
    block = min(width, max_threads)
    if block % 32 != 0:
        block = (block // 32 + 1) * 32
    potential_reg_cnt = (width + block - 1) // block
    if potential_reg_cnt < 10:
        return 'reg'
    smem_reduction_sz = (block // 32) * 4
    max_elements_per_smem = (_P4_SMEM_BUDGET_BYTES - smem_reduction_sz) // 4
    if width < max_elements_per_smem:
        return 'smem_unbuilt'
    return 'gridstride'


def insert_input2_arg(cuda, fact):
    """Two-input models: add 'const float* v2' after v when the chain
    uses *_input2 ops."""
    if '_input2' not in fact:
        return cuda
    sig_old = 'const float* __restrict__ v,'
    sig_new = 'const float* __restrict__ v, const float* __restrict__ v2,'
    if sig_old not in cuda:
        raise RuntimeError('input2-arg insertion: signature anchor missing.')
    return cuda.replace(sig_old, sig_new, 1)


def write_chain_file(fact, pid):
    """The WIRE step — the hop that was manual. Write lifter-output to a file
    the bridge consults. No human copy-paste."""
    path = os.path.join(REPO, 'lib', f'.chain_{pid}.pl')
    with open(path, 'w') as f:
        f.write('%% AUTO-GENERATED by auto_pipeline.py — do not edit.\n')
        f.write('%% Produced by lib/lift_chain.py from the problem source.\n')
        f.write(fact + '\n')
    return path


SPELLINGS = r"""
%% divide → RECIPROCAL-MULTIPLY (the substrate ruling, #57's 2nd
%% confirmation): torch's device tensor÷scalar IS reciprocal-multiply
%% (no true f32 division on-device) — a kernel /D (div.rn.f32)
%% DIVERGES from torch (hardswish saga + #57 16.8%). The reciprocal
%% is computed in f64 THEN narrowed (torch's Scalar path): emit
%% *(1.0/D) with the f64 quotient as the literal.
spell_stmt(divide(D), In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*', c_var(In), c_float_f(R)))) :-
    R is 1.0 / D.
spell_stmt(multiply(V), In, Out,
    c_decl_init(c_type(float), Out, c_binop('*', c_var(In), c_float_f(V)))).
spell_stmt(add_scalar(V), In, Out,
    c_decl_init(c_type(float), Out, c_binop('+', c_var(In), c_float_f(V)))).
spell_stmt(subtract(V), In, Out,
    c_decl_init(c_type(float), Out, c_binop('-', c_var(In), c_float_f(V)))).
spell_stmt(add_self, In, Out,
    c_decl_init(c_type(float), Out, c_binop('+', c_var(In), c_var(In)))).
spell_stmt(leaky_relu(S), In, Out,
    c_decl_init(c_type(float), Out,
        c_ternary(c_binop('>', c_var(In), c_float_f(0.0)), c_var(In),
                  c_binop('*', c_float_f(S), c_var(In))))).
%% relu, CUDA spelling: fmaxf(x, 0) — runtime max.f32 NORMALIZES -0.0
%% (verified: Doresh PTX-read + Mavdil #69 direct gate, 0/130M):
spell_stmt(relu, In, Out,
    c_decl_init(c_type(float), Out,
        c_call(fmaxf, [c_var(In), c_float_f(0.0)]))).
spell_stmt(swish, In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*', c_var(In),
            c_paren(c_binop('/', c_float_f(1.0),
                c_paren(c_binop('+', c_float_f(1.0),
                    c_call(expf, [c_unop('-', c_var(In))])))))))).
spell_stmt(clamp(Lo, Hi), In, Out,
    c_decl_init(c_type(float), Out,
        c_ternary(c_binop('<', c_var(In), c_float_f(Lo)), c_float_f(Lo),
            c_paren(c_ternary(c_binop('>', c_var(In), c_float_f(Hi)),
                    c_float_f(Hi), c_var(In)))))).
spell_stmt(tanh, In, Out,
    c_decl_init(c_type(float), Out, c_call(tanhf, [c_var(In)]))).
%% param-tensor ops: EACH DISTINCT param gets ITS OWN buffer (the
%% term's first arg is the BUFFER ATOM p0/p1/... assigned by the
%% python pre-pass — the multi-param aliasing fix, Doresh's 6th
%% catch: two distinct nn.Parameters must never share one pointer).
%% Per-buffer length args: plen_p0, plen_p1, ...
spell_stmt(add_param(P), In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('+', c_var(In),
            c_index(c_var(P), c_binop('%', c_var(i), c_var(PLen)))))) :-
    atom_concat(plen_, P, PLen).
spell_stmt(add_saved(_), In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('+', c_var(In), c_index(c_var(x2), c_var(i))))).
spell_stmt(mul_param(P), In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*', c_var(In),
            c_index(c_var(P), c_binop('%', c_var(i), c_var(PLen)))))) :-
    atom_concat(plen_, P, PLen).
spell_stmt(sub_param(P), In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('-', c_var(In),
            c_index(c_var(P), c_binop('%', c_var(i), c_var(PLen)))))) :-
    atom_concat(plen_, P, PLen).
%% scalar-param ops: the param is a SINGLE value (shape all-1s or 0-d);
%% arrives as p, indexed p[0] — valid in ANY template context:
spell_stmt(add_param_scalar(P), In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('+', c_var(In), c_index(c_var(P), c_int(0))))).
spell_stmt(mul_param_scalar(P), In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*', c_var(In), c_index(c_var(P), c_int(0))))).
spell_stmt(sub_param_scalar(P), In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('-', c_var(In), c_index(c_var(P), c_int(0))))).
spell_stmt(min_param_scalar(P), In, Out,
    c_decl_init(c_type(float), Out,
        c_call(fminf, [c_var(In), c_index(c_var(P), c_int(0))]))).
spell_stmt(max_param_scalar(P), In, Out,
    c_decl_init(c_type(float), Out,
        c_call(fmaxf, [c_var(In), c_index(c_var(P), c_int(0))]))).
%% channel-param ops: (C,1,1)-shaped params over (B,C,H,W) tensors —
%% index p[(i / hw) %% chn] (hw = H*W, chn = C; kernel args):
spell_stmt(add_param_chan(P), In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('+', c_var(In),
            c_index(c_var(P),
                c_binop('%', c_binop('/', c_var(i), c_var(HW)),
                        c_var(CHN)))))) :-
    atom_concat(hw_, P, HW), atom_concat(chn_, P, CHN).
spell_stmt(mul_param_chan(P), In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*', c_var(In),
            c_index(c_var(P),
                c_binop('%', c_binop('/', c_var(i), c_var(HW)),
                        c_var(CHN)))))) :-
    atom_concat(hw_, P, HW), atom_concat(chn_, P, CHN).
spell_stmt(sub_param_chan(P), In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('-', c_var(In),
            c_index(c_var(P),
                c_binop('%', c_binop('/', c_var(i), c_var(HW)),
                        c_var(CHN)))))) :-
    atom_concat(hw_, P, HW), atom_concat(chn_, P, CHN).
%% one-sided clamps:
spell_stmt(clamp_min(Lo), In, Out,
    c_decl_init(c_type(float), Out,
        c_ternary(c_binop('<', c_var(In), c_float_f(Lo)),
                  c_float_f(Lo), c_var(In)))).
spell_stmt(clamp_max(Hi), In, Out,
    c_decl_init(c_type(float), Out,
        c_ternary(c_binop('>', c_var(In), c_float_f(Hi)),
                  c_float_f(Hi), c_var(In)))).
%% two-input elementwise: the second forward-arg arrives as v2[i]
%% (same-shape as the output — the #26/#28 class):
spell_stmt(add_input2, In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('+', c_var(In), c_index(c_var(v2), c_var(i))))).
spell_stmt(mul_input2, In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*', c_var(In), c_index(c_var(v2), c_var(i))))).
spell_stmt(sub_input2, In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('-', c_var(In), c_index(c_var(v2), c_var(i))))).
%% swish_rev: sigmoid(x) * x — source order preserved (product same
%% value as swish but write it as the source does):
%% xsig3_div6: x * sigmoid(x+3) / 6 — the sigmoid-hardswish variant
%% (#58). RECIPROCAL-MULTIPLY per THE DIVISOR-TYPE RULE (Mavdil,
%% e118e6352): the problem's /6 is a PYTHON SCALAR (source line 16,
%% verbatim — zero torch.tensor in the file) → torch's c10::Scalar
%% path → reciprocal; a /6.0f spelling reproduces torch's TENSOR-
%% divisor arithmetic = a DIFFERENT function (1,398,520/4.2M differ).
%% Measured against the problem's own expression: reciprocal → 0
%% differ. The earlier true-div flip (e7229680e-era) was fixing to a
%% HARNESS reference — if a gate diverges on this spelling, CHECK THE
%% HARNESS's divisor materialization first (read the artefact that
%% DEFINES the thing, not the artefacts that USE it):
spell_stmt(xsig3_div6, In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*',
            c_binop('*', c_var(In),
                c_paren(c_binop('/', c_float_f(1.0),
                    c_paren(c_binop('+', c_float_f(1.0),
                        c_call(expf, [c_unop('-',
                            c_paren(c_binop('+', c_var(In),
                                c_float_f(3.0))))])))))),
            c_float_f(0.16666666666666666)))).
%% zero_fold: x − mean(x over width-1) ≡ 0 EXACTLY (derived identity;
%% the #80 degenerate class — PASS-DEGENERATE in the census):
spell_stmt(zero_fold, _In, Out,
    c_decl_init(c_type(float), Out, c_float_f(0.0))).
spell_stmt(swish_rev, In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*',
            c_paren(c_binop('/', c_float_f(1.0),
                c_paren(c_binop('+', c_float_f(1.0),
                    c_call(expf, [c_unop('-', c_var(In))]))))),
            c_var(In)))).
%% xclip_hswish: x * clamp((x+3)/6, 0, 1) — the WRITTEN-OUT clip form
%% (#57-style source); keep the SOURCE arithmetic (true division!):
spell_stmt(xclip_hswish, In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*', c_var(In),
            c_paren(c_ternary(
                c_binop('<',
                    c_paren(c_binop('*',
                        c_paren(c_binop('+', c_var(In), c_float_f(3.0))),
                        c_float_f(0.16666666666666666))),
                    c_float_f(0.0)),
                c_float_f(0.0),
                c_paren(c_ternary(
                    c_binop('>',
                        c_paren(c_binop('*',
                            c_paren(c_binop('+', c_var(In), c_float_f(3.0))),
                            c_float_f(0.16666666666666666))),
                        c_float_f(1.0)),
                    c_float_f(1.0),
                    c_paren(c_binop('*',
                        c_paren(c_binop('+', c_var(In), c_float_f(3.0))),
                        c_float_f(0.16666666666666666)))))))))).
%% xhardswish: x * F.hardswish(x) — self-gated, inner = the
%% MEASURED-EXACT vs INSTALLED torch 2.7.0 (true division; the 2.11 source uses *1/6 — version caveat):
spell_stmt(xhardswish, In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*', c_var(In),
            c_paren(c_binop('*',
                c_binop('*', c_var(In),
                    c_call(fminf,
                        [c_call(fmaxf,
                            [c_binop('+', c_var(In), c_float_f(3.0)),
                             c_float_f(0.0)]),
                         c_float_f(6.0)])),
                c_float_f(0.16666666666666666)))))).
%% xmish: x * mish(x) = x * (x * tanhf(softplus(x))) — the certified
%% imp22 epilogue spelling (thresh-20 softplus):
spell_stmt(xmish, In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*', c_var(In),
            c_paren(c_binop('*', c_var(In),
                c_call(tanhf,
                    [c_ternary(c_binop('>', c_var(In), c_float_f(20.0)),
                        c_var(In),
                        c_call(log1pf, [c_call(expf, [c_var(In)])]))])))))).
%% save/add_saved: pure SSA — save binds the current value to a named
%% slot (a plain temp); add_saved references it:
spell_stmt(save(_Name), In, Out,
    c_decl_init(c_type(float), Out, c_var(In))).
%% gelu, CUDA spelling: plain erff form 0.5*x*(1+erff(x/sqrt(2)))
%% (NOT the oneDNN polynomial — that's the CPU spelling; certified in
%% #53-CUDA 3.01x):
spell_stmt(gelu, In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*', c_binop('*', c_float_f(0.5), c_var(In)),
            c_paren(c_binop('+', c_float_f(1.0),
                c_call(erff, [c_binop('*', c_var(In),
                    c_float_f(0.7071067811865476))])))))).
%% sigmoid: f32 1/(1+expf(-x)) (transfers exactly; certified #70-CUDA):
spell_stmt(sigmoid, In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('/', c_float_f(1.0),
            c_paren(c_binop('+', c_float_f(1.0),
                c_call(expf, [c_unop('-', c_var(In))])))))).
%% mish: x*tanhf(softplus) with the thresh-20 softplus spelling
%% (sp = x>20 ? x : log1pf(expf(x)); certified #29/#87-CUDA):
spell_stmt(mish, In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*', c_var(In),
            c_call(tanhf,
                [c_ternary(c_binop('>', c_var(In), c_float_f(20.0)),
                    c_var(In),
                    c_call(log1pf, [c_call(expf, [c_var(In)])]))])))).
%% hardswish, CUDA spelling: RECIPROCAL-MULTIPLY (x+3)*(1/6), NOT /6
%% (IEEE-div diverges 21.8M/130M at 1-ulp; certified #57/#69-CUDA):
spell_stmt(hardswish, In, Out,
    c_decl_init(c_type(float), Out,
        c_binop('*',
            c_binop('*', c_var(In),
                c_call(fminf,
                    [c_call(fmaxf,
                        [c_binop('+', c_var(In), c_float_f(3.0)),
                         c_float_f(0.0)]),
                     c_float_f(6.0)])),
            c_float_f(0.16666666666666666)))).
%% ^ RESOLVED BY WIDE-RANGE TIE-BREAK (the dispute): at ±6 x 8.4M,
%% t*c*one_sixth differs 0 vs torch; (t*c)/6 differs 1.48M @1ulp.
%% The 2.7.0 SOURCE form (x*min(max(x+3,0),6)*one_sixth, LEFT-assoc)
%% is EXACT — read-the-source was right; the /6 form coincides only
%% on narrow distributions (why one 8.4M probe passed it). THE
%% DISTRIBUTION IS PART OF THE MEASUREMENT'S SCOPE. Replaces the clip
%% form ((x+3)*(1/6) clamped to [0,1]) which diverged 1-2 ulp on
%% wide-magnitude inputs (Doresh's #69 gate probe; read confirms his
%% relu6 instinct exactly).
%% hardtanh: clamp to [-1, 1]:
spell_stmt(hardtanh, In, Out,
    c_decl_init(c_type(float), Out,
        c_ternary(c_binop('<', c_var(In), c_float_f(-1.0)), c_float_f(-1.0),
            c_paren(c_ternary(c_binop('>', c_var(In), c_float_f(1.0)),
                    c_float_f(1.0), c_var(In)))))).

"""

BRIDGE_HARNESS = r"""
:- set_prolog_flag(double_quotes, codes).
:- use_module('lib/c_ast').

%% consult the lifter-generated chain-fact (the automated pipe):
:- consult('{chain_file}').

%% the bridge vocabulary + composition (unchanged from bridge_emit.pl):
{spellings}
compose_stmts(Ops, Cur, Final, Stmts) :-
    compose_stmts(Ops, Cur, Final, Stmts, []).
compose_stmts([], Cur, Cur, [], _).
compose_stmts([save(Name)|Rest], Cur, Final, Stmts, Saved) :-
    !, compose_stmts(Rest, Cur, Final, Stmts, [Name-Cur|Saved]).
compose_stmts([add_saved(Name)|Rest], Cur, Final, [Stmt|Stmts], Saved) :-
    member(Name-SavedVar, Saved), !,
    gensym(t, Next),
    Stmt = c_decl_init(c_type(float), Next,
        c_binop('+', c_var(Cur), c_var(SavedVar))),
    compose_stmts(Rest, Next, Final, Stmts, Saved).
%% the CROSS-SEGMENT saved tensor (#92: the save lives in another
%% segment — the kernel takes it as the x2 buffer, elementwise-
%% aligned; the route rewrites x2[i] to the template's index):
compose_stmts([add_saved(_Name)|Rest], Cur, Final, [Stmt|Stmts],
              Saved) :-
    !, gensym(t, Next),
    Stmt = c_decl_init(c_type(float), Next,
        c_binop('+', c_var(Cur), c_index(c_var(x2), c_var(i)))),
    compose_stmts(Rest, Next, Final, Stmts, Saved).
compose_stmts([mul_saved(Name)|Rest], Cur, Final, [Stmt|Stmts], Saved) :-
    !, member(Name-SavedVar, Saved),
    gensym(t, Next),
    Stmt = c_decl_init(c_type(float), Next,
        c_binop('*', c_var(Cur), c_var(SavedVar))),
    compose_stmts(Rest, Next, Final, Stmts, Saved).
compose_stmts([sub_saved(Name)|Rest], Cur, Final, [Stmt|Stmts], Saved) :-
    !, member(Name-SavedVar, Saved),
    gensym(t, Next),
    Stmt = c_decl_init(c_type(float), Next,
        c_binop('-', c_var(Cur), c_var(SavedVar))),
    compose_stmts(Rest, Next, Final, Stmts, Saved).
compose_stmts([Op|Rest], Cur, Final, [Stmt|Stmts], Saved) :-
    gensym(t, Next),
    spell_stmt(Op, Cur, Next, Stmt),
    compose_stmts(Rest, Next, Final, Stmts, Saved).

%% BASE signature only (v, out, n) — per-param buffer args (p0, plen_p0
%% / hw_p0+chn_p0, ...) are inserted by PYTHON from the param-manifest
%% (the multi-param aliasing fix: each DISTINCT param gets its OWN
%% buffer + its own length args; never one shared p).
kernel_params(_Ops, Params) :-
    Params = [param(c_type(named('const float* __restrict__')), v),
              param(c_type(named('float* __restrict__')), out),
              param(c_type(named('long long')), n)].

emit_kernel(Pid) :-
    chain(Pid, Ops),
    compose_stmts(Ops, x, Final, StageStmts),
    append([c_decl_init(c_type(float), x, c_index(c_var(v), c_var(i)))|StageStmts],
           [c_assign(c_index(c_var(out), c_var(i)), c_var(Final))], Body),
    kernel_params(Ops, Params),
    Kernel = c_func(['__global__'], c_type(void), k_auto, Params,
        [c_decl_init(c_type(named('long long')), i,
            c_binop('+',
                c_binop('*', c_var('blockIdx.x'),
                        c_cast(c_type(named('long long')), c_var('blockDim.x'))),
                c_var('threadIdx.x'))),
         c_if(c_binop('<', c_var(i), c_var(n)), Body)]),
    emit_c(Kernel, Str),
    write(Str), nl.

:- (emit_kernel({pid}) -> halt(0) ; halt(3)).
"""


REDUCTION_HARNESS = """
:- set_prolog_flag(double_quotes, codes).
:- use_module('lib/c_ast').
:- consult('{chain_file}').

{spellings}

compose_stmts(Ops, Cur, Final, Stmts) :-
    compose_stmts(Ops, Cur, Final, Stmts, []).
compose_stmts([], Cur, Cur, [], _).
compose_stmts([save(Name)|Rest], Cur, Final, Stmts, Saved) :-
    !, compose_stmts(Rest, Cur, Final, Stmts, [Name-Cur|Saved]).
compose_stmts([add_saved(Name)|Rest], Cur, Final, [Stmt|Stmts], Saved) :-
    member(Name-SavedVar, Saved), !,
    gensym(t, Next),
    Stmt = c_decl_init(c_type(float), Next,
        c_binop('+', c_var(Cur), c_var(SavedVar))),
    compose_stmts(Rest, Next, Final, Stmts, Saved).
%% the CROSS-SEGMENT saved tensor (#92): x2-buffer fallback:
compose_stmts([add_saved(_Name)|Rest], Cur, Final, [Stmt|Stmts],
              Saved) :-
    !, gensym(t, Next),
    Stmt = c_decl_init(c_type(float), Next,
        c_binop('+', c_var(Cur), c_index(c_var(x2), c_var(i)))),
    compose_stmts(Rest, Next, Final, Stmts, Saved).
compose_stmts([mul_saved(Name)|Rest], Cur, Final, [Stmt|Stmts], Saved) :-
    !, member(Name-SavedVar, Saved),
    gensym(t, Next),
    Stmt = c_decl_init(c_type(float), Next,
        c_binop('*', c_var(Cur), c_var(SavedVar))),
    compose_stmts(Rest, Next, Final, Stmts, Saved).
compose_stmts([sub_saved(Name)|Rest], Cur, Final, [Stmt|Stmts], Saved) :-
    !, member(Name-SavedVar, Saved),
    gensym(t, Next),
    Stmt = c_decl_init(c_type(float), Next,
        c_binop('-', c_var(Cur), c_var(SavedVar))),
    compose_stmts(Rest, Next, Final, Stmts, Saved).
compose_stmts([Op|Rest], Cur, Final, [Stmt|Stmts], Saved) :-
    gensym(t, Next),
    spell_stmt(Op, Cur, Next, Stmt),
    compose_stmts(Rest, Next, Final, Stmts, Saved).

emit_stmts_str([]).
emit_stmts_str([S|Ss]) :- emit_c(S, Str), write(Str), emit_stmts_str(Ss).

split_at_reduction([reduction(R)|Rest], [], R, Rest) :- !.
split_at_reduction([Op|Ops], [Op|Pre], R, Post) :-
    split_at_reduction(Ops, Pre, R, Post).

emit_sections(Pid) :-
    chain(Pid, Ops),
    split_at_reduction(Ops, PreOps, {red_kind}, PostOps),
    compose_stmts(PreOps, x, PreFinal, PreStmts),
    compose_stmts(PostOps, y, PostFinal, PostStmts),
    write('===PRE==='), nl, emit_stmts_str(PreStmts),
    write('===PREFINAL==='), nl, write(PreFinal), nl,
    write('===POST==='), nl, emit_stmts_str(PostStmts),
    write('===POSTFINAL==='), nl, write(PostFinal), nl.

:- (emit_sections({pid}) -> halt(0) ; halt(3)).
"""

# The certified Reg-path softmax template (#99-CUDA 0-ULP: element-strided
# reg cache, shuffle-halving blockReduce, RaW sync, exp-recomputed write;
# profiler-confirmed cunn_SoftMaxForwardReg<...,8>; covers N <= 8192):
SOFTMAX_TEMPLATE = """__device__ __forceinline__ float _wrs(float val) {{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}}
__device__ __forceinline__ float _wrm(float val) {{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}}
__device__ float _brs(float val, float* smem) {{
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    val = _wrs(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int nwarps = blockDim.x / 32;
    val = (threadIdx.x < nwarps) ? smem[lane] : 0.0f;
    if (wid == 0) {{ val = _wrs(val); if (threadIdx.x == 0) smem[0] = val; }}
    __syncthreads();
    return smem[0];
}}
__device__ float _brm(float val, float* smem) {{
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    val = _wrm(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int nwarps = blockDim.x / 32;
    val = (threadIdx.x < nwarps) ? smem[lane] : -3.402823466e+38f;
    if (wid == 0) {{ val = _wrm(val); if (threadIdx.x == 0) smem[0] = val; }}
    __syncthreads();
    return smem[0];
}}
__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out, int classes) {{
    /* buffered-row softmax template (certified Reg-path structure, imp99).
       pre-epilogue fuses into the load; post-epilogue into the write. */
    extern __shared__ float sred[];
    const float* input = v + (long long)blockIdx.x * classes;
    float* output = out + (long long)blockIdx.x * classes;
    float reg[8];
    float threadMax = -3.402823466e+38f;
    #pragma unroll
    for (int ri = 0; ri < 8; ri++) {{
        int off = threadIdx.x + ri * blockDim.x;
        if (off < classes) {{
            // i = the column (flat position within the row; chan-
            // param spellings i/hw%chn index correctly with hw=1
            // in the collapsed row-form — the #42 class):
            const long long i = off;
            float x = input[off];
{pre_stmts}            reg[ri] = {pre_final};
            threadMax = fmaxf(threadMax, {pre_final});
        }}
    }}
    float m = _brm(threadMax, sred);
    __syncthreads();   /* RaW sync between chained blockReduces (torch comment) */
    float texp = 0.0f;
    #pragma unroll
    for (int ri = 0; ri < 8; ri++) {{
        int off = threadIdx.x + ri * blockDim.x;
        if (off < classes) texp += expf(reg[ri] - m);
    }}
    float s = _brs(texp, sred);
    #pragma unroll
    for (int ri = 0; ri < 8; ri++) {{
        int off = threadIdx.x + ri * blockDim.x;
        if (off < classes) {{
            float y = expf(reg[ri] - m) / s;
{post_stmts}            output[off] = {post_final};
        }}
    }}
}}
"""


# The certified Reg-path lse template (#64-CUDA 5.19x certified, #22-CUDA
# 8.57-8.64x record: same Reg-path reductions, epilogue = logf(s)+m then
# the post-chain on the SCALAR, single value per row written by thread 0):
LSE_TEMPLATE = """__device__ __forceinline__ float _wrm(float val) {{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}}
__device__ float _brm(float val, float* smem) {{
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    val = _wrm(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int nwarps = blockDim.x / 32;
    val = (threadIdx.x < nwarps) ? smem[lane] : -3.402823466e+38f;
    if (wid == 0) {{ val = _wrm(val); if (threadIdx.x == 0) smem[0] = val; }}
    __syncthreads();
    return smem[0];
}}
__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out, int classes) {{
    /* Reduce.cuh-order LSE (the Sept-7 crack: reduce_kernel<512,1>
       = 32 lanes x 16 warps, ONE block per row; the SUM pass
       transcribes torch's exact accumulation:
       vec4 loads (vec-idx = lane + warp*32, stride 512 vec-units)
       -> FOUR independent accumulators -> serial combine
       -> BLOCK_Y smem-halving (8,4,2,1) FIRST
       -> BLOCK_X ASCENDING shfl_down (1,2,4,8,16).
       The MAX pass is order-insensitive (kept as block max).
       Emulation 0/256 vs torch.sum at (1024x8192), both benches. */
    extern __shared__ float sred[];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float* input = v + (long long)blockIdx.x * classes;
    const int nvec = classes / 4;
    float reg[16];
    float threadMax = -3.402823466e+38f;
    #pragma unroll
    for (int ri = 0; ri < 4; ri++) {{
        int idx = lane + warp * 32 + ri * 512;
        if (idx < nvec) {{
            #pragma unroll
            for (int j = 0; j < 4; j++) {{
                float x = input[idx * 4 + j];
{pre_stmts}                reg[ri * 4 + j] = {pre_final};
                threadMax = fmaxf(threadMax, reg[ri * 4 + j]);
            }}
        }}
    }}
    /* tail (classes %% 4): torch adds ONE scalar per threadIdx.x at
       tail_start+tid into accumulator 0 -- UNEXERCISED at 8192
       (the certified width); non-%%4 widths refuse upstream. */
    float m = _brm(threadMax, sred);
    __syncthreads();   /* RaW sync between chained blockReduces */
    float vl0 = 0.0f, vl1 = 0.0f, vl2 = 0.0f, vl3 = 0.0f;
    #pragma unroll
    for (int ri = 0; ri < 4; ri++) {{
        int idx = lane + warp * 32 + ri * 512;
        if (idx < nvec) {{
            vl0 += expf(reg[ri * 4 + 0] - m);
            vl1 += expf(reg[ri * 4 + 1] - m);
            vl2 += expf(reg[ri * 4 + 2] - m);
            vl3 += expf(reg[ri * 4 + 3] - m);
        }}
    }}
    /* combine accumulators SERIAL (Reduce.cuh line ~545): */
    float a = vl0 + vl1;
    a = a + vl2;
    a = a + vl3;
    /* BLOCK_Y FIRST: smem halving over warps (offset 8,4,2,1);
       own(y) + other(y+off) operand order: */
    sred[threadIdx.x] = a;
    __syncthreads();
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {{
        if (warp < off) {{
            sred[lane + warp * 32] = sred[lane + warp * 32] +
                                     sred[lane + (warp + off) * 32];
        }}
        __syncthreads();
    }}
    /* BLOCK_X: ASCENDING shfl_down (1,2,4,8,16) on warp 0: */
    if (warp == 0) {{
        float val = sred[lane];
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane == 0) {{
            float y = logf(val) + m;
{post_stmts}            out[blockIdx.x] = {post_final};
        }}
    }}
}}
"""

# NARROW single-warp LSE template (Doresh's #42 catch, Sept 2026):
# LSE_TEMPLATE's _wrs/_wrm use torch's SoftMax.cu DESCENDING warp-shuffle
# order (16,8,4,2,1 -- block_reduce.cuh's WarpReduceSum/Max), which is
# what a genuine torch.softmax dispatch uses. But torch.logsumexp is
# CompositeExplicitAutograd via TensorIterator/Reduce.cuh, NOT SoftMax.cu
# -- LSE_TEMPLATE's descending shuffle happens to match at width<=8192
# with a LARGE batch (dim1) ONLY because that combination hits
# Reduce.cuh's split_across_warps=TRUE regime (config.values_per_thread()
# >= min(block_height*16, 256)), which engages block_y_reduce (a
# cross-warp combine where the final warp-level shuffle is over a
# SMALLER, already-block_y_reduce'd set of values, not the raw per-
# thread partials) -- verified #22/#64 (width=8192, batch=1024) ARE in
# that regime, so LSE_TEMPLATE stays correct and UNTOUCHED for them.
#
# When split_across_warps is FALSE (small width AND/OR small batch, e.g.
# #42's width=128/batch=16: block_width=32, block_height=16,
# values_per_thread=4 < threshold=256), Reduce.cuh's block_x_reduce
# handles the ENTIRE final combine via a SINGLE warp-shuffle tree using
# ASCENDING offsets (1,2,4,8,16 -- block_reduce.cuh) -- the DESCENDING
# order LSE_TEMPLATE uses is provably wrong here (verified: 1-ULP
# residual on a real published shape, resolved to 0-ULP by switching to
# ascending order and a stride-32/vt0=4 accumulator layout matching
# Reduce.cuh's thread_reduce_impl exactly for this regime).
#
# This is a SEPARATE, narrower template -- LSE_TEMPLATE is NOT modified,
# so #22/#64's existing 0-ULP gates are untouched. Callers select this
# template only when they've confirmed split_across_warps is FALSE for
# their specific (dim_size, batch) pair -- see emit_reduction's
# 'lse_narrow' variant branch.
LSE_NARROW_TEMPLATE = """__device__ __forceinline__ float _wrs_asc(float val) {{
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
        val = val + __shfl_down_sync(0xffffffff, val, offset);
    return val;
}}
__device__ __forceinline__ float _wrm_asc(float val) {{
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}}
__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out, int classes) {{
    /* NARROW single-warp LSE (Doresh's #42 catch): block_width=32 (one
       warp per row), vt0=4 stride-32 accumulators (Reduce.cuh
       thread_reduce_impl), linear combine within thread, ASCENDING
       warp-shuffle for the final (and only) combine step. One row per
       block; launch with blockDim.x=32. NOTE: uses 'jj' (not 'i') as
       the unroll-loop var since the PRE-epilogue spelling (e.g.
       add_param_chan) aliases 'i' to the per-element offset itself
       (const long long i = off;) -- 'i' as an outer loop var would
       collide with that inner redeclaration. */
    const float* input = v + (long long)blockIdx.x * classes;
    const int stride = 32;
    int tid = threadIdx.x;

    float vmax[4] = {{-3.402823466e+38f, -3.402823466e+38f,
                      -3.402823466e+38f, -3.402823466e+38f}};
    /* torch's vec4 mapping (the Sept-7 #45 crack, 0-verified at
       widths 256..4096): accumulator j takes val[j] of each
       CONSECUTIVE vec4 at vec-index (lane + k*32). WIDTH < 256
       uses the strided-scalar order instead (torch's vec
       threshold: vpt >= 8) — that regime routes to
       LSE_NARROW_SCALAR_TEMPLATE. */
    const int nvec = classes / 4;
    int vidx = tid;
    while (vidx < nvec) {{
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int off = vidx * 4 + jj;
            float x = input[off];
{pre_stmts}            vmax[jj] = fmaxf(vmax[jj], {pre_final});
        }}
        vidx += stride;
    }}
    float threadMax = vmax[0];
    threadMax = fmaxf(threadMax, vmax[1]);
    threadMax = fmaxf(threadMax, vmax[2]);
    threadMax = fmaxf(threadMax, vmax[3]);
    float m = _wrm_asc(threadMax);
    m = __shfl_sync(0xffffffff, m, 0);

    float vsum[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    vidx = tid;
    while (vidx < nvec) {{
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int off = vidx * 4 + jj;
            float x = input[off];
{pre_stmts}            vsum[jj] = vsum[jj] + expf({pre_final} - m);
        }}
        vidx += stride;
    }}
    float texp = vsum[0];
    texp = texp + vsum[1];
    texp = texp + vsum[2];
    texp = texp + vsum[3];
    float s = _wrs_asc(texp);
    if (tid == 0) {{
        float y = logf(s) + m;
{post_stmts}        out[blockIdx.x] = {post_final};
    }}
}}
"""

SUB_OWN_MEAN_TEMPLATE = """__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out, long long rows, long long n) {{
    /* sub_own_mean (the #15 class): out[r,j] = x[r,j] - mean(row r).
       The mean's sum pass = the CRACKED Reduce.cuh order (Sept-7:
       vec4 loads at lane+warp*32 stride-512, four independent
       accumulators, serial combine, BLOCK_Y smem-halving first,
       BLOCK_X ascending shfl) — streaming (no register buffer;
       the subtract pass re-reads). mean = sum/n (decomposition
       measured bit-exact 0/512). One block per row, 32x16. */
    __shared__ float sred[512];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const long long row_base = (long long)blockIdx.x * n;
    const float* input = v + row_base;
    /* THE SHIFT HEAD (the Sept-7 unaligned crack, two-bench-verified
       via shift_head_emul.py): rows whose element-offset %% 4 != 0
       take a scalar head — lane x in [shift,4) of warp 0 loads
       REWOUND index x (= row element x - shift) into accumulator 0;
       the pointer rewinds; vec4 runs the aligned middle; the tail
       adds one scalar per tid.x. For aligned rows shift==0 and this
       reduces to the aligned form exactly. */
    const int shift = (int)(row_base % 4);
    const float* dat;
    long long end;
    float vl0 = 0.0f, vl1 = 0.0f, vl2 = 0.0f, vl3 = 0.0f;
    if (shift > 0) {{
        dat = input - shift;
        end = n + shift - 4;
        if (warp == 0 && lane >= shift && lane < 4)
            vl0 += dat[lane];
        dat += 4;
    }} else {{
        dat = input;
        end = n;
    }}
    for (long long idx = lane + warp * 32;
         idx * 4 + 3 < end; idx += 512) {{
        const float* p4 = dat + idx * 4;
        vl0 += p4[0]; vl1 += p4[1]; vl2 += p4[2]; vl3 += p4[3];
    }}
    {{
        const long long tail_start = end - (end % 4);
        const long long ti = tail_start + threadIdx.x;
        if (warp == 0 && ti < end) vl0 += dat[ti];
    }}
    float a = vl0 + vl1;
    a = a + vl2;
    a = a + vl3;
    sred[threadIdx.x] = a;
    __syncthreads();
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {{
        if (warp < off)
            sred[lane + warp * 32] = sred[lane + warp * 32] +
                                     sred[lane + (warp + off) * 32];
        __syncthreads();
    }}
    __shared__ float smean;
    if (warp == 0) {{
        float val = sred[lane];
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        /* torch's mean = sum * (1/n) — RECIPROCAL-MULTIPLY, not
           true division (measured: mean==S*(1/n) exactly,
           mean!=S/n — the ulp source at odd widths): */
        if (lane == 0) smean = val * (1.0f / (float)n);
    }}
    __syncthreads();
    const float m = smean;
    float* orow = out + (long long)blockIdx.x * n;
    for (long long j = threadIdx.x; j < n; j += 512) {{
        orow[j] = input[j] - m;
    }}
}}
"""

TUPLE_SHIFTHEAD_TEMPLATE = """__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out, long long rows, long long n) {{
    /* tuple-mean via the SHIFT-HEAD order (the #27 class: unaligned
       widths — N %% 4 != 0; verified 0/256 at (256,10933) incl the
       MeanOps combine = sum then reciprocal-multiply). One block
       per row, 32x16; per-row alignment shift; scalar head; vec4
       middle; tid tail; y-halving; x-ascending; out[r] = S*(1/n). */
    __shared__ float sred[512];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const long long row_base = (long long)blockIdx.x * n;
    const float* input = v + row_base;
    const int shift = (int)(row_base % 4);
    const float* dat;
    long long end;
    float vl0 = 0.0f, vl1 = 0.0f, vl2 = 0.0f, vl3 = 0.0f;
    if (shift > 0) {{
        dat = input - shift;
        end = n + shift - 4;
        if (warp == 0 && lane >= shift && lane < 4)
            vl0 += dat[lane];
        dat += 4;
    }} else {{
        dat = input;
        end = n;
    }}
    for (long long idx = lane + warp * 32;
         idx * 4 + 3 < end; idx += 512) {{
        const float* p4 = dat + idx * 4;
        float x0 = p4[0], x1 = p4[1], x2 = p4[2], x3 = p4[3];
{pre_vec}        vl0 += x0; vl1 += x1; vl2 += x2; vl3 += x3;
    }}
    {{
        const long long tail_start = end - (end % 4);
        const long long ti = tail_start + threadIdx.x;
        if (warp == 0 && ti < end) {{
            float x = dat[ti];
{pre_tail}            vl0 += x;
        }}
    }}
    float a = vl0 + vl1;
    a = a + vl2;
    a = a + vl3;
    sred[threadIdx.x] = a;
    __syncthreads();
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {{
        if (warp < off)
            sred[lane + warp * 32] = sred[lane + warp * 32] +
                                     sred[lane + (warp + off) * 32];
        __syncthreads();
    }}
    if (warp == 0) {{
        float val = sred[lane];
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane == 0) {{
            float y = val * (1.0f / (float)n);
{post_stmts}            out[blockIdx.x] = {post_final};
        }}
    }}
}}
"""

AVGPOOL3D_K2_TEMPLATE = """__global__ void k_auto{suffix}(const float* __restrict__ v, float* __restrict__ out,
                       long long outer, long long D, long long H, long long W) {{
    /* avg_pool3d(kernel_size=2) — the 2x2x2 block-mean (#72; the
       order measured trivial-sequential, 0/512 both divide forms;
       /8 is pow2-exact). outer = B*C; one thread per OUTPUT. */
    const long long OD = D / 2, OH = H / 2, OW = W / 2;
    const long long nout = outer * OD * OH * OW;
    long long o = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (o >= nout) return;
    const long long ow = o % OW;
    const long long oh = (o / OW) % OH;
    const long long od = (o / (OW * OH)) % OD;
    const long long bc = o / (OW * OH * OD);
    const float* base = v + ((bc * D + od * 2) * H + oh * 2) * W
                          + ow * 2;
    float s = 0.0f;
    #pragma unroll
    for (int dd = 0; dd < 2; dd++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++)
            #pragma unroll
            for (int ww = 0; ww < 2; ww++)
                s += base[(dd * H + hh) * W + ww];
    out[o] = s * 0.125f;
}}
"""



LSE_NARROW_SCALAR_TEMPLATE = """__device__ __forceinline__ float _wrs_asc(float val) {{
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
        val = val + __shfl_down_sync(0xffffffff, val, offset);
    return val;
}}
__device__ __forceinline__ float _wrm_asc(float val) {{
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}}
__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out, int classes) {{
    /* NARROW-SCALAR single-warp LSE (width < 256: torch's vec
       threshold unmet — input_vec_size=1; the strided-scalar
       accumulator mapping measured 0/128 at width 128; the vec4
       mapping measured WRONG there, 93/128). One warp per row. */
    const float* input = v + (long long)blockIdx.x * classes;
    const int stride = 32;
    int tid = threadIdx.x;
    float vmax[4] = {{-3.402823466e+38f, -3.402823466e+38f,
                      -3.402823466e+38f, -3.402823466e+38f}};
    int idx = tid;
    while (idx + 3 * stride < classes) {{
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int off = idx + jj * stride;
            float x = input[off];
{pre_stmts}            vmax[jj] = fmaxf(vmax[jj], {pre_final});
        }}
        idx += stride * 4;
    }}
    int idx_tail = idx;
    {{
        int k = 0;
        while (idx < classes) {{
            int off = idx;
            float x = input[off];
{pre_stmts}            vmax[k % 4] = fmaxf(vmax[k % 4], {pre_final});
            idx += stride; k++;
        }}
    }}
    float threadMax = fmaxf(fmaxf(vmax[0], vmax[1]),
                            fmaxf(vmax[2], vmax[3]));
    float m = _wrm_asc(threadMax);
    m = __shfl_sync(0xffffffff, m, 0);
    float vsum[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    idx = tid;
    while (idx + 3 * stride < classes) {{
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int off = idx + jj * stride;
            float x = input[off];
{pre_stmts}            vsum[jj] = vsum[jj] + expf({pre_final} - m);
        }}
        idx += stride * 4;
    }}
    {{
        int k = 0;
        idx = idx_tail;
        while (idx < classes) {{
            int off = idx;
            float x = input[off];
{pre_stmts}            vsum[k % 4] = vsum[k % 4] + expf({pre_final} - m);
            idx += stride; k++;
        }}
    }}
    float texp = vsum[0];
    texp = texp + vsum[1];
    texp = texp + vsum[2];
    texp = texp + vsum[3];
    float s = _wrs_asc(texp);
    if (tid == 0) {{
        float y = logf(s) + m;
{post_stmts}        out[blockIdx.x] = {post_final};
    }}
}}
"""



# Grid-stride softmax/LSE template — transcribes torch's cunn_SoftMaxForward
# path (SoftMax.cu, ILP=4 vectorized-load ilpReduce + cuda_utils::BlockReduce),
# the variant torch actually dispatches when potential_reg_cnt =
# ceil(dim_size/min(dim_size,1024)) >= 10 AND the row does NOT fit the
# smem budget (~49KB on Pascal/Tesla-P4). Distinct from the Reg-path
# above (SOFTMAX_TEMPLATE/LSE_TEMPLATE, torch's cunn_SoftMaxForwardReg,
# used when potential_reg_cnt < 10) and from a not-yet-built third
# regime (cunn_SoftMaxForwardSmem, potential_reg_cnt >= 10 AND row DOES
# fit smem) — see _softmax_regime() below for the trigger logic and the
# gap this leaves.
#
# Verified 0-ULP exact against real torch at classes=16384 (#66,
# Matmul_Dropout_Softmax, two independent random seeds) —
# bpd/gate_workspace/softmax_66_capacity/. Key structural correctness
# points transcribed bit-for-bit from torch's source (not "equivalent"
# substitutions): the ILP=4 float4-vectorized load (4 consecutive
# floats per thread per outer iteration, accumulated in sequence before
# striding by blockDim.x — NOT a naive one-float-per-thread-per-stride
# loop, which was verified to diverge by up to 4 ULP at this width);
# the exact cuda_utils::BlockReduce warp-shuffle-then-shared-memory
# structure; and torch's literal Max/Add functor definitions
# (`a < b ? b : a` for max, not fmaxf — NaN-propagation semantics
# differ and this project transcribes the functor literally rather
# than substituting an "equivalent" intrinsic).
GRIDSTRIDE_SOFTMAX_TEMPLATE = """__device__ __forceinline__ float _gs_max(float a, float b) {{ return a < b ? b : a; }}
__device__ __forceinline__ float _gs_add(float a, float b) {{ return a + b; }}
template <typename Op>
__device__ __forceinline__ float _gs_warp_reduce(float val, Op op) {{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = op(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}}
template <typename Op>
__device__ __forceinline__ float _gs_block_reduce(float val, Op op, float identity, float* shared) {{
    int tid = threadIdx.x;
    int lid = tid % 32, wid = tid / 32;
    val = _gs_warp_reduce(val, op);
    __syncthreads();
    if (lid == 0) shared[wid] = val;
    __syncthreads();
    int nwarps = blockDim.x / 32;
    val = (tid < nwarps) ? shared[lid] : identity;
    if (wid == 0) val = _gs_warp_reduce(val, op);
    return val;
}}
template <typename Op>
__device__ __forceinline__ float _gs_block_reduce_bc(float val, Op op, float identity, float* shared) {{
    float result = _gs_block_reduce(val, op, identity, shared);
    if (threadIdx.x == 0) shared[0] = result;
    __syncthreads();
    return shared[0];
}}
__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out, int classes) {{
    /* grid-stride softmax template (torch cunn_SoftMaxForward transcription,
       imp66). ILP=4-vectorized; re-reads global memory each pass (no
       reg-cache) -- this is what removes the reg[8]x1024=8192 capacity
       ceiling, at the cost of re-reading v three times instead of once.
       pre-epilogue fuses into EACH load (max-pass and sum-pass both
       re-apply it, since there is no cached reg[] to hold the
       already-transformed value); post-epilogue into the final write.
       Each unrolled ILP sub-iteration is its own {{ }} scope so any
       pre_stmts-declared locals don't collide across the 4 unrolled
       lanes. */
    extern __shared__ float sred[];
    const float* input = v + (long long)blockIdx.x * classes;
    float* output = out + (long long)blockIdx.x * classes;
    const int ILP = 4;
    int last = classes % (ILP * blockDim.x);

    float threadMax = -3.402823466e+38f;
    {{
        int offset = threadIdx.x;
        for (; offset * ILP < (classes - last); offset += blockDim.x) {{
            float4 v4 = reinterpret_cast<const float4*>(input)[offset];
            {{ float x = v4.x;
{pre_stmts}               threadMax = _gs_max(threadMax, {pre_final}); }}
            {{ float x = v4.y;
{pre_stmts}               threadMax = _gs_max(threadMax, {pre_final}); }}
            {{ float x = v4.z;
{pre_stmts}               threadMax = _gs_max(threadMax, {pre_final}); }}
            {{ float x = v4.w;
{pre_stmts}               threadMax = _gs_max(threadMax, {pre_final}); }}
        }}
        offset = classes - last + threadIdx.x;
        for (; offset < classes; offset += blockDim.x) {{
            float x = input[offset];
{pre_stmts}            threadMax = _gs_max(threadMax, {pre_final});
        }}
    }}
    float m = _gs_block_reduce_bc(threadMax, _gs_max, -3.402823466e+38f, sred);

    float threadExp = 0.0f;
    {{
        int offset = threadIdx.x;
        for (; offset * ILP < (classes - last); offset += blockDim.x) {{
            float4 v4 = reinterpret_cast<const float4*>(input)[offset];
            {{ float x = v4.x;
{pre_stmts}               threadExp = threadExp + expf(({pre_final}) - m); }}
            {{ float x = v4.y;
{pre_stmts}               threadExp = threadExp + expf(({pre_final}) - m); }}
            {{ float x = v4.z;
{pre_stmts}               threadExp = threadExp + expf(({pre_final}) - m); }}
            {{ float x = v4.w;
{pre_stmts}               threadExp = threadExp + expf(({pre_final}) - m); }}
        }}
        offset = classes - last + threadIdx.x;
        for (; offset < classes; offset += blockDim.x) {{
            float x = input[offset];
{pre_stmts}            threadExp = threadExp + expf(({pre_final}) - m);
        }}
    }}
    float s = _gs_block_reduce_bc(threadExp, _gs_add, 0.0f, sred);

    for (int offset = threadIdx.x; offset < classes; offset += blockDim.x) {{
        float x = input[offset];
{pre_stmts}        float y = expf(({pre_final}) - m) / s;
{post_stmts}        output[offset] = {post_final};
    }}
}}
"""

# Grid-stride LSE template -- same STRUCTURE as GRIDSTRIDE_SOFTMAX_TEMPLATE
# above but for logsumexp (scalar-per-row output, written once by thread 0,
# matching LSE_TEMPLATE's contract).
#
# *** SPECULATIVE -- torch.logsumexp is CompositeExplicitAutograd (5
# launches: amax -> exp -> sum -> log -> add through TensorIterator/
# Reduce.cuh, NOT SoftMax.cu). Verification target = the composition's
# per-reduction TI configs at the claimed width. DO NOT trust by
# association with the softmax fix (Doresh's trace, Sept 2026). ***
#
# NOT gated against real torch at any width -- no current L2 problem
# needs LSE above the reg-cache-verified band, and the dispatch in
# run() REFUSES (never emits this template) for any logsumexp width
# that resolves to the 'gridstride' regime, precisely because this
# template's structure is the wrong transcription target for that op.
#
# RESOLVED (Bocher, Sept 2026) -- why the existing certified
# LSE_TEMPLATE (reg-cache, #22/#64 at width=8192) already matches torch
# bit-exactly despite torch.logsumexp not going through SoftMax.cu at
# all: it's not coincidence, it's STRUCTURAL and REGIME-BOUND. torch's
# amax and sum (the two component TI reductions in the 5-launch
# composition) each get their OWN setReduceConfig for their own shape.
# At width=8192 with these tensor shapes, BOTH component reductions
# happen to dispatch to TI configurations whose per-element
# accumulation/visitation order matches what LSE_TEMPLATE's reg-cache
# stripe order produces -- so a single fused max-then-sum pass with
# that visitation order gives bit-identical intermediates to torch's
# two separate TI passes with the SAME order (max is order-insensitive/
# idempotent; sum sees identical addends in identical sequence either
# way). This equivalence does NOT generalize: at other widths, torch's
# amax and sum launches may each configure DIFFERENTLY (different
# block/thread/vec shapes per-reduction), breaking the alignment that
# makes the single-pass reg-cache template match. GRIDSTRIDE_LSE_TEMPLATE
# was written as a structural sibling of GRIDSTRIDE_SOFTMAX_TEMPLATE
# (which correctly transcribes SoftMax.cu's REAL wide-width dispatch)
# but has no analogous basis for correctness, since logsumexp's real
# wide-width behavior is a composition of two independently-configured
# TI reductions, not a SoftMax.cu kernel at all.
#
# If a future L2 problem needs LSE above the reg-cache band: build the
# REAL torch.logsumexp reference at that exact shape, determine BOTH
# component TI reductions' actual dispatch configs (amax's and sum's,
# independently -- do not assume they match each other or match
# anything softmax-shaped), and gate a purpose-built template against
# that composition precisely -- same discipline as #66, but transcribing
# TensorIterator/Reduce.cuh's config-selection logic, not SoftMax.cu's.
GRIDSTRIDE_LSE_TEMPLATE = """__device__ __forceinline__ float _gs_max(float a, float b) {{ return a < b ? b : a; }}
__device__ __forceinline__ float _gs_add(float a, float b) {{ return a + b; }}
template <typename Op>
__device__ __forceinline__ float _gs_warp_reduce(float val, Op op) {{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = op(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}}
template <typename Op>
__device__ __forceinline__ float _gs_block_reduce(float val, Op op, float identity, float* shared) {{
    int tid = threadIdx.x;
    int lid = tid % 32, wid = tid / 32;
    val = _gs_warp_reduce(val, op);
    __syncthreads();
    if (lid == 0) shared[wid] = val;
    __syncthreads();
    int nwarps = blockDim.x / 32;
    val = (tid < nwarps) ? shared[lid] : identity;
    if (wid == 0) val = _gs_warp_reduce(val, op);
    return val;
}}
template <typename Op>
__device__ __forceinline__ float _gs_block_reduce_bc(float val, Op op, float identity, float* shared) {{
    float result = _gs_block_reduce(val, op, identity, shared);
    if (threadIdx.x == 0) shared[0] = result;
    __syncthreads();
    return shared[0];
}}
__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out, int classes) {{
    /* grid-stride LSE template (torch cunn_SoftMaxForward transcription,
       imp66-sibling). See GRIDSTRIDE_SOFTMAX_TEMPLATE for the full
       regime rationale; this is the scalar-output (logsumexp) sibling. */
    extern __shared__ float sred[];
    const float* input = v + (long long)blockIdx.x * classes;
    const int ILP = 4;
    int last = classes % (ILP * blockDim.x);

    float threadMax = -3.402823466e+38f;
    {{
        int offset = threadIdx.x;
        for (; offset * ILP < (classes - last); offset += blockDim.x) {{
            float4 v4 = reinterpret_cast<const float4*>(input)[offset];
            {{ float x = v4.x;
{pre_stmts}               threadMax = _gs_max(threadMax, {pre_final}); }}
            {{ float x = v4.y;
{pre_stmts}               threadMax = _gs_max(threadMax, {pre_final}); }}
            {{ float x = v4.z;
{pre_stmts}               threadMax = _gs_max(threadMax, {pre_final}); }}
            {{ float x = v4.w;
{pre_stmts}               threadMax = _gs_max(threadMax, {pre_final}); }}
        }}
        offset = classes - last + threadIdx.x;
        for (; offset < classes; offset += blockDim.x) {{
            float x = input[offset];
{pre_stmts}            threadMax = _gs_max(threadMax, {pre_final});
        }}
    }}
    float m = _gs_block_reduce_bc(threadMax, _gs_max, -3.402823466e+38f, sred);

    float threadExp = 0.0f;
    {{
        int offset = threadIdx.x;
        for (; offset * ILP < (classes - last); offset += blockDim.x) {{
            float4 v4 = reinterpret_cast<const float4*>(input)[offset];
            {{ float x = v4.x;
{pre_stmts}               threadExp = threadExp + expf(({pre_final}) - m); }}
            {{ float x = v4.y;
{pre_stmts}               threadExp = threadExp + expf(({pre_final}) - m); }}
            {{ float x = v4.z;
{pre_stmts}               threadExp = threadExp + expf(({pre_final}) - m); }}
            {{ float x = v4.w;
{pre_stmts}               threadExp = threadExp + expf(({pre_final}) - m); }}
        }}
        offset = classes - last + threadIdx.x;
        for (; offset < classes; offset += blockDim.x) {{
            float x = input[offset];
{pre_stmts}            threadExp = threadExp + expf(({pre_final}) - m);
        }}
    }}
    float s = _gs_block_reduce_bc(threadExp, _gs_add, 0.0f, sred);
    if (threadIdx.x == 0) {{
        float y = logf(s) + m;
{post_stmts}        out[blockIdx.x] = {post_final};
    }}
}}
"""


# The certified TensorIterator reduce template (from the #22-CUDA record's
# read: Reduce.cuh two-phase block-combine — smem-halving dim_x/2..32 THEN
# warp shuffle-down doubling 1..16; block=512, vec4 lane-accumulators
# ident-seeded with ascending lane-combine; verified 0-ULP incl. saturated
# rows in imp22 8.57-8.64x certified).
# REGIME: row-contiguous reduce over the last dim, N % 4 == 0.
# OP: sum -> += / fmaxf-free; min -> fminf; mean -> sum then * (1/N).
TI_TEMPLATE = """__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out, int classes) {{
    /* TI-reduce template (certified two-phase structure, imp22).
       op={op_name}; pre-epilogue fuses into the lane loads;
       post-epilogue applies to the SCALAR result (thread 0). */
    __shared__ float sh[512];
    const float* input = v + (long long)blockIdx.x * classes;
    int tid = threadIdx.x;
    int nvec = classes / 4;
    float lane[4] = {{{ident}f, {ident}f, {ident}f, {ident}f}};
    for (int idx = tid; idx < nvec; idx += 512) {{
        #pragma unroll
        for (int li = 0; li < 4; li++) {{
            float x = input[idx*4 + li];
{pre_stmts}            lane[li] = {op}(lane[li], {pre_final});
        }}
    }}
    float tval = lane[0];
    #pragma unroll
    for (int li = 1; li < 4; li++) tval = {op}(tval, lane[li]);
    sh[tid] = tval;
    for (int off = 256; off >= 32; off >>= 1) {{
        __syncthreads();
        if (tid < off && tid + off < 512) sh[tid] = {op}(sh[tid], sh[tid + off]);
    }}
    __syncthreads();
    float sval = sh[tid < 32 ? tid : 0];
    if (tid < 32) {{
        for (int off = 1; off < 32; off <<= 1) {{
            float o = __shfl_down_sync(0xffffffff, sval, off);
            if (tid + off < 32) sval = {op}(sval, o);
        }}
    }}
    if (tid == 0) {{
        float y = {finalize};
{post_stmts}        out[blockIdx.x] = {post_final};
    }}
}}
"""

_TI_OPS = {
    'sum':  dict(op='_addf', op_name='sum', ident='0.0',
                 finalize='sval'),
    'mean': dict(op='_addf', op_name='mean', ident='0.0',
                 finalize='sval / (float)classes'),
    'min':  dict(op='fminf', op_name='min', ident='3.402823466e+38',
                 finalize='sval'),
    'max':  dict(op='fmaxf', op_name='max', ident='-3.402823466e+38',
                 finalize='sval'),
}
_ADDF_HELPER = "__device__ __forceinline__ float _addf(float a, float b) { return a + b; }\n"

# NARROW single-warp TI-reduce template (Doresh's #8 finding, generalized
# from the LSE_NARROW precedent, Sept 2026 -- REVISED after a real gate
# caught an over-broad first draft, see below): TI_TEMPLATE ALWAYS
# launches 512 threads with a fixed 256->32->32-warp-shuffle structure,
# regardless of `classes`. Real torch's ReduceConfig (Reduce.cuh)
# derives block_width = min(last_pow2(classes), 32) [注: this is itself
# batch-dependent for classes>=32, see below] and split_across_warps =
# (classes/block_width) >= min(block_height*16, 256).
#
# CONFIRMED SCOPE (narrower than first assumed -- IMPORTANT):
#   - split_across_warps is FALSE for ALL classes < 8192 (batch-
#     independent) -- the SAME boundary as LSE_NARROW_TEMPLATE's regime.
#   - For classes < 32: block_width = min(last_pow2(classes), 32) is
#     PROVABLY <= 16 and BATCH-INDEPENDENT (verified via ReduceConfig
#     simulation across classes in {2,4,8,16,31} x batch in
#     {1..16384}: block_width never exceeds 16 regardless of batch).
#     This template's fixed 32-wide warp with tail-masking correctly
#     emulates any narrower real block_width in this range -- CONFIRMED
#     via real gates: classes=8/batch=64 and classes=16/batch=128 (#8's
#     exact case), both 0-ULP for sum/mean/min/max, verified twice.
#   - For 32 <= classes < 8192: block_width is BATCH-DEPENDENT, not
#     fixed at 32 as first assumed. block_width==32 only when
#     batch>=16; for batch<16, block_width grows (up to 512), requiring
#     a DIFFERENT multi-warp-per-row structure this template does NOT
#     implement. Found via a real gate: classes=4096,batch=4 (batch<16)
#     showed a real 2-ULP residual with an earlier draft of this
#     dispatch rule that (wrongly) covered the whole classes<8192
#     range. Because emit_reduction doesn't currently plumb the batch
#     dimension through to the dispatch point, this range is NOT
#     currently routed to ti_narrow (stays on TI_TEMPLATE, which is
#     ALSO not fully correct here in general -- both templates have a
#     latent gap in this sub-range; TI_TEMPLATE's 512-thread structure
#     happens to be closer to correct at LARGE-batch/wide-classes
#     shapes than at small-batch ones, but neither is proven exactly
#     right for 32<=classes<8192 with batch<16). HONEST GAP, left for a
#     follow-up that plumbs batch through emit_reduction's dispatch.
#   - NOTE: this SAME batch-dependent block_width wrinkle likely also
#     affects the ALREADY-SHIPPED LSE_NARROW_TEMPLATE (same underlying
#     Reduce.cuh mechanism) -- LSE_NARROW's own verification (#42,
#     classes=128) happened to use batch=16, which is exactly the
#     boundary where block_width=32 first holds, so it was safe by
#     coincidence, not because the batch dimension was checked. Flagged
#     to the team as a shared, pre-existing risk worth a targeted
#     sweep, not just a note here.
#
# Structurally identical to LSE_NARROW_TEMPLATE where used: block_width
# assumed 32 (one warp per row), vt0=4 stride-32 accumulators
# (Reduce.cuh thread_reduce_impl), linear in-thread combine, ASCENDING
# warp-shuffle for the final combine (matches torch's real narrow-width
# warp-reduce order, NOT the DESCENDING order TI_TEMPLATE's shared-mem
# tree effectively uses). One row per block, launch blockDim.x=32. Uses
# 'jj' (not 'i') as the unroll-loop var for the same PRE-epilogue-
# aliasing reason documented on LSE_NARROW_TEMPLATE.
#
# DISPATCH: currently gated to classes<32 only (see emit_reduction's
# ti_variant selection) -- the safely-proven, batch-independent regime.
# Widening to the full classes<8192 range needs batch-plumbing first.
TI_NARROW_TEMPLATE = """__device__ __forceinline__ float _wr_asc(float val, int op_sel) {{
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {{
        float o = __shfl_down_sync(0xffffffff, val, offset);
        val = op_sel == 0 ? val + o : (op_sel == 1 ? fminf(val, o) : fmaxf(val, o));
    }}
    return val;
}}
__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out, int classes) {{
    /* NARROW single-warp TI-reduce (Doresh's #8 finding, generalized):
       block_width=32 (one warp per row), vt0=4 stride-32 accumulators
       (Reduce.cuh thread_reduce_impl), linear combine within thread,
       ASCENDING warp-shuffle for the final (and only) combine step.
       One row per block; launch with blockDim.x=32. op={op_name}. */
    const float* input = v + (long long)blockIdx.x * classes;
    const int stride = 32;
    int tid = threadIdx.x;

    float acc[4] = {{{ident}f, {ident}f, {ident}f, {ident}f}};
    int idx = tid;
    while (idx + 3 * stride < classes) {{
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int off = idx + jj * stride;
            float x = input[off];
{pre_stmts}            acc[jj] = {op}(acc[jj], {pre_final});
        }}
        idx += stride * 4;
    }}
    #pragma unroll
    for (int jj = 0; jj < 4; jj++) {{
        if (idx >= classes) break;
        int off = idx;
        float x = input[off];
{pre_stmts}        acc[jj] = {op}(acc[jj], {pre_final});
        idx += stride;
    }}
    float tval = acc[0];
    tval = {op}(tval, acc[1]);
    tval = {op}(tval, acc[2]);
    tval = {op}(tval, acc[3]);
    float sval = _wr_asc(tval, {op_sel});
    if (tid == 0) {{
        float y = {finalize};
{post_stmts}        out[blockIdx.x] = {post_final};
    }}
}}
"""

_TI_NARROW_OP_SEL = {'sum': 0, 'mean': 0, 'min': 1, 'max': 2}


# Channel-strided min/max template: reduce over dim=1 of a contiguous
# (B, C, spatial...) tensor. ORDER-INDEPENDENT ops (fminf/fmaxf are
# exact) -> ANY reduce order is bit-exact -> 0-ULP-safe BY CONSTRUCTION
# (no torch-order read needed — the one reduction class with no
# dispatch-regime sensitivity). One thread per output element (b, s);
# strided loop over c. Epilogue fuses on the reduced scalar.
CHAN_TEMPLATE = """__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out,
                       long long spatial, long long chn, long long nout) {{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < nout) {{
        long long b = i / spatial;
        long long s = i % spatial;
        const float* base = v + b * chn * spatial + s;
        float acc;
        for (long long ci = 0; ci < chn; ci++) {{
            float x = base[ci * spatial];
{pre_stmts}            float xe = {pre_final};
            acc = (ci == 0) ? xe : {op}(acc, xe);
        }}
        float y = acc;
{post_stmts}        out[i] = {post_final};
    }}
}}
"""

_CHAN_OPS = {'min': 'fminf', 'max': 'fmaxf'}

# Spatial mean/sum over dim=1 — TRANSCRIBES torch's TI-reduce order
# (Reduce.cuh thread_reduce_impl + block_y_reduce), validated by the
# 200/200 probe (read-and-probe AGREE, session 2). One thread per
# output EMULATES the y-split deterministically:
#   bh = min(last_pow2(C), 4)            [set_block_dimension mirror:
#       MAX_NUM_THREADS=256 (Reduce.cuh:66), /ovs4 = 64, bw=32 ->
#       bh cap 4 — VALIDATED 100/100 at C=16/32/64/128]
#   split = C >= min(bh*16, 256)         [warp_split_threshold]
#   split: stride=bh, nY=bh   else: stride=1, nY=1
#   per y: vt0=4 striped accumulators (v[i] ⊕= c = y + i*stride + 4k*stride),
#   sequential combine v0⊕v1⊕v2⊕v3, then y-tree halving.
# MEAN finalize: *(1.0f/C) — the factor-multiply; NOTE: at pow2 C this
# equals true division (verified C=64); non-pow2 C awaits the MeanOps
# read (stated regime).
# REGIME: dim=1 of contiguous (B,C,spatial), rank≥3, C ≤ 2047 (no
# global/CTA split: values_per_thread < 256 after y-split).
SPATIAL_MEANSUM_TEMPLATE = """__device__ __forceinline__ long long _lp2(long long n) {{
    long long p = 1; while (p * 2 <= n) p *= 2; return p;
}}
__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out,
                       long long spatial, long long chn, long long nout) {{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < nout) {{
        long long b = i / spatial;
        long long s = i % spatial;
        const float* base = v + b * chn * spatial + s;
        long long bh = _lp2(chn < 4 ? chn : 4);
        long long thr = bh * 16 < 256 ? bh * 16 : 256;
        long long nY = (chn >= thr) ? bh : 1;
        long long stride = (chn >= thr) ? bh : 1;
        float yacc[8];
        for (long long y = 0; y < nY; y++) {{
            float vl[4] = {{{ident}f, {ident}f, {ident}f, {ident}f}};
            long long idx = y;
            while (idx + 3 * stride < chn) {{
                #pragma unroll
                for (int t = 0; t < 4; t++) {{
                    float x = base[(idx + t * stride) * spatial];
{pre_stmts}                    vl[t] = vl[t] {op} ({pre_final});
                }}
                idx += 4 * stride;
            }}
            for (int t = 0; t < 4 && idx < chn; t++) {{
                float x = base[idx * spatial];
{pre_stmts2}                vl[t] = vl[t] {op} ({pre_final});
                idx += stride;
            }}
            float a = vl[0];
            a = a {op} vl[1]; a = a {op} vl[2]; a = a {op} vl[3];
            yacc[y] = a;
        }}
        for (long long off = nY / 2; off > 0; off >>= 1) {{
            for (long long y = 0; y < off; y++) {{
                yacc[y] = yacc[y] {op} yacc[y + off];
            }}
        }}
        float y = {finalize};
{post_stmts}        out[i] = {post_final};
    }}
}}
"""

# Trailing-contiguous dim-TUPLE mean/sum — transcribes TI's Case-1
# vectorize-along-input order (Reduce.cuh: 32-lane vec4 stripe,
# per-lane sequential combine, ASCENDING warp tree 1→16). Validated
# 50/50 at N=256/1024/4096 (pow2); non-pow2 N refused (named 46/50
# residual). One thread per OUTPUT ROW emulates the warp
# deterministically. REGIME: N=prod(tuple-dims) pow2, N≤8192,
# trailing-contiguous tuple, mean/sum.
TUPLE_MEANSUM_TEMPLATE = """__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out,
                       long long nrow, long long nred) {{
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r < nrow) {{
        const float* row = v + r * nred;
        long long nvec = nred / 4;
        // config mirror (2.7: mnt=512): bh = min(lp2(rows-context...
        // per-row: dim1 is the OUTPUT count; the y-split serves the
        // REDUCE dim: bh = min(lp2(outputs? no — dim1), 512/32=16).
        // For the tuple class rows are huge → bh=16 always; emulate
        // 32×16 threads. ORDER (Doresh's instrumented crack,
        // 67519dfb2): per-thread accumulate → per-LANE y-halving
        // FIRST (block_y_reduce before block_x_reduce in
        // ReduceOp::run()) → THEN the ascending x-warp tree.
        // nY: the y-split fires at N>=8192 (empirical boundary,
        // probe-verified 12 shapes + hardware-verified by Doresh's
        // instrumented run 67519dfb2); y-halving BEFORE the x-tree
        // (block_y_reduce precedes block_x_reduce in ReduceOp::run):
        const int nY = (nred >= 8192) ? 16 : 1;
        const long long stride = 32 * nY;
        float acc[32][16];
        for (int y = 0; y < nY; y++) {{
            for (int t = 0; t < 32; t++) {{
                float vl[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
                for (long long idx = t + 32 * y; idx < nvec; idx += stride) {{
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {{
                        float x = row[idx * 4 + i];
{pre_stmts}                        vl[i] = vl[i] + ({pre_final});
                    }}
                }}
                /* straggler tail (nred %% 4 != 0 — the #44 unlock,
                   mavhir's diagnosis): the remaining scalars stride
                   over the same thread grid into vl[0] (the
                   scalar-path lane); order gate-decided. */
                for (long long j = nvec * 4 + t + 32 * y; j < nred;
                     j += stride) {{
                    float x = row[j];
{pre_stmts}                    vl[0] = vl[0] + ({pre_final});
                }}
                float a = vl[0];
                a = a + vl[1]; a = a + vl[2]; a = a + vl[3];
                acc[t][y] = a;
            }}
        }}
        for (int off = nY / 2; off > 0; off >>= 1) {{
            for (int t = 0; t < 32; t++) {{
                for (int y = 0; y < off; y++) {{
                    acc[t][y] = acc[t][y] + acc[t][y + off];
                }}
            }}
        }}
        float lanes[32];
        for (int t = 0; t < 32; t++) lanes[t] = acc[t][0];
        for (int off = 1; off <= 16; off *= 2) {{
            for (int t = 0; t + off < 32; t++) {{
                lanes[t] = lanes[t] + lanes[t + off];
            }}
        }}
        float y = {finalize};
{post_stmts}        out[r] = {post_final};
    }}
}}
"""

_TUPLE_OPS = {
    'sum':  dict(finalize='lanes[0]'),
    # MeanOps multiplies by the reciprocal FACTOR (f64 1/N narrowed
    # to f32 — the reciprocal mechanism's third home; the non-pow2
    # residual's cause). (float)(1.0/(double)nred) matches torch:
    'mean': dict(finalize='lanes[0] * (float)(1.0 / (double)nred)'),
}

_SPATIAL_OPS = {
    'sum':  dict(op='+', ident='0.0', finalize='yacc[0]'),
    'mean': dict(op='+', ident='0.0',
                 finalize='yacc[0] * (1.0f / (float)chn)'),
}

# Spatial (channel-axis) softmax template — transcribes torch's
# cunn_SpatialSoftMaxForward blockDim.x==1 SPECIALIZATION (SoftMax.cu
# ~299: plain ASCENDING SEQUENTIAL loops — max, then exp-sum, then
# epilogue exp(x-max)/sum). DISPATCH PROOF (getBlockSize ~134):
# dim_threads stays 1 unless inner_size<=64 AND dim_size>=64; conv
# outputs have inner_size = spatial >= hundreds -> the sequential
# path runs for our whole problem class. acc_type<float,true>=float.
# Order matches torch's exactly BY TRANSCRIPTION -> bit-exact-to-
# torch's-order-per-regime (the settled standard). REGIME: dim=1 of
# contiguous (B,C,spatial), inner_size>64 (enforced at the lifter by
# rank>=3; the small-inner case would dispatch differently — refused
# implicitly since conv spatial sizes are large; stated honestly).
SPATIAL_SOFTMAX_TEMPLATE = """__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out,
                       long long outer_size, long long dim_size, long long inner_size) {{
    const long long outer_stride = inner_size * dim_size;
    const long long dim_stride = inner_size;
    for (long long outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {{
        const long long outer_offset = outer_index * outer_stride;
        for (long long inner_index = blockIdx.y * blockDim.y + threadIdx.y;
             inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {{
            const long long data_offset = outer_offset + inner_index;
            float max_input = -3.402823466e+38f;
            for (long long d = 0; d < dim_size; d++) {{
                float x = v[data_offset + d * dim_stride];
{pre_stmts}                float xe = {pre_final};
                max_input = fmaxf(max_input, xe);
            }}
            float sum = 0.0f;
            for (long long d = 0; d < dim_size; d++) {{
                float x = v[data_offset + d * dim_stride];
{pre_stmts2}                float xe = {pre_final};
                sum += expf(xe - max_input);
            }}
            for (long long d = 0; d < dim_size; d++) {{
                // 'i' = the flat-tensor position (aliased so param
                // spellings — p[i/hw%chn] etc — index correctly in
                // this template's scope; the #91 follow-on fix):
                const long long i = data_offset + d * dim_stride;
                float x = v[i];
{pre_stmts3}                float xe = {pre_final};
                float y = expf(xe - max_input) / sum;
{post_stmts}                out[i] = {post_final};
            }}
        }}
    }}
}}
"""


# lse over the channel axis (dim=1, rank≥4 — the #58/#43 shared
# class): the spatial-softmax sequential structure with the LSE
# finalize (log(sum)+max), output (B,spatial). Same regime argument
# (inner_size>64 → torch's spatial path is sequential per position;
# lse = CompositeExplicitAutograd amax→exp→sum→log→add BUT over the
# STRIDED chan axis both component TI reductions visit d ascending
# sequentially at inner_size≫warp — the same-order equivalence).
# TRANSCRIBES torch.logsumexp's ACTUAL dispatch (the #43 gate-catch,
# Doresh's root-cause + resolution 724d8c475): logsumexp is Composite
# (amax→exp→sum→log→add via TI); the component reduces over dim=1-of-
# rank≥4 hit TI Case-2 (vectorize-along-OUTPUT) with block_y firing:
# 4 y-lanes each reduce a strided quarter (stride 4) of the dim via
# vt0=4 accumulators folded LINEARLY; lanes combine PAIRWISE
# ((l0+l2)+(l1+l3) — the halving tree). Verified 0/262144 twice on
# hardware at #43's published shape. NOTE: numpy re-derivations of
# this structure hit host-libm-vs-device-expf 1-ulp differences —
# verify residuals against real CUDA before trusting them (Doresh's
# toolbox lesson).
SPATIAL_LSE_TEMPLATE = """__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out,
                       long long outer_size, long long dim_size,
                       long long inner_size) {{
    for (long long outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {{
        const long long outer_offset = outer_index * dim_size * inner_size;
        for (long long inner_index = blockIdx.y * blockDim.y + threadIdx.y;
             inner_index < inner_size;
             inner_index += gridDim.y * blockDim.y) {{
            const long long data_offset = outer_offset + inner_index;
            // amax pass (composite launch 1) — same lane structure;
            // max is order-insensitive so a flat loop is bit-safe:
            float max_input = -3.402823466e+38f;
            for (long long d = 0; d < dim_size; d++) {{
                float x = v[data_offset + d * inner_size];
                max_input = fmaxf(max_input, x);
            }}
            // sum pass (composite sum-reduce): 4 y-lanes; lane y's
            // vt0=4 accumulators are STRIDE-INTERLEAVED (acc[i] gets
            // d = y + i*4 + j*16 — stride 4 BETWEEN accumulators,
            // 16 between iterations; Doresh's verified kernel
            // 724d8c475), then folded LINEARLY:
            float lane[4];
            for (int y = 0; y < 4; y++) {{
                float vl[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
                long long d = y;
                while (d + 12 < dim_size) {{
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {{
                        float x = v[data_offset + (d + i * 4) * inner_size];
                        vl[i] = vl[i] + expf(x - max_input);
                    }}
                    d += 16;
                }}
                for (int i = 0; i < 4; i++) {{
                    if (d >= dim_size) break;
                    float x = v[data_offset + d * inner_size];
                    vl[i] = vl[i] + expf(x - max_input);
                    d += 4;
                }}
                float a = vl[0];
                a = a + vl[1]; a = a + vl[2]; a = a + vl[3];
                lane[y] = a;
            }}
            // pairwise y-combine (block_y_reduce halving tree):
            float s02 = lane[0] + lane[2];
            float s13 = lane[1] + lane[3];
            float sum = s02 + s13;
            float y = logf(sum) + max_input;
{post_stmts}            out[outer_index * inner_size + inner_index] = {post_final};
        }}
    }}
}}
"""


# SPATIAL_LSE narrow-channel variant (the #58 dispatch: ovs collapses
# to 1 when inner_size is odd → block_height=16 → vpt < threshold →
# split_across_warps FALSE → single-thread per output, vt0=4
# CONTIGUOUS-stride sub-accumulators (acc[i] ← d = i·(dim/4)+j? NO:
# stride-1 groups — acc[i] gets d ∈ {i, i+4, i+8...}? Per Doresh's
# verified kernel58_narrow.cu (9068415a8): contiguous-stride vt0
# striping, linear fold, NO y-combine. Hardware-verified 0/15748992
# ×2 at the published shape.
SPATIAL_LSE_NARROW_TEMPLATE = """__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out,
                       long long outer_size, long long dim_size,
                       long long inner_size) {{
    for (long long outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {{
        const long long outer_offset = outer_index * dim_size * inner_size;
        for (long long inner_index = blockIdx.y * blockDim.y + threadIdx.y;
             inner_index < inner_size;
             inner_index += gridDim.y * blockDim.y) {{
            const long long data_offset = outer_offset + inner_index;
            float max_input = -3.402823466e+38f;
            for (long long d = 0; d < dim_size; d++) {{
                float x = v[data_offset + d * inner_size];
                max_input = fmaxf(max_input, x);
            }}
            // single-thread vt0=4 striped accumulators (stride 1
            // between accumulators within an iteration, 4 between
            // iterations), linear fold, no cross-thread combine:
            float vl[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
            long long d = 0;
            while (d + 3 < dim_size) {{
                #pragma unroll
                for (int i = 0; i < 4; i++) {{
                    float x = v[data_offset + (d + i) * inner_size];
                    vl[i] = vl[i] + expf(x - max_input);
                }}
                d += 4;
            }}
            for (int i = 0; d < dim_size; i++, d++) {{
                float x = v[data_offset + d * inner_size];
                vl[i] = vl[i] + expf(x - max_input);
            }}
            float sum = vl[0];
            sum = sum + vl[1]; sum = sum + vl[2]; sum = sum + vl[3];
            float y = logf(sum) + max_input;
{post_stmts}            out[outer_index * inner_size + inner_index] = {post_final};
        }}
    }}
}}
"""


# THE BROADCAST-EXPANSION TEMPLATE (the #75/#51 class): a row-reduce
# (min/mean over width N) whose OUTPUT is re-expanded — out[b,j] =
# MID(red(b)) ⊕ vec[j] (a param OR a saved tensor). One block per
# row: the reduce uses the VALIDATED tuple-order structure (vec4,
# nY=16 iff N≥8192, y-halving-first, ascending x-tree — emulated
# single-thread, same bit-order); PRE ops ride the load (i = the
# column); MID ops apply to the reduced scalar; the POST-WRITE loop
# broadcasts across j. vec2 (the second input: bias param or the
# saved row) binds as 'bvec'.
BROADCAST_RED_P_TEMPLATE_SIG = """__global__ void k_auto(const float* __restrict__ v,
                       const float* __restrict__ p0,
                       const float* __restrict__ bvec,
                       float* __restrict__ out,
                       long long rows, long long n,
                       long long bn,
                       long long plen_p0) {"""

BROADCAST_RED_TEMPLATE = """__global__ void k_auto(const float* __restrict__ v,
                       const float* __restrict__ bvec,
                       float* __restrict__ out,
                       long long rows, long long n,
                       long long bn) {{
    for (long long r = blockIdx.x; r < rows; r += gridDim.x) {{
        const float* row = v + r * n;
        long long nvec = n / 4;
        const int nY = (n >= 8192) ? 16 : 1;
        const long long stride = 32 * nY;
        float acc[32][16];
        for (int y = 0; y < nY; y++) {{
            for (int t = 0; t < 32; t++) {{
                float vl[4] = {{{red_init}, {red_init}, {red_init}, {red_init}}};
                for (long long idx = t + 32 * y; idx < nvec; idx += stride) {{
                    #pragma unroll
                    for (int ii = 0; ii < 4; ii++) {{
                        const long long i = idx * 4 + ii;
                        float x = row[i];
{pre_stmts}                        vl[ii] = {red_op}(vl[ii], ({pre_final}));
                    }}
                }}
                float a = vl[0];
                a = {red_op}(a, vl[1]); a = {red_op}(a, vl[2]); a = {red_op}(a, vl[3]);
                acc[t][y] = a;
            }}
        }}
        for (int off = nY / 2; off > 0; off >>= 1) {{
            for (int t = 0; t < 32; t++) {{
                for (int y = 0; y < off; y++) {{
                    acc[t][y] = {red_op}(acc[t][y], acc[t][y + off]);
                }}
            }}
        }}
        float lanes[32];
        for (int t = 0; t < 32; t++) lanes[t] = acc[t][0];
        for (int off = 1; off <= 16; off *= 2) {{
            for (int t = 0; t + off < 32; t++) {{
                lanes[t] = {red_op}(lanes[t], lanes[t + off]);
            }}
        }}
        float y = {finalize};
{mid_stmts}        float redv = {mid_final};
        for (long long j = threadIdx.x; j < bn; j += blockDim.x) {{
            out[r * bn + j] = redv {bop} bvec[{bvec_idx}];
        }}
    }}
}}
"""


# softmax-chan → elementwise → RED-chan (the #89 class): the spatial-
# softmax 3-pass sequential structure + a 4TH sequential loop that
# applies the between-elementwise per-d and accumulates the trailing
# reduction; output is (B, spatial) — one value per (outer, inner).
# Same regime as SPATIAL_SOFTMAX (dispatch-proof inner_size>64 →
# torch's cunn_SpatialSoftMaxForward sequential; the trailing max/min
# over the SAME axis with identical visitation is order-matching by
# the same sequential argument).
SPATIAL_SOFTMAX_THEN_RED_TEMPLATE = """__global__ void k_auto(const float* __restrict__ v, float* __restrict__ out,
                       long long outer_size, long long dim_size,
                       long long inner_size) {{
    for (long long outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {{
        const long long outer_offset = outer_index * dim_size * inner_size;
        for (long long inner_index = blockIdx.y * blockDim.y + threadIdx.y;
             inner_index < inner_size;
             inner_index += gridDim.y * blockDim.y) {{
            const long long data_offset = outer_offset + inner_index;
            float max_input = -3.402823466e+38f;
            for (long long d = 0; d < dim_size; d++) {{
                float x = v[data_offset + d * inner_size];
                max_input = fmaxf(max_input, x);
            }}
            float sum = 0.0f;
            for (long long d = 0; d < dim_size; d++) {{
                float x = v[data_offset + d * inner_size];
                sum += expf(x - max_input);
            }}
            float racc = {red_init};
            for (long long d = 0; d < dim_size; d++) {{
                const long long i = data_offset + d * inner_size;
                float x = v[i];
                float y = expf(x - max_input) / sum;
{post_stmts}                float fin = {post_final};
                racc = {red_op}(racc, fin);
            }}
            out[outer_index * inner_size + inner_index] = racc;
        }}
    }}
}}
"""


def emit_reduction_chan(chain_file, pid, spellings, red_kind):
    """Channel-strided min/max: pre-epilogue must be EMPTY (ops before
    the reduce would need the strided-load fusion — refuse for now);
    post-epilogue fuses on the reduced value."""
    harness = REDUCTION_HARNESS.format(chain_file=chain_file, pid=pid,
                                       spellings=spellings,
                                       red_kind=f'chan_{red_kind}')
    # reuse the split by rewriting the chain fact: reduction_chan(K) -> reduction(chan_K):
    txt = open(chain_file).read().replace(
        f'reduction_chan({red_kind})', f'reduction(chan_{red_kind})')
    with open(chain_file, 'w') as f:
        f.write(txt)
    harness_path = os.path.join(REPO, 'lib', f'.bridge_chan_{pid}.pl')
    with open(harness_path, 'w') as f:
        f.write(harness)
    r = subprocess.run(['swipl', '-q', harness_path],
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        raise RuntimeError(
            f'{pid}: chan-reduce fuse failed (epilogue vocabulary-gap). '
            f'{r.stderr.strip()[:150]}')
    sections = {}
    cur = None
    for line in r.stdout.splitlines():
        if line.startswith('===') and line.endswith('==='):
            cur = line.strip('=')
            sections[cur] = []
        elif cur is not None:
            sections[cur].append(line)
    pre = '\n'.join('            ' + l.strip()
                    for l in sections.get('PRE', []) if l.strip())
    # THE #79 FIX (the param-chan spelling inside a COLLAPSING
    # chan-reduce): the Prolog spelling indexes p0 by the elementwise
    # output index (i / hw_P % chn_P) — correct in elementwise
    # kernels, WRONG inside the channel loop where the traversal
    # variable is ci. Rewrite the index to the loop var:
    pre = re.sub(r'\[i / hw_(\w+) % chn_\1\]', '[ci]', pre)
    pre_final = (sections.get('PREFINAL', ['x'])[0] or 'x').strip()
    if pre:
        pre += '\n'
    post = '\n'.join('        ' + l.strip()
                     for l in sections.get('POST', []) if l.strip())
    post_final = (sections.get('POSTFINAL', ['y'])[0] or 'y').strip()
    if post:
        post += '\n'
    if red_kind in ('mean', 'sum'):
        # pow2-C guard (honesty gap closed): the *(1/chn) finalize +
        # y-split emulation are validated at pow2-C only (100/100 at
        # 16/32/64/128); non-pow2 C = the MeanOps-finalize read (named).
        # C provable via conv out_channels tracking = the same morning
        # build as conv-shape arithmetic; until then: the template
        # emits with the regime STATED in the kernel comment and
        # Doresh's gate enforces at published shapes. (Full lifter
        # proof lands with item-2 of the loose-ends list.)
        cfg = _SPATIAL_OPS[red_kind]
        pre_sp = '\n'.join('                    ' + l.strip()
                           for l in sections.get('PRE', []) if l.strip())
        if pre_sp:
            pre_sp += '\n'
        post_sp = '\n'.join('        ' + l.strip()
                            for l in sections.get('POST', []) if l.strip())
        if post_sp:
            post_sp += '\n'
        return SPATIAL_MEANSUM_TEMPLATE.format(
            pre_stmts=pre_sp, pre_stmts2=pre_sp,
            pre_final=pre_final, post_stmts=post_sp,
            post_final=post_final, **cfg)
    if red_kind == 'softmax':
        pre_sp = '\n'.join('                ' + l.strip()
                           for l in sections.get('PRE', []) if l.strip())
        # the #13 fix (the #79 ci-lesson at this template): param
        # indexes inside the chan loop rewrite to the loop var 'd'
        # (the channel) — the elementwise spellings index by the
        # flat position 'i' which doesn't exist here:
        pre_sp = re.sub(r'\[i % plen_(\w+)\]', '[d]', pre_sp)
        pre_sp = re.sub(r'\[i / hw_(\w+) % chn_\1\]', '[d]', pre_sp)
        if pre_sp:
            pre_sp += '\n'
        post_sp = '\n'.join('                ' + l.strip()
                            for l in sections.get('POST', []) if l.strip())
        if post_sp:
            post_sp += '\n'
        return SPATIAL_SOFTMAX_TEMPLATE.format(
            pre_stmts=pre_sp, pre_stmts2=pre_sp, pre_stmts3=pre_sp,
            pre_final=pre_final, post_stmts=post_sp,
            post_final=post_final)
    return CHAN_TEMPLATE.format(op=_CHAN_OPS[red_kind], pre_stmts=pre,
                                pre_final=pre_final, post_stmts=post,
                                post_final=post_final)


def emit_reduction(chain_file, pid, spellings, red_kind='softmax',
                   variant='reg'):
    """The buffered-row path: compose pre/post epilogues via the bridge,
    wrap in the certified Reg-path template for the reduction kind.

    HONEST-GAP: param-tensor ops (add_param/mul_param/sub_param) are NOT
    yet plumbed into the reduction templates (their loop-var is 'off' not
    'i', and the signatures lack p/plen) — the elementwise add_param
    spelling emits UNDECLARED variables here (Doresh's 4th catch, #58/
    #91). Refuse rather than emit invalid CUDA."""
    chain_text = open(chain_file).read()
    if chain_text.count('reduction(') > 1:
        raise RuntimeError(
            f'{pid}: MULTI-reduction chain — nested/serial reductions need '
            f'the multi-boundary composition (named build, refused honestly).')
    # POSITION-AWARE (the #42 row-form fix): vector params BEFORE the
    # reduction ride the PRE section (the load loop has the column
    # index — i aliased to off; for the collapsed (B,C) row-form the
    # chan-spelling i/1%C = the column ✓). Params AFTER the reduction
    # = the broadcast-expansion class (still refused):
    red_pos = chain_text.find('reduction(')
    _post_param_softmax = False
    for pop in ('add_param(', 'mul_param(', 'sub_param(',
                'add_param_chan(', 'mul_param_chan(', 'sub_param_chan('):
        pp = chain_text.find(pop)
        if pp != -1 and red_pos != -1 and pp > red_pos:
            # SOFTMAX EXCEPTION (#38): softmax preserves shape — a
            # trailing vector param rides the write-loop elementwise
            # with the ROW's chan index (p0[blockIdx.x % chn]); no
            # broadcast-expansion:
            if red_kind == 'softmax':
                _post_param_softmax = True
                continue
            raise RuntimeError(
                f'{pid}: VECTOR {pop[:-1]} in a reduction EPILOGUE — post-'
                f'reduce vector params BROADCAST the output (out[b,j] = '
                f'red(b) ⊕ p[j]: 2D output from 1D reduce) = the broadcast-'
                f'expansion class; needs its own template (named build). '
                f'(Scalar params pass: p[0] is context-free. Pre-reduce '
                f'chan-params now plumb via the general insert_param_args.)')
    has_scalar_param = any(
        p in chain_text for p in
        ('add_param_scalar', 'mul_param_scalar', 'sub_param_scalar'))
    harness = REDUCTION_HARNESS.format(chain_file=chain_file, pid=pid,
                                       spellings=spellings,
                                       red_kind=red_kind)
    harness_path = os.path.join(REPO, 'lib', f'.bridge_red_{pid}.pl')
    with open(harness_path, 'w') as f:
        f.write(harness)
    r = subprocess.run(['swipl', '-q', harness_path],
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        raise RuntimeError(
            f'reduction fuse failed for {pid}: vocabulary-gap in pre/post '
            f'epilogue or non-softmax reduction. {r.stderr.strip()[:200]}')
    sections = {}
    cur = None
    for line in r.stdout.splitlines():
        if line.startswith('===') and line.endswith('==='):
            cur = line.strip('=')
            sections[cur] = []
        elif cur is not None:
            sections[cur].append(line)
    pre = '\n'.join('            ' + l.strip() for l in sections.get('PRE', []) if l.strip())
    post = '\n'.join('            ' + l.strip() for l in sections.get('POST', []) if l.strip())
    if _post_param_softmax:
        # the row-chan param index in the softmax write loop
        # (#38: scale (1,C,1,1,1) — the row = (b,c); chan =
        # blockIdx.x % C — chn_p0 arrives via insert_param_args):
        post = re.sub(r'\[i % plen_(\w+)\]',
                      r'[blockIdx.x % plen_\1]', post)
        post = re.sub(r'\[i / hw_(\w+) % chn_\1\]',
                      r'[blockIdx.x % chn_\1]', post)
    pre_final = (sections.get('PREFINAL', ['x'])[0] or 'x').strip()
    post_final = (sections.get('POSTFINAL', ['y'])[0] or 'y').strip()
    if pre: pre += '\n'
    if post: post += '\n'
    # per-param buffers (the aliasing fix): scalar params in reductions
    # are p0/p1/... — insert each as its own const float* arg:
    patoms = sorted(set(_re.findall(r"param_scalar\((p\d+)\)", chain_text)),
                    key=lambda a: int(a[1:]))
    # chan-params in reduction PREs (the #42 rowred hole — Doresh's
    # two-part catch): plumb p + hw_/chn_ length args AND alias the
    # chan-spelling's index var i to the template's loop var 'off'
    # (the row-form: hw=1 so i/1%chn = the column):
    chatoms = sorted(set(_re.findall(r"param_chan\((p\d+)\)", chain_text)),
                     key=lambda a: int(a[1:]))
    def _with_p(body):
        allp = patoms + [a for a in chatoms if a not in patoms]
        if allp:
            bufs = ', '.join(f'const float* __restrict__ {a}' for a in allp)
            body = body.replace(
                'const float* __restrict__ v, float* __restrict__ out',
                f'const float* __restrict__ v, {bufs}, '
                'float* __restrict__ out')
        if chatoms:
            lens = ', '.join(f'long long hw_{a}, long long chn_{a}'
                             for a in chatoms)
            # append the length args at the signature end:
            body = _re.sub(
                r'(__global__ void k_auto\([^)]*)\)',
                r'\1, ' + lens + ')', body, count=1)
        return body
    if chatoms and pre:
        # the chan spelling references i -- alias it to the PRE loop's
        # REAL per-element index variable. Doresh's #8 catch (Sept 2026):
        # this alias was previously hardcoded to 'off' unconditionally,
        # but 'off' is only the correct loop-index name for the
        # softmax/logsumexp-family templates (SOFTMAX_TEMPLATE,
        # LSE_TEMPLATE, LSE_NARROW_TEMPLATE, GRIDSTRIDE_*) -- the
        # fallback TI_TEMPLATE (used for sum/mean/min/max, e.g. #8's
        # add_param_chan+reduction(sum) chain) uses a DIFFERENT
        # loop-variable naming (idx/li, real per-element index =
        # idx*4+li), and 'off' doesn't even exist in that template's
        # scope at the PRE injection point (it's declared LATER, inside
        # the separate warp/block-reduce loop) -- causes an nvcc
        # compile failure (undeclared identifier 'off'), not a silent
        # wrong-answer bug, but a real codegen bug nonetheless. Select
        # the alias expression based on which template will actually be
        # used.
        # TI_NARROW_TEMPLATE (variant='ti_narrow') uses the SAME 'off'
        # loop-var naming as LSE_NARROW_TEMPLATE (both are single-warp,
        # stride-32, vt0=4 templates transcribed from the same Reduce.cuh
        # narrow regime) -- 'idx*4+li' is only correct for the WIDE
        # TI_TEMPLATE's 512-thread nvec-loop structure.
        if red_kind in ('softmax', 'logsumexp') or variant == 'ti_narrow':
            i_alias = 'off'
        else:
            i_alias = 'idx * 4 + li'
        pre = f'            const long long i = {i_alias};\n' + pre
    if red_kind == 'softmax':
        tmpl = GRIDSTRIDE_SOFTMAX_TEMPLATE if variant == 'gridstride' \
            else SOFTMAX_TEMPLATE
        return _with_p(tmpl.format(
            pre_stmts=pre, pre_final=pre_final,
            post_stmts=post, post_final=post_final))
    if red_kind == 'logsumexp':
        if variant == 'gridstride':
            tmpl = GRIDSTRIDE_LSE_TEMPLATE
        elif variant == 'lse_narrow':
            tmpl = LSE_NARROW_TEMPLATE
        elif variant == 'lse_narrow_scalar':
            tmpl = LSE_NARROW_SCALAR_TEMPLATE
        else:
            tmpl = LSE_TEMPLATE
        return _with_p(tmpl.format(
            pre_stmts=pre, pre_final=pre_final,
            post_stmts=post, post_final=post_final))
    cfg = _TI_OPS[red_kind]
    if variant == 'ti_narrow':
        body = TI_NARROW_TEMPLATE.format(
            pre_stmts=pre, pre_final=pre_final,
            post_stmts=post, post_final=post_final,
            op_sel=_TI_NARROW_OP_SEL[red_kind], **cfg)
    else:
        body = TI_TEMPLATE.format(pre_stmts=pre, pre_final=pre_final,
                                  post_stmts=post, post_final=post_final,
                                  **cfg)
    if '_addf' in body:
        body = _ADDF_HELPER + body
    body = _with_p(body)
    return body


def emit(chain_file, pid):
    """Run the bridge: consult the lifted chain-fact, emit CUDA. No hand-work."""
    harness = BRIDGE_HARNESS.format(chain_file=chain_file, pid=pid,
                                    spellings=SPELLINGS)
    harness_path = os.path.join(REPO, 'lib', f'.bridge_{pid}.pl')
    with open(harness_path, 'w') as f:
        f.write(harness)
    r = subprocess.run(['swipl', '-q', harness_path],
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode == 3:
        # emit_kernel failed: a spell_stmt is missing for some op in the chain.
        raise RuntimeError(
            f'fuse/codegen failed for {pid}: a lifted op has no spell_stmt '
            f'(vocabulary-gap — name it, add the fact). chain in {chain_file}')
    if r.returncode != 0:
        raise RuntimeError(f'bridge error: {r.stderr.strip()}')
    return r.stdout


def compile_check(cuda_text):
    """nvcc -c the emitted CUDA. Returns True (compiles), False (invalid),
    or None (nvcc absent on this host — check DEFERRED to the enclave,
    NOT assumed-pass). PASS-ladder: swipl-ran (weak) < compiles (floor)
    < gates-0-ULP (real improvement)."""
    import shutil
    import tempfile
    # prefer CUDA_HOME's real nvcc: the PATH nvcc can be a WRAPPER whose
    # nvvm/bin/cicc is not beside it (RC 127; Mavdil's build-fact):
    cuda_home = os.environ.get(
        'CUDA_HOME',
        '/nix/store/3y4mvymhwmnfi5d0vwyzcw7f7sqnqnkd-cuda-merged-12.8')
    nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
    if not os.path.exists(nvcc):
        nvcc = shutil.which('nvcc')
    if nvcc is None:
        return None
    d = tempfile.mkdtemp()
    src = os.path.join(d, 'k.cu')
    with open(src, 'w') as f:
        # no cuda_runtime.h needed: plain __global__ kernels compile
        # bare; the wrapper nvcc lacks default includes (Mavdil's
        # build-fact: CUDA_HOME include must be passed explicitly
        # when headers are needed).
        f.write(cuda_text)
    inc = os.path.join(cuda_home, 'include')
    # --fmad=false = THE EMISSION STANDARD (the 8th bug-class, Doresh's
    # #16 FMA-contraction catch): nvcc's default contraction changes
    # rounding vs torch's gpu_kernel compilation. Verified: #16 fixed
    # 0/536M (Doresh) + #25/#32 published-shape gates STAY 0-ULP with
    # fmad off — torch doesn't rely on contraction where we emit.
    # Full-board confirmation = Doresh's re-gate.
    cmd = [nvcc, '-c', src, '--fmad=false', '-o', os.path.join(d, 'k.o'),
           '-Wno-deprecated-gpu-targets']
    if os.path.isdir(inc):
        cmd += ['-I', inc]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r.returncode == 0


def run_segments(problem_path, pid):
    """Multi-stage path: one fused kernel PER inter-stage segment.
    All segments must express (rc=4 = inexpressible segment = honest
    GAP) — never partial. Segments are elementwise-only (reductions
    mid-chain refuse)."""
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, 'lib', 'lift_chain.py'),
         problem_path, pid, '--segments'],
        capture_output=True, text=True)
    if r.returncode == 4:
        raise RuntimeError(
            f'{pid}: multi-stage chain has an inexpressible segment '
            f'(honest GAP — all-or-nothing per problem).')
    if r.returncode != 0:
        raise RuntimeError(f'lift failed: {r.stderr.strip()[:200]}')
    # group each chain fact WITH its %% meta lines (they follow it):
    grouped = []
    for line in r.stdout.splitlines():
        if line.startswith('chain('):
            grouped.append([line])
        elif line.startswith('%%') and grouped and \
                not line.startswith('%% seg'):
            grouped[-1].append(line)
    facts = ['\n'.join(g) for g in grouped]
    kernels = []
    for fact in facts:
        seg_pid = fact.split('(', 1)[1].split(',', 1)[0]
        first_line = fact.splitlines()[0]
        # EMPTY SEGMENT FACT (chain(segN, []) — #8's seg0): stage-
        # only content between boundaries; NO kernel to emit (an
        # identity kernel here = the no-op-kernel class). Skip:
        _inner_seg = first_line.split('[', 1)[1].rsplit(']', 1)[0].strip()
        if _inner_seg == '':
            continue
        if 'reduction' in first_line:
            # REDUCTION SEGMENT (the day-2 generalization — the #42
            # split-route idea applied to all multi-stage problems):
            # route through run()'s full reduction machinery with the
            # segment fact + its metas. Chan/tuple/row routes all
            # apply; their guards all hold (prove-or-refuse rides).
            _, cuda = run(problem_path, seg_pid, fact_override=fact)
            cuda = cuda.replace('k_auto', f'k_auto_{seg_pid.rsplit("_", 1)[-1]}')
            kernels.append(cuda)
            continue
        fact, manifest = normalize_params(fact)  # aliasing fix per-segment
        chain_file = write_chain_file(fact, seg_pid)
        cuda = emit(chain_file, seg_pid)
        cuda = insert_param_args(cuda, manifest)
        cuda = insert_input2_arg(cuda, fact)
        # the manifest at THIS exit too (Mavdil's 20-unit finding —
        # the segments path was the remaining bypass; elementwise
        # segments have no reduce_width/spatial_inner):
        cuda = _call_manifest(
            seg_pid, manifest, None, None, fact,
            launch={'grid': '(n+255)/256', 'block': 256,
                    'smem': 0}) + cuda  # elementwise segments = the
        # canonical elementwise geometry (certified)
        # rename per-segment (the #35/#60/#96 nvcc-fail: TWO
        # elementwise segments both kept k_auto — duplicate
        # definition; the reduction path renamed, this path didn't):
        cuda = cuda.replace('k_auto',
                            f'k_auto_{seg_pid.rsplit("_", 1)[-1]}')
        # THE SEGMENT-SAVED SIGNATURE (#8's add_saved-in-a-segment):
        # the saved-tensor ops spell x2[i], but the elementwise
        # segment templates declare only (v, out, n) — add the x2
        # param when the body references it (the whole-chain path's
        # x2-splice, ported to the segment route):
        if 'x2[' in cuda and '__restrict__ x2' not in cuda:
            # spell it v2 (not x2): the runner's EXISTING
            # elementwise2 launch class is (v, v2, out, n) — the
            # #8 rc=-3 was the x2-spelling missing that class:
            cuda = cuda.replace('x2[', 'v2[')
            cuda = cuda.replace(
                'const float* __restrict__ v,',
                'const float* __restrict__ v,\n'
                '                       '
                'const float* __restrict__ v2,', 1)
        ok = compile_check(cuda)
        if ok is False:
            raise RuntimeError(f'{seg_pid}: segment failed nvcc compile check — GAP.')
        if ok is None:
            cuda = '/* COMPILE-CHECK DEFERRED */\n' + cuda
        kernels.append(cuda)
    if not kernels:
        raise RuntimeError(f'{pid}: no expressible segments.')
    return facts, kernels


class NoEpilogue(Exception):
    """The problem's whole forward is stage-boundaries (conv/bn/pool...):
    there is NOTHING to fuse. The identity emit would be mathematically
    correct but is NOT an improvement — torch does all the work. Distinct
    from a GAP (nothing missing) and from a PASS (nothing improved).
    Ruling (Bocher, per Doresh's #72 flag + Iyun's category question):
    compile-verified-but-not-an-improvement; excluded from the
    improvement count; the census reports it as NO-EPILOGUE."""
    pass


def _call_manifest(pid, manifest, reduce_width, spatial_inner, fact,
                   tuple_n=None, tuple_rows=None, launch=None,
                   chan_count=None):
    """The emitter's known shapes as embedded JSON (Mavdil's board-
    scale ask). ALL of run()'s exits must pass through this — the
    early-return-bypasses-the-tail pattern is twice-bitten (the
    census-grep hole's layer + the broadcast route's missing
    manifest, Doresh's flag)."""
    import json as _json
    mani = {
        'pid': pid,
        'params': [{'atom': a, 'attr': nm, 'kind': kd}
                   for a, nm, kd in (manifest or [])],
        'reduce_width': reduce_width,
        'spatial_inner': spatial_inner,
        'tuple_n': tuple_n,
        'tuple_rows': tuple_rows,
        'chan_count': chan_count,
        'launch': launch,
        'fact': (fact.splitlines()[0][:400] if fact else ''),
    }
    return f'/* CALL-MANIFEST {_json.dumps(mani)} */\n'


def run(problem_path, pid, fact_override=None):
    """The full pipe: lift -> wire -> fuse -> codegen. Returns emitted CUDA.
    fact_override: run the pipe on a pre-split fact (the two-red route)."""
    fact = (fact_override if fact_override is not None
            else lift(problem_path, pid))   # 1. LIFT (was: run + read)
    inner = fact.split('[', 1)[1].rsplit(']', 1)[0].strip()
    if inner == '':
        # THE NOMENCLATURE BRIDGE (Heath's ruling): route by whether
        # the walker DECLINED composites (named in the fact by the
        # lift's reporting bridge) or genuinely saw none. The
        # declined list IS the work queue.
        _opq = [ln.split()[2:] for ln in fact.splitlines()
                if ln.startswith('%% opaque_composite ')]
        if _opq:
            _named = ', '.join(
                f'{a}({c2}: {w})' for a, c2, w in _opq)
            raise NoEpilogue(
                f'{pid}: [NO-EPILOGUE-IN-REACH] the walker produced '
                f'no fusable stages. Opaque composites encountered: '
                f'{_named}. Nothing inspected inside them.')
        raise NoEpilogue(
            f'{pid}: [NOTHING-TO-FUSE-YET] whole model is '
            f'stage-boundaries with no unopened composite — nothing '
            f'for US to fuse yet; identity would be correct but is '
            f'NOT an improvement.')
    # reduce-width meta (the capacity check — Doresh's 7th catch:
    # the Reg-path covers N<=8192; beyond = silent half-zeros):
    spatial_inner = None
    if '%% spatial_inner ' in fact:
        for ln in fact.splitlines():
            if ln.startswith('%% spatial_inner '):
                spatial_inner = int(ln.split()[-1])
        fact = '\n'.join(l for l in fact.splitlines()
                         if not l.startswith('%% spatial_inner'))
    chan_count = None
    if '%% chan_count ' in fact:
        for ln in fact.splitlines():
            if ln.startswith('%% chan_count '):
                chan_count = int(ln.split()[-1])
        fact = '\n'.join(l for l in fact.splitlines()
                         if not l.startswith('%% chan_count'))
    fold_chan = fold_spatial = None
    if '%% fold_chan ' in fact:
        for ln in fact.splitlines():
            if ln.startswith('%% fold_chan '):
                fold_chan = int(ln.split()[-1])
            if ln.startswith('%% fold_spatial '):
                fold_spatial = int(ln.split()[-1])
        fact = '\n'.join(l for l in fact.splitlines()
                         if not l.startswith('%% fold_'))
    tuple_n = None      # populated by the tuple route (visible at
    tuple_rows = None   # the tail — Mavdil's null-width finding)
    _launch = None      # launch geometry (certified values only —
                        # WRONG geometry is worse than none; Mavdil's
                        # stopped-pass lesson)
    reduce_width = None
    if '%% reduce_width ' in fact:
        for ln in fact.splitlines():
            if ln.startswith('%% reduce_width '):
                reduce_width = int(ln.split()[-1])
        fact = '\n'.join(l for l in fact.splitlines()
                         if not l.startswith('%% reduce_width'))
    fact, manifest = normalize_params(fact)   # multi-param aliasing fix
    chain_file = write_chain_file(fact, pid)  # 2. WIRE (was: hand-paste)
    if re.search(r"reduction\((min|mean)\)", fact) and \
            re.search(r"(add_param|add_saved|mul_param|sub_param)"
                      r"(_chan)?\('?\w*'?\)\]\)\.", fact) and \
            'reduction_tuple(' not in fact and \
            fact.count('reduction(') == 1:
        # THE BROADCAST-EXPANSION ROUTE (#75/#51): row-reduce then a
        # trailing VEC op (param or saved) that re-expands the output.
        # The reduce structure = the validated tuple-order (min is
        # order-insensitive; mean is the validated regime + the
        # reciprocal finalize). PRE ops ride the load; MID (between
        # the reduce and the vec-op) applies to the scalar.
        m0 = re.search(r'chain\((\w+), \[(.*)\]\)\.', fact, re.S)
        opl = [o.strip() for o in re.split(r',\s*(?![^()]*\))', m0.group(2))]
        ri = next(i for i, o in enumerate(opl)
                  if o.startswith('reduction('))
        rkind = opl[ri][len('reduction('):-1]
        last = opl[-1]
        vm = re.match(r"(add|mul|sub)_(param|saved)(_chan)?\('?([\w.]+)'?\)",
                      last)
        if vm is None:
            raise RuntimeError(f'{pid}: broadcast route: unparsed vec op '
                               f'{last}')
        bop = {'add': '+', 'mul': '*', 'sub': '-'}[vm.group(1)]
        pre_ops, mid_ops = opl[:ri], opl[ri + 1:-1]
        if reduce_width is None:
            raise RuntimeError(f'{pid}: broadcast route needs proven '
                               f'width; refuse.')
        if reduce_width % 4 != 0 or not (256 <= reduce_width <= 577600):
            raise RuntimeError(f'{pid}: broadcast route: width '
                               f'{reduce_width} outside the validated '
                               f'tuple-order regime; refuse.')
        # build PRE/MID via the segment spelling machinery: emit a
        # fake elementwise chain for each section and extract stmts:
        def _spell(ops_list, seed_var):
            if not ops_list:
                return '', seed_var
            fake = f'chain({pid}_bx, [' + ', '.join(ops_list) + ']).'
            fk, mani = normalize_params(fake)
            cf = write_chain_file(fk, f'{pid}_bx')
            code = emit(cf, f'{pid}_bx')
            body = []
            fin = seed_var
            for ln in code.splitlines():
                ls = ln.strip()
                if ls.startswith('float t') and '=' in ls:
                    body.append(ls.replace(' x ', f' {seed_var} ')
                                  .replace('(x', f'({seed_var}')
                                  .replace(' x;', f' {seed_var};')
                                  .replace(' x)', f' {seed_var})'))
                    fin = ls.split('=')[0].replace('float', '').strip()
            return ('\n'.join('        ' + b for b in body) +
                    ('\n' if body else '')), fin
        _pre_has_param = any('param' in o for o in pre_ops)
        pre_stmts, pre_final = _spell(pre_ops, 'x')
        # re-indent pre for its position (deeper):
        pre_stmts = '\n'.join(('                        ' + l.strip())
                              for l in pre_stmts.splitlines() if l.strip())
        if pre_stmts:
            pre_stmts += '\n'
        mid_stmts, mid_final = _spell(mid_ops, 'y')
        red_cfg = {'min': ('fminf', '3.402823466e+38f', 'lanes[0]'),
                   'mean': ('_ba', '0.0f',
                            'lanes[0] * (float)(1.0 / (double)n)')}[rkind]
        cuda = BROADCAST_RED_TEMPLATE.format(
            red_op=red_cfg[0], red_init=red_cfg[1], finalize=red_cfg[2],
            pre_stmts=pre_stmts, pre_final=pre_final,
            mid_stmts=mid_stmts, mid_final=mid_final,
            bop=bop,
            # add_saved = a MATRIX bvec (the saved tensor is per-row
            # — original_x (B,bn); read bvec[r*bn+j]) vs
            # add_param-broadcast = a VECTOR (bias (bn,); read
            # bvec[j]) — the #51-vs-#75 distinction:
            bvec_idx=('r * bn + j' if vm.group(2) == 'saved'
                      else 'j'))
        if _pre_has_param:
            # THE P-BUFFER EXTENSION (#51): the PRE carries a vector
            # param — the spelling indexes p0[i %% plen_p0] which is
            # correct inside the reduce loop (i = the column). Swap
            # the signature to the p0 variant:
            _plain_sig = cuda[cuda.index('__global__'):
                              cuda.index(') {{'.replace('{{', '{')) + 3]
            _p_sig = BROADCAST_RED_P_TEMPLATE_SIG[
                BROADCAST_RED_P_TEMPLATE_SIG.index('__global__'):]
            cuda = cuda.replace(_plain_sig, _p_sig, 1)
        if rkind == 'mean':
            cuda = ('__device__ __forceinline__ float _ba(float a, float b)'
                    ' { return a + b; }\n' + cuda)
        # the broadcast-red kernel: ONE sequential thread per row
        # (r = blockIdx.x, gridDim stride; the 32x16 structure is
        # emulated in-thread) — launch <<<rows, 1>>> (the #75 stamp):
        cuda = _call_manifest(pid, manifest, reduce_width,
                              spatial_inner, fact,
                              launch={'grid': 'rows', 'block': 1,
                                      'smem': 0}) + cuda
        ok = compile_check(cuda)
        if ok is False:
            raise RuntimeError(f'{pid}: broadcast kernel FAILED nvcc.')
        return fact, cuda
    if re.search(r"chain\(\w+, \[(avgpool3d_k2(, )?)+\]\)", fact):
        # THE #72 CLASS: a pure avgpool chain — N sequential
        # 2x2x2-block-mean kernels (each halves D/H/W):
        _np = fact.count('avgpool3d_k2')
        kparts = []
        for _pi in range(_np):
            kparts.append(AVGPOOL3D_K2_TEMPLATE.format(
                suffix=(f'_p{_pi}' if _np > 1 else '')))
        cuda = '\n'.join(kparts)
        manis = []
        for _pi in range(_np):
            _sfx = (f'_p{_pi}' if _np > 1 else '')
            manis.append(_call_manifest(
                f'{pid}{_sfx}' if _sfx else pid, manifest, None,
                None,
                f'chain({pid}{_sfx}, [avgpool3d_k2]).',
                launch={'grid': '(nout+255)/256', 'block': 256,
                        'smem': 0}))
        cuda = '\n'.join(manis) + cuda
        return fact, cuda
    if 'reduction_chan_fold(' in fact:
        # THE FOLD SPLIT (#13/#24/#36 — the morning build): TWO
        # chan-reduces with DIFFERENT extents in one chain. Split
        # BEFORE the second reduction; each half re-runs with its
        # OWN extents (the #42 two-kernel pattern generalized). The
        # fold rewrites to reduction_chan with the FOLD extents.
        m0 = re.search(r'chain\((\w+), \[(.*)\]\)\.', fact, re.S)
        if not m0:
            raise RuntimeError(f'{pid}: fold split: unparseable fact.')
        opl = [o.strip() for o in re.split(r',\s*(?![^()]*\))',
                                           m0.group(2))]
        _redix = [i for i, o in enumerate(opl)
                  if o.startswith(('reduction_chan(',
                                   'reduction_chan_fold(',
                                   'reduction('))]
        if len(_redix) == 1:
            # the fold ALONE — rewrite in place, single kernel:
            fact2 = fact.replace('reduction_chan_fold(',
                                 'reduction_chan(')
            return run(problem_path, pid, fact_override=(
                fact2 + f'\n%% chan_count {fold_chan}'
                + f'\n%% spatial_inner {fold_spatial}'))
        if len(_redix) != 2:
            raise RuntimeError(
                f'{pid}: fold split: {len(_redix)} reductions — '
                f'named build.')
        # THE #13 LESSON (param-placement): cut right AFTER the
        # FIRST reduction — ops between the two reductions ride the
        # SECOND kernel's PRE, where the layout restores the
        # original chan (the fold half's spelling indexes the
        # FOLDED chan — wrong extents for between-ops params):
        _cut = _redix[0] + 1
        head, tail = opl[:_cut], opl[_cut:]
        def _requote(ops_):
            return [_re.sub(r"\((p\d+)\)", r"('\1')", o)
                    for o in ops_]
        def _meta_for(ops_):
            # the half containing the FOLD gets the fold extents;
            # the other half gets the original chan extents:
            if any(o.startswith('reduction_chan_fold(')
                   for o in ops_):
                return (f'\n%% chan_count {fold_chan}'
                        f'\n%% spatial_inner {fold_spatial}')
            return ((f'\n%% chan_count {chan_count}'
                     if chan_count is not None else '')
                    + (f'\n%% spatial_inner {spatial_inner}'
                       if spatial_inner is not None else ''))
        fact_a = (f'chain({pid}_f, ['
                  + ', '.join(_requote(head)).replace(
                      'reduction_chan_fold(', 'reduction_chan(')
                  + ']).' + _meta_for(head))
        fact_b = (f'chain({pid}_s, ['
                  + ', '.join(_requote(tail)).replace(
                      'reduction_chan_fold(', 'reduction_chan(')
                  + ']).' + _meta_for(tail))
        _, ka = run(problem_path, f'{pid}_f', fact_override=fact_a)
        _, kb = run(problem_path, f'{pid}_s', fact_override=fact_b)
        # the atom-as-attr rewrite (the #42-split lesson applied
        # here): the sub-runs' manifests carry the re-quoted ATOM
        # ('p0') as attr — rewrite to the real model attr:
        for _atom, _attr, _kind in (manifest or []):
            if _atom and _attr and _atom != _attr:
                ka = ka.replace(f'"attr": "{_atom}"',
                                f'"attr": "{_attr}"')
                kb = kb.replace(f'"attr": "{_atom}"',
                                f'"attr": "{_attr}"')
        ka = ka.replace('k_auto', 'k_auto_fold')
        kb = kb.replace('k_auto', 'k_auto_second')
        combined = (f'// fold split (the #13/#24/#36 class): kernel 1\n'
                    f'// carries one chan-reduce, kernel 2 the other —\n'
                    f'// different extents each, launched sequentially.\n'
                    + ka + '\n' + kb)
        return fact_a + '\n' + fact_b, combined
    if 'reduction_tuple(' in fact and ', reduction(' in fact:
        # THE #42 CLASS: tuple-reduce THEN a row-reduce (the shape-
        # collapse chain: keepdim tuple-mean → (B,C,1,1) ≡ (B,C) →
        # elementwise → lse/softmax over C). TWO KERNELS: [tuple] +
        # [row-reduce w/ the between-ops as PRE + trailing as POST].
        # Split the fact at the first op after the tuple-reduce:
        m0 = re.search(r'chain\((\w+), \[(.*)\]\)\.', fact, re.S)
        if not m0:
            raise RuntimeError(f'{pid}: two-red split: unparseable fact.')
        opl = [o.strip() for o in re.split(r',\s*(?![^()]*\))', m0.group(2))]
        ti = next(i for i, o in enumerate(opl)
                  if o.startswith('reduction_tuple('))
        head, tail = opl[:ti + 1], opl[ti + 1:]
        if any(o.startswith('reduction_tuple(') for o in tail):
            raise RuntimeError(f'{pid}: >1 tuple-reduce — named build.')
        # rebuild metas from parsed vars (run() strips %% lines early;
        # reduce_width/spatial_inner are already parsed; tuple_* still
        # ride in `fact` since the tuple route parses them later):
        meta = [l for l in fact.splitlines() if l.startswith('%%')]
        ma = '\n'.join(l for l in meta if l.startswith('%% tuple_'))
        mb = (f'%% reduce_width {reduce_width}'
              if reduce_width is not None else '')
        # RE-QUOTE already-normalized param atoms before the second
        # normalize_params pass (Doresh's #42 catch): `fact` here has
        # ALREADY been through this run()'s own normalize_params call,
        # so `tail`'s param ops read like add_param_chan(p0) -- a BARE
        # atom, not the quoted original name PARAM_OP_RE expects
        # ('([^']+)'). fact_b gets passed BACK through run() via
        # fact_override, which calls normalize_params(fact_b) AGAIN;
        # without re-quoting, that second pass silently fails to match
        # (bare p0 != 'name'), manifest comes back EMPTY, and
        # insert_param_args never adds p0/hw_p0/chn_p0 to the kernel B
        # signature -- yet the reduction template's body still
        # references them (a template/plumbing mismatch, not a template
        # bug). Re-quoting the atom name lets it round-trip: the second
        # normalize_params pass renames 'p0' -> p0 again (a no-op
        # rename, same atom, but now the manifest is populated).
        tail_requoted = [_re.sub(r"\((p\d+)\)", r"('\1')", o) for o in tail]
        fact_a = f'chain({pid}_t, [' + ', '.join(head) + ']).' + \
                 (('\n' + ma) if ma else '')
        fact_b = f'chain({pid}_r, [' + ', '.join(tail_requoted) + ']).' + \
                 (('\n' + mb) if mb else '')
        # NOTE for kernel B: the tensor is (rows=B, width=C); chan-
        # params index i % C — the elementwise chan spelling divides
        # by hw (spatial) which is 1 here, so i/1%C = i%C ✓ correct
        # by construction after the collapse.
        _, ka = run(problem_path, f'{pid}_t', fact_override=fact_a)
        _, kb = run(problem_path, f'{pid}_r', fact_override=fact_b)
        ka = ka.replace('k_auto', 'k_auto_tuple')
        kb = kb.replace('k_auto', 'k_auto_rowred')
        # the sub-runs carry their OWN manifests (imp42_t / imp42_r) —
        # the legacy whole-unit manifest would make 3 manifests for 2
        # kernels (the wrapper's mapping breaks). DROP it. And the _r
        # manifest's attr must be the REAL model attr (the re-quoted
        # atom 'p0' leaks as attr otherwise — the #42 AttributeError):
        for _atom, _attr, _kind in (manifest or []):
            if _atom and _attr and _atom != _attr:
                kb = kb.replace(f'"attr": "{_atom}"',
                                f'"attr": "{_attr}"')
        combined = (f'// two-red split (the #42 class): kernel 1 = the\n'
                    f'// tuple-reduce; kernel 2 = the row-reduce on its\n'
                    f'// (rows, chan) output — launched sequentially.\n'
                    + ka + '\n' + kb)
        return fact_a + '\n' + fact_b, combined
    if 'reduction_tuple(' in fact:
        # REGIME GUARD (probe-verified matrix + Doresh's hardware run):
        # validated for N%4==0, 256 ≤ N ≤ 16384 (the y-halving-first
        # order; nY=16 iff N≥8192). Beyond 16384 the CTA-split regime
        # engages (~vpt≥256: #65's 577600 fails 0/8) — named read.
        # N and rows must be PROVEN by the lifter (tuple_n/tuple_rows
        # meta from the conv-shape tracker).
        tuple_n = None
        tuple_rows = None
        for ln in fact.splitlines():
            if ln.startswith('%% tuple_n '):
                tuple_n = int(ln.split()[-1])
            if ln.startswith('%% tuple_rows '):
                tuple_rows = int(ln.split()[-1])
        # REGIME (Doresh's CTA-resolution 9af9fed24): the 32×16 block
        # config requires rows≥16 (below: 64×8/128×4/... — dim1 shapes
        # the block); CTA-split needs grid≤target (false at rows≥~80+);
        # verified 128/128 at N=475200 AND 577600 (the real published
        # shapes, two seeds). Guard: proven N%4==0 ∈ [256, 577600],
        # proven rows ≥ 16.
        # THE SHIFT-HEAD BRANCH (the #27 class, Sept-8): unaligned
        # N (%4 != 0) with a BARE tuple-mean — the shift-head order
        # verified 0/256 at (256,10933) incl MeanOps combine
        # (= sum + reciprocal-multiply). Bare-chain only (pre/post
        # ride other segments):
        _bare = re.search(
            r'chain\(\w+, \[reduction_tuple\((mean|sum)\)\]\)', fact)
        if _bare and tuple_n is not None and tuple_rows is not None \
                and tuple_n % 4 != 0 and 256 <= tuple_n <= 577600 \
                and tuple_rows >= 16:
            kind2 = _bare.group(1)
            if kind2 == 'mean':
                cuda = TUPLE_SHIFTHEAD_TEMPLATE.format(
                    pre_vec='', pre_tail='', post_stmts='',
                    post_final='y')
                fact2 = '\n'.join(
                    l for l in fact.splitlines()
                    if not l.startswith('%% tuple_'))
                cuda = _call_manifest(pid, manifest, None, None,
                                      fact2,
                                      tuple_n=tuple_n,
                                      tuple_rows=tuple_rows,
                                      launch={'grid': 'rows',
                                              'block': 512,
                                              'smem': 0}) + cuda
                return fact2, cuda
        if tuple_n is None or tuple_rows is None or \
                not (256 <= tuple_n <= 577600) or tuple_n % 4 != 0 or \
                tuple_rows < 16:
            raise RuntimeError(
                f'{pid}: dim-tuple reduce N={tuple_n} rows={tuple_rows} — '
                f'validated regime is proven-N%4==0 ∈ [256,577600] with '
                f'rows≥16; prove-or-refuse. (The straggler tail EXISTS '
                f'but FAILED its first real gate — #27 at N=10933: '
                f'3242/16384 rows at 3 ulp; the tail order vs torch at '
                f'non-%4 widths is unproven; mavhir dual-width caveat '
                f'vindicated. Re-relax when the tail matches torch.)')
        fact = '\n'.join(l for l in fact.splitlines()
                         if not l.startswith('%% tuple_'))
        kind = 'mean' if 'reduction_tuple(mean)' in fact else 'sum'
        txt = open(chain_file).read().replace(
            f'reduction_tuple({kind})', f'reduction(tuple_{kind})')
        with open(chain_file, 'w') as f:
            f.write(txt)
        harness = REDUCTION_HARNESS.format(chain_file=chain_file, pid=pid,
                                           spellings=SPELLINGS,
                                           red_kind=f'tuple_{kind}')
        hp = os.path.join(REPO, 'lib', f'.bridge_tuple_{pid}.pl')
        with open(hp, 'w') as f:
            f.write(harness)
        r2 = subprocess.run(['swipl', '-q', hp], capture_output=True,
                            text=True, cwd=REPO)
        if r2.returncode != 0:
            raise RuntimeError(f'{pid}: tuple-reduce fuse failed. '
                               f'{r2.stderr.strip()[:120]}')
        sections = {}
        cur = None
        for line in r2.stdout.splitlines():
            if line.startswith('===') and line.endswith('==='):
                cur = line.strip('=')
                sections[cur] = []
            elif cur is not None:
                sections[cur].append(line)
        pre_final = (sections.get('PREFINAL', ['x'])[0] or 'x').strip()
        post_final = (sections.get('POSTFINAL', ['y'])[0] or 'y').strip()
        pre = '\n'.join('                    ' + l.strip()
                        for l in sections.get('PRE', []) if l.strip())
        if pre:
            pre += '\n'
        post = '\n'.join('        ' + l.strip()
                         for l in sections.get('POST', []) if l.strip())
        if post:
            post += '\n'
        _launch = {'grid': '(nrow+255)/256', 'block': 256, 'smem': 0}
        # certified: gate_23/gate_65 (one thread per output row)
        cuda = TUPLE_MEANSUM_TEMPLATE.format(
            pre_stmts=pre, pre_final=pre_final, post_stmts=post,
            post_final=post_final, **_TUPLE_OPS[kind])
    elif ('reduction_chan(mean)' in fact or
          'reduction_chan(sum)' in fact) and not (
              chan_count is not None and chan_count > 0 and
              (chan_count & (chan_count - 1)) == 0):
        # THE POW2-C GUARD (loose-end item 7, closed properly): the
        # spatial mean/sum emulation + *(1/chn) finalize are validated
        # at pow2-C only (100/100 at 16/32/64/128); non-pow2 C = the
        # MeanOps-finalize read (named). Prove-or-refuse.
        raise RuntimeError(
            f'{pid}: chan-mean/sum with C={chan_count} — the spatial '
            f'order is validated at pow2-C only (non-pow2/unproven '
            f'refused; the MeanOps read = named build).')
    elif __import__('re').search(r"chain\(\w+, \[sub_own_mean\([\d,]+\)\]\)", fact):
        # THE #15 CLASS: out = x - mean(x, spatial dims, keepdim).
        # Guard: the op alone in the chain (the two-pass template
        # owns the whole tail) + %4 width (the vec4 sum pass; the
        # tail unexercised - refuse-not-hope):
        # width may be None (dynamic conv-output spatial) — the
        # kernel handles any n (the vec4 loop + torch's one-scalar-
        # per-tid.x tail, transcribed from Reduce.cuh):
        cuda = SUB_OWN_MEAN_TEMPLATE.format()
        cuda = _call_manifest(pid, manifest, reduce_width,
                              spatial_inner, fact,
                              launch={'grid': 'rows', 'block': 512,
                                      'smem': 0}) + cuda
        return fact, cuda
    elif 'reduction_chan(logsumexp)' in fact and \
            fact.count('reduction_chan(') == 1:
        # SPATIAL_LSE (#58/#43): pre ops must be empty; post rides:
        fact2 = fact.replace('reduction_chan(logsumexp)',
                             'reduction(chan_lse)')
        chain_file2 = write_chain_file(fact2, pid)
        harness = REDUCTION_HARNESS.format(chain_file=chain_file2, pid=pid,
                                           spellings=SPELLINGS,
                                           red_kind='chan_lse')
        hp = os.path.join(REPO, 'lib', f'.bridge_cl_{pid}.pl')
        with open(hp, 'w') as f:
            f.write(harness)
        r2 = subprocess.run(['swipl', '-q', hp], capture_output=True,
                            text=True, cwd=REPO)
        if r2.returncode != 0:
            raise RuntimeError(f'{pid}: chan-lse fuse failed. '
                               f'{r2.stderr.strip()[:120]}')
        sections = {}
        cur = None
        for line in r2.stdout.splitlines():
            if line.startswith('===') and line.endswith('==='):
                cur = line.strip('=')
                sections[cur] = []
            elif cur is not None:
                sections[cur].append(line)
        _pre_lines = [l.strip() for l in sections.get('PRE', [])
                      if l.strip()]
        _pre_final = (sections.get('PREFINAL', ['x'])[0] or 'x').strip()
        _has_pre = bool(_pre_lines)
        _has_x2 = any('x2[' in l for l in _pre_lines)
        post_final = (sections.get('POSTFINAL', ['y'])[0] or 'y').strip()
        post = '\n'.join('            ' + l.strip()
                         for l in sections.get('POST', []) if l.strip())
        if post:
            post += '\n'
        # THE DISPATCH SELECTOR (transcribing torch's own arithmetic,
        # Doresh's 9068415a8): ovs = 4 halved while inner % ovs != 0;
        # bh = 4 if ovs==4 else 16; threshold = min(bh*16, 256);
        # split_across_warps iff dim >= threshold → wide (y-lane)
        # structure; else narrow (single-thread). Refuse if the
        # shapes aren't PROVEN (prove-or-refuse):
        if chan_count is None or spatial_inner is None:
            raise RuntimeError(
                f'{pid}: chan-lse needs PROVEN dim/inner for the '
                f'dispatch selector (ovs/threshold arithmetic); '
                f'unproven → refuse.')
        def _splice_pre(cuda_):
            # THE STRIDED-LOAD FUSION (#92): insert the PRE after
            # every 'float x = v[IDX];' load, rewriting the
            # spelled x2[i] to x2[IDX] (the SAME strided index —
            # the saved tensor is elementwise-aligned with v), and
            # aliasing the pre-final back to x:
            def _repl(m):
                idx_expr = m.group(1)
                lines = [m.group(0)]
                for pl in _pre_lines:
                    lines.append(
                        pl.replace('x2[i]', f'x2[{idx_expr}]'))
                if _pre_final != 'x':
                    lines.append(f'x = {_pre_final};')
                return ' '.join(lines)
            out_ = re.sub(
                r'float x = v\[([^\]]+)\];', _repl, cuda_)
            if _has_x2:
                out_ = out_.replace(
                    'const float* __restrict__ v,',
                    'const float* __restrict__ v,\n'
                    '                       '
                    'const float* __restrict__ x2,', 1)
            return out_
        ovs = 4
        while ovs > 1 and spatial_inner % ovs != 0:
            ovs //= 2
        bh = 4 if ovs == 4 else 16
        threshold = min(bh * 16, 256)
        if chan_count >= threshold:
            cuda = _splice_pre(
                SPATIAL_LSE_TEMPLATE.format(post_stmts=post,
                                            post_final=post_final))
            # certified: spatial_lse_43_58_fixed/kernel43_fixed.cu
            _launch = {'form': '2d', 'gridx': 'outer',
                       'gridy': '(inner+31)/32',
                       'blockx': 32, 'blocky': 4, 'smem': 0}
        else:
            cuda = _splice_pre(SPATIAL_LSE_NARROW_TEMPLATE.format(
                post_stmts=post, post_final=post_final))
            # certified: narrow_channel_58/kernel58_narrow.cu
            _launch = {'form': '2d', 'gridx': 'outer', 'gridy': 128,
                       'blockx': 1, 'blocky': 32, 'smem': 0}
    elif fact.count('reduction_chan(') == 2 and \
            'reduction_chan(softmax)' in fact and \
            ('reduction_chan(min)' in fact or 'reduction_chan(max)' in fact):
        # THE #89 CLASS: softmax-chan → elementwise → min/max-chan.
        # Rewrite the trailing red as a marker, route the middle ops
        # as the POST section of the softmax, emit the 4-loop template:
        then_kind = 'max' if 'reduction_chan(max)' in fact else 'min'
        fact2 = fact.replace(f', reduction_chan({then_kind})', '')
        fact2 = fact2.replace('reduction_chan(softmax)',
                              'reduction(chan_softmax)')
        chain_file2 = write_chain_file(fact2, pid)
        harness = REDUCTION_HARNESS.format(chain_file=chain_file2, pid=pid,
                                           spellings=SPELLINGS,
                                           red_kind='chan_softmax')
        hp = os.path.join(REPO, 'lib', f'.bridge_c2_{pid}.pl')
        with open(hp, 'w') as f:
            f.write(harness)
        r2 = subprocess.run(['swipl', '-q', hp], capture_output=True,
                            text=True, cwd=REPO)
        if r2.returncode != 0:
            raise RuntimeError(f'{pid}: two-red fuse failed. '
                               f'{r2.stderr.strip()[:120]}')
        sections = {}
        cur = None
        for line in r2.stdout.splitlines():
            if line.startswith('===') and line.endswith('==='):
                cur = line.strip('=')
                sections[cur] = []
            elif cur is not None:
                sections[cur].append(line)
        if any(l.strip() for l in sections.get('PRE', [])):
            raise RuntimeError(f'{pid}: two-red pre-softmax ops — the '
                               f'strided-load fusion is a named build.')
        post_final = (sections.get('POSTFINAL', ['y'])[0] or 'y').strip()
        post = '\n'.join('                ' + l.strip()
                         for l in sections.get('POST', []) if l.strip())
        if post:
            post += '\n'
        red_cfg = {'max': ('fmaxf', '-3.402823466e+38f'),
                   'min': ('fminf', '3.402823466e+38f')}[then_kind]
        # certified: spatial_softmax_then_red_89/gate_89.cu —
        # grid(outer,32), block(1,32):
        _launch = {'form': '2d', 'gridx': 'outer', 'gridy': 32,
                   'blockx': 1, 'blocky': 32, 'smem': 0}
        cuda = SPATIAL_SOFTMAX_THEN_RED_TEMPLATE.format(
            post_stmts=post, post_final=post_final,
            red_op=red_cfg[0], red_init=red_cfg[1])
    elif 'reduction_chan(' in fact:
        for k in ('softmax', 'min', 'max', 'mean', 'sum'):
            if f'reduction_chan({k})' in fact:
                kind = k
                break
        chain_file2 = write_chain_file(fact, pid)
        cuda = emit_reduction_chan(chain_file2, pid, SPELLINGS, kind)
        if kind in ('softmax', 'logsumexp', 'log_softmax'):
            # certified: spatial_softmax_49 + _then_red_89 gates:
            # grid(outer,32), block(1,32)
            _launch = {'form': '2d', 'gridx': 'outer', 'gridy': 32,
                       'blockx': 1, 'blocky': 32, 'smem': 0}
        else:
            # CHAN_TEMPLATE (min/max/sum/mean): one thread per output
            # — certified gate_chanminmax_pub ((nout+255)/256, 256):
            _launch = {'grid': '(nout+255)/256', 'block': 256,
                       'smem': 0}
    elif any(f'reduction({k})' in fact for k in ('softmax', 'logsumexp')) \
            and _softmax_regime(reduce_width) is None:
        # UNPROVEN WIDTH: neither the reg-cache nor the grid-stride
        # template can be trusted without knowing the width (a width we
        # can't prove could silently half-cover the reg-cache path, or
        # dispatch to the wrong regime). Refuse rather than guess.
        raise RuntimeError(
            f'{pid}: softmax/lse width {reduce_width} is unproven at the '
            f'lifter — cannot select a regime (reg-cache/smem/grid-stride) '
            f'without a known width; silent half-coverage refused.')
    elif any(f'reduction({k})' in fact for k in ('softmax', 'logsumexp')) \
            and _softmax_regime(reduce_width) == 'smem_unbuilt':
        # PROVEN width, but it dispatches to torch's cunn_SoftMaxForwardSmem
        # regime (potential_reg_cnt>=10, row fits shared memory) — BPD has
        # NO template for this regime yet. Refuse honestly rather than
        # emit the reg-cache OR grid-stride kernel for a width neither is
        # verified at (the #66 lesson: emitting the wrong regime's kernel
        # can silently diverge from torch, even if it "looks" equivalent).
        raise RuntimeError(
            f'{pid}: softmax/lse width {reduce_width} dispatches to '
            f'torch\'s smem-cached regime (potential_reg_cnt>=10, row '
            f'fits shared memory) — BPD has no template for this middle '
            f'regime yet (only reg-cache <10 and grid-stride '
            f'>=10-and-no-smem are built); named build, refused honestly.')
    elif 'reduction(logsumexp)' in fact \
            and _softmax_regime(reduce_width) == 'gridstride':
        # LSE ABOVE THE REG-VERIFIED BAND: refuse, don't emit
        # GRIDSTRIDE_LSE_TEMPLATE. Per Bocher's resolution (Sept 2026):
        # torch.logsumexp is CompositeExplicitAutograd, NOT SoftMax.cu's
        # host_softmax -- it's 5 separate kernel launches (amax -> exp ->
        # sum -> log -> add) through TensorIterator/Reduce.cuh, each
        # reduction getting its OWN setReduceConfig for its own shape.
        # The existing certified LSE_TEMPLATE (reg-cache) matches torch
        # at width<=8192 not by coincidence but STRUCTURALLY: at that
        # width the reg-cache stripe order equals the TI vectorized-input
        # order both component reductions (amax, sum) dispatch to, so a
        # single fused pass reproduces the same per-element visitation
        # (and thus identical bits) as the two separate TI launches. That
        # equivalence is REGIME-BOUND -- at other widths torch's amax and
        # sum launches may each configure DIFFERENTLY, and a naive
        # ILP-4/SoftMax.cu-style grid-stride transcription (which targets
        # the WRONG source file entirely for this op) has no basis for
        # matching. Refuse rather than emit a template whose verification
        # target isn't even the right torch code path.
        raise RuntimeError(
            f'{pid}: logsumexp width {reduce_width} is above the '
            f'reg-cache-verified band (<=8192) -- GRIDSTRIDE_LSE_TEMPLATE '
            f'is unverified/speculative (its structure transcribes '
            f'SoftMax.cu, but torch.logsumexp is a 5-launch composition '
            f'via TensorIterator/Reduce.cuh, a different code path '
            f'entirely). The real verification target at this width is '
            f'the composition\'s per-reduction TI configs (amax\'s config '
            f'and sum\'s config, each independently set for this shape) '
            f'-- named build, refused honestly rather than emitted '
            f'unverified.')
    elif 'reduction(softmax)' in fact:
        variant = 'gridstride' if _softmax_regime(reduce_width) == \
            'gridstride' else 'reg'
        cuda = emit_reduction(chain_file, pid, SPELLINGS, 'softmax',
                              variant=variant)
        if variant in ('reg', 'gridstride'):
            # certified: gate_reduce_scalar (reg) + gate_66_gridstride
            # (same rows/1024/dynamic-smem shape)
            _launch = {'grid': 'rows', 'block': 1024,
                       'smem': '(block/32)*sizeof(float)'}
    elif 'reduction(logsumexp)' in fact:
        # Only 'reg'/'lse_narrow' reachable here -- the gridstride+
        # logsumexp combination was refused above.
        #
        # split_across_warps boundary (Doresh's #42 catch, Sept 2026):
        # Reduce.cuh's split_across_warps = values_per_thread() >=
        # min(block_height*16, 256) is FALSE for every (classes, batch)
        # pair when classes < 8192, and TRUE at classes == 8192
        # regardless of batch (verified via direct derivation of
        # set_block_dimension + split_input/split_output across a wide
        # batch sweep at each width -- classes<8192 never triggers
        # split for ANY batch size; classes==8192 always does). This is
        # a genuine, batch-INDEPENDENT boundary -- safe to use here
        # even though emit_reduction doesn't currently track the batch
        # dimension. LSE_TEMPLATE (descending shuffle, matches
        # split_across_warps=TRUE's block_y_reduce combine) stays for
        # classes>=8192 (where #22/#64 are already verified);
        # LSE_NARROW_TEMPLATE (ascending shuffle, matches
        # split_across_warps=FALSE's single-warp block_x_reduce) is
        # used below that boundary (verified: #42 at classes=128,
        # 0/16 twice through the real pipeline).
        # width-keyed narrow split (measured Sept-7): torch's vec
        # threshold at vpt>=8 — width<256 = strided-scalar order
        # (0/128 at 128), width>=256 = vec4-consecutive (0/128 at
        # 256..4096):
        if reduce_width is not None and reduce_width < 256:
            lse_variant = 'lse_narrow_scalar'
        elif reduce_width is not None and reduce_width < 8192:
            lse_variant = 'lse_narrow'
        else:
            lse_variant = 'reg'
        if reduce_width is None or reduce_width % 4 != 0 or \
                (lse_variant == 'reg' and reduce_width > 32768):
            raise RuntimeError(
                f'{pid}: reg-LSE (Reduce.cuh order) proven at %4==0 '
                f'widths <= 32768 (reg[16] ceiling; the vec4 tail '
                f'unexercised); W={reduce_width}; prove-or-refuse.')
        cuda = emit_reduction(chain_file, pid, SPELLINGS, 'logsumexp',
                              variant=lse_variant)
        if lse_variant == 'reg':
            # the Reduce.cuh transcription: 512 threads (32x16 flat),
            # smem = 512 floats (the y-halving buffer):
            _launch = {'grid': 'rows', 'block': 512,
                       'smem': '512*sizeof(float)'}
        else:
            # certified: gate_42_final (k_auto_rowred<<<outer, 32>>> —
            # one warp per row, ascending shuffle):
            _launch = {'grid': 'rows', 'block': 32, 'smem': 0}
    elif any(f'reduction({k})' in fact for k in ('sum', 'mean', 'min', 'max')):
        kind = next(k for k in ('sum', 'mean', 'min', 'max')
                    if f'reduction({k})' in fact)
        # SAME class of boundary as LSE's split_across_warps check
        # (Doresh's #8 finding, generalized Sept 2026): TI_TEMPLATE's
        # fixed 512-thread structure only matches torch's real dispatch
        # when split_across_warps is TRUE (classes >= 8192, batch-
        # independent). Below that, torch's real block_width is
        # min(last_pow2(classes), 32) -- BUT this is NOT simply "32 for
        # classes>=32" as first assumed: block_width is BATCH-DEPENDENT
        # for classes>=32 (empirically derived via ReduceConfig
        # simulation: block_width==32 requires batch>=16 when
        # 32<=classes<8192; for batch<16 block_width can grow up to 512,
        # a DIFFERENT multi-warp-per-row structure this template doesn't
        # implement -- found via a real gate at classes=4096,batch=4
        # showing a 2-ULP residual after this template's first pass).
        # emit_reduction doesn't currently plumb the batch dimension
        # through to this dispatch point, so classes>=32 can't be safely
        # routed to ti_narrow without further plumbing (HONEST GAP,
        # left for a follow-up). RESTRICTING to classes<32, where
        # block_width=min(last_pow2(classes),32) is PROVABLY <=16 and
        # batch-INDEPENDENT (verified via simulation across
        # classes in {2,4,8,16,31} x batch in {1..16384}: block_width
        # never exceeds 16) -- this template's fixed 32-wide warp with
        # tail-masking correctly emulates any narrower real block_width
        # for classes<32 (confirmed via real gates: classes=8 batch=64
        # and classes=16 batch=128, both 0-ULP for sum/mean/min/max).
        # FURTHER TIGHTENED after a real gate caught a second gap
        # (Sept 2026): classes=31 (block_width=16, NOT a power of 2)
        # showed a real 6-ULP residual with the "any classes<32" rule
        # above -- this template's stride is HARDCODED to 32, but the
        # real per-thread accumulation grouping only coincides with a
        # fixed-32-lane shuffle when classes<=block_width, i.e. when
        # EVERY active thread handles at most one element (no internal
        # multi-element accumulation happening in a DIFFERENT stride
        # than what the fixed-32 structure assumes). This holds
        # precisely when classes IS a power of 2 (so
        # last_pow2(classes)==classes) and classes<=32. Non-power-of-2
        # classes<32 (e.g. 31) have classes>block_width, meaning real
        # threads DO accumulate multiple strided elements internally
        # before the warp-shuffle -- a grouping this template doesn't
        # yet implement (would need the stride baked in as
        # last_pow2(reduce_width) rather than hardcoded 32, a real
        # follow-up, not done here to keep this fix strictly to what's
        # gate-verified). Restricting to exact powers of 2 is
        # conservative but honest -- covers #8's real case (classes=16)
        # and is trivially checkable.
        _classes_is_narrow_pow2 = (reduce_width is not None
                                    and reduce_width <= 32
                                    and (reduce_width & (reduce_width - 1)) == 0)
        ti_variant = 'ti_narrow' if _classes_is_narrow_pow2 else 'reg'
        cuda = emit_reduction(chain_file, pid, SPELLINGS, kind,
                              variant=ti_variant)
        # launch geometry (Mavdil's ask — certified-gate values only).
        # CORRECTED (his #18 fault-catch): the sum/mean/min/max 'reg'
        # variant is TI_TEMPLATE — 512 threads, STATIC __shared__
        # sh[512] (smem=0 dynamic) — NOT the softmax-family's 1024/
        # dynamic-smem shape (launching 1024 on the 512-shared
        # template = out-of-bounds sh[] = his illegal access):
        _launch = ({'grid': 'rows', 'block': 32, 'smem': 0}
                   if ti_variant == 'ti_narrow' else
                   {'grid': 'rows', 'block': 512, 'smem': 0})
    elif 'reduction(' in fact:
        raise RuntimeError(
            f'{pid}: unrecognized reduction kind (vocabulary-gap, honest).')
    else:
        cuda = emit(chain_file, pid)         # 3. FUSE + CODEGEN (elementwise)
    cuda = insert_param_args(cuda, manifest)  # per-param buffers (aliasing fix)
    cuda = insert_input2_arg(cuda, fact)      # two-input models
    # THE CALL-MANIFEST (Mavdil's board-scale ask, day-2): the emitter
    # KNOWS the real buffer shapes — embed them as structured JSON so
    # verification harnesses use REAL shapes, never invented ones
    # ('inventing inputs is how you get a check that passes on data
    # the kernel never sees'):
    if _launch is None and 'reduction' not in fact:
        _launch = {'grid': '(n+255)/256', 'block': 256, 'smem': 0}
    cuda = _call_manifest(pid, manifest, reduce_width,
                          spatial_inner, fact,
                          tuple_n=tuple_n, tuple_rows=tuple_rows,
                          launch=_launch,
                          chan_count=(chan_count if 'chan_count'
                                      in locals() else None)) + cuda
    ok = compile_check(cuda)                  # 4. COMPILE-FLOOR
    if ok is False:
        raise RuntimeError(
            f'{pid}: CUDA FAILED nvcc compile — codegen bug, '
            f'GAP not PASS (the compile-check floor).')
    if ok is None:
        cuda = ('/* COMPILE-CHECK DEFERRED: nvcc absent on this host — '
                'verify on the enclave before counting as PASS */\n') + cuda
    return fact, cuda


def run_auto(problem_path, pid):
    """Route: single-chain first; multi-stage segments as fallback."""
    try:
        fact, cuda = run(problem_path, pid)
        return [fact], [cuda], 'single'
    except NoEpilogue as e_noepi:
        # THE FALL-THROUGH RULING (Mavdil's discriminator, adopted):
        # a whole-chain NoEpilogue may still have expressible
        # SEGMENTS (#8: three of them, incl the residual). Try the
        # fallback FIRST; re-raise the ORIGINAL NoEpilogue only if
        # segments ALSO find nothing — "let the refusal come from
        # the path with more reach." The genuinely-one-opaque-stage
        # case stays protected: its segments are ALL EMPTY (the
        # empty-segment skip emits no kernels) → the original
        # refusal (with its composites-report) survives intact.
        try:
            facts, kernels = run_segments(problem_path, pid)
        except Exception:
            raise e_noepi
        if not kernels:
            raise e_noepi
        return facts, kernels, 'segments'
    except Exception as e_single:
        try:
            facts, kernels = run_segments(problem_path, pid)
            return facts, kernels, 'segments'
        except Exception:
            raise e_single


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: auto_pipeline.py <problem.py> <pid>', file=sys.stderr)
        sys.exit(2)
    problem_path, pid = sys.argv[1], sys.argv[2]
    try:
        facts, kernels, mode = run_auto(problem_path, pid)
        for fact in facts:
            sys.stderr.write(f'[lifted]  {fact}\n')
        total = sum(len(k.splitlines()) for k in kernels)
        sys.stderr.write(f'[emitted] {pid} {len(kernels)} kernel(s) '
                         f'({mode}), {total} lines, zero hand-work\n')
        # RETAIN the emitted kernels (Mavdil's board-scale ask): an
        # enumerable store — emitted/<pid>.cu, overwritten per run
        # (the census run therefore refreshes the whole store):
        emit_dir = os.path.join(REPO, 'emitted')
        os.makedirs(emit_dir, exist_ok=True)
        with open(os.path.join(emit_dir, f'{pid}.cu'), 'w') as ef:
            ef.write(f'// {pid} — auto-emitted ({mode}); census-refreshed.\n')
            for k in kernels:
                ef.write(k)
                ef.write('\n')
        for k in kernels:
            sys.stdout.write(k)
    except NoEpilogue as e:
        # legitimate empty: correct but not an improvement.
        sys.stderr.write(f'[NO-EPILOGUE] {e}\n')
        sys.exit(3)
    except RuntimeError as e:
        # honest failure: name the gap, don't fake a result.
        sys.stderr.write(f'[GAP] {e}\n')
        sys.exit(1)
