# janus_swi: empirical patterns from LlamaTov runner work

This file documents the term-shape conventions that janus_swi uses to
represent Python values in Prolog. Discovered empirically during the
LlamaTov substrate work (commits baac87f74 through 1692dfc98) and
relevant to anyone using py_call with PyTorch / numpy.

## Why this matters

When a Python function returns a value, janus_swi must convert it
into a Prolog term. The conversion isn't always obvious — especially
for tuple-shaped return values which can't directly be represented
in Prolog's syntax. Knowing these patterns prevents an entire class
of pattern-matching bugs.

## Pattern 1: Python booleans → Prolog atoms

```python
def is_finite(x):
    return torch.all(torch.isfinite(x)).item()  # Python bool
```

```prolog
?- py_call(my_module:is_finite(T), V).
V = @(true).    % NOT 1, NOT true_atom — the term `@(true)`
```

To compare:
```prolog
V == @(true)      % WORKS
V == true         % FAILS (true is a different atom)
V =:= 1           % FAILS (not a number)
```

## Pattern 2: Python 1-tuples → Prolog -(N)

```python
def shape_1d(t):
    return t.shape    # torch.Size([768]) is a 1-tuple
```

```prolog
?- py_call(my_module:shape_1d(T), S).
S = -(768).       % 1-ary negative-prefix term
```

To destructure:
```prolog
S = -(N)          % WORKS
S = (N)           % FAILS (not a parenthesized number)
S =.. [-, N]      % WORKS (canonical form via univ)
```

## Pattern 3: Python 2-tuples → Prolog A - B

```python
def cos_sin():
    return (cos_tensor, sin_tensor)  # 2-tuple
```

```prolog
?- py_call(my_module:cos_sin, Tuple).
Tuple = <py_Tensor>-<py_Tensor>.    % Binary - operator term
```

To destructure:
```prolog
Tuple = (Cos - Sin)     % WORKS — note the parentheses are required
                        % to override Prolog operator precedence
Tuple = Cos - Sin       % ALSO WORKS in most contexts
```

## Pattern 4: Python 3-tuples → Prolog -(A, B, C)

```python
def qkv_split(qkv):
    return (q, k, v)  # 3-tuple
```

```prolog
?- py_call(my_module:qkv_split(QKV), Tuple).
Tuple = -(<py_Tensor>, <py_Tensor>, <py_Tensor>).   % 3-ary - functor
```

Critical: the functor is the SAME (`-`) but the arity differs.
Pattern matching:
```prolog
Tuple = (Q - K - V)     % FAILS — that's `-(`-(Q, K), V)` (left-associative binary)
Tuple = (Q - (K - V))   % FAILS — different shape
Tuple =.. ['-', Q, K, V]   % WORKS — canonical univ form
```

## Pattern 5: General N-tuple destructure via univ

For any tuple shape, the safe destructuring is:
```prolog
Tuple =.. ['-' | Elements]
length(Elements, N)
```

This works for tuples of any arity ≥ 1. The `-` is the consistent
functor; only the arity changes.

## Common error: shape comparison

```prolog
%% Got tensor shape, want to check if it's [1, 4, 8]:

%% WRONG (3-tuple, but written as binary):
Shape == (1 - 4 - 8)        % Parses as `-(-(1, 4), 8)` (binary, left-assoc)

%% RIGHT:
Shape =.. ['-', 1, 4, 8]    % Univ form

%% Also right (for 2-tuple shapes like (768, 768)):
Shape == 768 - 768          % This DOES parse as `-(768, 768)`
```

The 2-tuple case has a syntactic shortcut because Prolog's binary `-`
operator naturally produces `-(A, B)`. For 3-tuples and higher, that
shortcut doesn't exist and you MUST use univ.

## Convenience helpers in llamatov_helpers.py

The `tensor_shape` function in `llamatov_helpers.py` returns
`tuple(t.shape)`. This makes the janus conversion explicit and
predictable:
- 1D tensor → `-(N)`
- 2D tensor → `A - B`
- 3D tensor → `-(A, B, C)` (use univ)
- 4D tensor → `-(A, B, C, D)` (use univ)

## Tests demonstrating each pattern

These behaviors are verified empirically:
- Pattern 1 (booleans): `test_q4_0_dequant.pl::test_q4_0_values_finite`
- Pattern 2 (1-tuple): `test_tensor_loader_adapter.pl::test_load_f32_tensor_directly`
- Pattern 3 (2-tuple): `test_tensor_loader_adapter.pl::test_load_f16_tensor_upcasts_to_f32`
- Pattern 4 (3-tuple): `test_llama_helpers.pl::test_llama_qkv_split_gqa`
- Pattern 5 (univ): `test_llama_helpers.pl::test_llama_qkv_split_mha`

## Origin

Discovered during the overnight LlamaTov substrate push 2026-05-15
while building the Prolog-maximal runner for TinyLlama. Three
integration bugs in the runner were caused by pattern-matching
mistakes covered here — see commit 1692dfc98 for the bug fixes.

Author: metayen (2026-05-15)
