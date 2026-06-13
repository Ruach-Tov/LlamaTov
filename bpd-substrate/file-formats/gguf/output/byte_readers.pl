%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
% Deployment copy of ir/include/byte_readers.pl
%
% Emitted alongside the generated format reader by
% emit_byte_readers_to_file/1 in prolog_reader_generator.pl.
% The canonical source lives at ir/include/byte_readers.pl.
% This copy lets the generated reader bundle be self-contained.
%
% --- begin copy ---

% byte_readers.pl — primitive byte-list operations for Prolog-based parsers.
%
% This is a small runtime library that generated Prolog parsers will
% call. It encapsulates the low-level operations of:
%   - Slicing byte lists by offset and length
%   - Decoding little-endian and big-endian unsigned integers
%   - Combined "read TYPE at OFFSET" convenience predicates
%
% The library has no dependencies on the BPD IR or on any specific
% format. Generated parsers consult this library plus their format's
% generated reader, then call read_<section>(+Bytes, -Result).
%
% Architecture: per Heath's design direction, "IR describes structure →
% Prolog rule reads IR → generates Prolog that reads bytes and prints
% outputs, consults that generated Prolog → result." This file is the
% runtime LIBRARY that generated code calls; the GENERATOR is a separate
% module (built in subsequent steps).
%
% Byte order conventions: little-endian decodes the FIRST byte in the
% list as the least-significant byte (matches how LE multi-byte values
% are laid out in memory). Big-endian decodes the FIRST byte as the
% most-significant byte (network byte order).
%
% Author: metayen, 2026-05-12, BPD Prolog parser-generator Step 1.


% --- Slicing ---

% bytes_slice(+Bytes, +Start, +Length, -Slice)
%
% Extracts Length bytes starting at Start (0-indexed). Length=0 yields
% an empty slice; Start past end-of-input fails.
%
% Per Path 1 (Heath's choice, 2026-05-13): polymorphic over BOTH
% byte-list AND binary-string representations of Bytes:
%
%   - If Bytes is a string (typically from open(..., type(binary)) +
%     read_string/3), Slice is a string. This is the EFFICIENT path:
%     no list-cell amplification.
%
%   - If Bytes is a list (test fixtures, small inline data), Slice
%     is a list. This preserves existing behavior.
%
% The polymorphism is type-checked at the head — no cascading changes
% needed in the typed readers because they all funnel through this
% primitive (and bytes_to_uint_*) and get the same representation
% back.

% String case (efficient, 1.0 bytes/byte — no amplification)
bytes_slice(Bytes, Start, Length, Slice) :-
    string(Bytes), !,
    integer(Start), Start >= 0,
    integer(Length), Length >= 0,
    sub_string(Bytes, Start, Length, _, Slice).

% List case (preserves existing semantics — test fixtures construct lists)
bytes_slice(Bytes, Start, Length, Slice) :-
    is_list(Bytes), !,
    integer(Start), Start >= 0,
    integer(Length), Length >= 0,
    length(Prefix, Start),
    length(Slice, Length),
    append(Prefix, Rest, Bytes),
    append(Slice, _Suffix, Rest).


% bytes_slice_from(+Bytes, +Start, -Slice)
%
% Slice from Start to the end of Bytes. Equivalent to bytes_slice
% with Length = (total - Start), but doesn't require knowing the
% remaining length upfront.
%
% Used by read_section_at to give a section's body view that starts
% at byte 0 of the section. Polymorphic over list/string per Path 1.

% String case
bytes_slice_from(Bytes, Start, Slice) :-
    string(Bytes), !,
    integer(Start), Start >= 0,
    string_length(Bytes, Total),
    Length is Total - Start,
    Length >= 0,
    sub_string(Bytes, Start, Length, 0, Slice).

% List case
bytes_slice_from(Bytes, Start, Slice) :-
    is_list(Bytes), !,
    integer(Start), Start >= 0,
    length(Prefix, Start),
    append(Prefix, Slice, Bytes).


% --- Integer decoding ---

% bytes_to_uint_le(+Bytes, -Value)
%
% Little-endian unsigned integer. The first byte is the least-significant.
% Examples:
%   [1, 0, 0, 0]   → 1            (u32 = 1)
%   [0, 1, 0, 0]   → 256          (u32 = 256)
%   [3, 0, 0, 0]   → 3            (GGUF version)
%
% Per Path 1: polymorphic over byte-list AND binary-string. Strings
% are processed via string_codes/2 which is O(N) but doesn't allocate
% intermediate cons cells. For typical reads of 1-8 byte primitives
% the cost is negligible.

% String case
bytes_to_uint_le(Bytes, Value) :-
    string(Bytes), !,
    string_codes(Bytes, Codes),
    bytes_to_uint_le_list(Codes, Value).

% List case
bytes_to_uint_le(Bytes, Value) :-
    is_list(Bytes), !,
    bytes_to_uint_le_list(Bytes, Value).

bytes_to_uint_le_list([], 0) :- !.
bytes_to_uint_le_list([B | Rest], Value) :-
    bytes_to_uint_le_list(Rest, RestValue),
    Value is B + (RestValue << 8).


% bytes_to_uint_be(+Bytes, -Value)
%
% Big-endian unsigned integer. The first byte is the most-significant.
% Examples:
%   [0, 0, 0, 1]   → 1
%   [0, 0, 0x89, 0x50]  → 0x8950
%   [137, 80, 78, 71]   → 0x89504E47  (PNG signature first 4 bytes)

% String case
bytes_to_uint_be(Bytes, Value) :-
    string(Bytes), !,
    string_codes(Bytes, Codes),
    bytes_to_uint_be_acc(Codes, 0, Value).

% List case
bytes_to_uint_be(Bytes, Value) :-
    is_list(Bytes), !,
    bytes_to_uint_be_acc(Bytes, 0, Value).

bytes_to_uint_be_acc([], Acc, Acc).
bytes_to_uint_be_acc([B | Rest], Acc, Value) :-
    NewAcc is (Acc << 8) + B,
    bytes_to_uint_be_acc(Rest, NewAcc, Value).


% --- Convenience: read TYPE at OFFSET ---
%
% These combine slicing with decoding for the common cases. The naming
% mirrors the BPD primitive type atoms (u8/u16/u32/u64) plus endianness.

% read_u32_le(+Bytes, +Offset, -Value)
read_u32_le(Bytes, Offset, Value) :-
    bytes_slice(Bytes, Offset, 4, Slice),
    bytes_to_uint_le(Slice, Value).

% read_u64_le(+Bytes, +Offset, -Value)
read_u64_le(Bytes, Offset, Value) :-
    bytes_slice(Bytes, Offset, 8, Slice),
    bytes_to_uint_le(Slice, Value).

% read_u16_le(+Bytes, +Offset, -Value)
read_u16_le(Bytes, Offset, Value) :-
    bytes_slice(Bytes, Offset, 2, Slice),
    bytes_to_uint_le(Slice, Value).

% read_u8(+Bytes, +Offset, -Value)
% Single-byte read; endianness is moot.
% read_u8 was destructuring bytes_slice's result as [Value], assuming a
% 1-byte list. Per Path 1 (polymorphic primitives), bytes_slice returns
% a STRING when input is a string, so we use bytes_to_uint_le (which IS
% polymorphic) instead. A 1-byte unsigned int is endian-agnostic, so
% bytes_to_uint_le and bytes_to_uint_be give the same result.

read_u8(Bytes, Offset, Value) :-
    bytes_slice(Bytes, Offset, 1, Slice),
    bytes_to_uint_le(Slice, Value).

% read_u32_be(+Bytes, +Offset, -Value)
read_u32_be(Bytes, Offset, Value) :-
    bytes_slice(Bytes, Offset, 4, Slice),
    bytes_to_uint_be(Slice, Value).

% read_u64_be(+Bytes, +Offset, -Value)
read_u64_be(Bytes, Offset, Value) :-
    bytes_slice(Bytes, Offset, 8, Slice),
    bytes_to_uint_be(Slice, Value).

% read_u16_be(+Bytes, +Offset, -Value)
read_u16_be(Bytes, Offset, Value) :-
    bytes_slice(Bytes, Offset, 2, Slice),
    bytes_to_uint_be(Slice, Value).

% read_bytes(+Bytes, +Offset, +Length, -Slice)
% Pass-through to bytes_slice, named to match the read_ convention.
read_bytes(Bytes, Offset, Length, Slice) :-
    bytes_slice(Bytes, Offset, Length, Slice).


% --- Variable-width readers (length-prefixed) ---

% read_length_prefixed_string(+Bytes, +Offset, -StringBytes, -BytesConsumed)
%
% Read a length-prefixed byte sequence with a u64 LITTLE-ENDIAN
% length prefix (8 bytes), followed by that many bytes of content.
% Returns the content as a list of byte values, plus the total bytes
% consumed (length + 8 for the prefix).
%
% This pattern (u64 LE length + bytes) appears in many binary file
% formats including GGUF metadata strings, GGUF tensor names, and
% any format following the same convention. The predicate is
% universal substrate; format-specific knowledge of how strings are
% encoded (UTF-8, ASCII, etc.) lives in the format-specific
% generated reader.
%
% Length-encoding limitation: this predicate hardcodes (u64, LE) for
% the length prefix. If a future format uses (u32, LE) or (u16, BE)
% or some other prefix encoding, that needs a sibling predicate
% (e.g., read_u32le_length_prefixed_bytes/4). The BPD vocabulary
% `length_prefixed(LengthType, ElementType)` captures the
% parameterization at the spec level; the generator picks the
% appropriate runtime predicate based on the length-type variant.
% Currently only length_prefixed(u64, u8) is supported.
read_length_prefixed_string(Bytes, Offset, StringBytes, BytesConsumed) :-
    read_u64_le(Bytes, Offset, Length),
    ContentOffset is Offset + 8,
    bytes_slice(Bytes, ContentOffset, Length, StringBytes),
    BytesConsumed is Length + 8.


% =====================================================================
% Signed integer readers (R-004)
% =====================================================================
%
% Two's-complement decoding: read the unsigned value, then if the
% high bit is set, subtract 2^N. For little-endian integers we reuse
% the existing unsigned readers; same for big-endian.
%
% read_iN_{le,be}(+Bytes, +Offset, -Value)
%
% Where Value is in the range [-2^(N-1), 2^(N-1) - 1].


% read_i8(+Bytes, +Offset, -Value)
%
% Single byte interpreted as i8. Sign bit is bit 7.
% No endianness applies for a single byte.

read_i8(Bytes, Offset, Value) :-
    read_u8(Bytes, Offset, Unsigned),
    (   Unsigned >= 128
    ->  Value is Unsigned - 256
    ;   Value = Unsigned
    ).


% read_i16_le(+Bytes, +Offset, -Value)

read_i16_le(Bytes, Offset, Value) :-
    read_u16_le(Bytes, Offset, Unsigned),
    (   Unsigned >= 0x8000
    ->  Value is Unsigned - 0x10000
    ;   Value = Unsigned
    ).


% read_i16_be(+Bytes, +Offset, -Value)

read_i16_be(Bytes, Offset, Value) :-
    read_u16_be(Bytes, Offset, Unsigned),
    (   Unsigned >= 0x8000
    ->  Value is Unsigned - 0x10000
    ;   Value = Unsigned
    ).


% read_i32_le(+Bytes, +Offset, -Value)

read_i32_le(Bytes, Offset, Value) :-
    read_u32_le(Bytes, Offset, Unsigned),
    (   Unsigned >= 0x80000000
    ->  Value is Unsigned - 0x100000000
    ;   Value = Unsigned
    ).


% read_i32_be(+Bytes, +Offset, -Value)

read_i32_be(Bytes, Offset, Value) :-
    read_u32_be(Bytes, Offset, Unsigned),
    (   Unsigned >= 0x80000000
    ->  Value is Unsigned - 0x100000000
    ;   Value = Unsigned
    ).


% read_i64_le(+Bytes, +Offset, -Value)

read_i64_le(Bytes, Offset, Value) :-
    read_u64_le(Bytes, Offset, Unsigned),
    (   Unsigned >= 0x8000000000000000
    ->  Value is Unsigned - 0x10000000000000000
    ;   Value = Unsigned
    ).


% read_i64_be(+Bytes, +Offset, -Value)

read_i64_be(Bytes, Offset, Value) :-
    read_u64_be(Bytes, Offset, Unsigned),
    (   Unsigned >= 0x8000000000000000
    ->  Value is Unsigned - 0x10000000000000000
    ;   Value = Unsigned
    ).


% =====================================================================
% IEEE 754 float readers (R-004)
% =====================================================================
%
% Decode IEEE 754 single (32-bit) and double (64-bit) precision floats
% from their bit patterns. The bit pattern is read as an unsigned
% integer (with appropriate endianness), then decoded via sign,
% exponent, and mantissa fields.
%
% Special cases:
%   - +0 and -0 both decode to 0 (mantissa=0, exponent=0)
%   - Subnormals (exponent=0, mantissa≠0) decoded as 2^(1-bias) * (mantissa/2^MantissaBits)
%   - +Inf and -Inf decoded as 1.0Inf and -1.0Inf
%   - NaN decoded as nan (SWI represents as float)
%
% read_fN_{le,be}(+Bytes, +Offset, -Value)


% read_f32_le(+Bytes, +Offset, -Value)

read_f32_le(Bytes, Offset, Value) :-
    read_u32_le(Bytes, Offset, Bits),
    decode_f32(Bits, Value).


% read_f32_be(+Bytes, +Offset, -Value)

read_f32_be(Bytes, Offset, Value) :-
    read_u32_be(Bytes, Offset, Bits),
    decode_f32(Bits, Value).


% read_f64_le(+Bytes, +Offset, -Value)

read_f64_le(Bytes, Offset, Value) :-
    read_u64_le(Bytes, Offset, Bits),
    decode_f64(Bits, Value).


% read_f64_be(+Bytes, +Offset, -Value)

read_f64_be(Bytes, Offset, Value) :-
    read_u64_be(Bytes, Offset, Bits),
    decode_f64(Bits, Value).


% decode_f32(+Bits, -Value)
%
% Decode a 32-bit IEEE 754 single-precision bit pattern.
% Layout: sign(1) | exponent(8, bias 127) | mantissa(23)

decode_f32(Bits, Value) :-
    Sign     is (Bits >> 31) /\ 0x1,
    Exponent is (Bits >> 23) /\ 0xFF,
    Mantissa is Bits /\ 0x7FFFFF,
    decode_ieee754(Sign, Exponent, Mantissa, 127, 23, Value).


% decode_f64(+Bits, -Value)
%
% Decode a 64-bit IEEE 754 double-precision bit pattern.
% Layout: sign(1) | exponent(11, bias 1023) | mantissa(52)

decode_f64(Bits, Value) :-
    Sign     is (Bits >> 63) /\ 0x1,
    Exponent is (Bits >> 52) /\ 0x7FF,
    Mantissa is Bits /\ 0xFFFFFFFFFFFFF,
    decode_ieee754(Sign, Exponent, Mantissa, 1023, 52, Value).


% decode_ieee754(+Sign, +Exponent, +Mantissa, +Bias, +MantissaBits, -Value)
%
% Generic IEEE 754 decoder. The exponent-all-ones value is (2*Bias+1)
% — for f32 with Bias=127 that's 255; for f64 with Bias=1023 that's
% 2047. The decoder branches on:
%
%   Exp = 0:
%     Mantissa = 0  → ±0.0 (sign preserved via ieee754_apply_sign)
%     Mantissa > 0  → subnormal:   ±2^(1-bias) * (mantissa / 2^mbits)
%
%   Exp = 2*Bias+1 (all ones):
%     Mantissa = 0  → ±Infinity
%     Mantissa > 0  → NaN (LOSSY: collapses all NaN variants to a
%                          single 'nan' atom — IEEE 754 NaN carries a
%                          sign bit and 22/51 bits of payload including
%                          the quiet/signaling distinction, all
%                          discarded here. Acceptable for BPD parsing
%                          where NaN in spec-defined metadata fields
%                          would be a spec violation anyway. If a
%                          future format requires NaN-payload roundtrip,
%                          this needs a richer return type.)
%
%   Otherwise (normal):
%     ±2^(exp-bias) * (1 + mantissa / 2^mbits)

decode_ieee754(Sign, Exponent, Mantissa, Bias, MantissaBits, Value) :-
    MaxExp is 2 * Bias + 1,
    (   Exponent =:= 0
    ->  (   Mantissa =:= 0
        ->  % ±0.0 — route through ieee754_apply_sign so -0.0 produces
            % a negatively-signed zero. Whether SWI-Prolog
            % distinguishes +0.0 from -0.0 in the float type is
            % runtime-dependent; this fix preserves the sign at the
            % BPD-decoder level regardless.
            ieee754_apply_sign(Sign, 0.0, Value)
        ;   % Subnormal
            Scale is 2.0 ** (1 - Bias),
            Frac is Mantissa / (2 ** MantissaBits),
            ieee754_apply_sign(Sign, Scale * Frac, Value)
        )
    ;   Exponent =:= MaxExp
    ->  (   Mantissa =:= 0
        ->  (   Sign =:= 1
            ->  Value is -inf
            ;   Value is inf
            )
        ;   Value is nan
        )
    ;   % Normal
        RealExp is Exponent - Bias,
        Scale is 2.0 ** RealExp,
        Frac is 1 + Mantissa / (2 ** MantissaBits),
        ieee754_apply_sign(Sign, Scale * Frac, Value)
    ).


% ieee754_apply_sign(+Sign, +Magnitude, -Value)
%
% Magnitude is an arithmetic expression (Scale * Frac). Apply the
% sign bit and evaluate to a final float Value.

ieee754_apply_sign(0, Magnitude, Value) :-
    Value is Magnitude.
ieee754_apply_sign(1, Magnitude, Value) :-
    Value is -(Magnitude).


% bytes_total_length(+Bytes, -Length)
%
% Polymorphic byte-length primitive. Works on both binary strings (where
% Length = string_length/2) and byte lists (where Length = length/2).
%
% Used by opaque_blob section readers (V.D.G3, 2026-05-14) to determine
% "remaining bytes from offset" without materializing the slice.

bytes_total_length(Bytes, Length) :-
    string(Bytes), !,
    string_length(Bytes, Length).
bytes_total_length(Bytes, Length) :-
    is_list(Bytes), !,
    length(Bytes, Length).


% align_up(+Value, +Alignment, -Aligned)
%
% Round Value UP to the next multiple of Alignment. If Value is already
% aligned, returns Value unchanged.
%
% Used by opaque_blob section readers when the spec declares
% align_to(:Section, byte_alignment(N)) — the section's content starts
% at the next N-byte boundary after the given Offset.
%
% Example: align_up(749979, 32, A) → A = 749984 (next 32-byte boundary).
%          align_up(749984, 32, A) → A = 749984 (already aligned).

align_up(Value, Alignment, Aligned) :-
    integer(Value), integer(Alignment), Alignment > 0,
    Remainder is Value mod Alignment,
    (   Remainder =:= 0
    ->  Aligned = Value
    ;   Aligned is Value + (Alignment - Remainder)
    ).

% --- end copy ---
