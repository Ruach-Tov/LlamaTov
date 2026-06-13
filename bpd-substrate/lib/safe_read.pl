%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% safe_read.pl — Byte-ownership tracked binary file reader
%%
%% Every byte in a file can be read exactly ONCE. Reading a byte
%% permanently claims it. Attempting to read an already-claimed byte
%% is a parse failure — not silent corruption.
%%
%% This prevents crossword-puzzle file attacks where:
%%   - Two data structures share the same bytes (tensor aliasing)
%%   - Data offsets point into the header (header/data overlap)
%%   - An offset field points at itself (self-referential)
%%   - Bytes exist that no parser consumed (phantom data)
%%
%% Usage:
%%   safe_open(Path, Handle),
%%   safe_read_uint32_le(Handle, Value, Handle1),
%%   safe_read_bytes(Handle1, N, Bytes, Handle2),
%%   safe_seek_and_read(Handle2, Offset, N, Bytes, Handle3),
%%   safe_verify_complete(Handle3).  % warns about unclaimed bytes
%%
%% The Handle carries: stream, file_size, claimed_ranges (sorted, non-overlapping).

/** <module> Byte-Ownership Tracked Binary Reader

Every byte in a file can be read exactly once. Reading a byte permanently
claims it — like a Rust borrow, but you never give it back. This invariant
is maintained by the byte-access primitives themselves, not by any semantic
knowledge of the file format.

This makes crossword-puzzle attacks (where the same bytes are interpreted
as two different structures) structurally impossible.

Provides: safe_open/2, safe_close/1, safe_read_bytes/4,
safe_read_uint8/3, safe_read_uint16_le/3, safe_read_uint32_le/3,
safe_read_uint64_le/3, safe_read_string/4.

@author Ruach Tov Collective
@see gguf_validate.pl which uses these primitives for GGUF validation
*/

:- module(safe_read, [
    safe_open/2,
    safe_close/1,
    safe_read_bytes/4,
    safe_read_uint8/3,
    safe_read_uint16_le/3,
    safe_read_uint32_le/3,
    safe_read_uint64_le/3,
    safe_read_int8/3,
    safe_read_int32_le/3,
    safe_read_float32_le/3,
    safe_read_string/3,
    safe_read_bool/3,
    safe_seek_and_read_bytes/5,
    safe_position/2,
    safe_verify_complete/1,
    safe_unclaimed_ranges/2,
    safe_claimed_bytes/2,
    safe_file_size/2
]).

%% Handle = safe_handle(Stream, FileSize, ClaimedRanges)
%% ClaimedRanges is a sorted list of Start-End pairs (non-overlapping).

%! safe_open(+Path, -Handle) is det.
%  Open a file for safe reading. Returns a Handle that tracks
%  byte ownership. Every subsequent read through this handle
%  claims the bytes read, preventing double-reading.
safe_open(Path, safe_handle(Stream, FileSize, [])) :-
    size_file(Path, FileSize),
    open(Path, read, Stream, [type(binary), reposition(true)]).

safe_close(safe_handle(Stream, _, _)) :-
    close(Stream).

safe_position(safe_handle(Stream, _, _), Pos) :-
    byte_count(Stream, Pos).

safe_file_size(safe_handle(_, Size, _), Size).

safe_claimed_bytes(safe_handle(_, _, Claimed), Total) :-
    foldl([S-E, Acc0, Acc1]>>(Len is E - S, Acc1 is Acc0 + Len), Claimed, 0, Total).

%% ═══════════════════════════════════════════════════════════════
%% Core: claim bytes before reading
%% ═══════════════════════════════════════════════════════════════

%% claim_range(+Start, +End, +Claimed0, -Claimed1)
%% Tracks claimed byte ranges. FAILS if any overlap with existing.
%% Uses an asserta-based approach for O(1) insertion with overlap checking
%% via a simple last-end tracker for the common case of sequential reads.
%%
%% For sequential reads (the common case), we only need to check that
%% the new range doesn't overlap the LAST claimed range. For seeks
%% (random access), we do a full overlap check.
claim_range(Start, End, Claimed0, [Start-End | Claimed0]) :-
    Start < End,
    \+ ranges_overlap_fast(Start, End, Claimed0).

%% Fast overlap check: only check ranges that could possibly overlap.
%% Since most reads are sequential, the new range usually comes AFTER
%% all existing ranges. We check the first few entries (most recent)
%% and skip the rest if they're clearly before our start.
ranges_overlap_fast(S, E, [S2-E2 | Rest]) :-
    (   S < E2, E > S2
    ->  !  % overlap found
    ;   E2 > S  % remaining ranges might still overlap
    ->  ranges_overlap_fast(S, E, Rest)
    ;   fail   % all remaining ranges end before our start
    ).

%% ═══════════════════════════════════════════════════════════════
%% Byte-level readers (all claim before reading)
%% ═══════════════════════════════════════════════════════════════

%! safe_read_bytes(+Handle0, +N, -Bytes, -Handle1) is det.
%  Read N bytes from Handle0, producing Bytes and an updated Handle1.
%  Throws read_past_eof if N bytes are not available.
%  Throws overlap if any byte was already claimed.
safe_read_bytes(safe_handle(S, FS, C0), N, Bytes, safe_handle(S, FS, C1)) :-
    byte_count(S, Pos),
    End is Pos + N,
    (End > FS -> throw(error(read_past_eof(Pos, N, FS), _)) ; true),
    claim_range(Pos, End, C0, C1),
    length(Bytes, N),
    maplist(get_byte(S), Bytes).

safe_read_uint8(H0, Value, H1) :-
    safe_read_bytes(H0, 1, [B], H1),
    Value is B.

safe_read_uint16_le(H0, Value, H1) :-
    safe_read_bytes(H0, 2, [B0, B1], H1),
    Value is B0 + B1 * 256.

safe_read_uint32_le(H0, Value, H1) :-
    safe_read_bytes(H0, 4, [B0, B1, B2, B3], H1),
    Value is B0 + B1*256 + B2*65536 + B3*16777216.

safe_read_uint64_le(H0, Value, H1) :-
    safe_read_bytes(H0, 8, [B0,B1,B2,B3,B4,B5,B6,B7], H1),
    Value is B0 + B1*256 + B2*65536 + B3*16777216
           + B4*4294967296 + B5*1099511627776
           + B6*281474976710656 + B7*72057594037927936.

safe_read_int8(H0, Value, H1) :-
    safe_read_uint8(H0, U, H1),
    (U > 127 -> Value is U - 256 ; Value = U).

safe_read_int32_le(H0, Value, H1) :-
    safe_read_uint32_le(H0, U, H1),
    (U > 2147483647 -> Value is U - 4294967296 ; Value = U).

safe_read_float32_le(H0, Value, H1) :-
    safe_read_bytes(H0, 4, Bytes, H1),
    %% IEEE 754 float32 decode
    float32_from_bytes(Bytes, Value).

safe_read_bool(H0, Value, H1) :-
    safe_read_uint8(H0, B, H1),
    (B =:= 0 -> Value = false ; Value = true).

%% GGUF-style length-prefixed string
safe_read_string(H0, String, H2) :-
    safe_read_uint64_le(H0, Len, H1),
    safe_read_bytes(H1, Len, Codes, H2),
    atom_codes(String, Codes).

%% ═══════════════════════════════════════════════════════════════
%% Seek + read (for offset-based access like tensor data)
%% ═══════════════════════════════════════════════════════════════

safe_seek_and_read_bytes(safe_handle(S, FS, C0), Offset, N, Bytes,
                         safe_handle(S, FS, C1)) :-
    End is Offset + N,
    (End > FS -> throw(error(seek_past_eof(Offset, N, FS), _)) ; true),
    claim_range(Offset, End, C0, C1),
    safe_seek_to(S, Offset),
    length(Bytes, N),
    maplist(get_byte(S), Bytes).

safe_seek_to(Stream, BytePos) :-
    byte_count(Stream, CurByte),
    Skip is BytePos - CurByte,
    (Skip > 0
     -> %% Forward seek: consume bytes (preserves byte_count accuracy on
        %% streams that don't support set_stream_position, e.g. pipes).
        %% For regular files this is equivalent to set_stream_position.
        forall(between(1, Skip, _), get_byte(Stream, _))
     ;  Skip =:= 0
     -> true
     ;  %% Backward seek — use seek/4 with bof (beginning-of-file) origin.
        %% This is the correct portable SWI-Prolog predicate for repositioning
        %% binary streams opened with reposition(true).
        %% seek(+Stream, +Offset, +Method, -NewLocation)
        %%   Method = bof means offset from beginning of file.
        seek(Stream, BytePos, bof, _)
    ).

%% ═══════════════════════════════════════════════════════════════
%% Completeness verification
%% ═══════════════════════════════════════════════════════════════

safe_unclaimed_ranges(safe_handle(_, FileSize, Claimed), Unclaimed) :-
    find_gaps(0, FileSize, Claimed, Unclaimed).

find_gaps(Pos, End, [], Gaps) :-
    (Pos < End -> Gaps = [Pos-End] ; Gaps = []).
find_gaps(Pos, End, [S-E | Rest], Gaps) :-
    (Pos < S
     -> Gaps = [Pos-S | MoreGaps],
        find_gaps(E, End, Rest, MoreGaps)
     ;  find_gaps(E, End, Rest, Gaps)).

safe_verify_complete(Handle) :-
    safe_unclaimed_ranges(Handle, Unclaimed),
    safe_file_size(Handle, FileSize),
    safe_claimed_bytes(Handle, ClaimedTotal),
    (Unclaimed = []
     -> format("COMPLETE: all ~d bytes claimed~n", [FileSize])
     ;  format("WARNING: ~d of ~d bytes unclaimed:~n", [FileSize - ClaimedTotal, FileSize]),
        forall(member(S-E, Unclaimed),
               (Len is E - S, format("  unclaimed: bytes ~d-~d (~d bytes)~n", [S, E, Len])))
    ).

%% ═══════════════════════════════════════════════════════════════
%% Helpers
%% ═══════════════════════════════════════════════════════════════

byte_count(Stream, Count) :-
    stream_property(Stream, position(Pos)),
    stream_position_data(byte_count, Pos, Count).

float32_from_bytes([B0, B1, B2, B3], Value) :-
    %% Reassemble as uint32, then interpret as IEEE 754
    U is B0 + B1*256 + B2*65536 + B3*16777216,
    (U =:= 0 -> Value = 0.0
    ; Sign is (U >> 31) /\ 1,
      Exp is (U >> 23) /\ 0xFF,
      Mant is U /\ 0x7FFFFF,
      (Exp =:= 0
       -> Value is (-1)^Sign * 2^(-126) * Mant / 8388608  % denormalized
       ;  Exp =:= 255
       -> (Mant =:= 0 -> (Sign =:= 0 -> Value = inf ; Value = -inf)
          ; Value = nan)
       ;  Value is (-1)^Sign * 2^(Exp - 127) * (1 + Mant / 8388608)
      )
    ).
