%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% =============================================================================
%% ir_compare_diagram.pl — Side-by-side LLVM IR / x86 asm comparison SVG.
%% =============================================================================
%%
%% Dimension (5) of the kernel-visualization atlas: the IR-LAYER drill-down.
%% Sits beneath the per-op detail page (mavchin's llvm_match_detail.pl) which
%% sits beneath the Table 10001 dashboard cell. When a reviewer wants to see
%% WHY two (or three) implementations diverge at the instruction level,
%% this is the view.
%%
%% Per Iyun's framing (2026-05-30): the SVG is the forcing function and
%% visible progress meter for AUTOMATED as-built IR extraction. The
%% diagnostic value: hand-decoding ggml's compiled IR is error-prone (Iyun
%% caught themselves reading AVX2 when the CPU runs AVX). Automated
%% extraction reads what's ACTUALLY compiled for the target.
%%
%% This module is rungs 2 (parse) + 3 (align + render) of the four-rung
%% extraction-automation climb:
%%   1. EXTRACT  — clang/objdump driver (separate; orchestration)
%%   2. PARSE    — .ll/.asm → structured op-list  [HERE]
%%   3. ALIGN+RENDER — side-by-side SVG with color coding   [HERE]
%%   4. POPULATE — derive ulp_match facts from measurement
%%                 (extends medayek's verify_llvm_ops_auto.py harness)
%%
%% Worked example: q8_0_dot from Iyun's ir_pairs/. Three implementations:
%%   ggml:          vpsignb → vpmaddubsw → vpmaddwd → vcvtdq2ps → vmulps → vaddps
%%   BPD intrinsic: same vpmaddubsw chain (proven 0 ULP, instruction match)
%%   BPD scalar:    imull → vmulss → vaddss (same int math, scalar fp accum)
%%
%% Three-color coding:
%%   GREEN  — identical mnemonic (literal instruction match)
%%   AMBER  — same purpose, different form (vectorization divergence)
%%   RED    — true divergence (different purpose, no semantic equivalent)
%%   STONE  — gap (instruction present on one side, missing on the other)
%%
%% Alignment strategy: semantic-group alignment per Iyun's block structure.
%%   Sections are detected from loop-back labels in the asm. Within each
%%   section, instructions are matched by purpose. The substantive divergence
%%   (vectorization choice at int_dot_step or fp_accumulate) lands as amber
%%   rows within aligned sections, rather than swamped by length asymmetry.
%%
%% Run:
%%   cd bpd/lib && swipl -g main ir_compare_diagram.pl
%%   (default: uses Iyun's ir_pairs at /tmp/output-only/ if scp'd locally)
%% Custom 2-way: swipl -g main ir_compare_diagram.pl -- LEFT.asm RIGHT.asm OUT.svg
%% =============================================================================

:- module(ir_compare_diagram, [
    parse_asm_file/2,
    align_instructions/3,
    align_sections/3,
    emit_ir_compare_svg/6,
    emit_ir_compare_three_way/8,
    instruction_purpose/3
]).

:- use_module(library(lists)).
:- use_module(library(readutil)).

:- discontiguous instruction_purpose/3.

%% ─────────────────────────────────────────────────────────────────────────────
%% Instruction-purpose fact base
%%
%% instruction_purpose(Mnemonic, PurposeAtom, OneLineDescription)
%%
%% Two instructions are semantically equivalent if they share PurposeAtom.
%% "Core" purposes participate in semantic block detection; "supporting"
%% purposes (loads, address calc, control flow) compose into a block but
%% don't define it.
%% ─────────────────────────────────────────────────────────────────────────────

%% Integer dot-product family (CORE)
instruction_purpose(imull,        int_dot_step,
    'scalar 32-bit signed multiply').
instruction_purpose(vpsignb,      int_dot_step,
    'packed sign byte (signed-x-unsigned setup)').
instruction_purpose(vpmaddubsw,   int_dot_step,
    'packed multiply-add unsigned-signed bytes → words').
instruction_purpose(vpmaddwd,     int_dot_step,
    'packed multiply-add words → dwords').

%% Integer accumulate
instruction_purpose(vpaddd,       int_accumulate,
    'packed add dwords (int accumulator update)').
instruction_purpose(addl,         int_accumulate,
    'scalar add (int accumulator update)').

%% Integer→float conversion (CORE)
instruction_purpose(vcvtdq2ps,    int_to_fp,
    'convert packed dwords to packed single-precision floats').
instruction_purpose(vcvtsi2ss,    int_to_fp,
    'convert scalar integer to scalar single-precision float').

%% Floating-point multiply-accumulate (CORE)
instruction_purpose(vmulps,       fp_multiply,
    'packed multiply (8-lane fp32)').
instruction_purpose(vmulss,       fp_multiply,
    'scalar fp32 multiply').
instruction_purpose(vaddps,       fp_accumulate,
    'packed add (8-lane fp32) — accumulator step').
instruction_purpose(vaddss,       fp_accumulate,
    'scalar fp32 add — accumulator step').
instruction_purpose(fadd,         fp_accumulate,
    'scalar fp add — accumulator step').

%% Vector lane management
instruction_purpose(vinsertf128,  vec_lane_construct,
    'insert 128-bit lane into 256-bit ymm (xmm→ymm promote)').
instruction_purpose(vextractf128, vec_lane_extract,
    'extract 128-bit lane from 256-bit ymm (hsum step)').
instruction_purpose(vshufps,      vec_lane_shuffle,
    'shuffle packed single-precision lanes').
instruction_purpose(vmovhlps,     vec_lane_extract,
    'move high-half pair to low-half (hsum step)').
instruction_purpose(vmovshdup,    vec_lane_shuffle,
    'duplicate odd-indexed singles (hsum step)').
instruction_purpose(vpsrldq,      vec_lane_shuffle,
    'packed shift-right by bytes (lane shift)').
instruction_purpose(vpmovsxbw,    vec_lane_extend,
    'sign-extend packed bytes to words').
instruction_purpose(vpshufd,      vec_lane_shuffle,
    'shuffle packed dwords').
instruction_purpose(vbroadcastss, vec_broadcast,
    'broadcast scalar to all lanes').

%% Loads / stores / register setup (SUPPORTING)
instruction_purpose(vmovdqu,      vec_load_unaligned,
    'unaligned packed load (xmm/ymm)').
instruction_purpose(vmovd,        vec_load_scalar,
    'move 32-bit value into low lane of xmm').
instruction_purpose(vmovss,       fp_load_scalar,
    'load scalar fp32').
instruction_purpose(vxorps,       fp_zero,
    'fp xor (commonly used to zero a register)').
instruction_purpose(movzwl,       int_load_unsigned,
    'load unsigned word, zero-extend to 32-bit').
instruction_purpose(movsbl,       int_load_signed,
    'load signed byte, zero-extend to 32-bit').
instruction_purpose(movsbq,       int_load_signed,
    'load signed byte, sign-extend to 64-bit').

%% Control flow / address calc / housekeeping (SUPPORTING)
instruction_purpose(test,        ctrl_flow_test, 'test for zero/sign').
instruction_purpose(push,        ctrl_flow_frame, 'push register').
instruction_purpose(pushq,       ctrl_flow_frame, 'push 64-bit register').
instruction_purpose(pop,         ctrl_flow_frame, 'pop register').
instruction_purpose(popq,        ctrl_flow_frame, 'pop 64-bit register').
instruction_purpose(mov,         data_move, 'move data').
instruction_purpose(movq,        data_move, '64-bit move').
instruction_purpose(movslq,      data_move, 'sign-extend 32→64-bit move').
instruction_purpose(lea,         address_calc, 'load effective address').
instruction_purpose(leal,        address_calc, '32-bit load effective address').
instruction_purpose(leaq,        address_calc, '64-bit load effective address').
instruction_purpose(add,         int_arith, 'add').
instruction_purpose(addq,        int_arith, '64-bit add').
instruction_purpose(sar,         int_arith, 'shift arithmetic right').
instruction_purpose(xor,         int_zero, 'xor (commonly used to zero)').
instruction_purpose(xorl,        int_zero, '32-bit xor (commonly used to zero)').
instruction_purpose(cmp,         ctrl_flow_test, 'compare').
instruction_purpose(cmpl,        ctrl_flow_test, '32-bit compare').
instruction_purpose(jle,         ctrl_flow_branch, 'jump if less-or-equal').
instruction_purpose(jl,          ctrl_flow_branch, 'jump if less').
instruction_purpose(jge,         ctrl_flow_branch, 'jump if greater-or-equal').
instruction_purpose(jg,          ctrl_flow_branch, 'jump if greater').
instruction_purpose(jmp,         ctrl_flow_branch, 'unconditional jump').
instruction_purpose(cmovns,      ctrl_flow_move, 'conditional move if not-sign').
instruction_purpose(inc,         int_arith, 'increment').
instruction_purpose(incl,        int_arith, '32-bit increment').
instruction_purpose(and,         int_bitwise, 'bitwise and').
instruction_purpose(cltq,        data_move, 'sign-extend eax→rax').
instruction_purpose(imul,        int_arith, 'integer multiply').
instruction_purpose(nopl,        ctrl_flow_nop, 'no-op (alignment padding)').
instruction_purpose(vzeroupper,  vec_state, 'zero upper halves of ymm/zmm').
instruction_purpose(leave,       ctrl_flow_frame, 'leave stack frame').
instruction_purpose(ret,         ctrl_flow_branch, 'return').
instruction_purpose(retq,        ctrl_flow_branch, '64-bit return').

%% Fallback for unknown mnemonics
instruction_purpose(_, unknown, 'unrecognized instruction (extend the fact base)').

%% ─────────────────────────────────────────────────────────────────────────────
%% Core-purpose classification: which purposes define a semantic block
%% ─────────────────────────────────────────────────────────────────────────────

is_core_purpose(int_dot_step).
is_core_purpose(int_accumulate).
is_core_purpose(int_to_fp).
is_core_purpose(fp_multiply).
is_core_purpose(fp_accumulate).
is_core_purpose(vec_lane_construct).
is_core_purpose(vec_lane_extract).

%% Purposes that signal a section boundary (e.g., end of main loop, start of hsum)
is_section_marker(ctrl_flow_branch).
is_section_marker(ctrl_flow_frame).

%% ─────────────────────────────────────────────────────────────────────────────
%% RUNG 2 — PARSE
%%
%% parse_asm_file/2 returns just the instructions (backward-compatible).
%% parse_asm_file/3 returns (Instructions, Labels) where Labels is a list of
%% label(InstrIdx, OriginalText, Letter) entries assigning A, B, C, … to each
%% label-anchor in source order. Operands in the returned Instructions are
%% rewritten so jump-target references show as (A) / (B) / … instead of
%% the raw label text or hex address.
%% ─────────────────────────────────────────────────────────────────────────────

parse_asm_file(Path, Instructions) :-
    parse_asm_file(Path, Instructions, _).

parse_asm_file(Path, RewrittenInstrs, Labels) :-
    read_file_to_string(Path, Content, []),
    split_string(Content, "\n", "", Lines),
    parse_lines_with_labels(Lines, 0, RawInstrs, RawLabels),
    %% RawLabels from explicit ".LBB0_N:" lines (llc format).
    %% For objdump format, additionally synthesize labels from jump targets.
    synthesize_objdump_labels(RawInstrs, ExtraLabels),
    append(RawLabels, ExtraLabels, AllLabels),
    sort_and_assign_letters(AllLabels, RawInstrs, Labels),
    rewrite_jump_operands(RawInstrs, Labels, RewrittenInstrs).

%% parse_lines_with_labels collects both instructions and explicit labels.
%% A label is any line that ends with ":" and starts with .L or a bare ident,
%% but NOT an objdump address-prefix line (which has a colon after the addr).
parse_lines_with_labels([], _, [], []).
parse_lines_with_labels([Line|Rest], Idx, Instrs, Labels) :-
    ( parse_explicit_label_line(Line, LabelText)
    -> %% Bind to the NEXT instruction index (which becomes Idx unchanged
       %% because we don't bump for label lines)
       Labels = [pending_label(Idx, LabelText) | LabelsRest],
       parse_lines_with_labels(Rest, Idx, Instrs, LabelsRest)
    ; parse_instruction_line_with_addr(Line, Mnemonic, OperandsString, AddrOpt)
    -> instruction_purpose(Mnemonic, Purpose, Desc),
       Instr = instr_addr(Idx, Mnemonic, OperandsString, Purpose, Desc, AddrOpt),
       NextIdx is Idx + 1,
       parse_lines_with_labels(Rest, NextIdx, RestInstrs, Labels),
       Instrs = [Instr | RestInstrs]
    ;  parse_lines_with_labels(Rest, Idx, Instrs, Labels)
    ).

%% parse_explicit_label_line: matches lines like ".LBB0_3:" or "bpd_q8_0_dot:".
%% Rejects objdump address-prefix lines (the colon is inside the line, not
%% at the very end, and the line has more content after).
parse_explicit_label_line(Line, LabelText) :-
    string_concat(_, "", Line),
    split_string(Line, "", " \t", [Stripped]),
    Stripped \= "",
    %% Must end with ":" (after optional comment)
    %% Strip trailing # comment if any
    split_string(Stripped, "#", "", [BeforeComment | _]),
    split_string(BeforeComment, "", " \t", [Trimmed]),
    string_length(Trimmed, L), L > 1,
    sub_string(Trimmed, _, 1, 0, LastChar),
    LastChar = ":",
    %% Strip the trailing colon
    LStem is L - 1,
    sub_string(Trimmed, 0, LStem, _, Stem),
    %% Must look like an identifier (start with . or letter, contain only ident chars)
    string_chars(Stem, [FirstChar | _]),
    ( FirstChar = '.' ; char_type(FirstChar, alpha) ; FirstChar = '_' ),
    %% Reject pure-hex (those are objdump address prefixes — but with content after)
    %% A real label line is JUST "label:" optionally followed by # comment.
    %% objdump lines have instruction text after the addr+colon, so they
    %% don't pass the "Trimmed ends with :" test.
    atom_string(LabelText, Stem).

%% parse_instruction_line_with_addr: like parse_instruction_line but also
%% extracts the address (for objdump format) so we can match jump targets.
parse_instruction_line_with_addr(Line, Mnemonic, Operands, AddrOpt) :-
    string_concat(_, "", Line),
    split_string(Line, "", " \t", [Stripped]),
    Stripped \= "",
    \+ string_concat(_, ":", Stripped),
    \+ string_concat(".", _, Stripped),
    \+ string_concat("#", _, Stripped),
    ( split_string(Stripped, ":", "", [AddrPart, RestOfLine]),
      string_length(AddrPart, AL), AL =< 10,
      split_string(RestOfLine, "", " \t", [Trimmed]),
      Trimmed \= ""
    -> parse_mnemonic_and_ops(Trimmed, Mnemonic, Operands),
       %% AddrPart is the hex address — store it
       split_string(AddrPart, "", " \t", [AddrClean]),
       AddrOpt = some(AddrClean)
    ;  parse_mnemonic_and_ops(Stripped, Mnemonic, Operands),
       AddrOpt = none
    ),
    is_mnemonic_atom(Mnemonic).

%% synthesize_objdump_labels: for objdump-format files, every address that
%% appears as a jump target should be marked as a label. We scan all
%% instructions, extract jump-target addresses from operands, and create
%% pending_label entries for the instr-indices at those addresses.
synthesize_objdump_labels(Instrs, ExtraLabels) :-
    %% Build address → instr-idx map
    findall(Addr-Idx,
        ( member(instr_addr(Idx, _, _, _, _, some(Addr)), Instrs) ),
        AddrMap),
    %% Find all jump-target addresses referenced in operands
    findall(TargetAddr,
        ( member(instr_addr(_, Mnem, Ops, _, _, _), Instrs),
          atom_concat('j', _, Mnem),    %% jump instruction
          extract_first_hex_word(Ops, TargetAddr)
        ),
        TargetAddrs),
    list_to_set(TargetAddrs, UniqueTargets),
    %% For each target address, find the instr-idx and emit a pending_label
    findall(pending_label(Idx, LabelText),
        ( member(TargetAddr, UniqueTargets),
          member(TargetAddr-Idx, AddrMap),
          atom_string(LabelText, TargetAddr)
        ),
        ExtraLabels).

%% extract_first_hex_word: from an operand string like "ebc <ggml...+0x20c>",
%% extract "ebc" (the leading hex address). Returns the string form.
extract_first_hex_word(OpsString, HexAddr) :-
    split_string(OpsString, " \t,", "", Words),
    member(W, Words),
    W \= "",
    %% All chars must be hex digits
    string_chars(W, Chars),
    forall(member(C, Chars), char_type(C, xdigit(_))),
    string_length(W, L), L >= 2,
    HexAddr = W, !.

%% sort_and_assign_letters: from a list of pending_label(Idx, Text) entries,
%% assign A, B, C, … in instr-index (source) order. Returns a list of
%% label(Idx, Text, Letter) entries plus a side-effect of validating against
%% RawInstrs (drop labels whose Idx is past the end).
sort_and_assign_letters(Pending, RawInstrs, Labels) :-
    length(RawInstrs, NInstrs),
    %% Filter to labels with valid Idx
    findall(pending_label(Idx, Text),
        ( member(pending_label(Idx, Text), Pending),
          Idx < NInstrs ),
        Valid),
    %% Sort by Idx, dedup (multiple pending_label for same Idx → keep first)
    sort(0, @<, Valid, Sorted),
    dedup_by_idx(Sorted, [], Dedup),
    assign_letters(Dedup, 0, Labels).

dedup_by_idx([], _, []).
dedup_by_idx([pending_label(Idx, Text) | Rest], Seen, Out) :-
    ( member(Idx, Seen)
    -> dedup_by_idx(Rest, Seen, Out)
    ;  Out = [pending_label(Idx, Text) | OutRest],
       dedup_by_idx(Rest, [Idx | Seen], OutRest)
    ).

assign_letters([], _, []).
assign_letters([pending_label(Idx, Text) | Rest], N, [label(Idx, Text, Letter) | RestLabels]) :-
    LetterCode is 0'A + N,
    char_code(LetterChar, LetterCode),
    atom_chars(Letter, [LetterChar]),
    N1 is N + 1,
    assign_letters(Rest, N1, RestLabels).

%% rewrite_jump_operands: for each jump instruction, rewrite its operand
%% string to replace label-text or hex-address with "(LETTER)". Also strips
%% the verbose objdump symbol+offset suffix.
rewrite_jump_operands([], _, []).
rewrite_jump_operands([instr_addr(Idx, Mnem, Ops, P, D, _) | Rest], Labels,
                      [instr(Idx, Mnem, NewOps, P, D) | RestNew]) :-
    ( atom_concat('j', _, Mnem)
    -> rewrite_one_jump(Ops, Labels, NewOps)
    ;  NewOps = Ops
    ),
    rewrite_jump_operands(Rest, Labels, RestNew).

rewrite_one_jump(OpsString, Labels, NewOps) :-
    %% Try to find the target by hex address (objdump) or by label text (llc)
    ( extract_first_hex_word(OpsString, HexAddr),
      member(label(_, HexLabel, Letter), Labels),
      atom_string(HexLabel, HexAddr)
    -> format(atom(NewOps), '(~w)', [Letter])
    ; extract_first_dot_label(OpsString, DotLabel),
      member(label(_, LabelAtom, Letter), Labels),
      atom_string(LabelAtom, DotLabel)
    -> format(atom(NewOps), '(~w)', [Letter])
    ;  NewOps = OpsString  %% fallback: leave as-is
    ).

extract_first_dot_label(OpsString, DotLabel) :-
    split_string(OpsString, " \t,", "", Words),
    member(W, Words),
    W \= "",
    string_concat(".", _, W),
    DotLabel = W, !.

%% Keep the old parse_lines/3 alive for any callers (currently none after
%% the change, but defensively keep the binding).
parse_lines([], _, []).
parse_lines([Line|Rest], Idx, Result) :-
    ( parse_instruction_line(Line, Mnemonic, OperandsString)
    -> instruction_purpose(Mnemonic, Purpose, Desc),
       Instr = instr(Idx, Mnemonic, OperandsString, Purpose, Desc),
       NextIdx is Idx + 1,
       parse_lines(Rest, NextIdx, RestInstructions),
       Result = [Instr | RestInstructions]
    ;  parse_lines(Rest, Idx, Result)
    ).

parse_instruction_line(Line, Mnemonic, Operands) :-
    string_concat(_, "", Line),
    split_string(Line, "", " \t", [Stripped]),
    Stripped \= "",
    \+ string_concat(_, ":", Stripped),
    \+ string_concat(".", _, Stripped),
    \+ string_concat("#", _, Stripped),
    ( split_string(Stripped, ":", "", [AddrPart, RestOfLine]),
      string_length(AddrPart, AL), AL =< 10,
      split_string(RestOfLine, "", " \t", [Trimmed]),
      Trimmed \= ""
    -> parse_mnemonic_and_ops(Trimmed, Mnemonic, Operands)
    ;  parse_mnemonic_and_ops(Stripped, Mnemonic, Operands)
    ),
    is_mnemonic_atom(Mnemonic).

parse_mnemonic_and_ops(Line, Mnemonic, Operands) :-
    split_string(Line, " \t", "", Parts),
    exclude([P]>>(P = ""), Parts, [MnStr | OpParts]),
    atom_string(Mnemonic, MnStr),
    atomic_list_concat(OpParts, ' ', OperandsAtom),
    atom_string(OperandsAtom, Operands).

is_mnemonic_atom(Atom) :-
    atom_length(Atom, L), L >= 2, L =< 12,
    atom_chars(Atom, [C|_]),
    char_type(C, alpha).

%% ─────────────────────────────────────────────────────────────────────────────
%% RUNG 3a — SECTION-AWARE ALIGNMENT
%%
%% A section is a maximal run of instructions sharing a dominant core purpose,
%% bounded by section-markers (control-flow branches) or shifts in core purpose.
%%
%% Section types we recognize:
%%   prologue       — initial frame setup (push/mov/xor before any core purpose)
%%   int_dot        — the integer-dot-product block
%%   int_to_fp_mac  — int→float conversion + fp multiply-accumulate (the main MAC)
%%   hsum           — horizontal sum (extract/shuffle/add cascade)
%%   epilogue       — tail (vzeroupper/pop/leave/ret)
%%   loop_overhead  — pure ctrl-flow + address calc (loop back-branches etc)
%%   other          — fallback
%% ─────────────────────────────────────────────────────────────────────────────

%% sectionize(+Instrs, -Sections)
%%   Sections: list of section(Kind, [Instr, ...])
sectionize(Instrs, Sections) :-
    classify_each(Instrs, Classified),
    group_runs(Classified, Sections).

%% classify_each: tag each instruction with its section-kind (based on purpose
%% + context). For purity, the per-instruction kind is just from the purpose;
%% the run-grouping handles the contextual merging.
classify_each([], []).
classify_each([Instr|Rest], [Kind-Instr | RestC]) :-
    Instr = instr(_, _, _, Purpose, _),
    purpose_section(Purpose, Kind),
    classify_each(Rest, RestC).

purpose_section(int_dot_step,       int_dot).
purpose_section(int_accumulate,     int_dot).
purpose_section(int_to_fp,          int_to_fp_mac).
purpose_section(fp_multiply,        int_to_fp_mac).
purpose_section(fp_accumulate,      int_to_fp_mac).
purpose_section(vec_lane_extract,   hsum).
purpose_section(vec_lane_shuffle,   hsum).
purpose_section(vec_lane_construct, int_to_fp_mac).  %% ymm-promote feeds the fp MAC
purpose_section(vec_lane_extend,    int_dot).        %% byte→word feeds int dot
purpose_section(ctrl_flow_frame,    framing).
purpose_section(ctrl_flow_branch,   loop_overhead).
purpose_section(ctrl_flow_test,     loop_overhead).
purpose_section(ctrl_flow_move,     loop_overhead).
purpose_section(ctrl_flow_nop,      loop_overhead).
purpose_section(data_move,          loop_overhead).
purpose_section(address_calc,       loop_overhead).
purpose_section(int_arith,          loop_overhead).
purpose_section(int_bitwise,        loop_overhead).
purpose_section(int_zero,           loop_overhead).
purpose_section(vec_load_unaligned, int_dot).        %% feeds the int-dot block
purpose_section(vec_load_scalar,    loop_overhead).
purpose_section(vec_broadcast,      loop_overhead).
purpose_section(fp_load_scalar,     int_to_fp_mac).  %% feeds fp MAC
purpose_section(fp_zero,            framing).        %% accumulator init
purpose_section(vec_state,          framing).        %% vzeroupper at end
purpose_section(int_load_signed,    int_dot).        %% feeds scalar int dot
purpose_section(int_load_unsigned,  int_dot).
purpose_section(unknown,            other).

%% group_runs: collapse consecutive same-kind instructions into sections.
%% But absorb short "loop_overhead" runs (≤2 instrs) into the surrounding
%% section so they don't fragment the picture.
group_runs([], []).
group_runs([Kind-Instr | Rest], [section(Kind, [Instr | MoreInRun]) | RestSections]) :-
    take_run(Kind, Rest, MoreInRun, Remaining),
    group_runs(Remaining, RestSections).

take_run(_, [], [], []).
take_run(Kind, [Kind-Instr | Rest], [Instr | MoreInRun], Remaining) :- !,
    take_run(Kind, Rest, MoreInRun, Remaining).
take_run(_, Other, [], Other).

%% align_sections(+SectionsLeft, +SectionsRight, -AlignedSectionPairs)
%%   AlignedSectionPairs: list of section_pair(Kind, LeftInstrs|gap, RightInstrs|gap)
%%   Aligned by kind using greedy first-match-wins.
align_sections([], [], []).
align_sections([], [section(K, R)|Rest], [section_pair(K, gap, R) | More]) :-
    align_sections([], Rest, More).
align_sections([section(K, L)|Rest], [], [section_pair(K, L, gap) | More]) :-
    align_sections(Rest, [], More).
align_sections([section(K, L)|RestL], [section(K, R)|RestR],
               [section_pair(K, L, R) | More]) :- !,
    %% Same kind — pair them up.
    align_sections(RestL, RestR, More).
align_sections([section(K1, L)|RestL], [section(K2, R)|RestR], Result) :-
    %% Different kinds — look ahead: does K1 appear in RestR? If yes, emit
    %% the K2 section as gap-on-left and recurse. Otherwise emit K1 as
    %% gap-on-right.
    ( member(section(K1, _), RestR)
    -> Result = [section_pair(K2, gap, R) | More],
       align_sections([section(K1, L)|RestL], RestR, More)
    ;  Result = [section_pair(K1, L, gap) | More],
       align_sections(RestL, [section(K2, R)|RestR], More)
    ).

%% align_instructions(+InstrsL, +InstrsR, -AlignedRows)
%%   Legacy line-by-line alignment for two-column emit (kept for backward compat).
align_instructions(L, R, Rows) :-
    align_loop(L, R, [], RowsRev),
    reverse(RowsRev, Rows).

align_loop([], [], Acc, Acc) :- !.
align_loop([], [R|Rs], Acc, Result) :- !,
    align_loop([], Rs, [row(gap, R, differ) | Acc], Result).
align_loop([L|Ls], [], Acc, Result) :- !,
    align_loop(Ls, [], [row(L, gap, differ) | Acc], Result).
align_loop([L|Ls], [R|Rs], Acc, Result) :-
    classify_match(L, R, Kind),
    align_loop(Ls, Rs, [row(L, R, Kind) | Acc], Result).

classify_match(instr(_, M, _, _, _), instr(_, M, _, _, _), identical) :- !.
classify_match(instr(_, _, _, P, _), instr(_, _, _, P, _), same_purpose) :-
    P \= unknown, !.
classify_match(_, _, differ).

%% ─────────────────────────────────────────────────────────────────────────────
%% RUNG 3b — RENDER
%% ─────────────────────────────────────────────────────────────────────────────

color_for_match(identical,    '#3f7c5c').  %% green
color_for_match(same_purpose, '#daa520').  %% amber
color_for_match(differ,       '#8a3a2a').  %% red
color_for_match(gap_only,     '#c4c2ba').  %% stone

%% Section-band colors (header strips above each section)
section_color(prologue,      '#7a8a7a').
section_color(framing,       '#7a8a7a').
section_color(int_dot,       '#3a6b88').   %% deep blue
section_color(int_to_fp_mac, '#7a5a8a').   %% violet
section_color(hsum,          '#9a6f5a').   %% terracotta
section_color(epilogue,      '#5a6a5a').
section_color(loop_overhead, '#aaa8a0').   %% pale stone
section_color(other,         '#6a6a6a').

section_label(prologue,      'prologue').
section_label(framing,       'framing').
section_label(int_dot,       'integer dot product').
section_label(int_to_fp_mac, 'int→fp multiply-accumulate').
section_label(hsum,          'horizontal sum').
section_label(epilogue,      'epilogue').
section_label(loop_overhead, 'loop overhead').
section_label(other,         'other').

%% emit_ir_compare_svg(+LeftLabel, +LeftPath, +RightLabel, +RightPath, +Title, +OutPath)
%%   Two-column section-aware render with per-side label-letter margin.
emit_ir_compare_svg(LeftLabel, LeftPath, RightLabel, RightPath, Title, OutPath) :-
    parse_asm_file(LeftPath, LeftInstrs, LeftLabels),
    parse_asm_file(RightPath, RightInstrs, RightLabels),
    sectionize(LeftInstrs, LeftSections),
    sectionize(RightInstrs, RightSections),
    align_sections(LeftSections, RightSections, AlignedSections),
    render_two_column(Title, LeftLabel, RightLabel, AlignedSections,
                      LeftLabels, RightLabels, OutPath).

%% emit_ir_compare_three_way(+LL, +LP, +ML, +MP, +RL, +RP, +Title, +OutPath)
%%   Three-column section-aware render with per-side label-letter margins.
emit_ir_compare_three_way(LL, LP, ML, MP, RL, RP, Title, OutPath) :-
    parse_asm_file(LP, LInstrs, LLabels),
    parse_asm_file(MP, MInstrs, MLabels),
    parse_asm_file(RP, RInstrs, RLabels),
    sectionize(LInstrs, LSections),
    sectionize(MInstrs, MSections),
    sectionize(RInstrs, RSections),
    three_way_align(LSections, MSections, RSections, AlignedTriples),
    render_three_column(Title, LL, ML, RL, AlignedTriples,
                        LLabels, MLabels, RLabels, OutPath).

three_way_align([], _, _, []).
three_way_align([section(K, L) | RestL], MSections, RSections,
                [triple(K, L, MInstrs, RInstrs) | RestTriples]) :-
    ( select(section(K, MInstrs), MSections, MRest)
    -> true
    ;  MInstrs = gap, MRest = MSections
    ),
    ( select(section(K, RInstrs), RSections, RRest)
    -> true
    ;  RInstrs = gap, RRest = RSections
    ),
    three_way_align(RestL, MRest, RRest, RestTriples).

%% ─────────────────────────────────────────────────────────────────────────────
%% Two-column SVG emission
%% ─────────────────────────────────────────────────────────────────────────────

render_two_column(Title, LeftLabel, RightLabel, AlignedSections,
                  LeftLabels, RightLabels, OutPath) :-
    RowH = 18,
    HeaderH = 80,
    SectionHeaderH = 22,
    FooterH = 70,
    PadX = 30,
    MarginW = 28,   %% per-side margin for label-letter (A:/B:/etc.)
    ColW = 466,
    GapX = 20,
    CanvasW is 2 * (MarginW + ColW) + PadX * 2 + GapX,
    total_rows_for_sections(AlignedSections, NRows, NSections),
    CanvasH is HeaderH + NSections * SectionHeaderH + NRows * RowH + FooterH,

    setup_call_cleanup(
        open(OutPath, write, S),
        ( emit_svg_header(S, Title, CanvasW, CanvasH, [LeftLabel, RightLabel],
                          PadX, MarginW + ColW, GapX, HeaderH),
          render_two_column_sections(S, AlignedSections, HeaderH, PadX,
                                     MarginW, ColW, GapX, RowH, SectionHeaderH,
                                     LeftLabels, RightLabels),
          LegendY is HeaderH + NSections * SectionHeaderH + NRows * RowH + 24,
          emit_svg_legend(S, PadX, LegendY),
          emit_label_legend(S, PadX, LegendY + 22,
                            LeftLabel, LeftLabels, RightLabel, RightLabels),
          format(S, '</svg>~n', [])
        ),
        close(S)),
    format(user_error, "Wrote: ~w (~w sections, ~w total rows)~n",
        [OutPath, NSections, NRows]).

total_rows_for_sections([], 0, 0).
total_rows_for_sections([section_pair(_, L, R) | Rest], Total, NSec) :-
    len_or_gap(L, NL),
    len_or_gap(R, NR),
    MaxLR is max(NL, NR),
    total_rows_for_sections(Rest, RestTotal, RestNSec),
    Total is MaxLR + RestTotal,
    NSec is RestNSec + 1.

len_or_gap(gap, 0) :- !.
len_or_gap(L, N) :- length(L, N).

render_two_column_sections(_, [], _, _, _, _, _, _, _, _, _).
render_two_column_sections(S, [section_pair(K, L, R) | Rest],
                           Y0, PadX, MarginW, ColW, GapX, RowH, SHH,
                           LeftLabels, RightLabels) :-
    section_color(K, SColor),
    section_label(K, SLabel),
    BandW is 2 * (MarginW + ColW) + GapX,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" fill-opacity="0.85"/>~n',
        [PadX, Y0, BandW, SHH, SColor]),
    BandTextY is Y0 + SHH - 6,
    BandTextX is PadX + 8,
    format(S, '  <text x="~w" y="~w" font-size="12" fill="#f8f5ee" font-weight="bold" font-family="Georgia, serif">~w</text>~n',
        [BandTextX, BandTextY, SLabel]),
    Y1 is Y0 + SHH,
    pad_to_equal_length(L, R, LPadded, RPadded),
    render_section_rows(S, LPadded, RPadded, Y1, PadX, MarginW, ColW, GapX, RowH,
                        LeftLabels, RightLabels),
    length(LPadded, N),
    Y2 is Y1 + N * RowH,
    render_two_column_sections(S, Rest, Y2, PadX, MarginW, ColW, GapX, RowH, SHH,
                               LeftLabels, RightLabels).

pad_to_equal_length(gap, R, LPad, R) :- !,
    length(R, NR),
    length(LPad, NR),
    maplist(=(gap), LPad).
pad_to_equal_length(L, gap, L, RPad) :- !,
    length(L, NL),
    length(RPad, NL),
    maplist(=(gap), RPad).
pad_to_equal_length(L, R, LPad, RPad) :-
    length(L, NL), length(R, NR),
    NMax is max(NL, NR),
    pad_list(L, NMax, LPad),
    pad_list(R, NMax, RPad).

pad_list(L, N, L) :- length(L, N), !.
pad_list(L, N, Padded) :-
    length(L, NL),
    NL < N,
    Diff is N - NL,
    length(Padding, Diff),
    maplist(=(gap), Padding),
    append(L, Padding, Padded).

render_section_rows(_, [], [], _, _, _, _, _, _, _, _).
render_section_rows(S, [L|Ls], [R|Rs], Y, PadX, MarginW, ColW, GapX, RowH,
                    LeftLabels, RightLabels) :-
    classify_pair(L, R, Kind),
    color_for_match(Kind, BgColor),
    %% Left margin (label-letter)
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="#e8e3d8" stroke="#3a2a1a" stroke-width="0.3"/>~n',
        [PadX, Y, MarginW, RowH]),
    %% Left main column
    LeftMainX is PadX + MarginW,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" fill-opacity="0.85" stroke="#3a2a1a" stroke-width="0.3"/>~n',
        [LeftMainX, Y, ColW, RowH, BgColor]),
    %% Right margin
    RightMarginX is PadX + MarginW + ColW + GapX,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="#e8e3d8" stroke="#3a2a1a" stroke-width="0.3"/>~n',
        [RightMarginX, Y, MarginW, RowH]),
    %% Right main column
    RightMainX is RightMarginX + MarginW,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" fill-opacity="0.85" stroke="#3a2a1a" stroke-width="0.3"/>~n',
        [RightMainX, Y, ColW, RowH, BgColor]),
    TextY is Y + RowH - 5,
    %% Letter-in-margin for left
    emit_label_letter(S, PadX + 4, TextY, L, LeftLabels),
    %% Letter-in-margin for right
    emit_label_letter(S, RightMarginX + 4, TextY, R, RightLabels),
    %% Instruction text
    LeftTextX is LeftMainX + 6,
    RightTextX is RightMainX + 6,
    instr_text(L, LeftText),
    instr_text(R, RightText),
    format(S, '  <text x="~w" y="~w" font-size="11" fill="#1a1a1a">~w</text>~n',
        [LeftTextX, TextY, LeftText]),
    format(S, '  <text x="~w" y="~w" font-size="11" fill="#1a1a1a">~w</text>~n',
        [RightTextX, TextY, RightText]),
    Y1 is Y + RowH,
    render_section_rows(S, Ls, Rs, Y1, PadX, MarginW, ColW, GapX, RowH,
                        LeftLabels, RightLabels).

%% emit_label_letter: if this instruction has a label, emit "X:" in the margin.
emit_label_letter(_, _, _, gap, _) :- !.
emit_label_letter(S, X, Y, instr(Idx, _, _, _, _), Labels) :-
    ( member(label(Idx, _, Letter), Labels)
    -> format(S, '  <text x="~w" y="~w" font-size="11" font-weight="bold" fill="#3a2a1a" font-family="Georgia, serif">~w:</text>~n',
          [X, Y, Letter])
    ;  true
    ).

classify_pair(gap, gap, gap_only) :- !.
classify_pair(gap, _, gap_only) :- !.
classify_pair(_, gap, gap_only) :- !.
classify_pair(instr(_, M, _, _, _), instr(_, M, _, _, _), identical) :- !.
classify_pair(instr(_, _, _, P, _), instr(_, _, _, P, _), same_purpose) :-
    P \= unknown, !.
classify_pair(_, _, differ).

%% ─────────────────────────────────────────────────────────────────────────────
%% Three-column SVG emission
%% ─────────────────────────────────────────────────────────────────────────────

render_three_column(Title, LL, ML, RL, Triples, LLabels, MLabels, RLabels, OutPath) :-
    RowH = 18,
    HeaderH = 80,
    SectionHeaderH = 22,
    FooterH = 70,
    PadX = 30,
    MarginW = 24,
    ColW = 356,
    GapX = 16,
    CanvasW is 3 * (MarginW + ColW) + PadX * 2 + 2 * GapX,
    total_rows_for_triples(Triples, NRows, NSections),
    CanvasH is HeaderH + NSections * SectionHeaderH + NRows * RowH + FooterH,

    setup_call_cleanup(
        open(OutPath, write, S),
        ( emit_svg_header(S, Title, CanvasW, CanvasH, [LL, ML, RL],
                          PadX, MarginW + ColW, GapX, HeaderH),
          render_three_column_sections(S, Triples, HeaderH, PadX, MarginW,
                                       ColW, GapX, RowH, SectionHeaderH,
                                       LLabels, MLabels, RLabels),
          LegendY is HeaderH + NSections * SectionHeaderH + NRows * RowH + 24,
          emit_svg_legend(S, PadX, LegendY),
          emit_label_legend(S, PadX, LegendY + 22,
                            LL, LLabels, ML, MLabels, RL, RLabels),
          format(S, '</svg>~n', [])
        ),
        close(S)),
    format(user_error, "Wrote: ~w (~w sections, ~w total rows, three-way)~n",
        [OutPath, NSections, NRows]).

total_rows_for_triples([], 0, 0).
total_rows_for_triples([triple(_, L, M, R) | Rest], Total, NSec) :-
    len_or_gap(L, NL),
    len_or_gap(M, NM),
    len_or_gap(R, NR),
    MaxLMR is max(NL, max(NM, NR)),
    total_rows_for_triples(Rest, RestTotal, RestNSec),
    Total is MaxLMR + RestTotal,
    NSec is RestNSec + 1.

render_three_column_sections(_, [], _, _, _, _, _, _, _, _, _, _).
render_three_column_sections(S, [triple(K, L, M, R) | Rest],
                              Y0, PadX, MarginW, ColW, GapX, RowH, SHH,
                              LLabels, MLabels, RLabels) :-
    section_color(K, SColor),
    section_label(K, SLabel),
    BandW is 3 * (MarginW + ColW) + 2 * GapX,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" fill-opacity="0.85"/>~n',
        [PadX, Y0, BandW, SHH, SColor]),
    BandTextY is Y0 + SHH - 6,
    BandTextX is PadX + 8,
    format(S, '  <text x="~w" y="~w" font-size="12" fill="#f8f5ee" font-weight="bold" font-family="Georgia, serif">~w</text>~n',
        [BandTextX, BandTextY, SLabel]),
    Y1 is Y0 + SHH,
    pad_to_equal_length3(L, M, R, LP, MP, RP),
    render_triple_rows(S, LP, MP, RP, Y1, PadX, MarginW, ColW, GapX, RowH,
                       LLabels, MLabels, RLabels),
    length(LP, N),
    Y2 is Y1 + N * RowH,
    render_three_column_sections(S, Rest, Y2, PadX, MarginW, ColW, GapX, RowH, SHH,
                                  LLabels, MLabels, RLabels).

pad_to_equal_length3(L, M, R, LP, MP, RP) :-
    len_or_gap(L, NL),
    len_or_gap(M, NM),
    len_or_gap(R, NR),
    NMax is max(NL, max(NM, NR)),
    coerce_to_padded(L, NMax, LP),
    coerce_to_padded(M, NMax, MP),
    coerce_to_padded(R, NMax, RP).

coerce_to_padded(gap, N, Padded) :-
    length(Padded, N),
    maplist(=(gap), Padded).
coerce_to_padded(L, N, Padded) :-
    is_list(L),
    pad_list(L, N, Padded).

render_triple_rows(_, [], [], [], _, _, _, _, _, _, _, _, _).
render_triple_rows(S, [L|Ls], [M|Ms], [R|Rs], Y, PadX, MarginW, ColW, GapX, RowH,
                   LLabels, MLabels, RLabels) :-
    classify_pair(L, M, MidKind),
    classify_pair(L, R, RightKind),
    color_for_match(identical, LeftBg),
    color_for_match(MidKind, MidBg),
    color_for_match(RightKind, RightBg),

    %% Column widths: each column = margin + main
    ColTotalW is MarginW + ColW,

    %% Left column: margin + main
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="#e8e3d8" stroke="#3a2a1a" stroke-width="0.3"/>~n',
        [PadX, Y, MarginW, RowH]),
    LMainX is PadX + MarginW,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" fill-opacity="0.85" stroke="#3a2a1a" stroke-width="0.3"/>~n',
        [LMainX, Y, ColW, RowH, LeftBg]),

    %% Middle column
    MMarginX is PadX + ColTotalW + GapX,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="#e8e3d8" stroke="#3a2a1a" stroke-width="0.3"/>~n',
        [MMarginX, Y, MarginW, RowH]),
    MMainX is MMarginX + MarginW,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" fill-opacity="0.85" stroke="#3a2a1a" stroke-width="0.3"/>~n',
        [MMainX, Y, ColW, RowH, MidBg]),

    %% Right column
    RMarginX is PadX + 2 * ColTotalW + 2 * GapX,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="#e8e3d8" stroke="#3a2a1a" stroke-width="0.3"/>~n',
        [RMarginX, Y, MarginW, RowH]),
    RMainX is RMarginX + MarginW,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" fill-opacity="0.85" stroke="#3a2a1a" stroke-width="0.3"/>~n',
        [RMainX, Y, ColW, RowH, RightBg]),

    TextY is Y + RowH - 5,
    emit_label_letter(S, PadX + 4, TextY, L, LLabels),
    emit_label_letter(S, MMarginX + 4, TextY, M, MLabels),
    emit_label_letter(S, RMarginX + 4, TextY, R, RLabels),
    instr_text(L, LText),
    instr_text(M, MText),
    instr_text(R, RText),
    LTX is LMainX + 6, MTX is MMainX + 6, RTX is RMainX + 6,
    format(S, '  <text x="~w" y="~w" font-size="11" fill="#1a1a1a">~w</text>~n',
        [LTX, TextY, LText]),
    format(S, '  <text x="~w" y="~w" font-size="11" fill="#1a1a1a">~w</text>~n',
        [MTX, TextY, MText]),
    format(S, '  <text x="~w" y="~w" font-size="11" fill="#1a1a1a">~w</text>~n',
        [RTX, TextY, RText]),
    Y1 is Y + RowH,
    render_triple_rows(S, Ls, Ms, Rs, Y1, PadX, MarginW, ColW, GapX, RowH,
                       LLabels, MLabels, RLabels).

%% ─────────────────────────────────────────────────────────────────────────────
%% SVG header + legend (shared by two- and three-column emit)
%% ─────────────────────────────────────────────────────────────────────────────

emit_svg_header(S, Title, CanvasW, CanvasH, ColLabels, PadX, ColW, GapX, HeaderH) :-
    format(S, '<svg xmlns="http://www.w3.org/2000/svg" width="~w" height="~w" viewBox="0 0 ~w ~w" font-family="\'Source Code Pro\', \'Menlo\', monospace">~n',
        [CanvasW, CanvasH, CanvasW, CanvasH]),
    format(S, '  <rect x="0" y="0" width="~w" height="~w" fill="#f8f5ee"/>~n',
        [CanvasW, CanvasH]),
    TitleX is CanvasW // 2,
    format(S, '  <text x="~w" y="32" font-size="20" font-weight="bold" fill="#3a2a1a" text-anchor="middle" font-family="Georgia, serif">~w</text>~n',
        [TitleX, Title]),
    format(S, '  <text x="~w" y="52" font-size="11" fill="#5a4a3a" text-anchor="middle" font-family="Georgia, serif">IR-layer drill-down · semantic-group alignment · 2026-05-30</text>~n',
        [TitleX]),
    emit_column_headers(S, ColLabels, PadX, ColW, GapX, HeaderH).

emit_column_headers(S, Labels, PadX, ColW, GapX, HeaderH) :-
    HeaderY is HeaderH - 6,
    emit_column_headers_loop(S, Labels, 0, PadX, ColW, GapX, HeaderY).

emit_column_headers_loop(_, [], _, _, _, _, _).
emit_column_headers_loop(S, [Label|Rest], Idx, PadX, ColW, GapX, HeaderY) :-
    HeaderX is PadX + Idx * (ColW + GapX) + ColW // 2,
    format(S, '  <text x="~w" y="~w" font-size="14" font-weight="bold" fill="#3a2a1a" text-anchor="middle" font-family="Georgia, serif">~w</text>~n',
        [HeaderX, HeaderY, Label]),
    NextIdx is Idx + 1,
    emit_column_headers_loop(S, Rest, NextIdx, PadX, ColW, GapX, HeaderY).

emit_svg_legend(S, X, Y) :-
    format(S, '  <text x="~w" y="~w" font-size="11" fill="#3a2a1a" font-family="Georgia, serif">Legend:</text>~n',
        [X, Y]),
    legend_swatch(S, X + 60,  Y, '#3f7c5c', 'identical mnemonic'),
    legend_swatch(S, X + 220, Y, '#daa520', 'same purpose, different form'),
    legend_swatch(S, X + 420, Y, '#8a3a2a', 'true divergence'),
    legend_swatch(S, X + 560, Y, '#c4c2ba', 'gap (missing in column)').

%% emit_label_legend (2-column variant): renders the mapping A→.LBB0_3, etc.
%% so the reader can decode the letters back to canonical label-text.
emit_label_legend(S, X, Y, LeftName, LeftLabels, RightName, RightLabels) :-
    labels_summary(LeftLabels, LSum),
    labels_summary(RightLabels, RSum),
    Y2 is Y + 14,
    format(S, '  <text x="~w" y="~w" font-size="10" fill="#3a2a1a" font-family="Georgia, serif">~w labels: ~w</text>~n',
        [X, Y, LeftName, LSum]),
    format(S, '  <text x="~w" y="~w" font-size="10" fill="#3a2a1a" font-family="Georgia, serif">~w labels: ~w</text>~n',
        [X, Y2, RightName, RSum]).

%% emit_label_legend (3-column variant)
emit_label_legend(S, X, Y, LName, LLabels, MName, MLabels, RName, RLabels) :-
    labels_summary(LLabels, LSum),
    labels_summary(MLabels, MSum),
    labels_summary(RLabels, RSum),
    Y2 is Y + 14,
    Y3 is Y + 28,
    format(S, '  <text x="~w" y="~w" font-size="10" fill="#3a2a1a" font-family="Georgia, serif">~w labels: ~w</text>~n',
        [X, Y, LName, LSum]),
    format(S, '  <text x="~w" y="~w" font-size="10" fill="#3a2a1a" font-family="Georgia, serif">~w labels: ~w</text>~n',
        [X, Y2, MName, MSum]),
    format(S, '  <text x="~w" y="~w" font-size="10" fill="#3a2a1a" font-family="Georgia, serif">~w labels: ~w</text>~n',
        [X, Y3, RName, RSum]).

%% labels_summary: format as "A=.LBB0_2, B=.LBB0_3, …"
labels_summary([], 'none').
labels_summary(Labels, Summary) :-
    Labels \= [],
    findall(Entry,
        ( member(label(_, Text, Letter), Labels),
          format(atom(Entry), '~w=~w', [Letter, Text])
        ),
        Entries),
    atomic_list_concat(Entries, ', ', Summary).

legend_swatch(S, X, Y, Color, Label) :-
    SwatchY is Y - 9,
    format(S, '  <rect x="~w" y="~w" width="14" height="12" fill="~w" stroke="#3a2a1a" stroke-width="0.5"/>~n',
        [X, SwatchY, Color]),
    TextX is X + 20,
    format(S, '  <text x="~w" y="~w" font-size="10" fill="#3a2a1a" font-family="Georgia, serif">~w</text>~n',
        [TextX, Y, Label]).

instr_text(gap, '·').
instr_text(instr(_, M, Ops, _, _), Text) :-
    xml_escape(Ops, OpsEscaped),
    format(atom(Text), '~w  ~w', [M, OpsEscaped]).

xml_escape(In, Out) :-
    ( atom(In) -> atom_string(In, Str) ; Str = In ),
    string_chars(Str, Chars),
    xml_escape_chars(Chars, Escaped),
    atom_chars(Out, Escaped).

xml_escape_chars([], []).
xml_escape_chars([C|Cs], Out) :-
    ( C = '<' -> Replacement = ['&','l','t',';']
    ; C = '>' -> Replacement = ['&','g','t',';']
    ; C = '&' -> Replacement = ['&','a','m','p',';']
    ; Replacement = [C]
    ),
    xml_escape_chars(Cs, Rest),
    append(Replacement, Rest, Out).

%% ─────────────────────────────────────────────────────────────────────────────
%% main — defaults to the three-way Iyun q8_0_dot worked example
%% ─────────────────────────────────────────────────────────────────────────────

main :-
    current_prolog_flag(argv, Argv),
    ( Argv = [LeftPath, RightPath, OutPath | _]
    -> emit_ir_compare_svg('left', LeftPath, 'right', RightPath,
                           'IR comparison', OutPath)
    ;  GgmlAsm  = '/tmp/output-only/ggml_q8_0_dot.asm',
       BpdIntr  = '/tmp/output-only/bpd_q8_0_dot_intrinsic.asm',
       BpdScal  = '/tmp/output-only/bpd_q8_0_dot_scalar.asm',
       OutPath  = '/tmp/output-only/q8_0_dot_compare.o.svg',
       emit_ir_compare_three_way(
           'ggml AVX (reference)',  GgmlAsm,
           'BPD intrinsic (proven 0 ULP)', BpdIntr,
           'BPD scalar (1.49e-8 vs ggml)', BpdScal,
           'q8_0 dot — three-way IR comparison',
           OutPath)
    ),
    halt(0).
