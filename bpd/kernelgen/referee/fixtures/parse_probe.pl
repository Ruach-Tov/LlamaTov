%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- initialization(main).
main :-
    consult('lib/c_ast'),
    %% real llama.cpp builder lines (from llm_build_llama body)
    Lines = [
        "ggml_tensor * cur = ggml_mul_mat(ctx0, model.layers[il].wq, inpL);",
        "ggml_tensor * inpL = build_inp_embd(model.tok_embd);",
        "cur = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);",
        "ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);"
    ],
    forall(member(L, Lines),
        ( string_codes(L, Codes),
          ( catch(c_tokenize(Codes, Toks), TE, (format("  TOKENIZE-ERR: ~w~n",[TE]), fail))
            -> ( catch((phrase(parse_stmt_v2(AST), Toks) ; phrase(parse_stmt(AST), Toks)), PE,
                       (format("  PARSE-ERR: ~w~n",[PE]), fail))
                 -> format("OK  ~w~n    -> ~q~n", [L, AST])
                 ;  format("NO-PARSE  ~w~n    tokens: ~q~n", [L, Toks]) )
            ; format("TOKENIZE-FAIL ~w~n",[L]) )
        )),
    halt.
