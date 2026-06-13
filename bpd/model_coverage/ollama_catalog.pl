%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
% ollama_catalog.pl — archived snapshot of ollama.com/library (2026-05-31, Iyun)
% 229 models. Each: ollama_model(Name, Description).
% Purpose: test-coverage reference. The llama-family models route through llm_build_llama
% (the compute graph Iyun reproduced bit-exact across 16 layers). graph_aliased(mistral,llama).
%
% ARCHITECTURE TAGGING: ollama_arch(Name, Arch) maps catalog names to their llama.cpp graph
% where known. Models with arch=llama use the EXACT graph we verified 0-ULP.
%
% :- module(ollama_catalog, [ollama_model/2, ollama_arch/2, llama_graph_model/1]).

:- discontiguous ollama_model/2.
:- discontiguous ollama_arch/2.

% A model uses the verified llama graph if its arch is llama (or aliases to it).
llama_graph_model(Name) :-
    ollama_arch(Name, Arch),
    ( Arch == llama
    ; catch(model_zoo:graph_aliased(llamacpp:Arch, llamacpp:llama), _, fail) ).

% ── ARCHITECTURE TAGS (known llama-graph models in the catalog) ──
% These report general.architecture="llama" in their GGUF and build via llm_build_llama.
ollama_arch(llama2, llama).
ollama_arch('llama2-uncensored', llama).
ollama_arch('llama2-chinese', llama).
ollama_arch(codellama, llama).
ollama_arch('phind-codellama', llama).
ollama_arch(tinyllama, llama).
ollama_arch(tinydolphin, llama).
ollama_arch(vicuna, llama).
ollama_arch('wizard-vicuna', llama).
ollama_arch('wizard-vicuna-uncensored', llama).
ollama_arch(zephyr, llama).        % Mistral/Mixtral fine-tune (mistral==llama graph)
ollama_arch(mistral, llama).       % mistral GGUF reports architecture=llama
ollama_arch('mistral-openorca', llama).
ollama_arch(mistrallite, llama).
ollama_arch('samantha-mistral', llama).
ollama_arch('dolphin-mistral', llama).
ollama_arch('neural-chat', llama).
ollama_arch(openhermes, llama).
ollama_arch('nous-hermes', llama).
ollama_arch(openchat, llama).
ollama_arch(starling-lm, llama).
ollama_arch(notus, llama).
ollama_arch('deepseek-coder', llama).   % the 1B/legacy deepseek-coder is llama arch
ollama_arch(wizardlm, llama).
ollama_arch('wizardlm-uncensored', llama).
ollama_arch('wizard-math', llama).
ollama_arch(wizardcoder, llama).
ollama_arch(orca2, llama).
ollama_arch('orca-mini', llama).
ollama_arch('stable-beluga', llama).
ollama_arch('llama-pro', llama).
ollama_arch(meditron, llama).
ollama_arch(medllama2, llama).
ollama_arch(everythinglm, llama).
ollama_arch(xwinlm, llama).
ollama_arch(goliath, llama).
ollama_arch('open-orca-platypus2', llama).
ollama_arch(codebooga, llama).
ollama_arch(megadolphin, llama).
ollama_arch(codeup, llama).
ollama_arch('yarn-llama2', llama).
ollama_arch('yarn-mistral', llama).
ollama_arch('stablelm-zephyr', llama).
ollama_arch(magicoder, llama).
ollama_arch(mathstral, llama).
ollama_arch(codestral, llama).
ollama_arch('llama3', llama).
ollama_arch('llama3.1', llama).
ollama_arch('llama3.2', llama).
ollama_arch('llama3.3', llama).
ollama_arch('llama3-chatqa', llama).
ollama_arch('llama3-gradient', llama).
ollama_arch('llama3-groq-tool-use', llama).
ollama_arch('llama-guard3', llama).
ollama_arch(dolphin3, llama).
ollama_arch('dolphin-llama3', llama).
ollama_arch(hermes3, llama).
ollama_arch('firefunction-v2', llama).
ollama_arch(nemotron, llama).
ollama_arch('nemotron-mini', llama).
ollama_arch(reflection, llama).
ollama_arch(tulu3, llama).
ollama_arch(aya, llama).
ollama_arch('aya-expanse', llama).
ollama_arch(solar, llama).
ollama_arch('llava-llama3', llama).

% Non-llama-graph examples (for contrast / future coverage):
ollama_arch(phi3, phi3).
ollama_arch('phi3.5', phi3).
ollama_arch(gemma, gemma).
ollama_arch(gemma2, gemma2).
ollama_arch(gemma3, gemma3).
ollama_arch(qwen2, qwen2).
ollama_arch('qwen2.5', qwen2).
ollama_arch(qwen3, qwen3).
ollama_arch('granite3.3', granite).
ollama_arch('granite3.1-moe', granite_moe).
ollama_arch(starcoder2, starcoder2).
ollama_arch(falcon, falcon).
ollama_arch(mixtral, llama).   % Mixtral = MoE but llama-family graph dispatch
