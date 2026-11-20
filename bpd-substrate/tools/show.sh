#!/usr/bin/env bash
# Usage: show.sh [VIEW]   VIEW = dashboard (default) | program | bandwidth | correspondence | taps
VIEW="${1:-dashboard}"
cd /mnt/data/home/iyun/Ruach-Tov
swipl -q -g "use_module('bpd-substrate/lib/tensor_schema'), \
  consult('/tmp/t10011/op_facts.pl'), \
  consult('/tmp/t10011/verdict_facts.pl'), \
  use_module('bpd-substrate/lib/render_ascii'), \
  render([mistral,layer(l),attn,qkv], ${VIEW}), halt" 2>/dev/null
