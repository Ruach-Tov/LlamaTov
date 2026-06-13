#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# Usage: show.sh [VIEW]   VIEW = dashboard (default) | program | bandwidth | correspondence | taps
VIEW="${1:-dashboard}"
cd .
swipl -q -g "use_module('bpd-substrate/lib/tensor_schema'), \
  consult('/tmp/t10011/op_facts.pl'), \
  consult('/tmp/t10011/verdict_facts.pl'), \
  use_module('bpd-substrate/lib/render_ascii'), \
  render([mistral,layer(l),attn,qkv], ${VIEW}), halt" 2>/dev/null
