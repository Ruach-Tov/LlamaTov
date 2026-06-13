#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# Show the full-model op listing. Usage: ops.sh [--summary|--f16-only|--layer N]
python3 "$(dirname "$0")"/../../bpd-substrate/tools/model_op_listing.py \
  "${DATA:-./data}" "$@"
