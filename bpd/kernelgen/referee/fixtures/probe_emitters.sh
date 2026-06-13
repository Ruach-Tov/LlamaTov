#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# CONSULT-AND-SINGLETON sweep over all 25 emitters: does each load clean?
# (The launch_bounds bug taught us: singleton warnings in EMITTERS are loaded guns.)
cd .
for f in bpd/kernelgen/emitters/*.pl; do
  out=$(swipl -q -g "consult('$f'), halt" -t 'halt(1)' 2>&1)
  if [ -z "$out" ]; then
    echo "CLEAN  $(basename $f)"
  else
    nwarn=$(echo "$out" | grep -c "Singleton\|Warning")
    nerr=$(echo "$out" | grep -c "ERROR")
    echo "DIRTY  $(basename $f) (errors=$nerr warn-lines=$nwarn)"
    echo "$out" | grep -E "ERROR|Singleton" | head -3 | sed "s/^/         /"
  fi
done
