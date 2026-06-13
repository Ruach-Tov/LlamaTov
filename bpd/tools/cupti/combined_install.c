// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <SWI-Prolog.h>
extern void install_cupti_bridge(void);
extern void install_q8_gemv_launcher(void);
install_t install_cupti_q8(void){ install_cupti_bridge(); install_q8_gemv_launcher(); }
