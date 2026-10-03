# PV module database (`cec_modules.csv.gz`)

- **What:** the California Energy Commission (CEC) PV module list as packaged in the
  module library of NREL's System Advisor Model (SAM), file `deploy/libraries/CEC Modules.csv`.
- **Source URL:** https://github.com/NREL/SAM/blob/develop/deploy/libraries/CEC%20Modules.csv
  (downloaded 2026-10-02, branch `develop`; library version `2026.7.3`, dated 6/23/2026).
- **SHA-256 of the downloaded file:** `a28cf8956a839ed98bb941809b439e4907f5b4c237bd9299bc123fb502c29aff`
- **Licence:** SAM is distributed under the BSD 3-Clause License, which permits
  redistribution with the notice below. The underlying module list is published by the
  California Energy Commission (https://www.energy.ca.gov/programs-and-topics/programs/solar-equipment-lists).
- **What was changed:** only columns were dropped (kept: Name, Manufacturer, Technology,
  Bifacial, STC, PTC, A_c, Length, Width, N_s, I_sc_ref, V_oc_ref, I_mp_ref, V_mp_ref,
  gamma_pmp, T_NOCT, BIPV, Version, Date) and the two SAM header rows were reduced to one
  units row. No value was edited. 21,677 modules.

Module efficiency is not a column of the list. `core/pv_modules.py` computes it as
`STC / (A_c x 1000 W/m²)` and says so wherever it is printed.

## SAM licence notice

BSD 3-Clause License

Copyright (c) 2017, 2022 Alliance for Energy Innovation, LLC
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
