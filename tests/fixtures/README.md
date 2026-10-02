# Test fixture provenance

`concrete_cracked_public_domain.jpg` is the Wikimedia Commons file
[`Concrete cracked.jpg`](https://commons.wikimedia.org/wiki/File:Concrete_cracked.jpg),
photographed by John Harvey and released into the public domain by its author. It was
downloaded from the Commons original-file endpoint on 2026-08-15.

- Purpose: real photographic input for installed-model assisted-annotation tests.
- SHA-256: `e020561bb21af625a19e820cbd75ce8246505fd4de68dd0ad6a12b57ba28e6ba`
- The test does not claim that one photograph is a field-validation dataset; it proves
  that the production inference and review path runs against a real, licensed input.

`ir_module_hot_spot.jpg` and `ir_module_no_anomaly.jpg` are images `6796.jpg`
(labelled Hot-Spot) and `13357.jpg` (labelled No-Anomaly) from Raptor Maps'
InfraredSolarModules dataset (2020-02-14 release), MIT License, Copyright (c) 2020
Raptor Maps, Inc. Each is one 24x40 px infrared crop of a single installed module.

- Purpose: real thermal-module input for the solar thermal classifier path of
  `detect_solar_defects`.
- SHA-256: `cf60655772e2b2f92f1e42e9062db01116e2a1e82ef8ba2ff26a7c80b1ac00b9` (hot spot), `62b70eb796670020c42b1689447d1d5c8a29e3db337b26629b24269f88c051df` (no anomaly)

`thedpv_module_regions.json` holds module outlines drawn on two THED-PV frames
(`20241221_120204_881_IR.TIFF`, `20241221_120227_869_IR.TIFF`; Zenodo
10.5281/zenodo.17404247, CC BY 4.0). It contains geometry only, no temperatures; the
frames themselves stay in the odk-testdata collection (`ODK_TESTDATA_DIR`).

- Purpose: real module geometry for image-space per-module temperature analysis
  (`module_temperatures_in_image`). Each frame shows the aluminium frame rail between two
  modules, in IR and in the paired RGB frame. Upper and lower regions stop 15 px short
  of the rail edges measured from the IR temperature profile.
- Annotated 2026-10-02 by Claude acting as the operator. The method is recorded in the
  file. Module IDs are positional; plant module and string IDs are not in the dataset.
