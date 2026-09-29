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

`esri_no_imagery_placeholder.jpg` is the tile Esri's World Imagery service returned on
2026-09-29 for `tile/20/391962/286167` (z20 over the Aukerman site, 41.3041 N 81.7521 W),
with HTTP 200, where the service holds no imagery: a flat grey (204,204,204) 256x256
JPEG reading "Map data not yet available". The same bytes came back for z20-z23 there,
from both server.arcgisonline.com and services.arcgisonline.com, and for World_Topo_Map.

- Purpose: the bytes `core/tile_cache.py` and `app/web/js/basemap-depth.js` recognise, so
  the placeholder is never cached, served or drawn in place of imagery.
- SHA-256: `9eafd300d61393184a4abc1d458564cfd1cd9b6f9c4e9c74687045c0a0e5b858` (2521 bytes)
