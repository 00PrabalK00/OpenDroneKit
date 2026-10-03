# Licence: shared_semantic_landcover (HELD BACK)

- **Status:** not published. Users may run weights they obtain themselves; the registry row
  and its sha256 check are unchanged.
- **Base model:** DINOv2 ViT-B/14, code and weights Apache-2.0
  (https://github.com/facebookresearch/dinov2).
- **Why held:** the training data mixes CC BY-SA 4.0 (SpaceNet 7; some OpenEarthMap regions)
  with CC BY-NC-SA 4.0 (OpenEarthMap regions admitted by commit 8e06ab5c1,
  `--include-noncommercial`; https://open-earth-map.org/attribution.html). Each licence's
  ShareAlike clause binds adaptations to its own licence elements. CC BY-SA 4.0 §3(b)(1):
  "The Adapter's License You apply must be a Creative Commons license with the same
  License Elements, this version or later, or a BY-SA Compatible License." CC BY-NC-SA 4.0
  §3(b)(1) says the same with its own elements, which include NonCommercial. No licence
  carries both element sets, so no compliant terms exist for these weights.
- **Fix:** retrain without the NC regions (the default corpus build, without
  `--include-noncommercial`), then publish under CC BY-SA 4.0.
