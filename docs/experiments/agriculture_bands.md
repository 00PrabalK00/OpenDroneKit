# Do red edge and near infrared earn their place?

Crop and weed separation is supposed to live in the bands human eyes cannot see: a weed
and a maize leaf differ far more in reflectance than in visible colour. The agriculture
corpus ships five bands per capture (B, G, R, RE, NIR) and the first model was trained on
an RGB composite built from three of them — the three that discriminate least.

This is the comparison that settles whether the other two are worth carrying, run as two
arms that differ in exactly one thing.

## Method

Same corpus (WeedsGalore, 104 train / 26 val / 26 test on the authors' own split), same
trainer, same SegFormer-B2 @512, same schedule. The only difference is `band_root`: the
5-band arm reads the cached R,G,B,RE,NIR stacks, the RGB arm reads the composite.

Both ran on Kaggle. Kernels: `odk-train-agriculture-segformer-b2-mc` (RGB) and
`odk-train-agriculture-segformer-b2-ms` (5-band).

## Result

| Arm | Epochs | Mean IoU | soil | maize | **weed** |
|---|---:|---:|---:|---:|---:|
| RGB | 70 | 0.5525 | 0.955 | 0.509 | **0.183** |
| 5-band | 76 | 0.5614 | 0.957 | 0.515 | **0.206** |

The extra bands help, and they help most where the theory said they would. Mean IoU moves
+0.009, which on its own would be noise on a corpus this size. Weed IoU moves 0.183 →
0.206: +0.023 absolute, **+12.6 per cent relative**, on the one class the capability
exists to find. Soil, which is easy in any band, does not move.

## What this does not show

**Weed IoU of 0.206 is not a usable weeding model.** Finding roughly a fifth of the weed
pixels is a direction of travel, not a deliverable, and no agriculture model is registered
on the strength of it.

The corpus is the limit rather than the bands or the architecture: 104 training images of
one field flown on four dates. A comparison at that size can show a consistent direction —
and this one does, on the class where the mechanism is understood — but it cannot separate
a small real effect from a lucky split. Neither arm was evaluated on Indian fields.

## What was nearly reported instead

The 5-band arm's first run trained on RGB. `band_root` in the config is a repo-relative
path that does not exist on a rented machine, the dataset returned `None` for the missing
stack, and the loader quietly fell back to the RGB composite. Both arms would have been
RGB, the difference would have been zero, and the honest-looking conclusion would have
been "the extra bands do not help".

A missing stack under a configured `band_root` is now a `FileNotFoundError` naming the
sample, and the kernel passes `--band-root` pointing at the mounted copy. The run above
logged `multispectral stacks at ...` before training, which is the evidence that this arm
read five bands.
