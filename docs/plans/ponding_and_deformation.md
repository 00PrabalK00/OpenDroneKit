# Water ponding and surface deformation: measure, do not classify

`ai.water_ponding` and `ai.deformation` were the only two AI capabilities in the
167-capability spec with neither a dataset nor a model plan behind them. This records
what searching for that data turned up, and why the conclusion is that neither should be
a learned model.

## What the search found

**Water ponding.** No open dataset exists for ponded water on roofs or paved surfaces.
The nearest neighbours are roof-segmentation corpora — AIRS covers 457 km² of
Christchurch at 7.5 cm GSD with ~220,000 buildings, and Nacala-Roof-Material covers
informal settlements in Mozambique — but both label *roofs and roof materials*, not
standing water on them. Flood-mapping work exists at the opposite scale: whole
inundated landscapes, not a 40 mm puddle on a warehouse roof.

**Deformation.** The literature is close to unanimous, and it is not about classifiers.
Bridge, dam, landslide and historic-building deformation are measured by **multi-temporal
photogrammetry and DEM differencing**: fly the same structure twice, build both surfaces,
subtract them. Reported accuracy is millimetre-level for bridges at 30 m flying height
(~1.3 mm GSD) and 0.2–0.9 m for open-pit subsidence at ~6.3 cm GSD. Nobody is training a
network to look at one photograph and say "this has deformed", because a single image does
not contain that information.

## Why that settles it

Both capabilities are **geometric measurements, not appearance classifications**.

Ponding is defined by where water can collect: a closed depression in a surface, below a
drainage path, deeper than some threshold. That is a question about a DSM. A model
trained on photographs would be guessing at it from colour and specularity — and would
guess confidently on wet-but-not-ponded surfaces, dark roofing membrane, shadow, and
solar glare, all of which look like standing water from above.

Deformation is the difference between two surfaces captured at different times. A model
shown one survey has nothing to compare against; it can only pattern-match to what
deformation *usually looks like*, which is exactly the plausible-but-wrong output this
project exists to refuse.

Building learned versions of these would produce two capabilities that answer confidently
and cannot be checked. Building measured versions produces two that state a number with
an error bound, or refuse.

## The plan

Neither needs new training data. Both are compositions of machinery the project already
has, plus one honest gate each.

### `ai.water_ponding` — depression analysis on the DSM

1. Take the DSM already produced by the reconstruction pipeline.
2. Fill sinks; the difference between filled and original is the depression depth field.
3. Threshold on depth and area to reject noise: a one-pixel dimple is not ponding, and
   the threshold must be tied to the DSM's own vertical error rather than picked.
4. Report each candidate as an area, a maximum depth, and a volume, with the DSM's
   vertical accuracy carried through as the uncertainty on all three.
5. **Refuse** when the reconstruction has no vertical accuracy estimate, when GSD is too
   coarse for the depth threshold to mean anything, or when the surface is not projected.

Optional and clearly separated: an RGB/thermal check on whether a depression is *currently*
holding water. That is a genuine appearance question, and it is a different claim from
"water can collect here" — the two must never be merged into one number.

### `ai.deformation` — surface differencing between epochs

1. Require two reconstructions of the same site with a shared, stated CRS.
2. Co-register on stable ground or on the existing GCP machinery, and report the
   registration residual — this is the error floor for everything downstream.
3. Difference the surfaces to get a vertical displacement field.
4. **Refuse to report any displacement smaller than the combined registration residual
   and the two surveys' vertical accuracies.** This is the whole gate: without it the
   output is noise with a decimal point.
5. Report displacement magnitude, extent, and the detection floor beside it, so a reader
   can see what the survey was capable of resolving.

This reuses `core/terrain_cache.py` for surfaces, the GCP residual work for
co-registration, and `core/slope.py`'s existing refusal on unprojected CRS.

## What this costs and what it buys

No datasets to license, no models to train, no weights to host. Both become deterministic
measurements with stated uncertainty, testable against synthetic surfaces with known
depressions and known displacements — which means they can reach `verified` on evidence
rather than on a metric from a corpus that does not exist.

The honest cost: neither will detect anything from a single flight. Ponding needs a
reconstruction, deformation needs two. That is a real limitation and belongs in the
capability description rather than being engineered around.

## Status

`ai.water_ponding` and `ai.deformation` stay `not_started` until implemented. They are no
longer *unplanned* — the reason they have no dataset is that they should not have one.
