# Tutorials

Three walkthroughs of the [OpenDroneKit Python SDK](../sdk/README.md) on open sample data.
Each one runs as written: `tests/test_tutorials.py` extracts the code and runs it on the
same files `tools/fetch_sample_data.py` downloads, so a tutorial that stops working fails
the test suite.

| Tutorial | Data | Covers |
|---|---|---|
| [1. Aukerman park, end to end](01-aukerman-end-to-end.md) | OpenDroneMap Aukerman, 8 photos (CC0) | project, mission plan + export, reconstruction, map measurement, report |
| [2. Crack detection](02-crack-detection.md) | RDD2022 road damage, 1 photo (CC BY 4.0) | crack segmentation, model provenance, comparison with the annotation, scale refusal |
| [3. A custom plugin](03-custom-plugin.md) | Aukerman footprint | writing, installing and using an exporter plugin; broken-plugin isolation |

## Sample data

```bash
python tools/fetch_sample_data.py list                  # every dataset, its licence and size
python tools/fetch_sample_data.py fetch odm_aukerman --dest sample_data --limit 8
python tools/fetch_sample_data.py fetch road_damage_rdd2022_subset --dest sample_data
python tools/fetch_sample_data.py verify odm_aukerman --dest sample_data --limit 8
```

Files are downloaded from the publishers (GitHub at a pinned commit, Zenodo, figshare);
OpenDroneKit does not mirror or redistribute any of them. Every file is checked against a
pinned sha256. Datasets behind an access request or a proprietary licence (the University
of Guam thermal sets, the DJI Thermal SDK) are never downloaded: the script prints where
to request them. Credit the authors of CC BY datasets as their licence requires; the
attribution is in `tools/sample_data/catalog.json` and printed by `list` and `fetch`.

The tutorials read the data from `sample_data/` (override with `ODK_SAMPLE_DATA`) and
write their workspaces to `odk-tutorial/` (override with `ODK_TUTORIAL_DIR`).

To run a tutorial's code without copying it by hand:

```bash
python tools/run_tutorial.py docs/tutorials/01-aukerman-end-to-end.md --workdir odk-tutorial
```

Tutorial 3 writes a plugin package that you install before its code runs:

```bash
python tools/run_tutorial.py docs/tutorials/03-custom-plugin.md --workdir odk-tutorial --write-files-only
pip install -e odk-tutorial/odk-geojson-path
python tools/run_tutorial.py docs/tutorials/03-custom-plugin.md --workdir odk-tutorial
```
