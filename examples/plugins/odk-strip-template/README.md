# odk-strip-template (example OpenDroneKit plugin)

A `mission_type` plugin named `strip_survey`: give it a centreline and a strip width and
it builds the strip polygon and asks the built-in planner for a high-overlap grid over it.

```bash
pip install -e examples/plugins/odk-strip-template
```

```python
plan = project.plan_mission([], template="plugin:strip_survey",
                            centreline_lonlat=[[-81.7538, 41.3039], [-81.7506, 41.3048]],
                            width_m=50, altitude_m=60)
```

See `docs/plugins.md` and `docs/tutorials/03-custom-plugin.md`. MIT licence.
