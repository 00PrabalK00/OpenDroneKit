# odk-csv-exporter (example OpenDroneKit plugin)

Two plugins in one package:

| Kind | Name | What it does |
|---|---|---|
| `exporter` | `waypoints_csv` | Writes the planned waypoints as `index,longitude,latitude,altitude_m,altitude_frame,gimbal_tilt_deg`. |
| `report_section` | `mission_waypoints` | Adds a table of the project's saved missions to a report. |

```bash
pip install -e examples/plugins/odk-csv-exporter
```

```python
plan.export(["qgc_plan", "waypoints_csv"])
project.generate_report(title="Survey", plugin_sections=["mission_waypoints"])
```

See `docs/plugins.md` and `docs/tutorials/03-custom-plugin.md`. MIT licence.
