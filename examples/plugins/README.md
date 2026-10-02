# Example plugins

Installable examples of the OpenDroneKit plugin API (`docs/plugins.md`):

- `odk-csv-exporter/`: an exporter (`waypoints_csv`) and a report section (`mission_waypoints`).
- `odk-strip-template/`: a mission template (`strip_survey`).

Install either into the environment that runs OpenDroneKit with `pip install -e <folder>`;
they then appear in the API's Environment Capabilities data (`capabilities()["plugins"]`)
and in the application's Plugins panel.
`tests/test_example_plugins.py` installs both in editable mode into a scratch prefix and
exercises them through the SDK.
