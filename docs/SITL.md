# ArduPilot SITL integration tests

The tests in `tests/sitl/` run OpenDroneKit's production mission planner, MAVLink
exporter and `MissionPlannerDroneClient` against a real ArduCopter Software In The
Loop process. They upload and download the mission, arm, take off, traverse the
planned waypoints, observe ArduPilot's `MISSION_ITEM_REACHED` messages, command RTL,
and wait for landing and disarm.

They are deliberately opt-in. A normal `pytest` run collects these tests but skips
them, so machines without a simulator stay green. Only the exact `-m sitl` selection
starts a simulator.

## Where the `fl.sitl` status comes from

That skip has a consequence worth stating plainly: feature status is computed from
passing tests, a skip counts as **no evidence** rather than as a pass, and so `fl.sitl`
reads *implemented* on any laptop no matter how many times the container succeeds.

CI is where the row is earned. The `sitl` job builds the image, flies both suites against
ArduPilot Copter-4.5.7, fails if they skipped rather than ran, and publishes the junit
report. The `status` job merges that report through
`tools/feature_status.py --extra-report`, so *verified* is backed by a flight run that
actually happened rather than by a status written down by hand. A missing or empty report
fails the job instead of computing status as though SITL had never been asked about.

This is the harness that caught the mission sequence bug: our missions put `NAV_TAKEOFF`
at sequence 0, which MAVLink reserves for home. ArduPilot silently overwrote it and the
aircraft would never have taken off — while every mock-based test passed, because a mock
stores what it is given and only a real autopilot has an opinion about sequence 0.

## Fixed-wing: ArduPlane

Fixed-wing missions are planned with `vehicle_class="fixed_wing"` (chosen automatically
when the connected vehicle's HEARTBEAT is `MAV_TYPE_FIXED_WING`) and flown by a real
ArduPlane in `tests/sitl/test_plane_missions.py`. Only the templates a plane can fly are
offered for it: `grid`, `double_grid`, `corridor` and `waypoints`; hover and structure
templates are refused with a reason. A fixed-wing plan orders its passes so every turn is
at least a turn diameter wide, extends each pass with a 5-radius lead-in and a flown
1-radius lead-out, uses airspeed, and ends in `DO_LAND_START` / `NAV_LOITER_TO_ALT` /
`NAV_LAND` at the vehicle's home along its heading (or `landing_lonlat` and
`landing_heading_deg`). The upload sets `RTL_AUTOLAND` and `RTL_ALTITUDE`; Land on a
plane jumps to the landing sequence.

The plane image is `infrastructure/docker/Dockerfile.sitl-plane` (Plane-4.5.7, the same
4.5 line as the copter image). It copies nothing from the context:

```bash
docker build -t opendronekit-sitl-plane:local - < infrastructure/docker/Dockerfile.sitl-plane
docker run --rm -e ODK_SITL_SPEEDUP=4 -e ODK_SITL_NO_REBUILD=1 -v "$PWD:/workspace"     -w /workspace opendronekit-sitl-plane:local     python3 -m pytest -m sitl tests/sitl/test_plane_missions.py
```

The plane tests skip unless `ODK_SITL_VEHICLE=ArduPlane` (the image sets it).

## Install ArduPilot SITL

ArduPilot's supported Linux setup starts with its complete build environment. The
official guides are:

- [Setting up SITL on Linux](https://ardupilot.org/dev/docs/setting-up-sitl-on-linux.html)
- [SITL on Windows using WSL](https://ardupilot.org/dev/docs/sitl-on-windows-wsl.html)
- [Using SITL](https://ardupilot.org/dev/docs/using-sitl-for-ardupilot-testing.html)

A typical Ubuntu/WSL installation is:

```bash
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile
cd ArduCopter
../Tools/autotest/sim_vehicle.py -v ArduCopter -f quad -w
```

Let that first invocation finish building and reach the MAVProxy prompt, then stop it
with Ctrl-C. The OpenDroneKit harness starts and stops subsequent instances itself.
The ArduPilot prerequisite script normally installs MAVProxy and pymavlink; the Python
environment running OpenDroneKit must also be able to import `pymavlink`.

On Windows, run both pytest and SITL inside the same WSL distribution. This avoids
depending on WSL networking modes for the two loopback UDP telemetry streams.

## Point OpenDroneKit at the clone

Either put `sim_vehicle.py` on `PATH`, or set one of:

```bash
export ARDUPILOT_HOME=/path/to/ardupilot
# Alternatively, including wrappers or an explicit Python interpreter:
export ARDUPILOT_SITL_COMMAND='/path/to/ardupilot/Tools/autotest/sim_vehicle.py'
```

Useful optional settings are:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `ODK_SITL_START_TIMEOUT_S` | `240` | Maximum build/boot/EKF wait |
| `ODK_SITL_FLIGHT_TIMEOUT_S` | `120` | Mission and RTL completion wait |
| `ODK_SITL_SPEEDUP` | `2` | ArduPilot simulation speed multiplier |
| `ODK_SITL_INSTANCE` | `0` | `sim_vehicle.py` instance number |
| `ODK_SITL_NO_REBUILD` | unset | Set to `1` to pass `--no-rebuild` |
| `ODK_SITL_EXTRA_ARGS` | unset | Additional `sim_vehicle.py` arguments |

Check discovery without starting anything:

```bash
python tools/sitl/launch.py --check
```

Run the integration tests:

```bash
python -m pytest -m sitl tests/sitl -v
```

If `sim_vehicle.py` or pymavlink is absent, an explicitly selected run is skipped with
the reason. If SITL is found but exits, fails to build, or never reports the required
EKF flags and 3D GPS fix, the run fails and includes the simulator log tail. The
launcher owns the complete process group and stops ArduCopter and MAVProxy during
fixture teardown, test failure, startup failure and normal interpreter exit.

## Production entry points under test

The mission is created with `MissionPlanner.generate(mode="waypoints", ...)` and
converted with `mission.exporters.build_mission_items(plan, include_rth=False)`.
Nothing in `mission/` is replaced or modified. The resulting command list is uploaded
and read back through `MissionPlannerDroneClient`; RTL is deliberately issued through
the client's `return_to_home()` instead of being appended to the mission.

The production client currently treats a successfully transmitted arm, takeoff,
mission-start or RTL command as a successful `CommandResult`; it does not await the
corresponding `COMMAND_ACK`. These tests therefore judge those commands by subsequent
autopilot state and telemetry. Also, the client records `MISSION_CURRENT` but does not
retain `MISSION_ITEM_REACHED`, so the harness uses the second passive MAVLink output
for exact reached-waypoint evidence. A simulated flight validates MAVLink/autopilot
integration, not real-aircraft dynamics, navigation accuracy or safety.
