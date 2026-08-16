# Pilot guide

For the person standing at the launch point. This covers what the software checks, what
it does not, and where it will stop you.

## Before the aircraft leaves the ground

```python
report = api.run_preflight()
```

The preflight report is a list of findings, not a pass/fail light. Read it. It checks
what software can check — geofence containment, battery capacity against the planned
distance, home position, payload compatibility with the planned capture commands, and
whether the mission's altitudes came from a terrain model or an assumption.

**What it cannot check**, and you must:

- Airspace and permissions.
- Wind at altitude, which is not the wind at your feet.
- People and property under the flight path.
- Whether the DEM you loaded is actually of *this* site. Nothing about a raster says
  which site it belongs to; the software checks that the DEM covers your area, not that
  it is the right DEM.

## The warning you must not ignore

> No terrain model loaded: altitudes are relative to a flat plane at the launch
> elevation.

A flat-earth plan over sloping ground flies at a constant height above your *launch
point*, not above the ground. Downhill, you gain clearance and lose resolution. Uphill,
you lose clearance. The waypoints look identical either way — only the meaning changes.

If the terrain source cannot be read, you get:

> Terrain following was requested but the source could not be read; the plan assumes
> flat ground.

That is the same situation with an extra failure in it. Do not fly a terrain-following
mission that reported this over anything but flat ground.

## Payload

A mission carries the capture commands the fitted payload understands. A LiDAR sent a
shutter trigger flies the entire mission and lands with nothing, so the planner resolves
commands against the payload database and refuses an unknown payload rather than
guessing. If you changed the payload, re-plan.

## In flight

```python
api.start_mission()
api.get_telemetry()      # position, battery, GPS, RC link, waypoint index
api.return_to_home()
```

The waypoint index is reported in **your plan's numbering**. ArduPilot counts from its
own home item at sequence 0, so its indices run one ahead; the software translates so
the progress you see matches the plan you drew.

Manual override is always available and does not require the software's permission. A
mode change is confirmed against the vehicle's own heartbeat rather than against having
sent the command — an unrecognised mode is treated as autonomous, which is the
conservative reading.

## Battery swap and resume

```python
api.resume_mission()
```

Missions resume at the segment level. What was captured is tracked per segment, so a
resume flies what is missing rather than starting again.

## After landing

```python
api.verify_captured_data()
```

This matches the images on the card against the planned capture points and quantifies
deviation. It is worth doing **before you leave site**, because a gap found now is a
ten-minute reflight and a gap found in the office is a return trip.

The check reports three things: images matched to planned points, planned points with
no image, and how far the actual capture positions deviated from the plan.

## Things the software will refuse to do

These are deliberate. Each one exists because the alternative is a plausible wrong
answer:

- Report a measurement from a reconstruction with arbitrary scale.
- Report ponding or deformation without a stated vertical accuracy.
- Report anything shallower than twice that accuracy.
- Count assets from detections that carry no model digest.
- Present a demo artefact as a survey.
- Clear a rail corridor. `rail_obstacle_detector` misses roughly one obstacle in four;
  an empty result means the model found nothing, not that the track is safe.
