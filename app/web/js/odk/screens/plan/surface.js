/* Plan: where every verb on this screen lives.

   Machine-readable placements, read by the reachability and parity tests instead of
   the old `toolbar: [` arrays. One entry per control:

     { action: "<ACTIONS key>", where: "plan.<region>", label: "<visible label>" }

   `action` must resolve in the merged registry (core ACTIONS plus ./actions.js), and
   the rendered button must carry data-action="<action>". Schema and rules:
   docs/ui/ARCHITECTURE.md, "surface.js". Every inventory row for this workspace
   (docs/ui/inventory.json) is here, in PLACEMENTS (a verb) or CONTROLS (an input or a
   control that changes the form, not the project).

   `where` regions: plan.left.* is the left panel, plan.main.* the mission bar, map and
   strip, plan.inspector.* the inspector, plan.field.* the Field screen. `via` names the
   data-control of the menu button a verb sits behind (the verb is in that popover). */

export const SCREEN = "plan";

export const PLACEMENTS = [
  /* left: area of interest */
  { action: "Edit Vertices", where: "plan.left.aoi", label: "Edit vertices", inventory: "plan.edit-vertices-delete", modes: ["desk"] },
  { action: "Import Boundary", where: "plan.left.aoi", label: "Import KML", inventory: "plan.import-boundary", modes: ["desk"] },
  { action: "Draw Area", where: "plan.left.aoi", via: "aoi:more", label: "Draw area", inventory: "plan.draw-area-of-interest", modes: ["desk"] },
  { action: "Area From Photos", where: "plan.left.aoi", via: "aoi:more", label: "Outline the camera positions", inventory: "plan.aoi-from-camera-hull", modes: ["desk"] },
  { action: "Facade Footprint", where: "plan.left.aoi", via: "aoi:more", label: "Read as a building footprint", inventory: "plan.multi-facade-irregular-facade", modes: ["desk"] },
  { action: "Mark Corner", where: "plan.left.aoi", via: "aoi:more", label: "Fly the corners: mark corner", inventory: "plan.fly-to-draw-boundary", modes: ["desk"] },
  { action: "Close Boundary", where: "plan.left.aoi", via: "aoi:more", label: "Close the flown boundary", inventory: "plan.fly-to-draw-boundary", modes: ["desk"] },
  { action: "Clear Corners", where: "plan.left.aoi", via: "aoi:more", label: "Clear flown corners", inventory: "plan.fly-to-draw-boundary", modes: ["desk"] },
  { action: "Zoom to Area", where: "plan.left.aoi", via: "aoi:more", label: "Fit to area", inventory: "plan.zoom-to-area-mission", modes: ["desk"] },
  { action: "Clear Area", where: "plan.left.aoi", via: "aoi:more", label: "Clear area", inventory: "plan.clear-area-clear-no-fly", modes: ["desk"] },
  { action: "Clear No-fly", where: "plan.left.aoi", via: "aoi:more", label: "Clear no-fly zones", inventory: "plan.clear-area-clear-no-fly", modes: ["desk"] },

  /* main: mission bar */
  { action: "Save", where: "plan.main.versions", via: "versions", label: "Save version", inventory: "plan.save-version", modes: ["desk"] },
  { action: "Load Version", where: "plan.main.versions", via: "versions", label: "Open a saved version", inventory: "plan.history-and-restore", modes: ["desk"] },
  { action: "History", where: "plan.main.versions", via: "versions", label: "Restore as newest", inventory: "plan.history-and-restore", modes: ["desk"] },
  { action: "Compare Versions", where: "plan.main.versions", via: "versions", label: "Compare versions", inventory: "plan.compare-versions-specifications", modes: ["desk"] },
  { action: "Compare Specifications", where: "plan.main.versions", via: "versions", label: "Compare specifications", inventory: "plan.compare-versions-specifications", modes: ["desk"] },
  { action: "Repeat Survey", where: "plan.main.versions", via: "versions", label: "Repeat survey", inventory: "plan.repeat-survey", modes: ["desk"] },
  { action: "Select", where: "plan.main.tools", label: "Select", inventory: "plan.draw-area-of-interest", modes: ["desk"] },
  { action: "Draw Area", where: "plan.main.tools", label: "Draw area", inventory: "plan.draw-area-of-interest", modes: ["desk"] },
  { action: "Edit Vertices", where: "plan.main.tools", label: "Edit vertices", inventory: "plan.edit-vertices-delete", modes: ["desk"] },
  { action: "Rotate Flight Lines", where: "plan.main.tools", label: "Rotate flight lines", inventory: "plan.line-heading", modes: ["desk"] },
  { action: "Distance", where: "plan.main.tools", label: "Measure distance", inventory: "plan.measure-on-the-map", modes: ["desk"] },
  { action: "Draw No-fly", where: "plan.main.tools", via: "tools:more", label: "No-fly zone", inventory: "plan.no-fly-zone", modes: ["desk"] },
  { action: "Draw Route", where: "plan.main.tools", via: "tools:more", label: "Route line", inventory: "plan.route-line-interest-region", modes: ["desk"] },
  { action: "Draw Interest Region", where: "plan.main.tools", via: "tools:more", label: "Interest region", inventory: "plan.route-line-interest-region", modes: ["desk"] },
  { action: "Clear Route", where: "plan.main.tools", via: "tools:more", label: "Clear route and regions", inventory: "plan.route-line-interest-region", modes: ["desk"] },
  { action: "Place Home", where: "plan.main.tools", via: "tools:more", label: "Place home", inventory: "plan.place-home", modes: ["desk"] },
  { action: "Add Marker", where: "plan.main.tools", via: "tools:more", label: "Site marker", inventory: "plan.site-markers", modes: ["desk"] },
  { action: "Import CAD", where: "plan.main.tools", via: "tools:more", label: "CAD or image overlay", inventory: "plan.cad-image-overlay", modes: ["desk"] },
  { action: "Area", where: "plan.main.tools", via: "tools:more", label: "Measure area", inventory: "plan.measure-on-the-map", modes: ["desk"] },
  { action: "Zoom to Mission", where: "plan.main.tools", via: "tools:more", label: "Fit to mission", inventory: "plan.zoom-to-area-mission", modes: ["desk"] },
  { action: "Undo Area Edit", where: "plan.main.undo", label: "Undo area edit", inventory: "plan.area-undo-redo", modes: ["desk"] },
  { action: "Redo Area Edit", where: "plan.main.undo", label: "Redo area edit", inventory: "plan.area-undo-redo", modes: ["desk"] },
  /* main: strip */
  { action: "Play Route", where: "plan.main.strip", label: "Play the route", inventory: "plan.simulate", modes: ["desk"] },
  /* main: map (the map module's Layers popover, on every map) */
  { action: "Cache Basemap View", where: "plan.main.map", via: "layers", label: "Cache this view", inventory: "plan.basemap-cache-view-cache-status", modes: ["desk"] },

  /* inspector: Plan tab */
  { action: "Plan", where: "plan.inspector.head", label: "Generate / Re-plan", inventory: "plan.generate-plan", modes: ["desk"] },
  { action: "Describe Camera", where: "plan.inspector.camera", via: "camera-picker", label: "Describe", inventory: "plan.camera-picker", modes: ["desk"] },
  { action: "Add Camera", where: "plan.inspector.camera", via: "camera-picker", label: "Add camera", inventory: "plan.camera-picker", modes: ["desk"] },
  { action: "Camera Capabilities", where: "plan.inspector.camera", via: "camera-picker", label: "Connected camera", inventory: "plan.camera-picker", modes: ["desk"] },
  { action: "Cameras", where: "plan.inspector.camera", via: "camera-picker", label: "All at an altitude", inventory: "plan.camera-picker", modes: ["desk"] },
  { action: "Payloads", where: "plan.inspector.camera", via: "camera-picker", label: "Describe payload", inventory: "plan.payload-picker", modes: ["desk"] },
  { action: "Add Payload", where: "plan.inspector.camera", via: "camera-picker", label: "Add payload", inventory: "plan.payload-picker", modes: ["desk"] },
  { action: "Load Terrain", where: "plan.inspector.safety", label: "Load terrain", inventory: "plan.terrain-source-cache-coverage", modes: ["desk"] },
  { action: "Cache Terrain", where: "plan.inspector.safety", label: "Cache", inventory: "plan.terrain-source-cache-coverage", modes: ["desk"] },
  { action: "Terrain Details", where: "plan.inspector.safety", label: "Details", inventory: "plan.terrain-source-cache-coverage", modes: ["desk"] },
  { action: "Check Hazards", where: "plan.inspector.safety", label: "Obstacle clearance", inventory: "plan.hazards", modes: ["desk"] },
  { action: "Validate", where: "plan.inspector.safety", label: "Check site", inventory: "plan.validate-site", modes: ["desk"] },
  { action: "Export Mission Files", where: "plan.inspector.export", label: "Export n formats", inventory: "plan.export-6-formats", modes: ["desk"] },
  { action: "Export", where: "plan.inspector.export", label: "All formats", inventory: "plan.export-6-formats", modes: ["desk"] },
  { action: "Send to Fly", where: "plan.inspector.export", label: "Send to Fly", inventory: "plan.send-to-fly", modes: ["desk"] },
  /* inspector: Sorties tab */
  { action: "Battery Segments", where: "plan.inspector.sorties", label: "Re-split", inventory: "plan.battery-segments", modes: ["desk"] },
  { action: "Mission Estimates", where: "plan.inspector.sorties", label: "Estimate again", inventory: "plan.estimates", modes: ["desk"] },
  /* inspector: Site tab */
  { action: "Add Marker", where: "plan.inspector.site", label: "Add marker", inventory: "plan.site-markers", modes: ["desk"] },
  { action: "Remove Marker", where: "plan.inspector.site", label: "Remove", inventory: "plan.site-markers", modes: ["desk"] },
  { action: "Place Home", where: "plan.inspector.site", label: "Place home", inventory: "plan.place-home", modes: ["desk"] },
  { action: "Draw No-fly", where: "plan.inspector.site", label: "Draw no-fly zone", inventory: "plan.no-fly-zone", modes: ["desk"] },
  { action: "Clear No-fly", where: "plan.inspector.site", label: "Clear", inventory: "plan.clear-area-clear-no-fly", modes: ["desk"] },
  { action: "Mark Corner", where: "plan.inspector.site", label: "Mark corner", inventory: "plan.fly-to-draw-boundary", modes: ["desk"] },

  /* Field */
  { action: "Send to Fly", where: "plan.field.panel", label: "Send to aircraft", inventory: "plan.send-to-fly", modes: ["field"] },
  { action: "Export Mission Files", where: "plan.field.panel", label: "Export", inventory: "plan.export-6-formats", modes: ["field"] },
  { action: "Edit Vertices", where: "plan.field.map", label: "Edit boundary", inventory: "plan.edit-vertices-delete", modes: ["field"] },
];

/* Rows realised by an input or a control rather than a verb: what the operator sets on
   the form (the next Generate sends it), and the two native menu items that open this
   screen or point at it. `control` is a data-control (a trailing * matches a prefix);
   `field` a data-field input. */
export const CONTROLS = [
  { control: "template:*", where: "plan.left.templates", label: "Mission templates", inventory: "plan.template-list", modes: ["desk"] },
  { control: "template:multi_facade", where: "plan.left.templates", label: "Multi-facade", inventory: "plan.multi-facade-irregular-facade", modes: ["desk"] },
  { control: "template:pylon_inspection", where: "plan.left.templates", label: "Utility pylon", inventory: "plan.pylon-thermal-multispectral-linked", modes: ["desk"] },
  { field: "altitude_m", where: "plan.inspector.parameters", label: "Altitude", inventory: "plan.parameters-altitude-gsd-overlaps-speed-gimbal-li", modes: ["desk"] },
  { field: "front_overlap_pct", where: "plan.inspector.parameters", label: "Front overlap", inventory: "plan.parameters-altitude-gsd-overlaps-speed-gimbal-li", modes: ["desk"] },
  { field: "gsd_cm", where: "plan.inspector.parameters", label: "GSD", inventory: "plan.two-way-gsd-altitude", modes: ["desk"] },
  { control: "advanced", where: "plan.inspector.parameters", label: "Advanced", inventory: "plan.advanced-dwell-min-max-altitude-rth-altitude-win", modes: ["desk"] },
  { field: "rth_altitude_m", where: "plan.inspector.parameters", label: "RTH altitude", inventory: "plan.advanced-dwell-min-max-altitude-rth-altitude-win", modes: ["desk"] },
  { field: "turbine_hub_height_m", where: "plan.inspector.inputs", label: "Hub height (wind turbine)", inventory: "plan.template-specific-inputs-thermal-camera-and-gsd-", modes: ["desk"], template: "wind_turbine" },
  { control: "camera-picker", where: "plan.inspector.camera", label: "Camera", inventory: "plan.camera-picker", modes: ["desk"] },
  { control: "template:*", where: "plan.field.panel", label: "Mission type", inventory: "plan.template-list", modes: ["field"] },
  { control: "altitude:up", where: "plan.field.panel", label: "Altitude +", inventory: "plan.parameters-altitude-gsd-overlaps-speed-gimbal-li", modes: ["field"] },
  { control: "profile-scrub", where: "plan.main.strip", label: "Altitude profile", inventory: "plan.estimates", modes: ["desk"] },
  { control: "profile-scrub", where: "plan.main.strip", label: "Scrub the route (waypoint card)", inventory: "plan.waypoint-inspector", modes: ["desk"] },
  { control: "footprints", where: "plan.main.bar", label: "Footprints", inventory: "plan.photo-footprints", modes: ["desk"] },
  { menu: "view.panel{panel=mission}", where: "plan", label: "View › Mission panel opens Plan", inventory: "plan.view-mission-panel" },
  { menu: "mission.settings", where: "plan", label: "Mission › Settings points at PARAMETERS", inventory: "plan.mission-settings-menu" },
];
