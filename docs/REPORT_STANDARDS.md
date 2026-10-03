# Standards-structured, signed reports

OpenDroneKit can write inspection reports whose sections follow the required contents of
four public standards or specifications. Each report is signed with a per-user Ed25519 key
and carries a SHA-256 manifest of everything it was built from.

Code: `core/standards_reports.py` (templates, who supplies each item, checks, HTML),
`core/standards_pdf.py` (the PDF layout), `core/standards_figures.py` (figures drawn from
the input files), `core/standards_collectors.py` (facts read or computed from real files),
`core/inspection_records.py` (operator profiles, procedure templates, site-conditions log,
inspection details), `core/pv_modules.py` (CEC module list lookup), `core/surface_planes.py`
(tilt and azimuth from a DSM), `core/report_signing.py` (manifest, signing, verification),
`app/standards_builder.py` + `app/api.py` (desktop Api), `tools/standards_examples.py`
(example reports from the test data).

## What these reports are, and what they are not

- A report is **structured following** a standard. It is not a certificate of compliance or
  conformity, it is not issued by an accredited body, and OpenDroneKit makes no claim that
  the inspection meets the standard. Every report prints that statement on its cover.
- Each required item shows a value **only** when that value was supplied, together with its
  source: a file named in the signed manifest (with the tag, column or formula), a saved
  operator profile or adopted procedure template (with who saved it and when), a public
  database entry (the CEC module list), the site-conditions log, or a person
  (`entered by <name> <date>`).
- An item that was not supplied is **open**. Open items are not scattered through the body;
  they are collected once on a "Before this report can be issued" page right after the
  cover, grouped by who can supply them, and the cover says `DRAFT — N items to complete`.
  Nothing is defaulted, estimated or inferred to fill an item. A check whose input is open
  is itself open, never passed. (`content.json` keeps the machine status `MISSING`.)
- A **suggestion** (a default procedure nobody has adopted, a CoA class proposed from ΔT, a
  roof facet segmented from a DSM, a detector box, a drafted results summary) is printed
  as "suggested" next to the open item it would fill and counts for nothing until a named
  person confirms it.
- Items that need a qualified person's judgement are never assigned by software. That covers
  IEC CoA classes (suggested only), SNBI condition ratings and condition states, roof defect
  calls and tower checklist assessments. Model detections are proposals, not findings,
  until an inspector accepts them.

## The report layout

A4 portrait, Helvetica (no open font ships with the repository), one accent colour for
structure, an amber outline for the draft status, and severity colours on findings only.
Every body page has a running header (report title, draft state) and a footer (report id,
page *n* of *N*).

1. **Cover**: report type, site, inspection date, client, report id, items supplied, a hero
   image (the visual frame paired with the first thermal frame, the frame nearest the
   centre of the surveyed area, or the steepest roof frame) and the status pill.
2. **Before this report can be issued** (drafts only): every open item once, with a tick box,
   its clause reference, how to complete it in the app, and any suggestion.
3. **Executive summary**: key numbers, findings counted by severity (suggested and proposed
   ones labelled as such), and the standard's condition/equipment checks as Met / Not met /
   Open.
4. **Site and inspected area**: the orthomosaic with camera positions, or camera positions
   over the DSM, or the outlined thermal frames.
5. **Findings**: one card per finding with the image it was read from (thermal frames with
   their temperature scale, module outlines and ΔT; detector boxes drawn on the frame),
   its numbers, its class or "model proposal" status and its source.
6. **Photo evidence**, then **Method and conditions** (procedure, equipment, conditions,
   derived measurements with their formulas, supporting notes).
7. **Appendix A**: the clause-by-clause item table, compact and grey, every value with its
   source. **Appendix B**: the sources the report structure follows.
8. **Signature and input manifest**: report id, signing key, verify command, approval lines
   for the people named in the report, and the SHA-256 of every input file and model.

## Who supplies each item

Every required item has one supplier (`core.standards_reports.ITEM_WHO`); the checklist and
the inspection-details form are driven by it.

| Supplier | Meaning |
|---|---|
| Derived from data | read or computed from the project's files; the source names the file and tag, column or formula |
| Saved operator profile | a person saved once in Settings and picked per inspection |
| Adopted procedure template | procedure text and inspection type, adopted or edited by a named person |
| Module database lookup | the operator picks the module model; specs come from the CEC module list |
| Site conditions log | recorded on site in Field mode, or imported from a meteo CSV or flight telemetry |
| Entered per inspection | typed once for this inspection |
| Qualified person's assessment | a judgement; software may only suggest |

Per template (required items; tables counted separately):

| Template | Derived from data | Entered once, reused (profile / procedure / lookup) | Per-inspection input (operator / site log) | Needs a qualified person |
|---|---|---|---|---|
| IEC TS 62446-3 (30) | 17: date/time, location, inspected-component listing, mounting tilt/azimuth (DSM), camera make/model, serial/lens, IR resolution, spectral response, NETD (camera registry), geometric resolution (distance × pitch / focal), emissivity and Trefl (radiometric metadata), air temperature, wind speed and direction, irradiance, soiling | 7: PV expert, thermographer, attending persons; procedure, inspection type; component type, module efficiency | 3: BOS location; cloud oktas and type | 3: results summary (drafted as a suggestion), next inspection, conclusions; plus CoA, Annex C type and action per anomaly (CoA suggested) |
| Bridge SNBI (23) | 6: latitude, longitude, begin and completion dates, due date (completion + interval), equipment | 2: inspector, inspection type | 6: bridge number, name, state and county codes, interval, note | 9 condition ratings, element condition states, deficiencies (detector boxes are proposals) |
| Roof InterNACHI 3.1 (13) | 3: site (GPS / place name), date/time, method (from gimbal pitch) | 1: inspector | 3: covering type, limitations; weather (site log) | 6 roof assessments plus the defects table; roof pitch from the DSM is a suggested measurement |
| Telecom tower TIA-222-H (19) | 2: location, date | 1: assessor | 5: structure id, type, height, interval; weather (site log) | 11 Annex J.1 headings plus the deficiencies table |

## Capabilities that fill items

- **Location**: the project georeference, else the centroid of the EXIF GPS positions; a
  place name from the offline gazetteer (`<project>/places.json`, nearest saved place within
  2 km) when one is saved, otherwise coordinates. For a bridge, the orthomosaic footprint
  centre, labelled as not the agency's B.L.05/06 record.
- **Emissivity and reflected temperature**: from the measurement parameters the file itself
  stores (DJI R-JPEG via the DJI Thermal SDK; FLIR FFF camera-info record). FLIR TLinear
  TIFFs (Duo Pro R) do not record them, so they stay open with that reason.
- **NETD**: from the camera registry (`mission/cameras.py`, `netd_mk` with the datasheet
  page it was read from). Thermal profiles with a published NETD: DJI H20T, M30T, Mavic 3T,
  FLIR Duo Pro R 640 (13 mm), FLIR Vue Pro R 640.
- **Geometric resolution**: `distance × pixel pitch / focal length`, with the distance from
  the laser range finder or relative-altitude XMP, else the camera distance entered for the
  inspection; the formula and every input are in the source column.
- **Mounting tilt / azimuth and roof pitch** (`core/surface_planes.py`): a least-squares
  plane per operator outline (or RANSAC facets above the DTM), tilt and true-north azimuth
  with fit RMSE; a fit with RMSE above the limit is rejected, not reported.
- **Module database** (`core/pv_modules.py`): the CEC PV module list from NREL SAM's library
  (BSD-3-Clause, shipped as `core/data/pv_modules/cec_modules.csv.gz` with its provenance and
  licence). Rated STC power, technology, cells, dimensions; efficiency = STC / (area × 1000
  W/m²), cited as a computation.
- **Site conditions log**: structured on-site records (oktas 0-8, cloud type, wind, air
  temperature, irradiance, notes, observer, time), imported from a meteo CSV row (refused
  beyond 5 minutes) or flight-log telemetry that reports wind. The record nearest the
  inspection time (within 90 minutes) fills cloud, wind and weather items.
- **CoA suggestion**: from ΔT, the anomaly pattern and the thermal classifier label, using
  the informative Annex C examples as publicly summarised (single hot cell 10-40 K → 2,
  > 40 K → 3; open module or sub-string → 2; broken glass → 3; no pattern and ΔTmax
  < 10 K → 1). Labelled "suggested" with its basis; the class counts only after
  `confirm_anomaly_class` records the confirmer and time.
- **Soiling evidence**: a soiling-sensor column of the logger (e.g. DustIQ soiling ratio)
  and the visual frames paired with the thermal frames as photos. Without a sensor reading,
  the photos are attached as a suggestion for the thermographer to confirm.

## Templates and their sources

Clause references below are the ones printed in the `Ref` column of every report.

### `iec62446_3`: PV thermographic inspection (IEC TS 62446-3:2017)

| Report section | Clause | Items |
|---|---|---|
| Personnel | 8 a | PV expert, thermographer (and qualification), attending persons |
| Inspection | 8 c, 8 d, 8 h, 7 | date and time, location, procedure, simplified or detailed |
| Scope as contracted | 8 e 1-4 | component types, module efficiency / BOS rating, inspected components, mounting, BOS location |
| Camera system | 8 b, 8 o, 8 p, 4.2 Table 1 | make and model, serial and lens, IR resolution, spectral response, NETD, geometric resolution, emissivity, Trefl |
| Environmental conditions | 8 f 1-4, 8 g, 5.3 Table 3 | air temperature, wind speed and direction, cloud cover and type, in-plane irradiance, soiling |
| Thermal abnormalities (table) | 8 i, 8 m-s, 7.3.2 Table 4, Annex C | object, image file and time, the five position identifications (8 i 1 i-v), temperature, reference, ΔT, abnormality type, CoA, action (8 k) |
| Results | 8 j, 8 l, 8 t | summary, next inspection, conclusions |

Checks run wherever values exist (open otherwise):
in-plane irradiance ≥ 600 W/m² (5.3 Table 3); wind ≤ 4 Bft / 28 km/h (m/s converted with
the WMO Beaufort limits); cloud ≤ 2 okta; IR resolution ≥ 320 × 240 (4.2); NETD ≤ 0.1 K at
30 °C (Table 1 d); ≤ 3 cm of module edge per pixel (Table 1 e); at least two of the five
position identifications per abnormality (8 i).

Sources:
- IEC TS 62446-3:2017, IEC webstore entry: https://webstore.iec.ch/en/publication/28628
- IEC TS 62446-3:2017 public preview (scope, terms, table of contents): https://cdn.standards.iteh.ai/samples/22683/4de66fa53f2c4b8591ebe1672331acb4/IEC-TS-62446-3-2017.pdf
- Quark Electric's summary of the conditions (> 600 W/m², < 28 km/h wind, < 2 okta, > 320 × 240): https://www.quark-electric.com/blog/iec-ts-62446-3-requirements-for-testing-documentation-and-maintenance-of-photovoltaic-systems-using-infrared-thermography/
- Thermal Variations, the report contents (conditions, equipment, per-finding location, ΔT, classification, action): https://thermalvariations.com/learn/iec-62446-3-thermal-inspection
- A UAV PV-diagnosis review that quotes the Table 1 NETD ≤ 0.1 K and ≤ 3 cm/pixel minimums: https://pmc.ncbi.nlm.nih.gov/articles/PMC10789596/

The full clause 8 list (a-t) comes from the standard's text. The secondary sources above
agree with the numerical limits. Anyone relying on the report should check the items
against their own licensed copy of the standard.

### `bridge_snbi`: US highway bridge, element level (FHWA NBIS / SNBI 2022)

| Report section | SNBI items |
|---|---|
| Identification and location | B.ID.01 Bridge Number, B.ID.02 Name, B.L.01 State, B.L.02 County, B.L.05/06 Latitude/Longitude |
| Inspection event | B.IE.01 Type, B.IE.02/03 Begin/Completion Date, B.IE.04 Nationally Certified Bridge Inspector, B.IE.05 Interval, B.IE.06 Due Date, B.IE.11 Note, B.IE.12 Inspection Equipment |
| Component condition ratings | B.C.01 Deck, B.C.02 Superstructure, B.C.03 Substructure, B.C.04 Culvert, B.C.05 Railings, B.C.07 Bearings, B.C.08 Joints, B.C.09 Channel, B.C.11 Scour |
| Element-level condition (table) | B.E.01 Element Number, B.E.02 Parent, B.E.03 Total Quantity, B.CS.01-04 quantities in CS1 Good / CS2 Fair / CS3 Poor / CS4 Severe |
| Deficiencies and critical findings (table) | 23 CFR 650.313(a) documentation, 650.313(q) critical findings |

Checks: each condition rating that is present must be an SNBI code (0-9 or N), and for each
element CS1 + CS2 + CS3 + CS4 must equal B.E.03. If there is no element record, the element
check is open.

Sources:
- FHWA, Specifications for the National Bridge Inventory, March 2022: https://www.fhwa.dot.gov/bridge/snbi/snbi_march_2022_publication.pdf
- 23 CFR 650.313 Inspection procedures: https://www.law.cornell.edu/cfr/text/23/650.313
- 23 CFR 650.315 Inventory (element data per the AASHTO Manual for Bridge Element Inspection and the SNBI): https://www.law.cornell.edu/cfr/text/23/650.315
- FHWA SNBI data crosswalk: https://www.fhwa.dot.gov/bridge/snbi/datacrosswalk.cfm
- FHWA NBIS final rule overview (B.IE.12 equipment recorded on every inspection): https://www.fhwa.dot.gov/bridge/pubs/P1-NBIS-SNBI-Final-Rule-Overview_toHPA_V2_508v2.pdf

### `roof`: Roof inspection (InterNACHI SOP section 3.1)

Section 3.1 says what the inspector shall inspect: I.A roof covering, I.B gutters, I.C
downspouts, I.D vents, flashing, skylights, chimney and penetrations, I.E general structure.
It says what the inspector shall describe (II.A covering type) and report (III.A active
leaks), and SOP 2 adds the limitations and areas not inspected. These, plus the
site/inspector/date/weather/method header, are the report items, followed by a
material-defects table. The template has no automatic checks, because every judgement in
the SOP belongs to the inspector.

Source: InterNACHI, International Standards of Practice for Performing a General Home
Inspection, 3.1 Roof: https://www.nachi.org/sop.htm

### `telecom_tower`: Telecom tower condition assessment (ANSI/TIA-222-H s14, Annex J.1)

There is a structure and assessment header (identifier, location, guyed / self-supporting /
pole, height, assessor, date, weather, and the s14.4 interval: 3 years guyed, 5 years
self-supporting, and after extreme events). It is followed by the Annex J.1 checklist
headings A Structure, B Finish, C Lighting, D Grounding, E Mounts/antennas/lines, F Other
appurtenances, G Base insulator, H Guys, I Foundations, J Alignment (plumb/twist) and K
Previous modifications, and then a deficiencies table. Check: a guyed structure needs a
guy (H) record. This check is open when the structure type is unknown and N/A for a
self-supporting structure.

Sources:
- Telecommunications Industry Foundation, Planning Advisory Notice "ANSI/TIA-222 Maintenance and Condition Assessment" (lists J.1 A-K and the s14.4 intervals): https://tifonline.org/wp-content/uploads/2022/08/PAN-ANSI-TIA-222.pdf
- Wireless Estimator, "New inspection requirements added in Revision H of ANSI/TIA-222": https://wirelessestimator.com/articles/2017/new-inspection-requirements-added-in-revision-h-of-ansitia-222/

## Signed, traceable reports

- **Manifest** (`odk-report-manifest/1`): the SHA-256, size and path of every input file
  (images, rasters, logger data, region outlines) and the digest of every model that
  produced something in the report. A model digest is either a hash of the weights file or
  the digest a pipeline run recorded, and each is labelled as one or the other. A missing
  input file, or a model with no digest, is refused instead of being listed without a hash.
  The manifest is printed in the PDF's appendix and embedded in the PDF as the attachment
  `odk-manifest.json`, next to `odk-report-content.json` (every item and whether it was
  provided or open).
- **Signature** (`odk-report-signature/1`): Ed25519 over a canonical-JSON statement naming the
  SHA-256 of the PDF bytes (attachments included) and of the manifest. It is appended as an
  incremental PDF update (`/ODKSignature`), so the signed bytes are an untouched prefix. A
  detached `signature.json` also signs the final `report.pdf` and `report.html`.
- **Key**: one key pair per user, created on first use in `<data root>/keys/`
  (`core.paths.data_root`; `ODK_DATA_DIR` moves it). The private key is never written into a
  report. The public key and its fingerprint are. If the `cryptography` or `pypdf` package
  is missing, the report is refused with that reason before anything is written; if signing
  fails part-way, the half-written folder is removed. A report is never left unsigned.
- **Trust**: a valid signature from a key this installation does not know is reported as
  *untrusted*, not verified. The `trust_report_key` call adds another signer's public key.

### Verify

```
python -m core.report_signing verify <report.pdf | report folder> [--inputs-root DIR] [--no-inputs] [--json]
python -m core.report_signing key        # show (create on first use) this installation's key
```

Verification detects these problems: a changed byte in the signed PDF, anything appended after
signing, a signature update that also redefines signed objects (a page or content stream
smuggled into the unsigned update, which a viewer would display), an edited HTML copy, a
manifest that is not the signed one, and an input whose hash changed. Inputs that are not present on this machine produce a warning, and
`--inputs-root` locates moved inputs by name. It also catches a strip-and-re-sign with
another key. What it does **not** prove is that the inputs were captured as described or
that the inspection meets the standard. Exit code 0 means verified and 1 means not verified.

## Desktop Api (backend; for the ui/v1 rebuild)

All methods act on the open project and return `{"ok": ..., ...}` dicts like the rest of `app/api.py`.

| Method | Purpose |
|---|---|
| `standards_templates()` | templates, sections, items (each with its supplier `who`), tables, cited sources |
| `get_inspection_details(template)` | the template-driven form (every item: who supplies it, current value, source, suggestion, how to complete), stored details, operator profiles, the procedure in force, site-conditions records |
| `set_inspection_details(template, details, entered_by)` | save per-inspection details: `details` (`people` item → profile ids, `client`, `contract_scope`, `module_model`, `camera_distance_m`, `workflow_id`, `site_name`, `array_outlines`), `fields` (item → value) and `tables`. Unknown keys, unknown profiles and unlisted modules are refused; an empty value clears; every value records who entered it and when |
| `confirm_anomaly_class(template, key, coa, confirmed_by, abnormality_type, action)` | the thermographer's CoA for one anomaly; replaces the suggestion and names the confirmer |
| `list_operator_profiles()` / `save_operator_profile(profile, saved_by)` / `delete_operator_profile(id)` | people: name, role, qualification, certificate number, organisation |
| `procedure_templates()` / `save_procedure_template(template, text, inspection_type, edited_by, workflow_id)` | built-in default procedures (suggestions) and adopted ones |
| `list_site_conditions()` / `add_site_conditions(record, recorded_by)` / `import_site_conditions(source, observed_at, columns, imported_by, flight_id, kind)` | the site-conditions log (Field mode entry, meteo CSV, flight-log telemetry) |
| `search_pv_modules(query, limit)` | search the CEC module list |
| `measure_surface_planes(dsm_path, outlines, dtm_path)` | tilt/azimuth/RMSE per outline, or RANSAC roof facets |
| `standards_facts(template)` / `save_standards_facts(template, facts, entered_by)` | the older per-template entry store, kept for compatibility |
| `standards_report_readiness(template)` | supplied counts by supplier, the open checklist, check states, draft status |
| `generate_standards_report(template, title)` | writes `<project>/reports/standards/<run>/report.pdf` (+ html/json/manifest/signature/figures). Dataset images, a project DSM/DTM and recorded model digests go into the manifest |
| `list_standards_reports()` | the reports already generated |
| `verify_report(path, inputs_root, check_inputs)` | the verification result. The call succeeds whenever the check ran, and the result carries whether the report verified |
| `report_signing_key()` | this installation's public key and fingerprint |
| `trust_report_key(public_key_path, name)` | trust another signer's PEM public key |

UI hooks the ui/v1 rebuild needs (not implemented here, because `app/web/**` belongs to that branch):
1. **Reports > Standards** picker fed by `standards_templates()`, showing each template's
   "structured following" line and its sources.
2. **Report > Inspection details** screen built from `get_inspection_details(template).form.fields`:
   one control per item by its `who` — a profile picker (`profile`), a link to the procedure
   editor (`procedure`), a module search box backed by `search_pv_modules` (`lookup`), the
   site-conditions log (`site`), a text box (`operator`, `judgement`), and a read-only value
   with its source plus an "override" action (`data`). Show each `suggestion` with an
   "Accept" button that saves it under the operator's name. Save with
   `set_inspection_details` (ask for the name and pass it as `entered_by`). Row editors for
   the table sections (anomalies, elements, findings, defects, deficiencies).
3. **Review** panel for IEC anomalies: each finding card with its suggested CoA and basis,
   and Confirm / Change buttons calling `confirm_anomaly_class`.
4. **Readiness panel** from `standards_report_readiness`: the open checklist grouped by
   supplier and the check states, shown before Generate. Generate stays possible while items
   are open; the report is then a DRAFT with its checklist page.
5. **Generate** button calling `generate_standards_report`, then a list from
   `list_standards_reports` with open-PDF links.
6. **Verify** action (file picker for a PDF or folder, optional inputs folder) calling
   `verify_report`, which shows signer, trust state, per-input status, problems and warnings.
7. **Settings > People** (`list/save/delete_operator_profile`), **Settings > Procedures**
   (`procedure_templates`, `save_procedure_template`) and **Settings > Report signing**
   (`report_signing_key`, `trust_report_key`).
8. **Field mode > Site conditions**: a quick form (oktas 0-8 picker, cloud type, wind,
   notes; time and flight pre-filled) calling `add_site_conditions`, and an import action
   calling `import_site_conditions` for a logger CSV or a flight log.

## Example reports (real data)

Generated by `tools/standards_examples.py --testdata <odk-testdata> --out <dir>` with
`ODK_DATA_DIR` set so the example key stays out of the user's data root. They are in
`odk-testdata/_outputs/standards_reports/reports/`, and every one verifies
(`verify_log.txt`). No person, client, adopted procedure or class is invented for them, so
each is a DRAFT whose checklist lists what a person must still supply.

| Template | Data | Supplied / required: before → now | Open checklist items | Checks |
|---|---|---|---|---|
| IEC TS 62446-3 | THED-PV (FLIR Duo Pro R radiometric TIFFs, site logger, CC BY 4.0) | 9 → 12 / 30 (all derived: + NETD from the registry, inspected-component listing from the outlines, soiling ratio from the DustIQ column with the paired photos) | 21 | irradiance 921 W/m² Met, wind 3.2 m/s (Bft 2) Met, IR 640×512 Met, NETD < 50 mK Met; cloud and geometric resolution open (mast captures record no distance); position identification Not met (only the in-frame position is known) |
| Bridge (SNBI) | Boga Cayi bridge, DJI H20T, 81 wide frames + thermal orthomosaic + structural detector boxes | 1 → 5 / 23 (equipment, latitude and longitude from the orthomosaic, begin and completion dates) | 20 | elements open; 66 detector proposals shown as proposals; GSD 1.70 cm/px on the structure from the laser range finder |
| Roof (SOP 3.1) | FH Hagenberg building, DJI M30T, 60 frames + sparse DSM/DTM | 2 → 3 / 13 (site from GPS, date, method) | 11 | none defined; one DSM facet suggested; the frames are all facade obliques (gimbal −8°), which the report states |
| Telecom tower | no open tower dataset is available | 0 → 0 / 19 | 20 | guys open; a clean draft that shows the structure and what is needed |

The Aukerman dataset (nadir RGB of a park) has no PV array, bridge, roof subject or tower,
so it does not supply any of these templates. It was not forced into one.

## Next capabilities

Follow-on work that would fill more items; each is a design note, not implemented here.

1. **Datasheet PDF import**: parse a module or camera datasheet PDF (pymupdf text + table
   extraction; match labelled rows such as "NETD", "Pmax", "Module efficiency" with unit
   checks) into a *suggested* registry or module entry the operator confirms; the PDF is
   hashed into the manifest as the source.
2. **Module and string detection from the orthomosaic** for the 8 e 3 component listing and
   the 8 i position identifications: run the PV panel detector on a thermal/RGB
   orthomosaic, cluster detections into rows and strings by geometry, give each module a
   stable plan ID and map coordinates, and offer the listing as a suggestion; confirmed
   IDs then satisfy two of the five 8 i identifications (X-Y position and plan marking).
3. **Optional online weather fetch**: with explicit opt-in (as geocoding does), fetch
   historical cloud cover, wind and temperature for the site and time from a named open
   service (e.g. Open-Meteo archive) into the site-conditions log as an import whose source
   names the service, grid cell and time; never used without the operator choosing it, and
   cloud *type* stays an on-site observation.
4. **BOS locations from map annotations or a plant KMZ**: import inverter, combiner and
   transformer placemarks from a KMZ/GeoJSON or from annotations drawn on the project map,
   and fill 8 e 4 ii with named positions and distances to the inspected arrays.
5. **Thermal pattern classifier per module**: classify each module's thermal pattern
   (single hot cell, sub-string, whole module, PID-like) from the decoded field so the CoA
   suggestion has a pattern input, not only ΔT.
6. **Offline reverse geocoding beyond saved places**: ship or download an open
   administrative-boundary set (e.g. Natural Earth / GeoNames cities, with their licences)
   so the site and the SNBI state/county codes can be suggested from coordinates.
