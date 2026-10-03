# Standards-structured, signed reports

OpenDroneKit can write inspection reports whose sections follow the required contents of
four public standards or specifications. Each report is signed with a per-user Ed25519 key
and carries a SHA-256 manifest of everything it was built from.

Code: `core/standards_reports.py` (templates, checks, HTML/PDF), `core/standards_collectors.py`
(facts read from real files), `core/report_signing.py` (manifest, signing, verification),
`app/standards_builder.py` + `app/api.py` (desktop Api), `tools/standards_examples.py`
(example reports from the test data).

## What these reports are, and what they are not

- A report is **structured following** a standard. It is not a certificate of compliance or
  conformity, it is not issued by an accredited body, and OpenDroneKit makes no claim that
  the inspection meets the standard. Every report prints that statement on page one.
- Each required item shows a value **only** when that value was supplied, together with its
  source: a file named in the signed manifest (with the tag, column or computation), or a
  person (`entered by <name>`).
- Any required item that was not supplied is printed as **MISSING**. Nothing is defaulted,
  estimated or inferred to fill it. A check whose input is MISSING is itself MISSING, never
  PASS. The first page states the count, e.g. "9 of 30 required items provided; 21 MISSING".
- Items that need a qualified person's judgement are never assigned by software. That covers
  IEC CoA classes, SNBI condition ratings and condition states, roof defect calls and tower
  checklist assessments. Model detections are not findings until an inspector accepts them.

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

Checks run wherever values exist (MISSING otherwise):
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
check is MISSING.

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
guy (H) record. This check is MISSING when the structure type is unknown and N/A for a
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
  provided or MISSING).
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
| `standards_templates()` | templates, sections, items, tables, cited sources |
| `standards_facts(template)` | the inspector-entered values for the template in this project |
| `save_standards_facts(template, facts, entered_by)` | store entered items and table rows. Only known item ids and columns are accepted, and each one is sourced `entered by <who>` |
| `standards_report_readiness(template)` | which required items are provided and which will print MISSING, plus check states |
| `generate_standards_report(template, title)` | writes `<project>/reports/standards/<run>/report.pdf` (+ html/json/manifest/signature). Dataset images and recorded model digests go into the manifest |
| `list_standards_reports()` | the reports already generated |
| `verify_report(path, inputs_root, check_inputs)` | the verification result. The call succeeds whenever the check ran, and the result carries whether the report verified |
| `report_signing_key()` | this installation's public key and fingerprint |
| `trust_report_key(public_key_path, name)` | trust another signer's PEM public key |

UI hooks the ui/v1 rebuild needs (not implemented here, because `app/web/**` belongs to that branch):
1. **Reports > Standards** picker fed by `standards_templates()`, showing each template's
   "structured following" line and its sources.
2. **Item form** per template. Load it with `standards_facts`, save it with
   `save_standards_facts` (ask for the inspector's name and pass it as `entered_by`). Add
   row editors for the table sections (anomalies, elements, findings, defects, deficiencies).
3. **Readiness panel** from `standards_report_readiness`: a MISSING list and check states,
   shown before Generate. Generate must stay possible while items are MISSING, because the
   report prints them as MISSING.
4. **Generate** button calling `generate_standards_report`, then a list from
   `list_standards_reports` with open-PDF links.
5. **Verify** action (file picker for a PDF or folder, optional inputs folder) calling
   `verify_report`, which shows signer, trust state, per-input status, problems and warnings.
6. **Settings > Report signing** showing `report_signing_key()` (fingerprint, export public
   key) and `trust_report_key` to import a colleague's key.

## Example reports (real data)

Generated by `tools/standards_examples.py --testdata <odk-testdata> --out <dir>` with
`ODK_DATA_DIR` set so the example key stays out of the user's data root. They are in
`odk-testdata/_outputs/standards_reports/reports/`, and every one verifies
(`verify_log.txt`).

| Template | Data | Provided / required | Checks |
|---|---|---|---|
| IEC TS 62446-3 | THED-PV (FLIR Duo Pro R radiometric TIFFs, site logger, CC BY 4.0) | 9 / 30 | irradiance 921 W/m² PASS, wind 3.2 m/s (Bft 2) PASS, IR 640×512 PASS; cloud, NETD, geometric resolution MISSING; position identification FAIL (only the in-frame position is known, not the plant module identity) |
| Bridge (SNBI) | Boga Cayi bridge, DJI H20T, 81 wide frames + thermal orthomosaic | 1 / 23 (B.IE.12 equipment from EXIF) | elements MISSING |
| Roof (SOP 3.1) | FH Hagenberg building, DJI M30T, 60 frames | 2 / 13 (date, method from EXIF and gimbal pitch) | none defined |
| Telecom tower | no open tower dataset is available | 0 / 19 | guys MISSING |

The Aukerman dataset (nadir RGB of a park) has no PV array, bridge, roof subject or tower,
so it does not supply any of these templates. It was not forced into one.

To fill the MISSING items, the owner or inspector has to supply them. These are the
agency/plant identifiers and locations, personnel and qualifications, emissivity and Trefl
(when the camera did not record them), cloud cover, camera NETD (from the datasheet),
geometric resolution (from flight altitude and lens), CoA classes, SNBI condition ratings
and element condition-state quantities, roof defect calls, and an actual telecom tower
survey with its Annex J.1 assessment.
