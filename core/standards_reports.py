"""Inspection reports structured following published inspection standards.

Four templates, each laid out after the required contents of a public standard or
specification (sources and clause references in docs/REPORT_STANDARDS.md):

* ``iec62446_3``  -- PV thermographic inspection, structured following IEC TS 62446-3:2017
  clause 8 (inspection report), with the inspection conditions of clause 5.3 / Table 3
  and the camera minimums of clause 4.2 / Table 1 checked where the values are known.
* ``bridge_snbi`` -- US highway bridge inspection, structured following the FHWA
  Specifications for the National Bridge Inventory (SNBI, 2022) items required by the
  NBIS (23 CFR 650.313 / 650.315), with element-level condition states CS1-CS4
  (AASHTO Manual for Bridge Element Inspection).
* ``roof``        -- roof inspection, structured following InterNACHI's Home Inspection
  Standards of Practice section 3.1 (Roof).
* ``telecom_tower`` -- telecommunication tower inspection, structured following the
  ANSI/TIA-222-H Annex J.1 maintenance and condition assessment checklist (A-K).

The rule every template obeys
-----------------------------
A value appears in a report only if it was *supplied*, with where it came from: read
from a file (and the file is in the signed manifest) or entered by a named person.
Every required item without a value is printed as **MISSING**. Nothing is defaulted,
estimated or inferred to fill a field, and a check whose input is MISSING is MISSING,
not passed. The report never states that it complies with, or is certified to, the
standard: it says it is *structured following* it and how many required items are
missing. A conformity assessment is a qualified person's job.
"""

from __future__ import annotations

import html
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

MISSING = "MISSING"

NOT_CERTIFIED = (
    "This report is structured following the cited standard so that its required "
    "contents can be checked item by item. It is not a certificate of compliance or "
    "conformity, it was not issued by an accredited body, and OpenDroneKit makes no "
    "claim that the inspection meets the standard. Items marked MISSING were not "
    "supplied and were not filled with defaults.")


# ── Template model ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Item:
    """One required report item."""

    id: str
    label: str
    ref: str           # clause / item reference in the standard
    unit: str = ""


@dataclass(frozen=True)
class Column:
    key: str
    label: str
    required: bool = True


@dataclass(frozen=True)
class Table:
    """A repeated required item (one row per anomaly, element, finding...)."""

    id: str
    title: str
    ref: str
    columns: tuple[Column, ...]
    note: str = ""


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    ref: str
    items: tuple[Item, ...] = ()
    tables: tuple[Table, ...] = ()


@dataclass(frozen=True)
class Source:
    label: str
    url: str


@dataclass(frozen=True)
class StandardTemplate:
    key: str
    title: str
    standard: str              # e.g. "IEC TS 62446-3:2017"
    scope: str
    sources: tuple[Source, ...]
    sections: tuple[Section, ...]
    checks: Callable[[dict[str, Any]], list[dict[str, Any]]] = field(
        default=lambda facts: [], compare=False)

    @property
    def structured_following(self) -> str:
        return f"Structured following {self.standard}"

    def items(self) -> list[Item]:
        return [item for section in self.sections for item in section.items]

    def tables(self) -> list[Table]:
        return [table for section in self.sections for table in section.tables]

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key, "title": self.title, "standard": self.standard,
            "structured_following": self.structured_following, "scope": self.scope,
            "sources": [{"label": s.label, "url": s.url} for s in self.sources],
            "sections": [{
                "id": s.id, "title": s.title, "ref": s.ref,
                "items": [{"id": i.id, "label": i.label, "ref": i.ref, "unit": i.unit}
                          for i in s.items],
                "tables": [{"id": t.id, "title": t.title, "ref": t.ref, "note": t.note,
                            "columns": [{"key": c.key, "label": c.label,
                                         "required": c.required} for c in t.columns]}
                           for t in s.tables],
            } for s in self.sections],
        }


# ── Fact access (the only way a value reaches a report) ───────────────────────

def fact(facts: dict[str, Any], item_id: str) -> dict[str, Any] | None:
    """``{"value", "source"}`` for an item, or None when it was not supplied.

    A bare value (not a dict) is accepted as entered by the operator. An empty string,
    None, or an empty list is not a value.
    """
    raw = (facts.get("fields") or {}).get(item_id)
    if raw is None:
        return None
    if isinstance(raw, dict) and "value" in raw:
        value, source = raw.get("value"), str(raw.get("source") or "entered by operator")
    else:
        value, source = raw, "entered by operator"
    if value is None or (isinstance(value, (str, list, dict)) and not value):
        return None
    return {"value": value, "source": source}


def number(facts: dict[str, Any], item_id: str) -> float | None:
    found = fact(facts, item_id)
    if found is None:
        return None
    try:
        return float(found["value"])
    except (TypeError, ValueError):
        return None


def _check(id: str, label: str, ref: str, status: str, detail: str) -> dict[str, Any]:
    return {"id": id, "label": label, "ref": ref, "status": status, "detail": detail}


def _missing_check(id: str, label: str, ref: str, needs: str) -> dict[str, Any]:
    return _check(id, label, ref, MISSING, f"Cannot be checked: {needs} was not supplied.")


# ── IEC TS 62446-3 ────────────────────────────────────────────────────────────

BEAUFORT_UPPER_M_S = (0.2, 1.5, 3.3, 5.4, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6)


def beaufort_from_m_s(speed: float) -> int:
    """WMO Beaufort number for a mean wind speed in m/s (IEC TS 62446-3 Annex E)."""
    for force, upper in enumerate(BEAUFORT_UPPER_M_S):
        if speed <= upper:
            return force
    return 12


def _iec_checks(facts: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    poa = number(facts, "env.irradiance_poa_w_m2")
    if poa is None:
        out.append(_missing_check("irradiance", "In-plane irradiance >= 600 W/m²",
                                  "5.3 Table 3 a", "the in-plane irradiance"))
    else:
        out.append(_check("irradiance", "In-plane irradiance >= 600 W/m²", "5.3 Table 3 a",
                          "PASS" if poa >= 600 else "FAIL",
                          f"{poa:g} W/m² recorded in the plane of the modules."))
    wind = number(facts, "env.wind_speed_m_s")
    bft = number(facts, "env.wind_bft")
    if wind is None and bft is None:
        out.append(_missing_check("wind", "Wind <= 4 Bft (28 km/h)", "5.3 Table 3 b",
                                  "the wind speed"))
    else:
        force = int(bft) if bft is not None else beaufort_from_m_s(float(wind))
        speed = f"{wind:g} m/s = " if wind is not None else ""
        out.append(_check("wind", "Wind <= 4 Bft (28 km/h)", "5.3 Table 3 b",
                          "PASS" if force <= 4 else "FAIL", f"{speed}Beaufort {force}."))
    okta = number(facts, "env.cloud_okta")
    if okta is None:
        out.append(_missing_check("cloud", "Cloud <= 2 okta cumulus", "5.3 Table 3 c",
                                  "the cloud coverage"))
    else:
        out.append(_check("cloud", "Cloud <= 2 okta cumulus", "5.3 Table 3 c",
                          "PASS" if okta <= 2 else "FAIL", f"{okta:g} okta recorded."))
    res = fact(facts, "camera.ir_resolution_px")
    if res is None:
        out.append(_missing_check("ir_resolution", "IR resolution >= 320 x 240",
                                  "4.2 (text after Table 1)", "the IR resolution"))
    else:
        dims = [int(v) for v in re.findall(r"\d+", str(res["value"]))[:2]]
        okay = len(dims) == 2 and max(dims) >= 320 and min(dims) >= 240
        out.append(_check("ir_resolution", "IR resolution >= 320 x 240",
                          "4.2 (text after Table 1)", "PASS" if okay else "FAIL",
                          f"{res['value']} ({res['source']})."))
    netd = number(facts, "camera.netd_k")
    out.append(_missing_check("netd", "NETD <= 0.1 K at 30 °C", "4.2 Table 1 d",
                              "the camera's NETD") if netd is None else
               _check("netd", "NETD <= 0.1 K at 30 °C", "4.2 Table 1 d",
                      "PASS" if netd <= 0.1 else "FAIL", f"{netd:g} K."))
    gsd = number(facts, "camera.geometric_resolution_cm_px")
    out.append(_missing_check("geometric_resolution", "<= 3 cm of module edge per pixel",
                              "4.2 Table 1 e", "the geometric resolution on the module")
               if gsd is None else
               _check("geometric_resolution", "<= 3 cm of module edge per pixel",
                      "4.2 Table 1 e", "PASS" if gsd <= 3 else "FAIL", f"{gsd:g} cm/px."))
    rows = (facts.get("tables") or {}).get("anomalies") or []
    id_keys = ("serial_number", "array_photo", "xy_position", "string_or_plan_marking",
               "site_marking")
    for row in rows:
        have = [k for k in id_keys if row.get(k) not in (None, "")]
        out.append(_check(f"identification:{row.get('object', '?')}",
                          f"{row.get('object', '?')}: at least 2 position identifications",
                          "8 i", "PASS" if len(have) >= 2 else "FAIL",
                          f"{len(have)} of the five listed identifications supplied"
                          + (f" ({', '.join(have)})." if have else ".")))
    return out


IEC62446_3 = StandardTemplate(
    key="iec62446_3",
    title="PV thermographic inspection report",
    standard="IEC TS 62446-3:2017 (clause 8, inspection report)",
    scope=("Outdoor infrared thermography of PV modules and plants in operation. Items "
           "follow clause 8 a)-t); conditions follow clause 5.3 Table 3; camera minimums "
           "follow clause 4.2 Table 1."),
    sources=(
        Source("IEC TS 62446-3:2017, Photovoltaic (PV) systems - Requirements for testing, "
               "documentation and maintenance - Part 3: Photovoltaic modules and plants - "
               "Outdoor infrared thermography (IEC webstore)",
               "https://webstore.iec.ch/en/publication/28628"),
        Source("Thermal Variations, 'IEC TS 62446-3 thermal inspections, explained'",
               "https://thermalvariations.com/learn/iec-62446-3-thermal-inspection"),
        Source("Quark Electric, 'IEC TS 62446-3: Requirements for Testing, Documentation "
               "and Maintenance of PV Systems Using Infrared Thermography'",
               "https://www.quark-electric.com/blog/iec-ts-62446-3-requirements-for-testing-"
               "documentation-and-maintenance-of-photovoltaic-systems-using-infrared-"
               "thermography/"),
    ),
    sections=(
        Section("people", "Personnel", "8 a", (
            Item("people.pv_expert", "PV expert", "8 a"),
            Item("people.thermographer", "Thermographer (and qualification level)", "8 a"),
            Item("people.attending", "Attending persons", "8 a"),
        )),
        Section("inspection", "Inspection", "8 c, 8 d, 8 h", (
            Item("inspection.datetime", "Day and time of the inspection", "8 c"),
            Item("inspection.location", "Location of the inspection", "8 d"),
            Item("inspection.procedure", "Description of the inspection procedure", "8 h"),
            Item("inspection.type", "Inspection type (simplified / detailed)", "7"),
        )),
        Section("scope", "Scope as contracted", "8 e", (
            Item("scope.component_types", "Type designation of components", "8 e 1"),
            Item("scope.module_efficiency", "PV module efficiency / BOS nominal rating",
                 "8 e 2", "%"),
            Item("scope.inspected_components", "Listing of all inspected components", "8 e 3"),
            Item("scope.mounting", "Mounting (type, orientation, inclination)", "8 e 4 i"),
            Item("scope.bos_location", "BOS component location", "8 e 4 ii"),
        )),
        Section("camera", "Camera system", "8 b, 8 o, 4.2 Table 1", (
            Item("camera.make_model", "Camera make and model", "8 b"),
            Item("camera.serial_lens", "Camera serial number and lens", "8 o"),
            Item("camera.ir_resolution_px", "IR detector resolution", "4.2"),
            Item("camera.spectral_response", "Spectral response", "4.2 Table 1 a"),
            Item("camera.netd_k", "Thermal sensitivity (NETD)", "4.2 Table 1 d", "K"),
            Item("camera.geometric_resolution_cm_px", "Geometric resolution on the module",
                 "4.2 Table 1 e", "cm/px"),
            Item("camera.emissivity", "Emissivity used", "8 p, 5.4.3"),
            Item("camera.reflected_temperature_c", "Reflected temperature (Trefl) recorded",
                 "8 p", "°C"),
        )),
        Section("environment", "Environmental conditions", "8 f, 8 g, 5.3 Table 3", (
            Item("env.air_temperature_c", "Air temperature", "8 f 1", "°C"),
            Item("env.wind_speed_m_s", "Wind speed", "8 f 2", "m/s"),
            Item("env.wind_direction_deg", "Wind direction", "8 f 2", "°"),
            Item("env.cloud_okta", "Cloud coverage", "8 f 3", "okta"),
            Item("env.cloud_type", "Cloud type", "8 f 3"),
            Item("env.irradiance_poa_w_m2", "Irradiance in plane of module", "8 f 4", "W/m²"),
            Item("env.soiling", "Soiling of components (with photo evidence)", "8 g"),
        )),
        Section("abnormalities", "Thermal abnormalities", "8 i, 8 k, 8 m-s, 7.3.2", tables=(
            Table("anomalies", "Identified thermal noticeable spots", "8 i, 8 m-s, 7.3.2 Table 4",
                  (Column("object", "Object (8 m)"),
                   Column("image_file", "Thermal image file, date, time (8 n)"),
                   Column("serial_number", "Serial number (8 i 1 i)", False),
                   Column("array_photo", "Photo showing position (8 i 1 ii)", False),
                   Column("xy_position", "X-Y position (8 i 1 iii)", False),
                   Column("string_or_plan_marking", "String / plan marking (8 i 1 iv)", False),
                   Column("site_marking", "Permanent site marking (8 i 1 v)", False),
                   Column("temperature_c", "Temperature at abnormality (8 s)"),
                   Column("reference_c", "Temperature of regular spot (8 s)"),
                   Column("delta_t_k", "ΔT (8 s)"),
                   Column("abnormality_type", "Abnormality type (Annex C)"),
                   Column("coa", "Class of abnormality CoA 1/2/3 (7.3.2)"),
                   Column("action", "Recommended action (8 k)")),
                  note="CoA is assigned by the thermographer against the patterns of Annex "
                       "C; OpenDroneKit never assigns it."),
        )),
        Section("conclusions", "Results", "8 j, 8 l, 8 t", (
            Item("results.summary", "Summary of the results", "8 l"),
            Item("results.next_inspection", "Recommended next periodic inspection", "8 j"),
            Item("results.conclusions", "Conclusions and recommendations for further action",
                 "8 t"),
        )),
    ),
    checks=_iec_checks,
)


# ── Bridge (FHWA NBIS / SNBI, element level) ──────────────────────────────────

CONDITION_RATING_CODES = {"N", *map(str, range(10))}


def _bridge_checks(facts: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in BRIDGE.items():
        if item.id.startswith("cond.") and fact(facts, item.id) is not None:
            code = str(fact(facts, item.id)["value"]).strip().upper()
            out.append(_check(f"code:{item.id}", f"{item.label} is an SNBI code (0-9 or N)",
                              item.ref, "PASS" if code in CONDITION_RATING_CODES else "FAIL",
                              f"Recorded {code!r}."))
    rows = (facts.get("tables") or {}).get("elements") or []
    if not rows:
        out.append(_missing_check("elements", "Element quantities by condition state",
                                  "B.E.01-B.E.03, B.CS.01-B.CS.04",
                                  "any element-level inspection record"))
    for row in rows:
        try:
            total = float(row["total_quantity"])
            parts = [float(row[f"cs{i}"]) for i in range(1, 5)]
        except (KeyError, TypeError, ValueError):
            out.append(_missing_check(f"cs_sum:{row.get('element_number', '?')}",
                                      f"Element {row.get('element_number', '?')}: CS1-CS4 "
                                      "sum to total quantity", "B.CS.01-04 / B.E.03",
                                      "a total quantity or a condition-state quantity"))
            continue
        out.append(_check(f"cs_sum:{row.get('element_number', '?')}",
                          f"Element {row.get('element_number', '?')}: CS1-CS4 sum to total "
                          "quantity", "B.CS.01-04 / B.E.03",
                          "PASS" if abs(sum(parts) - total) < 1e-9 else "FAIL",
                          f"CS sum {sum(parts):g} vs total {total:g}."))
    return out


BRIDGE = StandardTemplate(
    key="bridge_snbi",
    title="Bridge inspection report (element level)",
    standard=("the FHWA Specifications for the National Bridge Inventory (SNBI, 2022) under "
              "the NBIS (23 CFR 650 Subpart C), with AASHTO element condition states"),
    scope=("Routine highway bridge inspection data the NBIS requires to be documented: "
           "identification, inspection event, component condition ratings and, for NHS "
           "bridges, element-level quantities in condition states 1-4."),
    sources=(
        Source("FHWA, Specifications for the National Bridge Inventory (SNBI), March 2022",
               "https://www.fhwa.dot.gov/bridge/snbi/snbi_march_2022_publication.pdf"),
        Source("23 CFR 650.313 Inspection procedures",
               "https://www.law.cornell.edu/cfr/text/23/650.313"),
        Source("23 CFR 650.315 Inventory (element data per the Manual for Bridge Element "
               "Inspection and the SNBI)", "https://www.law.cornell.edu/cfr/text/23/650.315"),
        Source("FHWA SNBI data crosswalk", "https://www.fhwa.dot.gov/bridge/snbi/datacrosswalk.cfm"),
    ),
    sections=(
        Section("identification", "Bridge identification and location", "SNBI 1, 2", (
            Item("id.bridge_number", "Bridge Number", "B.ID.01"),
            Item("id.bridge_name", "Bridge Name", "B.ID.02"),
            Item("loc.state_code", "State Code", "B.L.01"),
            Item("loc.county_code", "County Code", "B.L.02"),
            Item("loc.latitude", "Latitude", "B.L.05"),
            Item("loc.longitude", "Longitude", "B.L.06"),
        )),
        Section("inspection", "Inspection event", "SNBI 6 (B.IE)", (
            Item("ie.type", "Inspection Type", "B.IE.01"),
            Item("ie.begin_date", "Inspection Begin Date", "B.IE.02"),
            Item("ie.completion_date", "Inspection Completion Date", "B.IE.03"),
            Item("ie.inspector", "Nationally Certified Bridge Inspector", "B.IE.04"),
            Item("ie.interval_months", "Inspection Interval", "B.IE.05", "months"),
            Item("ie.due_date", "Inspection Due Date", "B.IE.06"),
            Item("ie.note", "Inspection Note", "B.IE.11"),
            Item("ie.equipment", "Inspection Equipment", "B.IE.12"),
        )),
        Section("condition", "Component condition ratings", "SNBI 7 (B.C)", (
            Item("cond.deck", "Deck Condition Rating", "B.C.01"),
            Item("cond.superstructure", "Superstructure Condition Rating", "B.C.02"),
            Item("cond.substructure", "Substructure Condition Rating", "B.C.03"),
            Item("cond.culvert", "Culvert Condition Rating", "B.C.04"),
            Item("cond.railing", "Bridge Railings Condition Rating", "B.C.05"),
            Item("cond.bearings", "Bridge Bearings Condition Rating", "B.C.07"),
            Item("cond.joints", "Bridge Joints Condition Rating", "B.C.08"),
            Item("cond.channel", "Channel Condition Rating", "B.C.09"),
            Item("cond.scour", "Scour Condition Rating", "B.C.11"),
        )),
        Section("elements", "Element-level condition", "SNBI 7 (B.E, B.CS); AASHTO MBEI", tables=(
            Table("elements", "Elements and quantities by condition state",
                  "B.E.01-03, B.CS.01-04",
                  (Column("element_number", "Element Number (B.E.01)"),
                   Column("parent_number", "Parent Number (B.E.02)", False),
                   Column("total_quantity", "Total Quantity (B.E.03)"),
                   Column("cs1", "CS1 Good (B.CS.01)"),
                   Column("cs2", "CS2 Fair (B.CS.02)"),
                   Column("cs3", "CS3 Poor (B.CS.03)"),
                   Column("cs4", "CS4 Severe (B.CS.04)")),
                  note="Condition states are assigned by the inspector per the AASHTO "
                       "Manual for Bridge Element Inspection; CS1-CS4 must sum to B.E.03."),
        )),
        Section("findings", "Deficiencies and critical findings", "23 CFR 650.313(a), (q)", tables=(
            Table("findings", "Documented deficiencies", "650.313(a), 650.313(q)(2)",
                  (Column("location", "Location on the bridge"),
                   Column("description", "Description"),
                   Column("photo", "Photo"),
                   Column("critical", "Critical finding (yes/no)"),
                   Column("action", "Corrective action")),
                  note="A finding is documented by the inspector; model proposals are not "
                       "findings until accepted."),
        )),
    ),
    checks=_bridge_checks,
)


# ── Roof ──────────────────────────────────────────────────────────────────────

ROOF = StandardTemplate(
    key="roof",
    title="Roof inspection report",
    standard="InterNACHI Home Inspection Standards of Practice, section 3.1 (Roof)",
    scope=("Visual roof inspection: what the standard says the inspector shall inspect, "
           "describe and report, plus the limitations it requires to be stated."),
    sources=(
        Source("InterNACHI, International Standards of Practice for Performing a General "
               "Home Inspection, section 3.1 Roof", "https://www.nachi.org/sop.htm"),
    ),
    sections=(
        Section("general", "Inspection", "SOP 1, 2", (
            Item("general.address", "Property address / site", "SOP 2"),
            Item("general.inspector", "Inspector", "SOP 2"),
            Item("general.datetime", "Date and time", "SOP 2"),
            Item("general.weather", "Weather at inspection", "SOP 2"),
            Item("general.method", "Method of inspecting the roof", "SOP 3.1 I"),
        )),
        Section("roof", "Roof", "SOP 3.1 I-II", (
            Item("roof.covering_type", "Type of roof-covering materials (describe)", "3.1 II.A"),
            Item("roof.covering_condition", "Roof-covering materials", "3.1 I.A"),
            Item("roof.gutters", "Gutters", "3.1 I.B"),
            Item("roof.downspouts", "Downspouts", "3.1 I.C"),
            Item("roof.penetrations", "Vents, flashing, skylights, chimney and other "
                 "roof penetrations", "3.1 I.D"),
            Item("roof.structure", "General structure of the roof", "3.1 I.E"),
            Item("roof.active_leaks", "Observed indications of active roof leaks", "3.1 III.A"),
            Item("roof.limitations", "Areas not inspected and why", "SOP 2, 3.1 IV"),
        ), tables=(
            Table("defects", "Material defects observed", "SOP 1.2, 2",
                  (Column("location", "Location"), Column("component", "Component"),
                   Column("description", "Description"), Column("photo", "Photo"),
                   Column("recommendation", "Recommendation"))),
        )),
    ),
)


# ── Telecom tower ─────────────────────────────────────────────────────────────

def _tower_checks(facts: dict[str, Any]) -> list[dict[str, Any]]:
    kind = fact(facts, "site.structure_type")
    if kind is None:
        return [_missing_check("guys", "Guys assessed for a guyed mast", "Annex J.1 H",
                               "the structure type")]
    guyed = "guy" in str(kind["value"]).lower()
    if not guyed:
        return [_check("guys", "Guys assessed for a guyed mast", "Annex J.1 H", "N/A",
                       f"Structure type {kind['value']!r} has no guys.")]
    have = fact(facts, "j.h_guys") is not None
    return [_check("guys", "Guys assessed for a guyed mast", "Annex J.1 H",
                   "PASS" if have else MISSING,
                   "Guy condition recorded." if have else "Guyed mast with no guy record.")]


TELECOM_TOWER = StandardTemplate(
    key="telecom_tower",
    title="Telecommunication tower condition assessment",
    standard="ANSI/TIA-222-H, section 14 and Annex J.1 (maintenance and condition assessment)",
    scope=("Maintenance and condition assessment of an antenna-supporting structure, item "
           "by item through the Annex J.1 checklist headings A-K."),
    sources=(
        Source("Telecommunications Industry Foundation (NATE Tower Times), Planning "
               "Advisory Notice 'ANSI/TIA-222 Maintenance and Condition Assessment of "
               "Telecommunication Towers' (lists Annex J.1 A-K and the s14.4 intervals)",
               "https://tifonline.org/wp-content/uploads/2022/08/PAN-ANSI-TIA-222.pdf"),
        Source("Wireless Estimator, 'New inspection requirements added in Revision H of "
               "ANSI/TIA-222'", "https://wirelessestimator.com/articles/2017/new-inspection-"
               "requirements-added-in-revision-h-of-ansitia-222/"),
    ),
    sections=(
        Section("site", "Structure and assessment", "s14", (
            Item("site.structure_id", "Structure / site identifier", "s14"),
            Item("site.location", "Location", "s14"),
            Item("site.structure_type", "Structure type (guyed / self-supporting / pole)", "s14.4"),
            Item("site.height_m", "Structure height", "s14", "m"),
            Item("site.assessor", "Assessed by", "s14"),
            Item("site.datetime", "Date of assessment", "s14"),
            Item("site.weather", "Weather at assessment", "s14"),
            Item("site.interval", "Assessment interval / reason (3 yr guyed, 5 yr "
                 "self-supporting, after extreme events)", "s14.4"),
        )),
        Section("checklist", "Annex J.1 checklist", "Annex J.1 A-K", (
            Item("j.a_structure", "A. Structure condition", "J.1 A"),
            Item("j.b_finish", "B. Finish", "J.1 B"),
            Item("j.c_lighting", "C. Lighting", "J.1 C"),
            Item("j.d_grounding", "D. Grounding", "J.1 D"),
            Item("j.e_mounts_antennas_lines", "E. Appurtenances: mounts, antennas and lines",
                 "J.1 E"),
            Item("j.f_other_appurtenances", "F. Other appurtenances", "J.1 F"),
            Item("j.g_base_insulator", "G. Base insulator (AM towers)", "J.1 G"),
            Item("j.h_guys", "H. Guys", "J.1 H"),
            Item("j.i_foundations", "I. Concrete foundations", "J.1 I"),
            Item("j.j_alignment", "J. Structure alignment (plumb and twist)", "J.1 J"),
            Item("j.k_modifications", "K. Previous modifications to structure", "J.1 K"),
        ), tables=(
            Table("deficiencies", "Deficiencies and maintenance items", "s14",
                  (Column("checklist_item", "Checklist item (A-K)"),
                   Column("location", "Location / elevation"),
                   Column("description", "Description"), Column("photo", "Photo"),
                   Column("action", "Maintenance action"))),
        )),
    ),
    checks=_tower_checks,
)


TEMPLATES: dict[str, StandardTemplate] = {
    t.key: t for t in (IEC62446_3, BRIDGE, ROOF, TELECOM_TOWER)
}


def get_template(key: str) -> StandardTemplate:
    try:
        return TEMPLATES[str(key)]
    except KeyError:
        raise ValueError(f"Unknown standards template {key!r}. Use one of: "
                         f"{', '.join(TEMPLATES)}.") from None


# ── Resolution: template + facts -> content ───────────────────────────────────

def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_cell(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def resolve(template: StandardTemplate, facts: dict[str, Any]) -> dict[str, Any]:
    """Every required item with its value and source, or MISSING. Pure: no I/O."""
    facts = dict(facts or {})
    sections, missing, provided = [], [], 0
    for section in template.sections:
        items = []
        for item in section.items:
            found = fact(facts, item.id)
            if found is None:
                missing.append(f"{item.ref} {item.label}")
                items.append({"id": item.id, "label": item.label, "ref": item.ref,
                              "unit": item.unit, "status": MISSING, "value": None,
                              "source": None})
            else:
                provided += 1
                items.append({"id": item.id, "label": item.label, "ref": item.ref,
                              "unit": item.unit, "status": "PROVIDED",
                              "value": found["value"], "source": found["source"]})
        tables = []
        for table in section.tables:
            rows = list((facts.get("tables") or {}).get(table.id) or [])
            out_rows = []
            for row in rows:
                cells = {}
                for column in table.columns:
                    value = row.get(column.key)
                    if value in (None, ""):
                        cells[column.key] = MISSING if column.required else "-"
                        if column.required:
                            missing.append(f"{table.ref} {table.title}: "
                                           f"{column.label} ({row.get(table.columns[0].key, '?')})")
                    else:
                        cells[column.key] = _cell(value)
                out_rows.append({"cells": cells, "source": str(row.get("source") or
                                                               "entered by operator")})
            if not rows:
                missing.append(f"{table.ref} {table.title}: no rows recorded")
            tables.append({"id": table.id, "title": table.title, "ref": table.ref,
                           "note": table.note,
                           "columns": [{"key": c.key, "label": c.label} for c in table.columns],
                           "rows": out_rows, "status": "PROVIDED" if rows else MISSING})
        sections.append({"id": section.id, "title": section.title, "ref": section.ref,
                         "items": items, "tables": tables})
    item_count = len(template.items())
    return {
        "template": template.key,
        "title": template.title,
        "standard": template.standard,
        "structured_following": template.structured_following,
        "statement": NOT_CERTIFIED,
        "sources": [{"label": s.label, "url": s.url} for s in template.sources],
        "sections": sections,
        "checks": template.checks(facts),
        "observations": list(facts.get("observations") or []),
        "evidence": list(facts.get("evidence") or []),
        "completeness": {
            "required_items": item_count,
            "provided_items": provided,
            "missing_items": item_count - provided,
            "missing": missing,
        },
    }


# ── Rendering ─────────────────────────────────────────────────────────────────

_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#18211e;max-width:1100px}
h1{font-size:22px;margin-bottom:4px} h2{font-size:17px;border-bottom:1px solid #ccd;
padding-bottom:3px;margin-top:28px} .ref{color:#556;font-size:12px}
.statement{background:#fff6dc;border:1px solid #e0c060;padding:10px;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border:1px solid #ccd;padding:4px 6px;text-align:left;vertical-align:top}
th{background:#eef1f4} .MISSING{color:#b00020;font-weight:700}
.PASS{color:#11683a;font-weight:700} .FAIL{color:#b00020;font-weight:700}
.src{color:#556;font-size:11px} code{font-size:11px;word-break:break-all}
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _value_text(item: dict[str, Any]) -> str:
    if item["status"] == MISSING:
        return MISSING
    unit = f" {item['unit']}" if item.get("unit") else ""
    return f"{_cell(item['value'])}{unit}"


def render_html(content: dict[str, Any], manifest: dict[str, Any], meta: dict[str, Any]) -> str:
    parts = [f"<h1>{_e(meta.get('title') or content['title'])}</h1>",
             f"<p><b>{_e(content['structured_following'])}</b></p>",
             f"<p class='ref'>Report {_e(meta.get('report_id'))} · generated "
             f"{_e(meta.get('generated_at'))} · project {_e(meta.get('project'))}</p>",
             f"<div class='statement'>{_e(content['statement'])}</div>"]
    comp = content["completeness"]
    parts.append(f"<p><b>Completeness:</b> {comp['provided_items']} of "
                 f"{comp['required_items']} required items provided; "
                 f"<span class='MISSING'>{comp['missing_items']} MISSING</span>"
                 f" (plus {len(comp['missing']) - comp['missing_items']} missing table "
                 "entries).</p>")
    for section in content["sections"]:
        parts.append(f"<h2>{_e(section['title'])} <span class='ref'>{_e(section['ref'])}"
                     "</span></h2>")
        if section["items"]:
            parts.append("<table><tr><th>Ref</th><th>Item</th><th>Value</th><th>Source</th></tr>")
            for item in section["items"]:
                cls = "MISSING" if item["status"] == MISSING else ""
                parts.append(f"<tr><td class='ref'>{_e(item['ref'])}</td><td>{_e(item['label'])}"
                             f"</td><td class='{cls}'>{_e(_value_text(item))}</td>"
                             f"<td class='src'>{_e(item['source'] or '')}</td></tr>")
            parts.append("</table>")
        for table in section["tables"]:
            parts.append(f"<h3>{_e(table['title'])} <span class='ref'>{_e(table['ref'])}</span></h3>")
            if table["note"]:
                parts.append(f"<p class='ref'>{_e(table['note'])}</p>")
            if not table["rows"]:
                parts.append("<p class='MISSING'>MISSING: no rows recorded.</p>")
                continue
            parts.append("<table><tr>" + "".join(f"<th>{_e(c['label'])}</th>"
                                                 for c in table["columns"]) + "<th>Source</th></tr>")
            for row in table["rows"]:
                parts.append("<tr>" + "".join(
                    f"<td class='{'MISSING' if row['cells'][c['key']] == MISSING else ''}'>"
                    f"{_e(row['cells'][c['key']])}</td>" for c in table["columns"])
                    + f"<td class='src'>{_e(row['source'])}</td></tr>")
            parts.append("</table>")
    if content["checks"]:
        parts.append("<h2>Checks against the standard's stated limits</h2><table><tr><th>Ref"
                     "</th><th>Check</th><th>Result</th><th>Detail</th></tr>")
        for check in content["checks"]:
            parts.append(f"<tr><td class='ref'>{_e(check['ref'])}</td><td>{_e(check['label'])}"
                         f"</td><td class='{_e(check['status'])}'>{_e(check['status'])}</td>"
                         f"<td>{_e(check['detail'])}</td></tr>")
        parts.append("</table>")
    if content["observations"]:
        parts.append("<h2>Supporting measurements (not standard items)</h2><ul>")
        for obs in content["observations"]:
            parts.append(f"<li>{_e(obs.get('text'))} <span class='src'>"
                         f"[{_e(obs.get('source'))}]</span></li>")
        parts.append("</ul>")
    parts.append("<h2>Missing items</h2>")
    parts.append("<ul>" + "".join(f"<li class='MISSING'>{_e(m)}</li>" for m in comp["missing"])
                 + "</ul>" if comp["missing"] else "<p>None.</p>")
    parts.append("<h2>Sources for the report structure</h2><ul>" + "".join(
        f"<li>{_e(s['label'])}: {_e(s['url'])}</li>" for s in content["sources"]) + "</ul>")
    parts.append(_manifest_html(manifest, meta))
    return ("<!doctype html><html><head><meta charset='utf-8'><title>"
            f"{_e(meta.get('title') or content['title'])}</title><style>{_CSS}</style></head>"
            f"<body>{''.join(parts)}</body></html>")


def _manifest_html(manifest: dict[str, Any], meta: dict[str, Any]) -> str:
    rows = "".join(f"<tr><td>{_e(i['role'])}</td><td>{_e(i['name'])}</td><td>{i['bytes']}</td>"
                   f"<td><code>{_e(i['sha256'])}</code></td></tr>" for i in manifest["inputs"])
    models = "".join(f"<tr><td>{_e(m['name'])}</td><td><code>{_e(m['sha256'])}</code></td>"
                     f"<td>{_e(m['digest_source'])}</td></tr>" for m in manifest["models"])
    return ("<h2>Input manifest (SHA-256)</h2>"
            f"<p class='ref'>Signed with Ed25519 key {_e(meta.get('fingerprint'))}. Verify "
            "with <code>python -m core.report_signing verify report.pdf</code>.</p>"
            + (f"<table><tr><th>Role</th><th>File</th><th>Bytes</th><th>SHA-256</th></tr>"
               f"{rows}</table>" if rows else "<p>No input files.</p>")
            + (f"<table><tr><th>Model</th><th>SHA-256</th><th>Digest source</th></tr>{models}"
               "</table>" if models else "<p>No model produced any content in this report.</p>"))


def render_pdf(path: Path, content: dict[str, Any], manifest: dict[str, Any],
               meta: dict[str, Any]) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate, Spacer,
                                    Table as RLTable, TableStyle)

    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=7.5, leading=9)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=11)
    red = ParagraphStyle("red", parent=small, textColor=colors.HexColor("#b00020"),
                         fontName="Helvetica-Bold")
    state = {"MISSING": red,
             "PASS": ParagraphStyle("pass", parent=small, textColor=colors.HexColor("#11683a"),
                                    fontName="Helvetica-Bold"),
             "FAIL": red}

    def p(text: Any, style=small) -> Paragraph:
        return Paragraph(html.escape(str(text if text is not None else "")).replace("\n", "<br/>"),
                         style)

    grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#99a")),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f4")),
                       ("VALIGN", (0, 0), (-1, -1), "TOP")])
    page = landscape(A4)
    width = page[0] - 30 * mm
    story: list[Any] = [Paragraph(html.escape(meta.get("title") or content["title"]),
                                  styles["Title"]),
                        p(content["structured_following"], styles["Heading3"]),
                        p(f"Report {meta.get('report_id')} | generated {meta.get('generated_at')} "
                          f"| project {meta.get('project')} | signing key {meta.get('fingerprint')}"),
                        Spacer(1, 4), p(content["statement"], body)]
    comp = content["completeness"]
    story.append(p(f"Completeness: {comp['provided_items']} of {comp['required_items']} "
                   f"required items provided; {comp['missing_items']} MISSING "
                   f"(plus {len(comp['missing']) - comp['missing_items']} missing table "
                   "entries).", body))
    for section in content["sections"]:
        story.append(p(f"{section['title']}  ({section['ref']})", styles["Heading2"]))
        if section["items"]:
            data = [[p("Ref"), p("Item"), p("Value"), p("Source")]]
            for item in section["items"]:
                value = _value_text(item)
                data.append([p(item["ref"]), p(item["label"]),
                             p(value, red if item["status"] == MISSING else small),
                             p(item["source"] or "")])
            t = RLTable(data, colWidths=[width * 0.1, width * 0.3, width * 0.3, width * 0.3],
                        repeatRows=1)
            t.setStyle(grid)
            story.append(t)
        for table in section["tables"]:
            story.append(p(f"{table['title']}  ({table['ref']})", styles["Heading4"]))
            if table["note"]:
                story.append(p(table["note"]))
            if not table["rows"]:
                story.append(p("MISSING: no rows recorded.", red))
                continue
            cols = table["columns"]
            data = [[p(c["label"]) for c in cols] + [p("Source")]]
            for row in table["rows"]:
                data.append([p(row["cells"][c["key"]],
                               red if row["cells"][c["key"]] == MISSING else small)
                             for c in cols] + [p(row["source"])])
            t = RLTable(data, colWidths=[width / (len(cols) + 1)] * (len(cols) + 1), repeatRows=1)
            t.setStyle(grid)
            story.append(t)
    if content["checks"]:
        story.append(p("Checks against the standard's stated limits", styles["Heading2"]))
        data = [[p("Ref"), p("Check"), p("Result"), p("Detail")]]
        for check in content["checks"]:
            data.append([p(check["ref"]), p(check["label"]),
                         p(check["status"], state.get(check["status"], small)), p(check["detail"])])
        t = RLTable(data, colWidths=[width * 0.12, width * 0.33, width * 0.1, width * 0.45],
                    repeatRows=1)
        t.setStyle(grid)
        story.append(t)
    if content["observations"]:
        story.append(p("Supporting measurements (not standard items)", styles["Heading2"]))
        for obs in content["observations"]:
            story.append(p(f"- {obs.get('text')}  [{obs.get('source')}]", body))
    for ev in content["evidence"]:
        image = Path(str(ev.get("path", "")))
        if image.is_file() and image.suffix.lower() in (".jpg", ".jpeg", ".png"):
            import io

            from PIL import Image as PILImage

            # A downscaled copy is drawn; the original is the file in the manifest.
            with PILImage.open(image) as im:
                im = im.convert("RGB")
                im.thumbnail((1400, 1400))
                w, h = im.size
                buffer = io.BytesIO()
                im.save(buffer, "JPEG", quality=85)
            buffer.seek(0)
            scale = min(120 * mm / w, 80 * mm / h)
            story += [Spacer(1, 4), RLImage(buffer, w * scale, h * scale),
                      p(f"{ev.get('caption', '')} ({image.name}; downscaled for print, "
                        "original hashed in the manifest)")]
    story.append(p("Missing items", styles["Heading2"]))
    for item in comp["missing"] or ["None."]:
        story.append(p(item, red if comp["missing"] else small))
    story.append(p("Sources for the report structure", styles["Heading2"]))
    for source in content["sources"]:
        story.append(p(f"{source['label']}: {source['url']}"))
    story.append(p("Input manifest (SHA-256)", styles["Heading2"]))
    story.append(p(f"Signed with Ed25519 key {meta.get('fingerprint')}. The manifest is also "
                   f"embedded in this PDF as {meta.get('manifest_attachment')}. Verify with: "
                   "python -m core.report_signing verify report.pdf"))
    if manifest["inputs"]:
        data = [[p("Role"), p("File"), p("Bytes"), p("SHA-256")]]
        data += [[p(i["role"]), p(i["name"]), p(i["bytes"]), p(i["sha256"])]
                 for i in manifest["inputs"]]
        t = RLTable(data, colWidths=[width * 0.12, width * 0.33, width * 0.1, width * 0.45],
                    repeatRows=1)
        t.setStyle(grid)
        story.append(t)
    else:
        story.append(p("No input files."))
    if manifest["models"]:
        data = [[p("Model"), p("SHA-256"), p("Digest source")]]
        data += [[p(m["name"]), p(m["sha256"]), p(m["digest_source"])] for m in manifest["models"]]
        t = RLTable(data, colWidths=[width * 0.3, width * 0.45, width * 0.25], repeatRows=1)
        t.setStyle(grid)
        story.append(t)
    else:
        story.append(p("No model produced any content in this report."))
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=page, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm,
                            title=str(meta.get("title") or content["title"]),
                            author="OpenDroneKit", subject=content["structured_following"])
    doc.build(story)
    return path


# ── End to end ────────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()[:40] or "report"


def generate_standards_report(template_key: str, facts: dict[str, Any], reports_root: str | Path,
                              *, title: str = "", project: str = "",
                              inputs: list[dict[str, Any]] | None = None,
                              models: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Write ``report.html`` + signed ``report.pdf`` + ``manifest.json`` into a new folder.

    ``inputs`` are the files the facts were read from (each hashed into the manifest);
    every evidence image in ``facts`` is added as an input too, so nothing printed in
    the report is outside the manifest.
    """
    from . import report_signing as rs

    template = get_template(template_key)
    content = resolve(template, facts)
    all_inputs = list(inputs or [])
    for ev in content["evidence"]:
        all_inputs.append({"path": ev.get("path"), "role": "evidence image"})
    report_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(reports_root) / f"{stamp}_{template.key}_{_slug(title or template.title)}_{report_id[:8]}"
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = rs.build_manifest(all_inputs, models or [], report={
        "report_id": report_id, "template": template.key,
        "structured_following": template.structured_following,
        "title": title or template.title, "project": project, "generated_at": generated_at,
    })
    key_info = rs.key_info()
    meta = {"title": title or template.title, "report_id": report_id,
            "generated_at": generated_at, "project": project,
            "fingerprint": key_info["fingerprint"],
            "manifest_attachment": rs.MANIFEST_ATTACHMENT}
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    (out_dir / "manifest.json").write_bytes(manifest_bytes)
    content_bytes = json.dumps(content, indent=2, sort_keys=True, default=str).encode("utf-8")
    (out_dir / "content.json").write_bytes(content_bytes)
    (out_dir / "report.html").write_text(render_html(content, manifest, meta), encoding="utf-8")
    pdf = render_pdf(out_dir / "report.pdf", content, manifest, meta)
    rs.embed_attachments(pdf, {rs.MANIFEST_ATTACHMENT: manifest_bytes,
                               rs.CONTENT_ATTACHMENT: content_bytes})
    rs.sign_pdf(pdf, manifest)
    rs.sign_files(out_dir, ["report.pdf", "report.html", "manifest.json", "content.json"],
                  manifest)
    result = {
        "id": report_id, "template": template.key, "title": meta["title"],
        "structured_following": template.structured_following,
        "output_dir": str(out_dir), "pdf_path": str(pdf),
        "html_path": str(out_dir / "report.html"), "manifest_path": str(out_dir / "manifest.json"),
        "signature_path": str(out_dir / "signature.json"),
        "signing_key_fingerprint": key_info["fingerprint"],
        "signing_key_created": key_info["created"],
        "inputs": len(manifest["inputs"]), "models": len(manifest["models"]),
        "completeness": content["completeness"], "checks": content["checks"],
        "created_at": generated_at,
    }
    (out_dir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
