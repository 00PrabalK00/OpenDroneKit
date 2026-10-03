"""Signed, traceable reports: manifest of inputs, Ed25519 signature, verification.

Everything runs on real files through the real signing code: a real PDF written by
reportlab, real SHA-256 over real bytes, a real Ed25519 key created in a temporary data
root. Tampering is done by editing the bytes on disk, the way a forger would.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pypdf = pytest.importorskip("pypdf")
pytest.importorskip("reportlab")

from core import report_signing as rs  # noqa: E402
from core.standards_reports import generate_standards_report  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "odk-data"
    monkeypatch.setenv("ODK_DATA_DIR", str(root))
    return root


@pytest.fixture
def inputs(tmp_path) -> list[dict]:
    folder = tmp_path / "inputs"
    folder.mkdir()
    (folder / "frame_001.jpg").write_bytes(b"\xff\xd8 first frame bytes")
    (folder / "ortho.tif").write_bytes(b"II*\x00 raster bytes" * 100)
    return [{"path": str(folder / "frame_001.jpg"), "role": "inspection image"},
            {"path": str(folder / "ortho.tif"), "role": "orthomosaic"}]


@pytest.fixture
def report(data_root, inputs, tmp_path) -> dict:
    facts = {"fields": {"site.structure_id": {"value": "T-17", "source": "entered by A. Tester"}}}
    return generate_standards_report(
        "telecom_tower", facts, tmp_path / "reports", title="Signed test", project="p",
        inputs=inputs, models=[{"name": "unit-model", "sha256": "ab" * 32,
                                "digest_source": "recorded by test"}])


class TestKey:
    def test_the_key_is_created_on_first_use_in_the_data_root_and_reused(self, data_root):
        first = rs.key_info()
        assert first["created"] is True
        assert Path(first["private_key_path"]).parent == data_root / "keys"
        assert Path(first["private_key_path"]).is_file()
        second = rs.key_info()
        assert second["created"] is False
        assert second["fingerprint"] == first["fingerprint"]

    def test_the_private_key_is_never_written_into_a_report(self, report):
        private = Path(rs.key_info()["private_key_path"]).read_bytes()
        body = private.split(b"-----")[2].strip()
        for path in Path(report["output_dir"]).iterdir():
            assert body not in path.read_bytes(), path.name


class TestManifest:
    def test_every_input_and_model_is_hashed_into_the_signed_pdf(self, report, inputs):
        import hashlib

        reader = pypdf.PdfReader(report["pdf_path"])
        manifest = json.loads(reader.attachments[rs.MANIFEST_ATTACHMENT][0])
        by_name = {e["name"]: e for e in manifest["inputs"]}
        for item in inputs:
            data = Path(item["path"]).read_bytes()
            assert by_name[Path(item["path"]).name]["sha256"] == hashlib.sha256(data).hexdigest()
        assert manifest["models"] == [{"name": "unit-model", "path": "", "sha256": "ab" * 32,
                                       "digest_source": "recorded by test"}]
        # Printed in the PDF too, not only attached.
        text = "".join(page.extract_text() for page in reader.pages)
        assert by_name["ortho.tif"]["sha256"][:40] in text.replace("\n", "")

    def test_an_input_that_does_not_exist_is_refused(self, data_root, tmp_path):
        with pytest.raises(FileNotFoundError):
            rs.build_manifest([{"path": str(tmp_path / "nope.jpg")}])

    def test_a_model_with_no_file_and_no_digest_is_refused(self, data_root):
        with pytest.raises(ValueError, match="neither"):
            rs.build_manifest([], [{"name": "ghost"}])


class TestVerify:
    def test_an_untouched_report_verifies(self, report):
        result = rs.verify_report(report["output_dir"])
        assert result["ok"], result["problems"]
        assert result["signature_valid"] and result["signer_trusted"]
        assert result["signer"] == "this installation's key"
        assert {i["status"] for i in result["inputs"]} == {"match"}
        assert set(result["files"].values()) == {"match"}

    def test_a_changed_byte_in_the_signed_pdf_fails(self, report):
        pdf = Path(report["pdf_path"])
        data = bytearray(pdf.read_bytes())
        index = data.find(b"Signed test")
        assert index > 0
        data[index] ^= 0x01
        pdf.write_bytes(bytes(data))
        result = rs.verify_pdf(pdf)
        assert not result["ok"]
        assert any("altered" in p for p in result["problems"])

    def test_content_appended_after_signing_fails(self, report):
        pdf = Path(report["pdf_path"])
        writer = pypdf.PdfWriter(str(pdf), incremental=True)
        writer.add_metadata({"/Title": "edited after signing"})
        with open(pdf, "wb") as handle:
            writer.write(handle)
        result = rs.verify_pdf(pdf)
        assert not result["ok"]
        assert any("after signing" in p for p in result["problems"])

    def test_page_content_smuggled_into_the_signature_update_fails(self, report, tmp_path):
        """Rewrite the unsigned signature update so it also adds a page content stream.

        One %%EOF, the signed prefix untouched and the same signature entry: only the
        object-level check catches that the file now shows unsigned content.
        """
        import io

        from pypdf.generic import ArrayObject, DecodedStreamObject, NameObject

        pdf = Path(report["pdf_path"])
        data = pdf.read_bytes()
        entry = pypdf.PdfReader(io.BytesIO(data)).metadata[rs.SIGNATURE_INFO_KEY]
        prefix = tmp_path / "prefix.pdf"
        prefix.write_bytes(data[:json.loads(entry)["statement"]["signed_length"]])
        writer = pypdf.PdfWriter(str(prefix), incremental=True)
        writer.add_metadata({rs.SIGNATURE_INFO_KEY: entry})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 40 Tf 50 400 Td (FORGED) Tj ET")
        page = writer.pages[0]
        contents = page["/Contents"]
        parts = ArrayObject(list(contents) if isinstance(contents, ArrayObject) else [contents])
        parts.append(writer._add_object(stream))
        page[NameObject("/Contents")] = parts
        with open(pdf, "wb") as handle:
            writer.write(handle)
        assert pdf.read_bytes().count(b"%%EOF") == data.count(b"%%EOF")
        result = rs.verify_pdf(pdf)
        assert result["signature_valid"]  # the maths still checks...
        assert not result["ok"]  # ...but the file shows content it does not cover
        assert any("not signed" in p for p in result["problems"])

    def test_a_changed_input_is_reported(self, report, inputs):
        Path(inputs[1]["path"]).write_bytes(b"different raster")
        result = rs.verify_report(report["pdf_path"])
        assert not result["ok"]
        status = {i["name"]: i["status"] for i in result["inputs"]}
        assert status == {"frame_001.jpg": "match", "ortho.tif": "changed"}

    def test_a_moved_input_is_found_by_name_under_inputs_root(self, report, inputs, tmp_path):
        moved = tmp_path / "elsewhere"
        moved.mkdir()
        for item in inputs:
            Path(item["path"]).rename(moved / Path(item["path"]).name)
        unchecked = rs.verify_report(report["pdf_path"])
        assert unchecked["ok"] and unchecked["warnings"]  # not found here: a warning
        found = rs.verify_report(report["pdf_path"], inputs_root=moved)
        assert found["ok"] and {i["status"] for i in found["inputs"]} == {"match"}

    def test_the_html_is_covered_by_the_detached_signature(self, report):
        html = Path(report["html_path"])
        html.write_text(html.read_text(encoding="utf-8").replace("T-17", "T-18"), encoding="utf-8")
        result = rs.verify_report(report["output_dir"])
        assert not result["ok"]
        assert result["files"]["report.html"] == "changed"

    def test_a_valid_signature_from_an_unknown_key_is_not_trusted(self, report, tmp_path,
                                                                    monkeypatch):
        # Another installation (another data root, another key) checks the report.
        monkeypatch.setenv("ODK_DATA_DIR", str(tmp_path / "other-machine"))
        rs.key_info()
        result = rs.verify_pdf(report["pdf_path"])
        assert result["signature_valid"] is True
        assert result["signer_trusted"] is False and not result["ok"]
        # Importing the signer's public key makes it trusted.
        original = tmp_path / "odk-data" / "keys" / rs.PUBLIC_KEY_NAME
        rs.trust_public_key(original, "colleague")
        assert rs.verify_pdf(report["pdf_path"])["ok"]

    def test_a_re_signed_forgery_names_a_different_key(self, report, tmp_path, monkeypatch):
        """Strip the signature, edit, re-sign with another key: valid but untrusted."""
        pdf = Path(report["pdf_path"])
        signed = json.loads(pypdf.PdfReader(str(pdf)).metadata[rs.SIGNATURE_INFO_KEY])
        prefix = pdf.read_bytes()[:signed["statement"]["signed_length"]]
        forged = tmp_path / "forged.pdf"
        forged.write_bytes(prefix)
        manifest = json.loads(pypdf.PdfReader(str(forged)).attachments[rs.MANIFEST_ATTACHMENT][0])
        monkeypatch.setenv("ODK_DATA_DIR", str(tmp_path / "forger"))
        rs.sign_pdf(forged, manifest)
        monkeypatch.setenv("ODK_DATA_DIR", str(tmp_path / "odk-data"))
        result = rs.verify_pdf(forged)
        assert result["signature_valid"] and not result["signer_trusted"]
        assert result["signer_fingerprint"] != report["signing_key_fingerprint"]
        assert not result["ok"]

    def test_an_unsigned_pdf_is_not_verified(self, tmp_path, data_root):
        from reportlab.pdfgen import canvas

        path = tmp_path / "plain.pdf"
        c = canvas.Canvas(str(path))
        c.drawString(72, 720, "not signed")
        c.save()
        result = rs.verify_report(path)
        assert not result["ok"]
        assert "no OpenDroneKit signature" in result["problems"][0]


class TestCommandLine:
    def test_the_verify_command_exits_zero_on_a_good_report_and_one_on_a_bad_one(
            self, report, data_root):
        env = {**__import__("os").environ, "ODK_DATA_DIR": str(data_root)}
        good = subprocess.run([sys.executable, "-m", "core.report_signing", "verify",
                               report["output_dir"]], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=120)
        assert good.returncode == 0, good.stdout + good.stderr
        assert good.stdout.startswith("VERIFIED")
        Path(report["html_path"]).write_text("tampered", encoding="utf-8")
        bad = subprocess.run([sys.executable, "-m", "core.report_signing", "verify",
                              report["output_dir"], "--json"], cwd=REPO, env=env,
                             capture_output=True, text=True, timeout=120)
        assert bad.returncode == 1
        assert json.loads(bad.stdout)["files"]["report.html"] == "changed"


class TestRefusal:
    """A report that cannot be signed is refused, and nothing unsigned is left on disk."""

    def _generate(self, inputs, root):
        return generate_standards_report("telecom_tower", {}, root, title="Refused",
                                         project="p", inputs=inputs)

    def test_missing_cryptography_refuses_before_writing(self, data_root, inputs, tmp_path,
                                                         monkeypatch):
        def unavailable():
            raise rs.SigningUnavailable("Report signing needs the 'cryptography' package")

        monkeypatch.setattr(rs, "_crypto", unavailable)
        root = tmp_path / "reports"
        with pytest.raises(rs.SigningUnavailable, match="cryptography"):
            self._generate(inputs, root)
        assert not root.exists() or not any(root.iterdir())
        assert not (data_root / "keys").exists()

    def test_missing_pypdf_refuses_before_writing(self, data_root, inputs, tmp_path,
                                                  monkeypatch):
        monkeypatch.setitem(sys.modules, "pypdf", None)  # import pypdf -> ImportError
        root = tmp_path / "reports"
        with pytest.raises(rs.SigningUnavailable, match="pypdf"):
            self._generate(inputs, root)
        assert not root.exists() or not any(root.iterdir())

    def test_a_signing_failure_removes_the_unsigned_report(self, data_root, inputs, tmp_path,
                                                           monkeypatch):
        def broken(*_args, **_kwargs):
            raise RuntimeError("disk full while signing")

        monkeypatch.setattr(rs, "sign_pdf", broken)
        root = tmp_path / "reports"
        with pytest.raises(RuntimeError, match="disk full"):
            self._generate(inputs, root)
        assert list(root.glob("*/report.pdf")) == []
        assert list(root.glob("*/report.html")) == []
