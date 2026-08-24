"""PDF 语料只读审计脚本测试。"""

import importlib.util
from pathlib import Path

from agent_material.rag_engine.mineru_loader import PdfProfile


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "audit_pdf_corpus", ROOT / "scripts" / "audit_pdf_corpus.py"
)
audit_pdf_corpus = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit_pdf_corpus)


def test_audit_pdf_corpus_aggregates_profiles(monkeypatch, tmp_path):
    (tmp_path / "native.pdf").write_bytes(b"pdf")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "scan.pdf").write_bytes(b"pdf")
    monkeypatch.setattr(
        audit_pdf_corpus,
        "inspect_pdf",
        lambda path: (
            PdfProfile(2, 2, 0, 0, "native", False)
            if path.endswith("native.pdf")
            else PdfProfile(3, 0, 3, 0, "scanned", True)
        ),
    )

    report = audit_pdf_corpus.audit(tmp_path)

    assert report["total_files"] == 2
    assert report["modes"] == {"native": 1, "scanned": 1}
    assert report["ocr_required_files"] == 1
    assert report["inspection_failed_pages"] == 0
    assert len(report["items"]) == 2


def test_audit_pdf_corpus_reports_invalid_file_without_aborting(monkeypatch, tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    monkeypatch.setattr(
        audit_pdf_corpus,
        "inspect_pdf",
        lambda _path: (_ for _ in ()).throw(ValueError("missing signature")),
    )

    report = audit_pdf_corpus.audit(tmp_path)

    assert report["total_files"] == 1
    assert report["modes"] == {"invalid": 1}
    assert report["items"][0]["pdf_mode"] == "invalid"
    assert "missing signature" in report["items"][0]["error"]
