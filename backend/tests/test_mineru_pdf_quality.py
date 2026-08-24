"""扫描版/混合型 PDF 的 P0 预检、OCR 路由与质量门禁测试。"""

import sys
from types import SimpleNamespace

from langchain_core.documents import Document

from agent_material.rag_engine import file_loader
from agent_material.rag_engine import mineru_loader as loader
from agent_material.rag_engine.db.worker import validate_ingest_chunk_quality


class _Page:
    def __init__(self, text=None, error=False):
        self.text = text
        self.error = error

    def extract_text(self):
        if self.error:
            raise ValueError("broken page")
        return self.text


def _install_fake_pypdf(monkeypatch, pages):
    class Reader:
        def __init__(self, _path):
            self.pages = pages

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=Reader))
    monkeypatch.setattr(loader, "_has_pdf_signature", lambda _path: True)


def test_inspect_pdf_classifies_native_scanned_mixed_and_unknown(monkeypatch):
    native_text = "这是有文本层的原生 PDF 页面，字符数量足够用于可靠识别。"

    _install_fake_pypdf(monkeypatch, [_Page(native_text), _Page(native_text)])
    native = loader.inspect_pdf("native.pdf")
    assert native.pdf_mode == "native"
    assert native.ocr_required is False

    _install_fake_pypdf(monkeypatch, [_Page(""), _Page(None)])
    scanned = loader.inspect_pdf("scan.pdf")
    assert scanned.pdf_mode == "scanned"
    assert scanned.image_only_pages == 2
    assert scanned.ocr_required is True

    _install_fake_pypdf(monkeypatch, [_Page(native_text), _Page("")])
    mixed = loader.inspect_pdf("mixed.pdf")
    assert mixed.pdf_mode == "mixed"
    assert mixed.text_layer_pages == 1
    assert mixed.ocr_required is True

    _install_fake_pypdf(monkeypatch, [_Page(error=True)])
    unknown = loader.inspect_pdf("unknown.pdf")
    assert unknown.pdf_mode == "unknown"
    assert unknown.inspection_failed_pages == 1
    assert unknown.ocr_required is True


def test_scanned_pdf_success_carries_ocr_and_quality_metadata():
    profile = loader.PdfProfile(2, 0, 2, 0, "scanned", True)
    client = SimpleNamespace(
        flash_extract=lambda _path: SimpleNamespace(
            state="done",
            markdown="# OCR 结果\n\n扫描页面已经识别出足够多的正文内容，可以安全进入后续切分和索引流程。",
            error=None,
        )
    )

    docs = loader._extract_single("scan.pdf", client, 1, 2, profile=profile)

    assert len(docs) == 1
    assert docs[0].metadata["pdf_mode"] == "scanned"
    assert docs[0].metadata["ocr_required"] is True
    assert docs[0].metadata["ocr_route_selected"] is True
    assert docs[0].metadata["parse_quality_status"] == "passed"


def test_empty_ocr_result_is_rejected():
    profile = loader.PdfProfile(1, 0, 1, 0, "scanned", True)
    client = SimpleNamespace(
        flash_extract=lambda _path: SimpleNamespace(
            state="done", markdown="![page](image.png)", error=None
        )
    )

    assert loader._extract_single("scan.pdf", client, 1, 1, profile=profile) == []


def test_pdf_over_size_limit_is_split_even_when_under_page_limit(
    monkeypatch, tmp_path,
):
    pdf = tmp_path / "large.pdf"
    pdf.write_bytes(b"pdf")
    profile = loader.PdfProfile(2, 2, 0, 0, "native", False)
    fake_mineru = SimpleNamespace(
        MinerU=lambda: (_ for _ in ()).throw(
            AssertionError("大文件分片前不应初始化 MinerU client")
        )
    )
    monkeypatch.setitem(sys.modules, "mineru", fake_mineru)
    monkeypatch.setattr(loader, "inspect_pdf", lambda _path: profile)
    monkeypatch.setattr(loader, "_has_pdf_signature", lambda _path: True)
    monkeypatch.setattr(loader.os.path, "getsize", lambda _path: 11 * 1024 * 1024)
    monkeypatch.setattr(
        loader, "_split_all_chunks", lambda *_args: [("part.pdf", 1, 2)]
    )
    monkeypatch.setattr(
        loader,
        "_batch_extract",
        lambda *_args, **_kwargs: [(1, 2, "正文" * 30)],
    )
    monkeypatch.setattr(
        loader,
        "_merge_and_cleanup",
        lambda parsed, chunks: (parsed[0][2], 1, 0),
    )

    docs = loader.mineru_pdf_loader(str(pdf))

    assert len(docs) == 1
    assert docs[0].metadata["parts"] == 1
    assert docs[0].metadata["parse_quality_status"] == "passed"


def test_oversized_generated_chunk_is_recursively_split(monkeypatch):
    pages = [object(), object(), object(), object()]

    class Reader:
        def __init__(self, _path):
            self.pages = pages

        def close(self):
            pass

    class Writer:
        def __init__(self):
            self.pages = []

        def add_page(self, page):
            self.pages.append(page)

        def write(self, stream):
            stream.write(b"pdf")

    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        SimpleNamespace(PdfReader=Reader, PdfWriter=Writer),
    )
    real_getsize = loader.os.path.getsize
    monkeypatch.setattr(
        loader.os.path,
        "getsize",
        lambda path: (
            11 * 1024 * 1024 if "p1-4.pdf" in str(path) else real_getsize(path)
        ),
    )

    chunks = loader._split_all_chunks("large.pdf", [(1, 4)])

    assert [(start, end) for _, start, end in chunks] == [(1, 2), (3, 4)]
    loader._merge_and_cleanup(
        [(start, end, "正文内容足够通过清理") for _, start, end in chunks],
        chunks,
    )


def test_single_oversized_page_fails_closed(monkeypatch):
    class Reader:
        pages = [object()]

        def __init__(self, _path):
            pass

        def close(self):
            pass

    class Writer:
        def add_page(self, _page):
            pass

        def write(self, stream):
            stream.write(b"pdf")

    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        SimpleNamespace(PdfReader=Reader, PdfWriter=Writer),
    )
    monkeypatch.setattr(loader.os.path, "getsize", lambda _path: 11 * 1024 * 1024)

    assert loader._split_all_chunks("one-huge-page.pdf", [(1, 1)]) == []


def test_any_failed_pdf_part_rejects_whole_version(monkeypatch, tmp_path):
    pdf = tmp_path / "many-pages.pdf"
    pdf.write_bytes(b"pdf")
    profile = loader.PdfProfile(21, 0, 21, 0, "scanned", True)
    monkeypatch.setitem(sys.modules, "mineru", SimpleNamespace(MinerU=lambda: object()))
    monkeypatch.setattr(loader, "inspect_pdf", lambda _path: profile)
    monkeypatch.setattr(loader, "_has_pdf_signature", lambda _path: True)
    monkeypatch.setattr(
        loader,
        "_split_all_chunks",
        lambda *_args: [("part1.pdf", 1, 20), ("part2.pdf", 21, 21)],
    )
    monkeypatch.setattr(
        loader,
        "_batch_extract",
        lambda *_args, **_kwargs: [(1, 20, "OCR正文" * 100), (21, 21, None)],
    )
    monkeypatch.setattr(
        loader,
        "_merge_and_cleanup",
        lambda *_args: ("OCR正文" * 100, 1, 1),
    )

    assert loader.mineru_pdf_loader(str(pdf)) == []


def test_native_pdf_can_fallback_but_scanned_pdf_cannot(monkeypatch):
    monkeypatch.setattr(loader, "mineru_pdf_loader", lambda _path: [])
    monkeypatch.setattr(
        file_loader,
        "pdf_text_loader",
        lambda path: [Document(page_content="原生文本回退内容" * 5, metadata={"source": path})],
    )

    native = loader.PdfProfile(1, 1, 0, 0, "native", False)
    monkeypatch.setattr(loader, "inspect_pdf", lambda _path: native)
    docs = file_loader.mineru_pdf_loader("native.pdf")
    assert len(docs) == 1
    assert docs[0].metadata["loader"] == "pdf_text_fallback"
    assert docs[0].metadata["parse_quality_status"] == "fallback_text"

    scanned = loader.PdfProfile(1, 0, 1, 0, "scanned", True)
    monkeypatch.setattr(loader, "inspect_pdf", lambda _path: scanned)
    assert file_loader.mineru_pdf_loader("scan.pdf") == []


def test_worker_rejects_missing_ocr_and_failed_parts():
    chunks = [
        Document(
            page_content="content",
            metadata={
                "source": "scan.pdf",
                "file_type": "pdf",
                "ocr_required": True,
                "ocr_route_selected": False,
                "parts_failed": 1,
                "parse_quality_status": "rejected",
            },
        )
    ]

    problems = validate_ingest_chunk_quality(chunks)

    assert "PDF 解析质量门禁未通过" in problems
    assert "PDF 存在 1 个失败分片" in problems
    assert "PDF 包含扫描页但未路由到 OCR 能力解析器" in problems


def test_worker_keeps_backward_compatibility_for_old_non_pdf_chunks():
    chunks = [Document(page_content="content", metadata={"source": "note.md"})]
    assert validate_ingest_chunk_quality(chunks) == []


def test_invalid_pdf_signature_is_rejected(monkeypatch, tmp_path):
    fake_pdf = tmp_path / "wrapped.pdf"
    fake_pdf.write_bytes(b'--boundary\r\nContent-Type: application/pdf\r\n\r\n%PDF-1.4')

    assert loader._has_pdf_signature(str(fake_pdf)) is False
    assert loader.mineru_pdf_loader(str(fake_pdf)) == []
