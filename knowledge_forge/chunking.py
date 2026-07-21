"""Deterministic document routing, extraction and hierarchy-aware chunking."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import importlib.util
import logging
import math
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from .config import Config
from .sentence_spans import split_sentences


LOGGER = logging.getLogger(__name__)
_DOCLING_LOCK = threading.Lock()


PARSER_OPTIONS = {
    "auto": "自动选择",
    "native_docx": "Word 原生解析",
    "native_text": "文本/Markdown 解析",
    "plaintext_pdf": "PDF 文本层解析",
    "docling": "Docling 版面解析",
    "spreadsheet": "电子表格解析",
}
CHUNK_STRATEGY_OPTIONS = {
    "auto": "自动选择",
    "default": "兼容分块",
    "general": "通用分块",
    "paper": "论文分块",
    "table": "表格分块",
}

_PARSER_SUFFIXES = {
    "native_docx": {".docx"},
    "native_text": {".md", ".markdown", ".txt"},
    "plaintext_pdf": {".pdf"},
    "docling": {".pdf", ".docx"},
    "spreadsheet": {".xlsx", ".xls", ".csv"},
}
_STRATEGY_SUFFIXES = {
    "default": {".docx", ".pdf", ".md", ".markdown", ".txt"},
    "general": set(Config.allowed_suffixes),
    "paper": {".docx", ".pdf", ".md", ".markdown", ".txt"},
    "table": {".docx", ".pdf", ".xlsx", ".xls", ".csv"},
}
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,7})[、.． ]+(.+)$")
_PAPER_MARKERS = re.compile(
    r"(?:摘要|关键词|引言|研究方法|实验(?:结果)?|结论|参考文献|abstract|keywords?|introduction|methodology|results?|conclusion|references)",
    re.IGNORECASE,
)
_REFERENCE_TITLE = re.compile(r"^(?:参考文献|references?|bibliography)\s*$", re.IGNORECASE)
_ABSTRACT_TITLE = re.compile(r"^(?:摘要|abstract)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class DocumentBlock:
    key: str
    block_type: str
    text: str
    order: int
    section_key: str
    section_title: str
    level: int = 1
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentProfile:
    suffix: str
    page_count: int = 0
    sampled_pages: int = 0
    text_chars: int = 0
    has_text_layer: bool = True
    empty_page_ratio: float = 0.0
    replacement_ratio: float = 0.0
    heading_count: int = 0
    table_count: int = 0
    image_count: int = 0
    paper_score: int = 0
    layout_complexity: str = "simple"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessingPlan:
    requested_parser: str
    requested_strategy: str
    effective_parser: str
    effective_strategy: str
    routing_reason: str
    profile: DocumentProfile


@dataclass(frozen=True)
class ParsedDocument:
    blocks: list[DocumentBlock]
    parser_type: str
    parser_version: str
    quality: dict[str, Any]


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    level: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    key: str
    section_key: str
    title: str
    part_index: int
    part_total: int
    content: str
    content_hash: str
    token_count: int = 0
    parent_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessingResult:
    plan: ProcessingPlan
    parsed: ParsedDocument
    sections: list[Section]
    chunks: list[TextChunk]

    def payload(self) -> dict[str, Any]:
        return {
            "plan": {**asdict(self.plan), "profile": asdict(self.plan.profile)},
            "quality": self.parsed.quality,
            "blocks": [asdict(item) for item in self.parsed.blocks],
            "sections": [asdict(item) for item in self.sections],
            "chunks": [asdict(item) for item in self.chunks],
        }


def docling_available() -> bool:
    return Config.docling_enabled and importlib.util.find_spec("docling") is not None


def validate_processing_request(filename: str, parser: str = "auto", strategy: str = "auto") -> None:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in Config.allowed_suffixes:
        supported = "、".join(sorted(Config.allowed_suffixes))
        raise ValueError(f"不支持 {suffix or '无后缀'} 文件；当前支持：{supported}")
    if parser not in PARSER_OPTIONS:
        raise ValueError(f"未知解析器：{parser}")
    if strategy not in CHUNK_STRATEGY_OPTIONS:
        raise ValueError(f"未知分块策略：{strategy}")
    if parser != "auto" and suffix not in _PARSER_SUFFIXES[parser]:
        raise ValueError(f"解析器“{PARSER_OPTIONS[parser]}”不能处理 {suffix} 文件")
    if strategy != "auto" and suffix not in _STRATEGY_SUFFIXES[strategy]:
        raise ValueError(f"分块策略“{CHUNK_STRATEGY_OPTIONS[strategy]}”不适用于 {suffix} 文件")
    if parser == "docling" and not docling_available():
        raise RuntimeError("已选择 Docling，但当前环境未安装可选依赖；请安装 requirements-docling.txt")


def _heading_info(paragraph: Paragraph) -> tuple[bool, int, str]:
    text = paragraph.text.strip()
    style_name = getattr(paragraph.style, "name", "") or ""
    match = re.search(r"(?:Heading|标题)\s*([1-9])", style_name, flags=re.IGNORECASE)
    if match and text:
        return True, int(match.group(1)), text
    numbered = _NUMBERED_HEADING.match(text)
    if numbered and len(text) <= 140:
        return True, min(numbered.group(1).count(".") + 1, 6), text
    return False, 0, text


def _slug(value: str, index: int) -> str:
    safe = re.sub(r"[^\w\u4e00-\u9fff]+", "_", value)[:48].strip("_") or "section"
    return f"s{index:04d}_{safe}"


def _paper_score(text: str) -> int:
    return len({item.group(0).lower() for item in _PAPER_MARKERS.finditer(text[:60000])})


def inspect_document(path: Path) -> DocumentProfile:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF 解析依赖未安装，请执行 pip install -r requirements.txt") from exc
        reader = PdfReader(str(path))
        sample_count = min(len(reader.pages), max(1, Config.pdf_probe_pages))
        texts = [(reader.pages[index].extract_text() or "") for index in range(sample_count)]
        sample = "\n".join(texts)
        empty = sum(not text.strip() for text in texts)
        replacement = sample.count("�")
        table_lines = sum(1 for line in sample.splitlines() if line.count("|") >= 2 or re.search(r"\S\s{3,}\S", line))
        complexity = "complex" if table_lines >= 4 or replacement / max(1, len(sample)) > 0.01 else "simple"
        return DocumentProfile(
            suffix=suffix,
            page_count=len(reader.pages),
            sampled_pages=sample_count,
            text_chars=len(sample.strip()),
            has_text_layer=bool(sample.strip()),
            empty_page_ratio=round(empty / max(1, sample_count), 4),
            replacement_ratio=round(replacement / max(1, len(sample)), 6),
            table_count=1 if table_lines >= 4 else 0,
            paper_score=_paper_score(sample),
            layout_complexity=complexity,
            metadata={"sample_table_lines": table_lines},
        )
    if suffix == ".docx":
        document = Document(path)
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        headings = sum(_heading_info(p)[0] for p in document.paragraphs)
        images = len(document.inline_shapes)
        text = "\n".join(paragraphs)
        complexity = "complex" if document.tables or images > 4 else "simple"
        return DocumentProfile(suffix=suffix, text_chars=len(text), heading_count=headings,
                               table_count=len(document.tables), image_count=images,
                               paper_score=_paper_score(text), layout_complexity=complexity)
    if suffix in {".xlsx", ".xls", ".csv"}:
        return DocumentProfile(suffix=suffix, table_count=1, layout_complexity="tabular")
    text = path.read_text(encoding="utf-8", errors="ignore")
    headings = len(re.findall(r"^#{1,6}\s+", text, flags=re.MULTILINE))
    return DocumentProfile(suffix=suffix, text_chars=len(text.strip()), heading_count=headings,
                           paper_score=_paper_score(text))


def route_processing(path: Path, requested_parser: str = "auto", requested_strategy: str = "auto") -> ProcessingPlan:
    validate_processing_request(path.name, requested_parser, requested_strategy)
    profile = inspect_document(path)
    suffix = profile.suffix
    reasons: list[str] = []
    parser = requested_parser
    if parser == "auto":
        if suffix == ".docx":
            parser = "native_docx"; reasons.append("DOCX 使用原生结构解析")
        elif suffix in {".md", ".markdown", ".txt"}:
            parser = "native_text"; reasons.append("文本文件使用标题与段落解析")
        elif suffix in {".xlsx", ".xls", ".csv"}:
            parser = "spreadsheet"; reasons.append("电子表格使用工作表与行解析")
        elif not profile.has_text_layer:
            if not docling_available():
                raise RuntimeError("PDF 未检测到可用文本层，且当前环境未安装 Docling，无法可靠解析扫描件")
            parser = "docling"; reasons.append("PDF 无文本层，改用 Docling")
        elif profile.layout_complexity == "complex" and docling_available():
            parser = "docling"; reasons.append("PDF 版面较复杂，改用 Docling")
        else:
            parser = "plaintext_pdf"; reasons.append("PDF 文本层完整，使用轻量解析")
    else:
        reasons.append(f"按用户选择使用{PARSER_OPTIONS[parser]}")

    strategy = requested_strategy
    if strategy == "auto":
        if suffix in {".xlsx", ".xls", ".csv"}:
            strategy = "table"; reasons.append("表格文件按行构造知识块")
        elif profile.paper_score >= 4:
            strategy = "paper"; reasons.append("检测到摘要/章节/参考文献等论文结构")
        else:
            strategy = "general"; reasons.append("采用通用结构与 token 预算分块")
    else:
        reasons.append(f"按用户选择使用{CHUNK_STRATEGY_OPTIONS[strategy]}")
    return ProcessingPlan(requested_parser, requested_strategy, parser, strategy, "；".join(reasons), profile)


def _heading_path(stack: list[str], level: int, title: str) -> tuple[str, ...]:
    del stack[max(0, level - 1):]
    while len(stack) < max(0, level - 1):
        stack.append("")
    stack.append(title)
    return tuple(item for item in stack if item)


def _table_text(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    return "\n".join([
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *("| " + " | ".join(row) + " |" for row in normalized[1:]),
    ])


def _append_block(blocks: list[DocumentBlock], block_type: str, text: str, section_key: str,
                  section_title: str, level: int, path: tuple[str, ...], *, page: int | None = None,
                  metadata: dict[str, Any] | None = None) -> None:
    text = text.strip()
    if not text:
        return
    order = len(blocks) + 1
    blocks.append(DocumentBlock(f"b{order:05d}", block_type, text, order, section_key, section_title,
                                level, path, page, metadata or {}))


def _parse_docx(path: Path) -> ParsedDocument:
    document = Document(path)
    blocks: list[DocumentBlock] = []
    title, level, section_index = path.stem, 1, 1
    section_key = _slug(title, section_index)
    stack = [title]
    table_index = 0
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            is_heading, candidate_level, text = _heading_info(item)
            if not text:
                continue
            if is_heading:
                title, level = text, candidate_level
                section_index += 1
                section_key = _slug(title, section_index)
                path_parts = _heading_path(stack, level, title)
                _append_block(blocks, "heading", title, section_key, title, level, path_parts)
            else:
                _append_block(blocks, "paragraph", text, section_key, title, level, tuple(stack))
        elif isinstance(item, Table):
            table_index += 1
            rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in item.rows]
            table = _table_text(rows)
            _append_block(blocks, "table", table, section_key, title, level, tuple(stack),
                          metadata={"table_index": table_index, "rows": len(rows)})
            headers = rows[0] if rows else []
            for row_index, values in enumerate(rows[1:] or rows, start=1):
                pairs = [f"{headers[i] or f'列{i + 1}'}: {value}" for i, value in enumerate(values) if value]
                _append_block(blocks, "table_row", "；".join(pairs), section_key, title, level, tuple(stack),
                              metadata={"table_index": table_index, "row_index": row_index,
                                        "headers": headers, "values": values})
    return _parsed(blocks, "native_docx", "python-docx", allow_empty=True)


def _text_to_blocks(text: str, stem: str, *, page: int | None = None,
                    initial_order: int = 0) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    title, level, section_index = stem, 1, 1
    section_key = _slug(title, section_index)
    stack = [title]
    paragraph: list[str] = []
    table_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            _append_block(blocks, "paragraph", "\n".join(paragraph), section_key, title, level,
                          tuple(stack), page=page)
            paragraph.clear()

    def flush_table() -> None:
        if table_lines:
            _append_block(blocks, "table", "\n".join(table_lines), section_key, title, level,
                          tuple(stack), page=page)
            rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in table_lines]
            rows = [row for row in rows if not all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in row)]
            headers = rows[0] if rows else []
            for row_index, values in enumerate(rows[1:], start=1):
                pairs = [f"{headers[i] or f'列{i + 1}'}: {value}" for i, value in enumerate(values) if value]
                _append_block(blocks, "table_row", "；".join(pairs), section_key, title, level,
                              tuple(stack), page=page,
                              metadata={"row_index": row_index, "headers": headers, "values": values})
            table_lines.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        markdown = re.match(r"^(#{1,6})\s+(.+)$", line)
        numbered = _NUMBERED_HEADING.match(line)
        is_table = line.startswith("|") and line.endswith("|") and line.count("|") >= 2
        if markdown or (numbered and len(line) <= 140):
            flush_paragraph(); flush_table()
            if markdown:
                level, title = len(markdown.group(1)), markdown.group(2).strip()
            else:
                level, title = min(numbered.group(1).count(".") + 1, 6), line
            section_index += 1
            section_key = _slug(title, section_index)
            path_parts = _heading_path(stack, level, title)
            _append_block(blocks, "heading", title, section_key, title, level, path_parts, page=page)
        elif is_table:
            flush_paragraph(); table_lines.append(line)
        elif not line:
            flush_paragraph(); flush_table()
        else:
            flush_table(); paragraph.append(line)
    flush_paragraph(); flush_table()
    if initial_order:
        blocks = [DocumentBlock(f"b{initial_order + i:05d}", item.block_type, item.text,
                                initial_order + i, item.section_key, item.section_title, item.level,
                                item.heading_path, item.page, item.metadata)
                  for i, item in enumerate(blocks, start=1)]
    return blocks


def _parse_text(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _parsed(_text_to_blocks(text, path.stem), "native_text", "builtin-text-v2")


def _parse_pdf(path: Path) -> ParsedDocument:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    blocks: list[DocumentBlock] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_blocks = _text_to_blocks(text, f"第 {index} 页", page=index, initial_order=len(blocks))
        blocks.extend([
            DocumentBlock(item.key, item.block_type, item.text, item.order,
                          f"p{index:04d}_{item.section_key}", item.section_title, item.level,
                          item.heading_path, item.page, item.metadata)
            for item in page_blocks
        ])
    return _parsed(blocks, "plaintext_pdf", f"pypdf-{_package_version('pypdf')}",
                   {"page_count": len(reader.pages)}, allow_empty=True)


def _docling_cuda_memory() -> tuple[bool, int, int, str]:
    """Return CUDA readiness and free/total MiB without making CUDA mandatory."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, 0, 0, "容器未检测到可用 CUDA"
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_mb = int(free_bytes / 1024 / 1024)
        total_mb = int(total_bytes / 1024 / 1024)
        if free_mb < Config.docling_gpu_min_free_mb:
            return False, free_mb, total_mb, (
                f"剩余显存 {free_mb} MiB 低于安全门槛 {Config.docling_gpu_min_free_mb} MiB"
            )
        return True, free_mb, total_mb, ""
    except Exception as exc:
        return False, 0, 0, f"CUDA 状态检查失败：{exc}"


def _select_docling_device() -> tuple[str, dict[str, Any]]:
    requested = Config.docling_device or "auto"
    metadata: dict[str, Any] = {"docling_accelerator_requested": requested}
    if requested == "cpu":
        return "cpu", metadata
    if requested not in {"auto", "cuda"} and not requested.startswith("cuda:"):
        raise ValueError("DOCLING_DEVICE 仅支持 auto、cpu、cuda 或 cuda:N")

    ready, free_mb, total_mb, reason = _docling_cuda_memory()
    metadata.update({"docling_gpu_free_mb_before": free_mb, "docling_gpu_total_mb": total_mb})
    if ready:
        return "cuda:0" if requested == "auto" else requested, metadata
    if not Config.docling_cuda_fallback and requested.startswith("cuda"):
        raise RuntimeError(reason)
    metadata["docling_gpu_fallback_reason"] = reason
    return "cpu", metadata


def _is_cuda_failure(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "cuda", "cudnn", "cublas", "out of memory", "device-side assert",
        "no kernel image", "nvidia driver", "hip out of memory",
    )
    return any(marker in message for marker in markers)


def _release_docling_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _docling_markdown(path: Path, device: str) -> str:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    batch_size = Config.docling_batch_size
    accelerator = AcceleratorOptions(
        num_threads=Config.docling_num_threads,
        device=device,
        cuda_use_flash_attention2=False,
    )
    pipeline = PdfPipelineOptions(
        accelerator_options=accelerator,
        ocr_options=RapidOcrOptions(backend="torch"),
        ocr_batch_size=batch_size,
        layout_batch_size=batch_size,
        table_batch_size=batch_size,
        queue_max_size=max(4, batch_size * 4),
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
    )
    try:
        result = converter.convert(str(path))
        return result.document.export_to_markdown()
    finally:
        del converter


def _parse_docling(path: Path) -> ParsedDocument:
    if not docling_available():
        raise RuntimeError("Docling 未安装；请安装 requirements-docling.txt 后重试")
    with _DOCLING_LOCK:
        device, accelerator_metadata = _select_docling_device()
        if device.startswith("cuda"):
            try:
                import torch
                fraction = min(1.0, max(0.05, Config.docling_gpu_memory_fraction))
                torch.cuda.set_per_process_memory_fraction(fraction, 0)
                accelerator_metadata["docling_gpu_memory_fraction"] = fraction
            except Exception as exc:
                if not Config.docling_cuda_fallback:
                    raise
                accelerator_metadata["docling_gpu_fallback_reason"] = f"显存限额配置失败：{exc}"
                device = "cpu"
        LOGGER.info("Docling parsing %s with %s", path.name, device)
        try:
            markdown = _docling_markdown(path, device)
        except Exception as exc:
            if not (device.startswith("cuda") and Config.docling_cuda_fallback and _is_cuda_failure(exc)):
                raise
            LOGGER.warning("Docling CUDA failed for %s; retrying on CPU: %s", path.name, exc)
            accelerator_metadata["docling_gpu_fallback_reason"] = f"CUDA 解析失败：{exc}"
            _release_docling_cuda()
            device = "cpu"
            markdown = _docling_markdown(path, device)
        finally:
            if device.startswith("cuda"):
                _release_docling_cuda()
    accelerator_metadata.update({
        "docling_accelerator": device,
        "docling_batch_size": Config.docling_batch_size,
    })
    return _parsed(_text_to_blocks(markdown, path.stem), "docling",
                   f"docling-{_package_version('docling')}", accelerator_metadata,
                   allow_empty=path.suffix.lower() in {".pdf", ".docx"})


def _sheet_blocks(sheet_name: str, rows: Iterable[list[Any]], blocks: list[DocumentBlock]) -> None:
    cleaned = [["" if value is None else str(value).strip() for value in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return
    section_key = _slug(sheet_name, len({item.section_key for item in blocks}) + 1)
    headers = cleaned[0]
    _append_block(blocks, "heading", sheet_name, section_key, sheet_name, 1, (sheet_name,),
                  metadata={"sheet": sheet_name})
    _append_block(blocks, "table", _table_text(cleaned), section_key, sheet_name, 1, (sheet_name,),
                  metadata={"sheet": sheet_name, "rows": len(cleaned)})
    for row_index, values in enumerate(cleaned[1:] or cleaned, start=1):
        pairs = [f"{headers[i] or f'列{i + 1}'}: {value}" for i, value in enumerate(values) if value]
        _append_block(blocks, "table_row", "；".join(pairs), section_key, sheet_name, 1, (sheet_name,),
                      metadata={"sheet": sheet_name, "row_index": row_index,
                                "headers": headers, "values": values})


def _parse_spreadsheet(path: Path) -> ParsedDocument:
    blocks: list[DocumentBlock] = []
    suffix = path.suffix.lower()
    if suffix == ".csv":
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|") if text.strip() else csv.excel
        except csv.Error:
            dialect = csv.excel
        _sheet_blocks(path.stem, list(csv.reader(text.splitlines(), dialect)), blocks)
        version = "csv-stdlib"
    elif suffix == ".xlsx":
        try:
            import openpyxl
        except ImportError as exc:
            raise RuntimeError("XLSX 解析依赖未安装，请执行 pip install -r requirements.txt") from exc
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for sheet in workbook.worksheets:
            _sheet_blocks(sheet.title, [list(row) for row in sheet.iter_rows(values_only=True)], blocks)
        version = f"openpyxl-{_package_version('openpyxl')}"
    else:
        try:
            import xlrd
        except ImportError as exc:
            raise RuntimeError("XLS 解析依赖未安装，请执行 pip install -r requirements.txt") from exc
        workbook = xlrd.open_workbook(path)
        for sheet in workbook.sheets():
            _sheet_blocks(sheet.name, [sheet.row_values(index) for index in range(sheet.nrows)], blocks)
        version = f"xlrd-{_package_version('xlrd')}"
    return _parsed(blocks, "spreadsheet", version)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _parsed(blocks: list[DocumentBlock], parser: str, version: str,
            extra: dict[str, Any] | None = None, *, allow_empty: bool = False) -> ParsedDocument:
    text_chars = sum(len(item.text) for item in blocks if item.block_type != "heading")
    quality = {
        "text_chars": text_chars,
        "block_count": len(blocks),
        "heading_count": sum(item.block_type == "heading" for item in blocks),
        "table_count": sum(item.block_type == "table" for item in blocks),
        "table_row_count": sum(item.block_type == "table_row" for item in blocks),
        "warnings": [],
        **(extra or {}),
    }
    if (not blocks or text_chars == 0) and not allow_empty:
        raise ValueError("解析完成但未提取到可入库文本")
    return ParsedDocument(blocks, parser, version, quality)


def parse_document(path: Path, parser: str) -> ParsedDocument:
    return {
        "native_docx": _parse_docx,
        "native_text": _parse_text,
        "plaintext_pdf": _parse_pdf,
        "docling": _parse_docling,
        "spreadsheet": _parse_spreadsheet,
    }[parser](path)


def blocks_to_sections(blocks: list[DocumentBlock]) -> list[Section]:
    grouped: dict[str, dict[str, Any]] = {}
    for block in blocks:
        if block.block_type in {"heading", "table_row"}:
            continue
        item = grouped.setdefault(block.section_key, {
            "title": block.section_title,
            "level": block.level,
            "parts": [],
            "pages": set(),
            "block_types": set(),
            "heading_path": list(block.heading_path),
            "media_assets": [],
        })
        item["parts"].append(block.text)
        item["block_types"].add(block.block_type)
        if block.page is not None:
            item["pages"].add(block.page)
        if block.block_type == "image" and block.metadata:
            item["media_assets"].append(dict(block.metadata))
    sections = []
    for key, item in grouped.items():
        content = "\n\n".join(item["parts"]).strip()
        if content:
            sections.append(Section(key, item["title"], item["level"], content, {
                "pages": sorted(item["pages"]),
                "block_types": sorted(item["block_types"]),
                "heading_path": item["heading_path"],
                "media_assets": item["media_assets"],
            }))
    return sections


def extract_docx_sections(path: Path) -> list[Section]:
    return blocks_to_sections(_parse_docx(path).blocks)


def extract_plain_sections(path: Path) -> list[Section]:
    parsed = _parse_pdf(path) if path.suffix.lower() == ".pdf" else _parse_text(path)
    return blocks_to_sections(parsed.blocks)


def extract_sections(path: Path) -> list[Section]:
    if path.suffix.lower() == ".docx":
        return extract_docx_sections(path)
    return extract_plain_sections(path)


def estimate_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    words = len(re.findall(r"[A-Za-z0-9_]+", text))
    other = len(re.findall(r"[^\s\u3400-\u9fffA-Za-z0-9_]", text))
    return max(1, cjk + math.ceil(words * 1.3) + math.ceil(other / 4))


def _segment_spans(text: str) -> list[tuple[int, int, str]]:
    """Return shared sentence spans without normalizing away source coordinates."""

    return split_sentences(text)


def _segments(text: str) -> list[str]:
    return [value.strip() for _, _, value in _segment_spans(text) if value.strip()]


def _token_pieces(text: str, target_tokens: int, max_tokens: int, overlap_percent: int) -> list[str]:
    segments: list[str] = []
    for segment in _segments(text):
        tokens = estimate_tokens(segment)
        if tokens <= max_tokens:
            segments.append(segment)
            continue
        chars_per_token = max(1.0, len(segment) / tokens)
        width = max(1, int(max_tokens * chars_per_token * 0.95))
        segments.extend(segment[start:start + width] for start in range(0, len(segment), width))
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for segment in segments:
        segment_tokens = estimate_tokens(segment)
        if current and current_tokens + segment_tokens > max_tokens:
            pieces.append("\n".join(current).strip())
            overlap_budget = max(0, int(target_tokens * overlap_percent / 100))
            overlap: list[str] = []
            overlap_tokens = 0
            for previous in reversed(current):
                amount = estimate_tokens(previous)
                if overlap and overlap_tokens + amount > overlap_budget:
                    break
                if amount <= overlap_budget:
                    overlap.insert(0, previous); overlap_tokens += amount
            current, current_tokens = overlap, overlap_tokens
        current.append(segment); current_tokens += segment_tokens
        if current_tokens >= target_tokens:
            pieces.append("\n".join(current).strip())
            current, current_tokens = [], 0
    if current:
        pieces.append("\n".join(current).strip())
    return [piece for piece in pieces if piece] or [""]


def _enforce_chunk_token_limit(chunks: list[TextChunk], max_tokens: int) -> list[TextChunk]:
    """Split any oversized strategy output while retaining its project metadata."""
    if max_tokens < 64:
        raise ValueError("单切片 Token 上限不能小于 64")
    bounded: list[TextChunk] = []
    parent_key_map: dict[str, str] = {}
    marker = re.compile(
        r"(?s)^(# .*?\n\n章节ID: .*?\n来源文档: .*?\n\n)分块:\s*\d+/\d+\n\n(.*)$"
    )
    for chunk in chunks:
        if chunk.token_count <= max_tokens:
            bounded.append(chunk)
            parent_key_map[chunk.key] = chunk.key
            continue

        match = marker.match(chunk.content)
        prefix, body = (match.group(1), match.group(2)) if match else ("", chunk.content)
        framing_tokens = estimate_tokens(prefix + "分块: 1/1\n\n") if prefix else 0
        payload_limit = max_tokens - framing_tokens
        if payload_limit >= 16:
            pieces = _token_pieces(body, payload_limit, payload_limit, 0)
            contents = [f"{prefix}分块: {index}/{len(pieces)}\n\n{piece}".strip()
                        for index, piece in enumerate(pieces, start=1)]
        else:
            contents = _token_pieces(chunk.content, max_tokens, max_tokens, 0)

        created: list[TextChunk] = []
        for index, content in enumerate(contents, start=1):
            # Estimation and splitting use the same tokenizer proxy, so this is a real hard bound.
            if estimate_tokens(content) > max_tokens:
                raise ValueError(f"知识块固定头信息已超过项目分块上限 {max_tokens}")
            key = f"{chunk.key}_s{index:03d}"
            created.append(TextChunk(
                key=key,
                section_key=chunk.section_key,
                title=chunk.title,
                part_index=index,
                part_total=len(contents),
                content=content,
                content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                token_count=estimate_tokens(content),
                parent_key=chunk.parent_key,
                metadata={**chunk.metadata, "token_limit_split": True},
            ))
        bounded.extend(created)
        parent_key_map[chunk.key] = created[0].key

    return [
        TextChunk(
            key=item.key,
            section_key=item.section_key,
            title=item.title,
            part_index=item.part_index,
            part_total=item.part_total,
            content=item.content,
            content_hash=item.content_hash,
            token_count=item.token_count,
            parent_key=parent_key_map.get(item.parent_key, item.parent_key),
            metadata=item.metadata,
        )
        for item in bounded
    ]


def _chunk(section: Section, source_name: str, body: str, index: int, total: int, strategy: str,
           *, key_suffix: str | None = None, parent_key: str | None = None,
           metadata: dict[str, Any] | None = None) -> TextChunk:
    prefix = f"# {section.title}\n\n章节ID: {section.key}\n来源文档: {source_name}\n\n"
    content = f"{prefix}分块: {index}/{total}\n\n{body}".strip()
    key = f"{section.key}_{key_suffix or f'p{index:03d}'}"
    data = {"strategy": strategy, "heading_path": section.metadata.get("heading_path", []),
            "pages": section.metadata.get("pages", []),
            "media_assets": section.metadata.get("media_assets", []), **(metadata or {})}
    return TextChunk(key, section.key, section.title, index, total, content,
                     hashlib.sha256(content.encode("utf-8")).hexdigest(), estimate_tokens(content),
                     parent_key, data)


def split_sections(sections: list[Section], *, max_chars: int, overlap_chars: int,
                   source_name: str) -> list[TextChunk]:
    """Compatibility strategy: preserve the original character-budget behavior."""
    chunks: list[TextChunk] = []
    for section in sections:
        prefix_size = len(f"# {section.title}\n\n章节ID: {section.key}\n来源文档: {source_name}\n\n")
        payload_limit = max(240, max_chars - prefix_size)
        pieces: list[str] = []
        current = ""
        for segment in _segments(section.content):
            if len(segment) > payload_limit:
                if current:
                    pieces.append(current); current = ""
                pieces.extend(segment[start:start + payload_limit] for start in range(0, len(segment), payload_limit))
                continue
            candidate = f"{current}\n{segment}".strip()
            if current and len(candidate) > payload_limit:
                pieces.append(current)
                current = f"{current[-overlap_chars:].strip()}\n{segment}".strip() if overlap_chars else segment
            else:
                current = candidate
        if current:
            pieces.append(current)
        pieces = pieces or [""]
        chunks.extend(_chunk(section, source_name, body, index, len(pieces), "default")
                      for index, body in enumerate(pieces, start=1))
    return chunks


def _general_chunks(sections: list[Section], source_name: str, *, target: int, maximum: int,
                    overlap: int, strategy: str = "general") -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for section in sections:
        pieces = _token_pieces(section.content, target, maximum, overlap)
        parent_key = None
        if len(pieces) > 1:
            overview = _token_pieces(section.content, min(220, target), min(320, maximum), 0)[0]
            parent = _chunk(section, source_name, overview, 0, len(pieces), strategy,
                            key_suffix="parent", metadata={"chunk_role": "parent",
                            "graph_extraction_enabled": False})
            chunks.append(parent); parent_key = parent.key
        for index, body in enumerate(pieces, start=1):
            chunks.append(_chunk(section, source_name, body, index, len(pieces), strategy,
                                 parent_key=parent_key, metadata={"chunk_role": "child"}))
    return chunks


def _paper_chunks(sections: list[Section], source_name: str) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for section in sections:
        title = section.title.strip()
        if _ABSTRACT_TITLE.match(title):
            chunks.append(_chunk(section, source_name, section.content, 1, 1, "paper",
                                 metadata={"paper_part": "abstract", "chunk_role": "child"}))
            continue
        pieces = _token_pieces(section.content, Config.paper_chunk_target_tokens,
                               Config.paper_chunk_max_tokens, Config.paper_chunk_overlap_percent)
        is_reference = bool(_REFERENCE_TITLE.match(title))
        for index, body in enumerate(pieces, start=1):
            chunks.append(_chunk(section, source_name, body, index, len(pieces), "paper",
                                 metadata={"paper_part": "references" if is_reference else "body",
                                           "chunk_role": "child",
                                           "graph_extraction_enabled": not is_reference}))
    return chunks


def _table_chunks(blocks: list[DocumentBlock], sections: list[Section], source_name: str) -> list[TextChunk]:
    section_map = {item.key: item for item in sections}
    rows = [block for block in blocks if block.block_type == "table_row"]
    if not rows:
        raise ValueError("表格分块策略未发现可用表格行，请改用通用策略或选择含表格的文档")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.section_key] = counts.get(row.section_key, 0) + 1
    positions: dict[str, int] = {}
    chunks: list[TextChunk] = []
    for row in rows:
        positions[row.section_key] = positions.get(row.section_key, 0) + 1
        section = section_map.get(row.section_key) or Section(row.section_key, row.section_title, row.level, row.text)
        metadata = {"chunk_role": "table_row", "table": row.metadata,
                    "graph_extraction_enabled": True}
        chunks.append(_chunk(section, source_name, row.text, positions[row.section_key],
                             counts[row.section_key], "table", key_suffix=f"row{row.order:04d}",
                             metadata=metadata))
    return chunks


def parse_document_structure(path: Path, *, requested_parser: str = "auto",
                             requested_strategy: str = "auto") -> tuple[ProcessingPlan, ParsedDocument]:
    """Route and parse a document without creating any retrieval chunks."""
    plan = route_processing(path, requested_parser, requested_strategy)
    try:
        parsed = parse_document(path, plan.effective_parser)
    except Exception:
        if not (plan.effective_parser == "plaintext_pdf" and requested_parser == "auto" and docling_available()):
            raise
        parsed = parse_document(path, "docling")
        plan = ProcessingPlan(plan.requested_parser, plan.requested_strategy, "docling",
                              plan.effective_strategy, plan.routing_reason + "；轻量 PDF 解析失败，回退 Docling",
                              plan.profile)
    if (plan.effective_parser == "plaintext_pdf" and requested_parser == "auto"
            and parsed.quality.get("text_chars", 0) < max(200, plan.profile.page_count * 40)
            and docling_available()):
        parsed = parse_document(path, "docling")
        plan = ProcessingPlan(plan.requested_parser, plan.requested_strategy, "docling",
                              plan.effective_strategy, plan.routing_reason + "；PDF 文本质量不足，回退 Docling",
                              plan.profile)
    return plan, parsed


def build_processing_result(path: Path, plan: ProcessingPlan, parsed: ParsedDocument, *,
                            source_name: str = "", chunk_max_tokens: int | None = None) -> ProcessingResult:
    """Build sections and chunks from an already persisted structure parse."""
    sections = blocks_to_sections(parsed.blocks)
    if not sections:
        raise ValueError("未提取到可入库章节")
    strategy = plan.effective_strategy
    if strategy == "default":
        chunks = split_sections(sections, max_chars=Config.chunk_max_chars,
                                overlap_chars=Config.chunk_overlap_chars, source_name=source_name or path.name)
    elif strategy == "general":
        chunks = _general_chunks(sections, source_name or path.name,
                                 target=Config.general_chunk_target_tokens,
                                 maximum=Config.general_chunk_max_tokens,
                                 overlap=Config.general_chunk_overlap_percent)
    elif strategy == "paper":
        chunks = _paper_chunks(sections, source_name or path.name)
    else:
        chunks = _table_chunks(parsed.blocks, sections, source_name or path.name)
    if chunk_max_tokens is not None:
        chunks = _enforce_chunk_token_limit(chunks, int(chunk_max_tokens))
    if not chunks:
        raise ValueError("分块完成但没有生成知识块")
    return ProcessingResult(plan, parsed, sections, chunks)


def process_document(path: Path, *, requested_parser: str = "auto", requested_strategy: str = "auto",
                     source_name: str = "", chunk_max_tokens: int | None = None) -> ProcessingResult:
    plan, parsed = parse_document_structure(
        path,
        requested_parser=requested_parser,
        requested_strategy=requested_strategy,
    )
    return build_processing_result(
        path,
        plan,
        parsed,
        source_name=source_name,
        chunk_max_tokens=chunk_max_tokens,
    )
