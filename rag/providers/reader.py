"""Read (OCR / layout): PDF -> Markdown + cropped figures.

  docling   on-prem (profile C). Open source; first run downloads its models, then works offline
            (set HF_HUB_OFFLINE=1 after the models are installed).
  azure_di  Azure Document Intelligence prebuilt-layout (profiles A and B).

Both return the same Markdown conventions, so the chunker is shared:
  <!-- PageNumber=3 -->  page markers
  ## Heading             headings
  <table>...</table>    tables (HTML, one line)
  [[FIGURE:7]]          where a figure sat
"""
import io
from dataclasses import dataclass, field
from functools import lru_cache

from rag.config import settings


@dataclass
class Figure:
    id: str
    page: int
    caption: str
    image_png: bytes


@dataclass
class ParsedDoc:
    markdown: str
    figures: list[Figure] = field(default_factory=list)
    pages: int = 0


# ---------------- Docling (profile C) ----------------
def _ocr_needed(pdf_path) -> bool:
    if settings.ocr_mode == "on":
        return True
    if settings.ocr_mode == "off":
        return False
    import pypdfium2 as pdfium   # installed with Docling
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        with_text = sum(1 for page in pdf if len(page.get_textpage().get_text_range().strip()) > 50)
        return with_text < 0.8 * len(pdf)       # mostly scanned -> OCR
    finally:
        pdf.close()


@lru_cache
def _docling_converter(ocr: bool):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    opts = PdfPipelineOptions()
    opts.do_table_structure = True
    opts.do_ocr = ocr
    opts.generate_picture_images = True
    opts.images_scale = 2.0
    try:
        import os
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        opts.accelerator_options = AcceleratorOptions(num_threads=os.cpu_count() or 4,
                                                      device=AcceleratorDevice.AUTO)  # GPU if present
    except ImportError:
        pass
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def _read_docling(pdf_path) -> ParsedDoc:
    from docling_core.types.doc import (DocItemLabel, PictureItem, SectionHeaderItem, TableItem,
                                        TextItem, TitleItem)
    doc = _docling_converter(_ocr_needed(pdf_path)).convert(str(pdf_path)).document
    lines, figures, page = [], [], None
    caption_refs = {c.cref for item in list(doc.tables) + list(doc.pictures) for c in item.captions}
    for item, _level in doc.iterate_items():   # body only: page headers/footers left out
        if getattr(item, "self_ref", None) in caption_refs:
            continue
        p = item.prov[0].page_no if getattr(item, "prov", None) else page
        if p and p != page:
            page = p
            lines += [f"<!-- PageNumber={page} -->", ""]
        if isinstance(item, TitleItem):
            lines += [f"# {item.text}", ""]
        elif isinstance(item, SectionHeaderItem):
            lines += ["#" * min(item.level + 1, 6) + f" {item.text}", ""]
        elif isinstance(item, TableItem):
            lines += [item.export_to_html(doc=doc, add_caption=True).replace("\n", ""), ""]
        elif isinstance(item, PictureItem):
            img = item.get_image(doc)
            if img is None:
                continue
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            fid = str(len(figures) + 1)
            figures.append(Figure(fid, page or 0, item.caption_text(doc) or "", buf.getvalue()))
            lines += [f"[[FIGURE:{fid}]]", ""]
        elif isinstance(item, TextItem):
            if item.label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER) or not item.text.strip():
                continue
            lines += [("- " if item.label == DocItemLabel.LIST_ITEM else "") + item.text.strip(), ""]
    return ParsedDoc("\n".join(lines), figures, pages=len(doc.pages))


# ---------------- Azure Document Intelligence (profiles A/B) ----------------
def _read_azure(pdf_path) -> ParsedDoc:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeOutputOption, DocumentContentFormat
    from azure.identity import DefaultAzureCredential
    client = DocumentIntelligenceClient(settings.docintel_endpoint, DefaultAzureCredential())
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout", body=f.read(),
            output_content_format=DocumentContentFormat.MARKDOWN,
            output=[AnalyzeOutputOption.FIGURES])
    result = poller.result()
    result_id = poller.details["operation_id"]
    md, figures = result.content, []
    # replace each figure's span with a placeholder (from the end so offsets stay valid)
    for f in sorted(result.figures or [], key=lambda f: f.spans[0].offset, reverse=True):
        png = b"".join(client.get_analyze_result_figure(model_id=result.model_id,
                                                        result_id=result_id, figure_id=f.id))
        fid = f.id.replace(".", "_")
        figures.append(Figure(fid, f.bounding_regions[0].page_number if f.bounding_regions else 0,
                              f.caption.content if f.caption else "", png))
        s = f.spans[0]
        md = md[: s.offset] + f"\n[[FIGURE:{fid}]]\n" + md[s.offset + s.length:]
    figures.reverse()
    return ParsedDoc(md, figures, pages=len(result.pages or []))


def read_pdf(pdf_path) -> ParsedDoc:
    if settings.reader == "docling":
        return _read_docling(pdf_path)
    if settings.reader == "azure_di":
        return _read_azure(pdf_path)
    raise ValueError(f"Unknown READER {settings.reader}")
