"""Convert: Office files -> PDF (LibreOffice); images -> PDF; PDFs stay as they are.
One reader then handles every format."""
import subprocess
from pathlib import Path

from rag.config import settings

OFFICE = {".doc", ".docx", ".ppt", ".pptx", ".rtf", ".odt", ".odp", ".xls", ".xlsx"}
IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED = OFFICE | IMAGES | {".pdf"}


def to_pdf(path: Path, out_dir: Path) -> Path:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return path
    if ext in IMAGES:   # scans / photos: one page per image frame
        from PIL import Image, ImageSequence
        img = Image.open(path)
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(img)]
        out = out_dir / (path.stem + ".pdf")
        frames[0].save(out, save_all=True, append_images=frames[1:], resolution=200)
        return out
    if ext in OFFICE:
        subprocess.run([settings.soffice_path, "--headless", "--convert-to", "pdf", "--outdir",
                        str(out_dir), str(path)], check=True, timeout=900, capture_output=True)
        out = out_dir / (path.stem + ".pdf")
        if not out.exists():
            raise RuntimeError("LibreOffice produced no PDF")
        return out
    raise ValueError(f"Unsupported file type: {ext}")
