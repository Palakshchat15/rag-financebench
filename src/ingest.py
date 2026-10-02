"""Parse PDFs page by page with pypdfium2 (Apache-2.0/BSD). Pages are 0-indexed, as in FinanceBench."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium


def parse_pdf(pdf_path: Path) -> list[dict]:
    doc_name = pdf_path.stem
    pdf = pdfium.PdfDocument(str(pdf_path))
    rows = []
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            tp = page.get_textpage()
            text = tp.get_text_bounded()
            tp.close()
            page.close()
            # Normalise line endings; keep layout newlines (tables rely on them).
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            rows.append({"doc_name": doc_name, "page": i, "text": text})
    finally:
        pdf.close()
    return rows


def parse_all(pdf_dir: Path, out_path: Path) -> pd.DataFrame:
    rows = []
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    for n, p in enumerate(pdfs, 1):
        r = parse_pdf(p)
        rows.extend(r)
        print(f"[{n}/{len(pdfs)}] {p.stem}: {len(r)} pages", flush=True)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def load_pages(pages_path: Path) -> pd.DataFrame:
    return pd.read_parquet(pages_path)


def render_page_png(pdf_path: Path, page: int, scale: float = 1.4) -> bytes:
    """Render one PDF page to PNG bytes (used by the app's citation viewer)."""
    import io

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        img = pdf[page].render(scale=scale).to_pil()
    finally:
        pdf.close()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
