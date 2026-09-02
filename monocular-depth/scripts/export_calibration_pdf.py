"""Export the existing binary ChArUco pattern as an exact-size, vector A4 PDF.

Run with the bundled document Python (ReportLab, Pillow, pypdf), not the ML environment.
This does not regenerate or change the board's marker IDs or physical specification.
"""
from __future__ import annotations

import argparse
import json
from itertools import groupby
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


def export(project: Path, output: Path, font: Path):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    board_dir = project / "calibration" / "board"
    spec = json.loads((board_dir / "board.json").read_text(encoding="utf-8"))
    expected = {
        "squares_x": 7, "squares_y": 5, "square_length_m": 0.025,
        "marker_length_m": 0.018, "dictionary": "DICT_4X4_50", "legacy_pattern": False,
    }
    if spec != expected:
        raise ValueError("This print layout requires the existing 7x5, 25 mm board specification")
    with Image.open(board_dir / "board.png") as source:
        gray = source.convert("L")
        width, height = gray.size
        pixels = gray.tobytes()
    if (width, height) != (1400, 1000) or not set(pixels).issubset({0, 255}):
        raise ValueError("Expected the original 1400x1000 binary board; no resampling allowed")

    pdfmetrics.registerFont(TTFont("Chinese", str(font)))
    output.parent.mkdir(parents=True, exist_ok=True)
    page = Canvas(str(output), pagesize=A4, pageCompression=1)
    page.setTitle("ChArUco calibration board - A4 - 25 mm squares")
    page.setAuthor("Monocular Depth Lab")
    page.setSubject("Print at actual size / 100%; board 175 x 125 mm; ruler 100 mm")
    page.setViewerPreference("PrintScaling", "None")
    page.setViewerPreference("Duplex", "Simplex")
    page_height = A4[1]

    def text(x, top, value, size=10, font_name="Chinese"):
        page.setFont(font_name, size)
        page.drawString(x * mm, page_height - top * mm, value)

    text(17.5, 21, "ChArUco 相机标定板", 20)
    text(17.5, 30, "A4 纵向  |  单面打印  |  实际大小 / 100%", 11)
    text(17.5, 39, "关闭“适应页面”和缩放；不要截图后打印。", 10)
    text(17.5, 47, "打印后尺量：每格 25 mm，整板 175 x 125 mm。", 10)

    # Convert runs of identical binary rows into exact vector rectangles.
    # The original pixels map to 0.125 mm each; marker geometry is unchanged.
    left, top, physical_w, physical_h = 17.5, 60.0, 175.0, 125.0
    rows = [pixels[row * width:(row + 1) * width] for row in range(height)]
    row_start = 0
    rectangles = 0
    page.setFillColorRGB(0, 0, 0)
    for row, equal_rows in groupby(rows):
        count = sum(1 for _ in equal_rows)
        column = 0
        for shade, run in groupby(row):
            run_width = sum(1 for _ in run)
            if shade == 0:
                x = left + column * physical_w / width
                y = 297 - top - (row_start + count) * physical_h / height
                page.rect(x * mm, y * mm, run_width * physical_w / width * mm,
                          count * physical_h / height * mm, stroke=0, fill=1)
                rectangles += 1
            column += run_width
        row_start += count

    text(17.5, 197, "7 x 5 squares  |  DICT_4X4_50  |  marker 18 mm  |  legacy=false",
         9, "Helvetica")
    text(17.5, 209, "打印尺寸校验尺：两端刻线之间应为 100 mm", 10)
    ruler_y = page_height - 217 * mm
    page.setLineWidth(.3 * mm)
    page.line(17.5 * mm, ruler_y, 117.5 * mm, ruler_y)
    for tick in range(11):
        x = (17.5 + tick * 10) * mm
        size = 2.5 if tick % 5 == 0 else 1.5
        page.line(x, ruler_y - size * mm, x, ruler_y + size * mm)
    text(17.5, 226, "0", 9, "Helvetica")
    text(65.5, 226, "50", 9, "Helvetica")
    text(112, 226, "100 mm", 9, "Helvetica")
    text(17.5, 242, "贴在平整硬板上，保留棋盘四周白边，勿折弯或覆反光膜。", 10)
    text(17.5, 251, "拍摄时保持清晰、无反光；变换距离、位置及左右/上下倾角。", 10)
    text(17.5, 260, "S 保存一张，Q 退出；建议拍摄 30 张不同姿态。", 10)
    text(17.5, 278, "本页仅用于相机标定；不代表深度模型的距离精度已验证。", 9)
    page.showPage()
    page.save()

    reader = PdfReader(output)
    assert len(reader.pages) == 1
    dimensions = [float(reader.pages[0].mediabox.width) / mm,
                  float(reader.pages[0].mediabox.height) / mm]
    assert all(abs(a - b) < 0.001 for a, b in zip(dimensions, (210, 297), strict=True))
    assert reader.trailer["/Root"]["/ViewerPreferences"]["/PrintScaling"] == "/None"
    assert "25 mm" in reader.pages[0].extract_text()
    print(json.dumps({"pdf": str(output), "page_mm": dimensions,
                      "board_mm": [physical_w, physical_h], "square_mm": 25,
                      "ruler_mm": 100, "vector_rectangles": rectangles,
                      "print_scaling_preference": "None"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/simhei.ttf"))
    args = parser.parse_args()
    export(args.project, args.output or args.project / "output/pdf/charuco-calibration-A4.pdf",
           args.font)
