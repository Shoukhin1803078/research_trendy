#!/usr/bin/env python3
"""Render every ASCII-art fenced block of a chapter into a PNG image and build
an illustrated Markdown copy that references the generated images.

For each fenced code block in the source document:
  * blocks tagged with a programming language (e.g. ```python) are kept as code;
  * untagged blocks (the ASCII diagrams) are rendered to a high-resolution PNG
    under ./images/ and replaced by a Markdown image plus a figure caption.

Run with an interpreter that has Pillow available, e.g.:
    /opt/miniconda3/bin/python3 render_chapter6_blocks.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
SOURCE_MD = (
    BASE_DIR / "chapter_6_ai_driven_precision_irrigation_and_water_management.md"
)
OUTPUT_MD = (
    BASE_DIR
    / "chapter_6_ai_driven_precision_irrigation_and_water_management_illustrated.md"
)
IMAGE_DIR = BASE_DIR / "images"

FONT_SIZE = 30
LINE_GAP = 8
PADDING_X = 48
PADDING_Y = 40

BG_COLOR = (246, 248, 250)
BORDER_COLOR = (208, 215, 222)
FG_COLOR = (31, 35, 40)

# Curated captions for the diagrams in this chapter, keyed by diagram order.
# Any diagram not listed falls back to the nearest preceding heading.
CAPTION_OVERRIDES = {
    1: "Traditional Schedule-Based Irrigation vs. Precision AI Irrigation",
    2: "Soil-Plant-Atmosphere Continuum (SPAC) Water Potential Gradient",
    3: "Partitioning of Total Evapotranspiration into Soil Evaporation and Canopy Transpiration",
    4: "Crop Coefficient (Kc) Curve Across Phenological Growth Stages",
    5: "Soil Moisture Zones From Saturation to Permanent Wilting Point",
    6: "Multi-Source Data Ingestion Engine",
    7: "Multi-Depth In-Situ IoT Soil Probe Node",
    8: "Crop Water Stress Index (CWSI) Scale From No Stress to Maximum Stress",
    9: "AI Predictive Architecture: Physics-Informed and Temporal Models",
    10: "Physics-Informed Neural Network (PINN) Training Loop",
    11: "Closed-Loop Automated Irrigation Control Architecture",
    12: "Sample 4x4 Spatial Management-Zone VRI Prescription Map",
    13: "Nebraska Center-Pivot Maize Implementation Schematic",
    14: "Mendoza Vineyard Precision Drip Irrigation Schematic",
    15: "Challenges and Bottleneck Taxonomy",
    16: "Future Horizons: LEO Satellite IoT, Edge Quantum-AI, and Autonomous Robotics",
}

LANGUAGE_TAGS = {
    "python",
    "py",
    "bash",
    "sh",
    "json",
    "yaml",
    "yml",
    "js",
    "javascript",
    "ts",
    "typescript",
    "c",
    "cpp",
    "java",
    "sql",
    "r",
    "text",
    "txt",
    "console",
}


def find_monospace_font() -> Path:
    """Locate a monospace TTF, preferring the one bundled with matplotlib."""
    candidates: list[Path] = []
    try:
        import matplotlib

        candidates.append(
            Path(matplotlib.__file__).parent
            / "mpl-data/fonts/ttf/DejaVuSansMono.ttf"
        )
    except Exception:
        pass
    candidates += [
        Path("/System/Library/Fonts/Menlo.ttc"),
        Path("/System/Library/Fonts/SFNSMono.ttf"),
        Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSansMono.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "No monospace font found. Add one to find_monospace_font()."
    )


def normalize_fences(lines: list[str]) -> tuple[list[str], list[str]]:
    """Repair an unterminated diagram whose opening fence was corrupted.

    Returns the (possibly repaired) lines and a list of human-readable repair
    notes. The first diagram in the source document lost its opening ``` and
    instead shows the stray token 'lik' before the art.
    """
    lines = list(lines)
    fences = [i for i, line in enumerate(lines) if line.lstrip().startswith("```")]
    notes: list[str] = []
    if len(fences) % 2 == 0:
        return lines, notes

    first_fence = fences[0]
    start = first_fence
    j = first_fence - 1
    while j >= 0 and lines[j].strip() != "":
        start = j
        j -= 1

    if start < first_fence:
        head = lines[start].strip()
        # A genuine art row contains ruler/side glyphs; a mangled fence does not.
        if head and not any(ch in head for ch in "+|^<>"):
            lines[start] = "```"
            notes.append(
                f"line {start + 1}: replaced malformed fence token {head!r} with '```'"
            )
    return lines, notes


def parse_blocks(lines: list[str]):
    """Yield ('text', str) and ('block', (fence_line, info, body_lines)) items."""
    buffer: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            info = line.lstrip()[3:].strip()
            body: list[str] = []
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith("```"):
                body.append(lines[j])
                j += 1
            if j < len(lines):
                if buffer:
                    yield "text", "\n".join(buffer)
                    buffer = []
                yield "block", (line, info, body)
                i = j + 1
                continue
        buffer.append(line)
        i += 1
    if buffer:
        yield "text", "\n".join(buffer)


def render_art(body: list[str], out_path: Path, font: ImageFont.FreeTypeFont) -> None:
    """Draw a block of monospace ASCII art onto a light code-panel PNG."""
    art = [line.expandtabs(4).rstrip() for line in body]
    while art and not art[0].strip():
        art.pop(0)
    while art and not art[-1].strip():
        art.pop()
    if not art:
        return

    ascent, descent = font.getmetrics()
    line_height = ascent + descent + LINE_GAP
    char_width = font.getlength("M")

    max_cols = max((len(line) for line in art), default=1)
    width = int(char_width * max_cols) + 2 * PADDING_X
    height = line_height * len(art) + 2 * PADDING_Y

    image = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [0, 0, width - 1, height - 1], outline=BORDER_COLOR, width=2
    )

    y = float(PADDING_Y)
    for line in art:
        draw.text((PADDING_X, y), line, font=font, fill=FG_COLOR)
        y += line_height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, "PNG", optimize=True)


def slugify(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if len(slug) <= limit:
        return slug or "figure"
    truncated = slug[:limit]
    if "_" in truncated:
        truncated = truncated.rsplit("_", 1)[0]
    return truncated.rstrip("_") or slug[:limit]


def caption_from_heading(heading: str | None, fallback: str) -> str:
    if not heading:
        return fallback
    text = re.sub(r"^#+\s*", "", heading).strip()
    text = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", text).strip()
    return text or fallback


def main() -> int:
    if not SOURCE_MD.exists():
        raise SystemExit(f"Source document not found: {SOURCE_MD}")

    font_path = find_monospace_font()
    font = ImageFont.truetype(str(font_path), FONT_SIZE, index=0)

    raw = SOURCE_MD.read_text(encoding="utf-8")
    lines, repair_notes = normalize_fences(raw.split("\n"))

    if repair_notes:
        print("Repairs applied to source:")
        for note in repair_notes:
            print(f"  - {note}")

    output: list[str] = []
    current_heading: str | None = None
    figure_number = 0
    images_written: list[Path] = []

    if IMAGE_DIR.exists():
        for stale in IMAGE_DIR.glob("fig_*.png"):
            stale.unlink()

    for kind, payload in parse_blocks(lines):
        if kind == "text":
            segment = payload
            for line in segment.split("\n"):
                if line.lstrip().startswith("#"):
                    current_heading = line.strip()
            output.append(segment)
            continue

        _, info, body = payload
        if info.lower() in LANGUAGE_TAGS:
            fence = "```" + info
            output.append("\n".join([fence, *body, "```"]))
            continue

        figure_number += 1
        first_content = next(
            (line.strip() for line in body if line.strip()), ""
        )
        caption = CAPTION_OVERRIDES.get(figure_number) or caption_from_heading(
            current_heading, first_content or f"Diagram {figure_number}"
        )
        caption = re.sub(r"\s+", " ", caption).strip()
        filename = f"fig_{figure_number:02d}_{slugify(caption)}.png"
        out_path = IMAGE_DIR / filename
        render_art(body, out_path, font)
        images_written.append(out_path)

        output.append(
            f"![Figure {figure_number}. {caption}](images/{filename})"
        )
        output.append("")
        output.append(f"*Figure {figure_number}. {caption}.*")
        print(f"  rendered Figure {figure_number}: {filename}")

    OUTPUT_MD.write_text("\n".join(output), encoding="utf-8")

    print(f"\nFont used   : {font_path}")
    print(f"Images dir  : {IMAGE_DIR} ({len(images_written)} PNGs)")
    print(f"Output file : {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
