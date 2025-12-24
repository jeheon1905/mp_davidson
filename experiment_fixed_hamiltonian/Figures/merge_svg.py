#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_svg.py
Merge given SVG files into a grid (vector preserved).

핵심 변경사항
- 기본 스케일 방식을 'cover'로 변경 (셀을 꽉 채움)  → 패널 사이가 멀어 보이는 현상 해소
- 배치 정렬을 '좌상단 정렬'로 변경 (중앙정렬 제거)
- 기본적으로 셀 경계로 'clip' 적용 (넘치는 부분은 잘라냄)

옵션
- --contain  : 이전처럼 'contain'(전부 보이기) 스케일 사용
- --no-clip  : 클리핑 끄기 (디버그용)
- --png PNG_PATH [--dpi 300 | --png-width W --png-height H | --png-scale S]
  : SVG 저장 후 PNG도 함께 저장 (CairoSVG 우선, 미설치 시 Inkscape CLI 시도)
"""

import os
import argparse
import shutil
import subprocess
from typing import List, Tuple
from svgutils.transform import SVGFigure, fromfile
from lxml import etree

# ---------------------- helpers ---------------------- #

def parse_rc(text: str) -> Tuple[int, int]:
    try:
        r, c = text.split(',')
        return int(r), int(c)
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid cell spec '{text}'. Use 'row,col' (zero-based).")

def _parse_svg_size(value: str) -> float:
    if value is None:
        return 0.0
    s = str(value).strip().lower()
    num, unit = "", ""
    for ch in s:
        if ch.isdigit() or ch == ".":
            num += ch
        else:
            unit += ch
    if not num:
        return 0.0
    x = float(num)
    unit = unit.strip()
    if unit in ("", "px"):
        return x
    if unit == "in":
        return x * 96.0
    if unit == "cm":
        return x * (96.0 / 2.54)
    if unit == "mm":
        return x * (96.0 / 25.4)
    if unit == "pt":  # 1pt = 1/72in
        return x * (96.0 / 72.0)
    if unit == "pc":  # 1pc = 12pt
        return x * (96.0 / 6.0)
    return x

def _get_src_size(fig, root) -> Tuple[float, float]:
    """width/height → fallback: viewBox."""
    w = _parse_svg_size(fig.width)
    h = _parse_svg_size(fig.height)
    if w > 0 and h > 0:
        return w, h
    vb = root.get("viewBox")
    if vb:
        parts = [float(x) for x in vb.strip().split()]
        if len(parts) == 4:
            _, _, vw, vh = parts
            if vw > 0 and vh > 0:
                return vw, vh
    return 0.0, 0.0

def build_canvas(rows: int, cols: int,
                 cell_w: int, cell_h: int,
                 gutter_x: int, gutter_y: int,
                 margin_l: int, margin_t: int,
                 margin_r: int, margin_b: int) -> Tuple[SVGFigure, int, int]:
    total_w = margin_l + margin_r + cols * cell_w + (cols - 1) * gutter_x
    total_h = margin_t + margin_b + rows * cell_h + (rows - 1) * gutter_y
    canvas = SVGFigure(f"{total_w}px", f"{total_h}px")
    return canvas, total_w, total_h

def _ensure_defs(fig_elem):
    nsmap = fig_elem.nsmap
    defs = fig_elem.find("{http://www.w3.org/2000/svg}defs")
    if defs is None:
        defs = etree.Element("{http://www.w3.org/2000/svg}defs", nsmap=nsmap)
        fig_elem.insert(0, defs)
    return defs

def _make_clip(defs, clip_id: str, x: float, y: float, w: float, h: float):
    ns = "{http://www.w3.org/2000/svg}"
    clipPath = etree.SubElement(defs, f"{ns}clipPath", id=clip_id)
    etree.SubElement(clipPath, f"{ns}rect", x=str(x), y=str(y), width=str(w), height=str(h))
    return f"url(#{clip_id})"

# ---------------------- core ---------------------- #

def place_svgs_on_canvas(svg_paths: List[str],
                         rows: int, cols: int,
                         empty_cells: List[Tuple[int, int]],
                         cell_w: int, cell_h: int,
                         gutter_x: int, gutter_y: int,
                         margin_l: int, margin_t: int,
                         use_contain: bool,
                         do_clip: bool,
                         canvas: SVGFigure) -> List:
    # grid positions excluding empty cells (row-major)
    positions: List[Tuple[int, int]] = []
    empty_set = set(empty_cells)
    for r in range(rows):
        for c in range(cols):
            if (r, c) in empty_set:
                continue
            positions.append((r, c))

    max_slots = len(positions)
    if len(svg_paths) > max_slots:
        print(f"[WARN] {len(svg_paths)} inputs but only {max_slots} slots; extra files will be ignored.")
        svg_paths = svg_paths[:max_slots]

    elements = []
    canvas_root = canvas.root
    defs = _ensure_defs(canvas_root) if do_clip else None
    ns = "{http://www.w3.org/2000/svg}"

    for path, (r, c) in zip(svg_paths, positions):
        fig = fromfile(path)
        root = fig.getroot()

        src_w, src_h = _get_src_size(fig, root)
        if src_w <= 0 or src_h <= 0:
            src_w, src_h = float(cell_w), float(cell_h)

        # --- scale: cover (기본) vs contain ---
        if use_contain:
            scale = min(cell_w / src_w, cell_h / src_h)
        else:
            scale = max(cell_w / src_w, cell_h / src_h)

        # --- 셀 좌상단 정렬(중앙정렬 제거) ---
        x0 = margin_l + c * (cell_w + gutter_x)   # gutter가 '셀 간 거리'
        y0 = margin_t + r * (cell_h + gutter_y)

        x_offset = x0
        y_offset = y0

        # 이동/스케일 적용
        root.moveto(x_offset, y_offset, scale_x=scale, scale_y=scale)

        if do_clip:
            g = etree.Element(f"{ns}g")
            clip_id = f"clip_{r}_{c}"
            clip_url = _make_clip(defs, clip_id, x0, y0, cell_w, cell_h)
            g.set("clip-path", clip_url)
            g.append(root.root)  # GroupElement 내부 <g>
            elements.append(g)
        else:
            elements.append(root)

    return elements

# ---------------------- png export ---------------------- #

def export_png(svg_path: str, png_path: str,
               dpi: int = None, png_w: int = None, png_h: int = None, scale: float = None):
    """
    Try CairoSVG first (pure Python). If unavailable, try Inkscape CLI.
    Priority of sizing: png_w/png_h > scale > dpi.
    """
    # 1) CairoSVG
    try:
        import cairosvg
        kwargs = {"url": svg_path, "write_to": png_path}
        if png_w is not None:
            kwargs["output_width"] = int(png_w)
        if png_h is not None:
            kwargs["output_height"] = int(png_h)
        if scale is not None:
            kwargs["scale"] = float(scale)
        if dpi is not None and png_w is None and png_h is None and scale is None:
            kwargs["dpi"] = int(dpi)
        cairosvg.svg2png(**kwargs)
        print(f"[OK] PNG saved via CairoSVG: {png_path}")
        return
    except ImportError:
        pass  # Fall back to Inkscape
    except Exception as e:
        print(f"[WARN] CairoSVG failed: {e}. Falling back to Inkscape if available...")

    # 2) Inkscape CLI
    inkscape = shutil.which("inkscape")
    if inkscape is None:
        raise RuntimeError(
            "PNG export failed: CairoSVG not installed and Inkscape not found.\n"
            "Install one of the following:\n"
            "  pip install cairosvg\n"
            "  or install Inkscape and ensure 'inkscape' is in PATH."
        )

    cmd = [inkscape, svg_path, "--export-type=png", f"--export-filename={png_path}"]
    # Inkscape sizing flags
    if png_w is not None and png_h is not None:
        cmd += [f"--export-width={int(png_w)}", f"--export-height={int(png_h)}"]
    elif png_w is not None:
        cmd += [f"--export-width={int(png_w)}"]
    elif png_h is not None:
        cmd += [f"--export-height={int(png_h)}"]
    elif dpi is not None:
        cmd += [f"--export-dpi={int(dpi)}"]
    elif scale is not None:
        # Inkscape는 scale 인자를 직접 받지 않음 → width/height나 dpi로 대체 권장
        pass

    subprocess.run(cmd, check=True)
    print(f"[OK] PNG saved via Inkscape: {png_path}")

# ---------------------- main ---------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Merge given SVG files into a grid (vector preserved)."
    )
    parser.add_argument("files", nargs="+", help="SVG files to merge (order matters)")
    parser.add_argument("--rows", type=int, default=3, help="Grid rows (default 3)")
    parser.add_argument("--cols", type=int, default=8, help="Grid cols (default 8)")
    parser.add_argument("--empty", type=parse_rc, action="append", default=None,
                        help="Empty cells as 'row,col'. Can repeat. Default: 1,0 and 2,1 for 3x8 grid, none otherwise")
    parser.add_argument("--cell-size", type=int, nargs=2, metavar=("W", "H"),
                        default=[400, 300], help="Cell size in px (default 400 300)")
    parser.add_argument("--gutter", type=int, nargs=2, metavar=("GX", "GY"),
                        default=[5, 5], help="Gutter (default 5 5)")
    parser.add_argument("--margin", type=int, nargs=4, metavar=("L", "T", "R", "B"),
                        default=[0, 0, 0, 0], help="Margins (default 0 0 0 0)")
    parser.add_argument("--output", default="merged.svg",
                        help="Output SVG filename (default merged.svg)")

    # contain/cover 선택: 기본은 cover
    parser.add_argument("--contain", action="store_true",
                        help="Use 'contain' fit instead of default 'cover' (may look spaced).")

    # clip 제어: 기본 켬
    parser.add_argument("--no-clip", action="store_true",
                        help="Disable clipping to cell bounds (default: clip enabled).")

    # PNG 옵션
    parser.add_argument("--png", help="Also export a PNG to this path (e.g., merged.png)")
    parser.add_argument("--dpi", type=int, default=None,
                        help="PNG DPI (used when width/height/scale are not specified)")
    parser.add_argument("--png-width", type=int, default=None,
                        help="PNG output width in pixels")
    parser.add_argument("--png-height", type=int, default=None,
                        help="PNG output height in pixels")
    parser.add_argument("--png-scale", type=float, default=None,
                        help="Scale factor for PNG (CairoSVG only).")

    args = parser.parse_args()

    # Handle empty cells default
    if args.empty is None:
        args.empty = []

    rows, cols = args.rows, args.cols
    for r, c in args.empty:
        if not (0 <= r < rows and 0 <= c < cols):
            raise SystemExit(f"Empty cell {r,c} out of range for grid {rows}x{cols}.")

    cell_w, cell_h = args.cell_size
    gutter_x, gutter_y = args.gutter
    margin_l, margin_t, margin_r, margin_b = args.margin

    canvas, total_w, total_h = build_canvas(
        rows, cols, cell_w, cell_h, gutter_x, gutter_y,
        margin_l, margin_t, margin_r, margin_b
    )

    elements = place_svgs_on_canvas(
        args.files, rows, cols, args.empty,
        cell_w, cell_h, gutter_x, gutter_y,
        margin_l, margin_t,
        use_contain=args.contain,
        do_clip=(not args.no_clip),
        canvas=canvas
    )

    canvas.append(elements)
    canvas.save(args.output)

    slots = rows * cols - len(args.empty)
    mode = "contain" if args.contain else "cover"
    print(f"[OK] Saved {args.output}")
    print(f"     inputs: {len(args.files)} / slots: {slots}, "
          f"canvas: {int(total_w)}×{int(total_h)} px, grid {rows}×{cols}, empty: {args.empty}, "
          f"fit={mode}, clip={not args.no_clip}")

    # Optional PNG export
    if args.png:
        try:
            export_png(
                svg_path=args.output,
                png_path=args.png,
                dpi=args.dpi,
                png_w=args.png_width,
                png_h=args.png_height,
                scale=args.png_scale
            )
        except Exception as e:
            raise SystemExit(f"[ERROR] PNG export failed: {e}")

if __name__ == "__main__":
    main()
