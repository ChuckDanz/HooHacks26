from PIL import Image, ImageDraw, ImageFont


FIT_LABELS = {
    "tight":     ("Tight Fit",    (220, 53,  69)),   # red
    "fitted":    ("True to Size", (40,  167, 69)),   # green
    "loose":     ("Loose Fit",    (13,  110, 253)),   # blue
    "oversized": ("Oversized",    (111, 66,  193)),   # purple
}


def add_fit_badge(img: Image.Image, size_style: str) -> Image.Image:
    """Paste a colored fit label pill onto the bottom of the result image."""
    label, color = FIT_LABELS.get(size_style, ("Unknown", (128, 128, 128)))
    img = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    W, H = img.size
    pill_w, pill_h = 220, 44
    x0 = (W - pill_w) // 2
    y0 = H - pill_h - 16
    x1, y1 = x0 + pill_w, y0 + pill_h
    r, g, b = color

    # Semi-transparent pill background
    draw.rounded_rectangle([x0, y0, x1, y1], radius=22, fill=(r, g, b, 210))

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # Center text
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = x0 + (pill_w - tw) // 2
    ty = y0 + (pill_h - th) // 2
    draw.text((tx, ty), label, font=font, fill=(255, 255, 255, 255))

    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


def make_comparison_grid(
    results: dict,       # {size_style_str: PIL Image}
    person: Image.Image,
    garment: Image.Image,
    cell_w: int = 300,
    cell_h: int = 400,
) -> Image.Image:
    """
    Layout:
      [Person] [Garment] | [Tight] [Fitted] [Loose]
    with size labels beneath each result column.
    """
    padding = 12
    label_h = 32
    divider = 4

    size_order = ["tight", "fitted", "loose", "oversized"]
    size_keys = [s for s in size_order if s in results]

    n_results = len(size_keys)
    n_cols = 2 + n_results
    total_w = n_cols * cell_w + (n_cols + 1) * padding + divider
    total_h = cell_h + label_h + padding * 2

    grid = Image.new("RGB", (total_w, total_h), (240, 240, 240))
    draw = ImageDraw.Draw(grid)

    try:
        label_font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        label_font = ImageFont.load_default()

    def paste_cell(img, col, label=None, label_color=(80, 80, 80)):
        x = padding + col * (cell_w + padding)
        y = padding
        thumb = img.resize((cell_w, cell_h), Image.LANCZOS)
        grid.paste(thumb, (x, y))
        if label:
            lx = x + (cell_w - draw.textlength(label, font=label_font)) // 2
            ly = y + cell_h + 6
            draw.text((lx, ly), label, font=label_font, fill=label_color)

    paste_cell(person,  col=0, label="Person")
    paste_cell(garment, col=1, label="Garment")

    # Divider line
    div_x = padding * 2 + 2 * (cell_w + padding) - padding // 2
    draw.rectangle([div_x, padding, div_x + divider, total_h - padding], fill=(180, 180, 180))

    for i, key in enumerate(size_keys):
        _, color = FIT_LABELS.get(key, ("", (80, 80, 80)))
        label_text = FIT_LABELS[key][0]
        paste_cell(results[key], col=2 + i, label=label_text, label_color=color)

    return grid
