import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries
from scipy.spatial import cKDTree
import io
import base64
import json
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Cell Viability Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size:2rem; font-weight:700; color:#1a7a4a; margin-bottom:0.2rem; }
    .sub-header  { font-size:0.95rem; color:#555; margin-bottom:1.5rem; }
    .section-title {
        font-size:1.1rem; font-weight:600; color:#1a7a4a;
        border-bottom:2px solid #c8e6d0; padding-bottom:4px; margin:1.2rem 0 0.8rem 0;
    }
    .section-title-red {
        font-size:1.1rem; font-weight:600; color:#b03030;
        border-bottom:2px solid #f5c0c0; padding-bottom:4px; margin:1.2rem 0 0.8rem 0;
    }
    .section-title-green {
        font-size:1.1rem; font-weight:600; color:#1a7a4a;
        border-bottom:2px solid #c8e6d0; padding-bottom:4px; margin:1.2rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────

def load_image(uploaded_file):
    return np.array(Image.open(uploaded_file))


def extract_channel(img_array, channel):
    if img_array.ndim == 2:
        return img_array.astype(np.float32)
    cmap = {'red': 0, 'green': 1, 'blue': 2}
    if channel in cmap and img_array.shape[2] > cmap[channel]:
        return img_array[:, :, cmap[channel]].astype(np.float32)
    nc = min(img_array.shape[2], 3)
    return img_array[:, :, :nc].mean(axis=2).astype(np.float32)


def get_mi(region):
    try:
        return region.intensity_mean
    except AttributeError:
        return region.mean_intensity


def segment_cells(cell_channel, min_area, min_circ, max_circ,
                  min_intensity, max_intensity=255, threshold_method='otsu', manual_threshold=None):
    thresh = manual_threshold if (threshold_method == 'manual' and manual_threshold is not None) \
             else threshold_otsu(cell_channel)
    binary = cell_channel > thresh
    binary = remove_small_objects(binary, max_size=max(5, min_area // 2 - 1))
    labels = label(binary)
    valid = []
    # cache_active=True (default) lazily computes only the properties we access below
    props = regionprops(labels, intensity_image=cell_channel, cache=True)
    for region in props:
        area  = region.area
        perim = region.perimeter
        circ  = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
        try:
            mi = region.intensity_mean
        except AttributeError:
            mi = region.mean_intensity
        if (area >= min_area and min_circ <= circ <= max_circ
                and min_intensity <= mi <= max_intensity):
            valid.append((region, circ))
    return labels, thresh, valid


def build_overlay(img_array, labels, valid_cells, outline_color):
    """
    Fast single-pass overlay: draw all cell boundaries in one find_boundaries call.
    For 1000 cells this is ~665x faster than the per-cell loop.
    """
    if img_array.ndim == 2:
        disp = np.stack([img_array] * 3, axis=-1)
    else:
        disp = img_array[:, :, :3].copy()
    if disp.dtype != np.uint8:
        pmax = disp.max()
        disp = ((disp / pmax) * 255).astype(np.uint8) if pmax > 0 else disp.astype(np.uint8)

    color_map = {'red': [255, 50, 50], 'green': [50, 255, 50]}
    color = color_map.get(outline_color, [255, 255, 0])

    # Build a validity mask: only label pixels that belong to valid cells
    valid_label_ids = {region.label for region, _ in valid_cells}
    valid_mask = np.isin(labels, list(valid_label_ids))

    # Zero out invalid labels so find_boundaries only traces valid cells
    labels_valid = np.where(valid_mask, labels, 0)

    # Single call — O(H×W) regardless of cell count
    outline = find_boundaries(labels_valid, mode='outer')
    disp[outline] = color
    return disp


def spatial_distribution_score(centroids, img_shape):
    """
    Clark-Evans index using cKDTree for O(n log n) nearest-neighbour queries.
    Old O(n²) loop took ~2s for 1000 cells; this takes <2ms.
    """
    n = len(centroids)
    if n < 2:
        return 0.0
    pts = np.array(centroids)
    if n == 2:
        d = np.linalg.norm(pts[0] - pts[1])
        return round(min(1.0, d / np.hypot(img_shape[0], img_shape[1]) * 2), 4)
    area        = img_shape[0] * img_shape[1]
    density     = n / area
    expected_nn = 1.0 / (2 * np.sqrt(density))
    # k=2 because index 0 is the point itself (distance 0)
    tree        = cKDTree(pts)
    dists, _    = tree.query(pts, k=2, workers=-1)
    R           = dists[:, 1].mean() / expected_nn
    return round(float(np.clip(R / 2.15, 0, 1)), 4)


def build_results_df(valid_cells, use_microns=False, scale_px_per_um=None):
    px_to_um2 = (1.0 / scale_px_per_um ** 2) if (use_microns and scale_px_per_um) else None
    area_col  = "Area (µm²)" if px_to_um2 else "Area (px²)"
    rows = []
    for idx, (region, circ) in enumerate(valid_cells, 1):
        area_val = region.area * px_to_um2 if px_to_um2 else int(region.area)
        rows.append({
            "Cell #":         idx,
            area_col:         round(float(area_val), 4) if px_to_um2 else int(area_val),
            "Mean Intensity": round(float(get_mi(region)), 2),
            "Circularity":    round(circ, 4),
            "Centroid X":     round(region.centroid[1], 2),
            "Centroid Y":     round(region.centroid[0], 2),
        })
    return pd.DataFrame(rows), area_col


def build_channel_summary(df, area_col, cell_channel, valid_cells, labels, img_shape,
                          use_microns=False, scale_px_per_um=None):
    if df.empty:
        return pd.DataFrame()
    total_px     = img_shape[0] * img_shape[1]
    area_px_vals = df[area_col] * (scale_px_per_um ** 2) if (use_microns and scale_px_per_um) else df[area_col]
    pct_area     = round(100.0 * area_px_vals.sum() / total_px, 4)
    # Fast: labels>0 covers all valid pixels in O(H×W) — no per-cell loop needed
    cell_mask    = np.isin(labels, [r.label for r, _ in valid_cells])
    bg_intensity = float(cell_channel[~cell_mask].mean())
    centroids    = [(r.centroid[0], r.centroid[1]) for r, _ in valid_cells]
    spatial      = spatial_distribution_score(centroids, img_shape)
    area_label   = "Average Area (µm²)" if (use_microns and scale_px_per_um) else "Average Area (px²)"
    return pd.DataFrame([{
        "Total Cells Detected":         len(df),
        area_label:                     round(float(df[area_col].mean()), 4),
        "Average Circularity":          round(df["Circularity"].mean(), 4),
        "Average Cell Intensity":       round(df["Mean Intensity"].mean(), 2),
        "Average Background Intensity": round(bg_intensity, 2),
        "% Area Occupied by Cells":     pct_area,
        "Spatial Distribution (0–1)":   spatial,
    }])


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf.read()


def img_array_to_b64(arr):
    """
    Convert numpy uint8 RGB array to base64 for HTML embedding.
    Uses JPEG (quality=82) for large images — 5-10x smaller than PNG,
    fast to encode, and visually indistinguishable for cell images.
    """
    pil  = Image.fromarray(arr.astype(np.uint8))
    buf  = io.BytesIO()
    n_px = arr.shape[0] * arr.shape[1]
    if n_px > 500_000:          # large image → JPEG
        pil.save(buf, format='JPEG', quality=82, optimize=True)
        mime = 'jpeg'
    else:
        pil.save(buf, format='PNG', optimize=True)
        mime = 'png'
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return b64, mime


def build_cell_data_json(valid_cells, results_df, area_col,
                          img_h, img_w, use_microns, scale_px_per_um):
    """
    Build a JSON list of cell dicts for the JS tooltip layer.
    Each cell stores its bounding box (scaled to display) and all metrics.
    """
    px_to_um2 = (1.0 / scale_px_per_um ** 2) if (use_microns and scale_px_per_um) else None
    area_unit = "µm²" if px_to_um2 else "px²"
    cells = []
    for idx, (region, circ) in enumerate(valid_cells, 1):
        # bounding box in original pixel coords
        rmin, cmin, rmax, cmax = region.bbox
        cy_px, cx_px = region.centroid
        area_val = region.area * px_to_um2 if px_to_um2 else int(region.area)
        cells.append({
            "id":        idx,
            "rmin":      int(rmin),
            "cmin":      int(cmin),
            "rmax":      int(rmax),
            "cmax":      int(cmax),
            "cx":        float(round(cx_px, 2)),       # centroid col (x)  in image pixels
            "cy":        float(round(cy_px, 2)),       # centroid row (y)  in image pixels
            "area":      round(float(area_val), 4),
            "area_unit": area_unit,
            "intensity": round(float(get_mi(region)), 2),
            "circ":      round(float(circ), 4),
            "img_h":     img_h,
            "img_w":     img_w,
        })
    return json.dumps(cells)


def render_interactive_image(orig_img, overlay_img, valid_cells, results_df,
                              area_col, ch_name, threshold_used,
                              use_microns, scale_px_per_um, file_prefix):
    """
    Replace st.pyplot with an interactive HTML canvas component.
    Hovering over a detected cell shows a tooltip with its measurements.
    """
    img_h, img_w = overlay_img.shape[:2]
    orig_arr     = orig_img[:, :, :3] if orig_img.ndim == 3 else np.stack([orig_img]*3, -1)
    orig_b64, orig_mime       = img_array_to_b64(orig_arr)
    overlay_b64, overlay_mime = img_array_to_b64(overlay_img)
    cells_json  = build_cell_data_json(
        valid_cells, results_df, area_col,
        img_h, img_w, use_microns, scale_px_per_um
    )
    is_green    = ch_name == "Green"
    accent      = "#3ecf7a" if is_green else "#f07070"
    accent_dark = "#1a7a4a" if is_green else "#c03030"
    n_cells     = len(valid_cells)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: transparent; }}

  .panel-wrap {{
    display: flex;
    gap: 10px;
    width: 100%;
  }}

  .img-panel {{
    flex: 1;
    background: #111;
    border-radius: 10px;
    overflow: hidden;
    position: relative;
    min-width: 0;
  }}

  .img-panel canvas {{
    display: block;
    width: 100%;
    cursor: crosshair;
  }}

  .panel-label {{
    position: absolute;
    top: 8px; left: 10px;
    background: rgba(0,0,0,0.6);
    color: #ddd;
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 6px;
    pointer-events: none;
  }}

  /* ── Tooltip ── */
  #tooltip {{
    position: fixed;
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 10px 14px;
    min-width: 200px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.12s ease;
    z-index: 9999;
    box-shadow: 0 6px 20px rgba(0,0,0,0.14);
    font-size: 13px;
    color: #222;
  }}
  #tooltip.show {{ opacity: 1; }}

  .tt-header {{
    font-size: 13px;
    font-weight: 600;
    color: #111;
    margin-bottom: 8px;
    padding-bottom: 7px;
    border-bottom: 1px solid #eee;
    display: flex;
    align-items: center;
    gap: 7px;
  }}
  .tt-dot {{
    width: 9px; height: 9px;
    border-radius: 50%;
    background: {accent};
    border: 1.5px solid {accent_dark};
    flex-shrink: 0;
  }}
  .tt-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0;
    font-size: 12px;
    color: #555;
    gap: 12px;
  }}
  .tt-val {{
    font-weight: 600;
    color: #111;
    white-space: nowrap;
  }}
  .tt-badge {{
    display: inline-block;
    padding: 1px 7px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    background: {"#e8faf0" if is_green else "#fdeaea"};
    color: {accent_dark};
  }}

  /* ── Info bar below images ── */
  .info-bar {{
    display: flex;
    gap: 16px;
    margin-top: 7px;
    font-size: 12px;
    color: #666;
    align-items: center;
    flex-wrap: wrap;
  }}
  .info-chip {{
    background: #f4f4f4;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    color: #444;
  }}
  .hint {{
    font-size: 11px;
    color: #aaa;
    margin-left: auto;
  }}
</style>
</head>
<body>

<div id="tooltip">
  <div class="tt-header">
    <span class="tt-dot"></span>
    <span id="tt-title">Cell #1</span>
    <span class="tt-badge" id="tt-badge">{"Live" if is_green else "Dead"}</span>
  </div>
  <div class="tt-row"><span>Area</span>         <span class="tt-val" id="tt-area">—</span></div>
  <div class="tt-row"><span>Mean intensity</span><span class="tt-val" id="tt-int">—</span></div>
  <div class="tt-row"><span>Circularity</span>  <span class="tt-val" id="tt-circ">—</span></div>
  <div class="tt-row"><span>Centroid X</span>   <span class="tt-val" id="tt-cx">—</span></div>
  <div class="tt-row"><span>Centroid Y</span>   <span class="tt-val" id="tt-cy">—</span></div>
</div>

<div class="panel-wrap">
  <div class="img-panel">
    <canvas id="origCanvas"></canvas>
    <div class="panel-label">Original</div>
  </div>
  <div class="img-panel">
    <canvas id="overlayCanvas"></canvas>
    <div class="panel-label">Detected: {n_cells} &nbsp;|&nbsp; Threshold: {threshold_used:.1f} &nbsp;|&nbsp; Channel: {ch_name}</div>
  </div>
</div>

<div class="info-bar">
  <span class="info-chip" style="background:{"#e8faf0" if is_green else "#fdeaea"};color:{accent_dark};">
    {"🟢 Green — Live cells" if is_green else "🔴 Red — Dead cells"}
  </span>
  <span class="info-chip">{n_cells} cells detected</span>
  <span class="hint">Hover over a cell in the right image for details</span>
</div>

<script>
const CELLS    = {cells_json};
const IMG_H    = {img_h};
const IMG_W    = {img_w};
const ACCENT   = "{accent}";
const IS_GREEN = {"true" if is_green else "false"};

// ── Load both images ──────────────────────────────
const origImg    = new Image();
const overlayImg = new Image();
origImg.src    = "data:image/{orig_mime};base64,{orig_b64}";
overlayImg.src = "data:image/{overlay_mime};base64,{overlay_b64}";

const origCanvas    = document.getElementById('origCanvas');
const overlayCanvas = document.getElementById('overlayCanvas');
const origCtx       = origCanvas.getContext('2d');
const overlayCtx    = overlayCanvas.getContext('2d');
const tooltip       = document.getElementById('tooltip');

let imagesLoaded = 0;
function onLoad() {{
  imagesLoaded++;
  if (imagesLoaded < 2) return;
  origCanvas.width    = IMG_W;
  origCanvas.height   = IMG_H;
  overlayCanvas.width  = IMG_W;
  overlayCanvas.height = IMG_H;
  origCtx.drawImage(origImg, 0, 0);
  overlayCtx.drawImage(overlayImg, 0, 0);
  drawHighlightRings(null);
}}
origImg.onload    = onLoad;
overlayImg.onload = onLoad;

// ── Draw cell-number labels + hover ring ──────────
function drawHighlightRings(hoveredId) {{
  overlayCtx.drawImage(overlayImg, 0, 0);
  CELLS.forEach(c => {{
    const cx = c.cmin + (c.cmax - c.cmin) / 2;
    const cy = c.rmin + (c.rmax - c.rmin) / 2;
    const rx = (c.cmax - c.cmin) / 2 + 4;
    const ry = (c.rmax - c.rmin) / 2 + 4;
    const isHov = hoveredId === c.id;

    if (isHov) {{
      overlayCtx.save();
      overlayCtx.beginPath();
      overlayCtx.ellipse(cx, cy, rx + 6, ry + 6, 0, 0, Math.PI * 2);
      overlayCtx.strokeStyle = ACCENT;
      overlayCtx.lineWidth   = 3;
      overlayCtx.globalAlpha = 0.5;
      overlayCtx.stroke();
      overlayCtx.restore();

      overlayCtx.save();
      overlayCtx.beginPath();
      overlayCtx.ellipse(cx, cy, rx + 14, ry + 14, 0, 0, Math.PI * 2);
      overlayCtx.strokeStyle = ACCENT;
      overlayCtx.lineWidth   = 1.5;
      overlayCtx.globalAlpha = 0.2;
      overlayCtx.stroke();
      overlayCtx.restore();
    }}

    // cell number label
    overlayCtx.save();
    overlayCtx.font      = isHov ? 'bold 13px sans-serif' : '11px sans-serif';
    overlayCtx.fillStyle = isHov ? ACCENT : 'rgba(255,255,255,0.7)';
    overlayCtx.textAlign = 'center';
    overlayCtx.textBaseline = 'middle';
    overlayCtx.fillText(c.id, cx, cy);
    overlayCtx.restore();
  }});
}}

// ── Hit test: is (mx,my) inside a cell bounding box? ──
function cellAtCanvasPoint(mx, my) {{
  const rect   = overlayCanvas.getBoundingClientRect();
  const scaleX = IMG_W / rect.width;
  const scaleY = IMG_H / rect.height;
  const px     = mx * scaleX;
  const py     = my * scaleY;
  for (const c of CELLS) {{
    const cx = c.cmin + (c.cmax - c.cmin) / 2;
    const cy = c.rmin + (c.rmax - c.rmin) / 2;
    const rx = (c.cmax - c.cmin) / 2;
    const ry = (c.rmax - c.rmin) / 2;
    const dx = (px - cx) / (rx + 6);
    const dy = (py - cy) / (ry + 6);
    if (dx * dx + dy * dy <= 1) return c;
  }}
  return null;
}}

// ── Tooltip position: keep inside viewport ──
function positionTooltip(e) {{
  const TW = 215, TH = 160;
  let left = e.clientX + 16;
  let top  = e.clientY - 20;
  if (left + TW > window.innerWidth  - 10) left = e.clientX - TW - 10;
  if (top  + TH > window.innerHeight - 10) top  = e.clientY - TH - 10;
  if (top < 6) top = 6;
  tooltip.style.left = left + 'px';
  tooltip.style.top  = top  + 'px';
}}

// ── Mouse handlers ────────────────────────────────
let lastHovered = null;

overlayCanvas.addEventListener('mousemove', e => {{
  const rect = overlayCanvas.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const my   = e.clientY - rect.top;
  const cell = cellAtCanvasPoint(mx, my);

  if (cell !== lastHovered) {{
    lastHovered = cell;
    drawHighlightRings(cell ? cell.id : null);
  }}

  if (cell) {{
    document.getElementById('tt-title').textContent = 'Cell #' + cell.id;
    document.getElementById('tt-area').textContent  = cell.area.toLocaleString() + ' ' + cell.area_unit;
    document.getElementById('tt-int').textContent   = cell.intensity.toFixed(2);
    document.getElementById('tt-circ').textContent  = cell.circ.toFixed(4);
    document.getElementById('tt-cx').textContent    = cell.cx.toFixed(1) + ' px';
    document.getElementById('tt-cy').textContent    = cell.cy.toFixed(1) + ' px';
    positionTooltip(e);
    tooltip.classList.add('show');
  }} else {{
    tooltip.classList.remove('show');
  }}
}});

overlayCanvas.addEventListener('mouseleave', () => {{
  tooltip.classList.remove('show');
  lastHovered = null;
  drawHighlightRings(null);
}});
</script>
</body>
</html>
"""
    # height = aspect-ratio-derived + info bar + padding
    display_w   = 900
    display_h   = int(display_w / 2 * img_h / img_w) + 60
    components.html(html, height=max(display_h, 340), scrolling=False)


# ─────────────────────────────────────────────────
#  CHANNEL SECTION RENDERER
# ─────────────────────────────────────────────────

def show_channel_section(title_html, ch_name, img_array, overlay, valid_cells, labels,
                         threshold_used, results_df, area_col, summary_df,
                         use_microns, scale_px_per_um, cell_channel, file_prefix):
    st.markdown(title_html, unsafe_allow_html=True)
    n = len(results_df)

    # ── Interactive image with hover tooltip ──
    render_interactive_image(
        orig_img=img_array,
        overlay_img=overlay,
        valid_cells=valid_cells,
        results_df=results_df,
        area_col=area_col,
        ch_name=ch_name,
        threshold_used=threshold_used,
        use_microns=use_microns,
        scale_px_per_um=scale_px_per_um,
        file_prefix=file_prefix,
    )

    # ── Download annotated image button ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#111')
    orig = img_array[:, :, :3] if img_array.ndim == 3 else img_array
    axes[0].imshow(orig, cmap='gray' if img_array.ndim == 2 else None)
    axes[0].set_title("Original", color='white', fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Detected = {n}  |  Threshold = {threshold_used:.1f}  |  Channel = {ch_name}",
        color='white', fontsize=12
    )
    axes[1].axis('off')
    plt.tight_layout(pad=0.4)
    st.download_button(
        f"⬇️ Download Annotated Image ({ch_name})",
        data=fig_to_bytes(fig),
        file_name=f"{file_prefix}_annotated.png",
        mime="image/png",
        key=f"dl_img_{file_prefix}",
    )
    plt.close(fig)

    # ── Measurements table ──
    st.markdown(f"**Individual Cell Measurements — {ch_name} channel**")
    if results_df.empty:
        st.warning("No cells detected. Try adjusting sidebar parameters.")
    else:
        fmt = {
            area_col:         "{:,.4f}" if (use_microns and scale_px_per_um) else "{:,}",
            "Mean Intensity": "{:.2f}",
            "Circularity":    "{:.4f}",
            "Centroid X":     "{:.2f}",
            "Centroid Y":     "{:.2f}",
        }
        st.dataframe(
            results_df.style.format(fmt),
            use_container_width=True,
            height=min(380, 40 + 36 * n),
        )
        st.download_button(
            f"🖨️ Download {ch_name} Cell Measurements CSV",
            data=df_to_csv_bytes(results_df),
            file_name=f"{file_prefix}_measurements.csv",
            mime="text/csv",
            key=f"dl_meas_{file_prefix}",
        )

    # ── Channel summary ──
    st.markdown(f"**Channel Summary — {ch_name}**")
    if not summary_df.empty:
        st.dataframe(summary_df.T.rename(columns={0: "Value"}), use_container_width=True)
        st.download_button(
            f"🖨️ Download {ch_name} Summary CSV",
            data=df_to_csv_bytes(summary_df),
            file_name=f"{file_prefix}_summary.csv",
            mime="text/csv",
            key=f"dl_sum_{file_prefix}",
        )

    # ── Histogram ──
    with st.expander(f"📈 {ch_name} Intensity Histogram"):
        fig2, ax = plt.subplots(figsize=(8, 3))
        hist_color = '#cc3333' if ch_name == 'Red' else '#1a7a4a'
        ax.hist(cell_channel.ravel(), bins=128, color=hist_color, alpha=0.85)
        ax.axvline(threshold_used, color='gold', linestyle='--', lw=1.5,
                   label=f"Threshold = {threshold_used:.1f}")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Count")
        ax.set_title(f"{ch_name} Channel Intensity Distribution")
        ax.legend()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


# ─────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Analysis Parameters")

    st.markdown("**Units**")
    use_microns = st.checkbox("Show area in µm²", value=False)
    scale_px_per_um = None
    if use_microns:
        scale_px_per_um = st.number_input(
            "Scale (pixels per µm)",
            min_value=0.01, max_value=1000.0, value=1.0, step=0.1,
            help="e.g. if 1 µm = 4.5 pixels, enter 4.5"
        )
        st.caption(f"1 px² = {1 / scale_px_per_um ** 2:.4f} µm²")

    st.markdown("**Segmentation Filters**")
    if use_microns and scale_px_per_um:
        um2_per_px2  = 1.0 / scale_px_per_um ** 2
        min_area_um2 = st.slider(
            "Min Cell Area (µm²)",
            min_value=round(10   * um2_per_px2, 4),
            max_value=round(2000 * um2_per_px2, 2),
            value=round(50       * um2_per_px2, 4),
            step=round(10        * um2_per_px2, 4),
            format="%.4f",
        )
        st.caption(f"= {min_area_um2 / um2_per_px2:.0f} px²")
        min_area = int(min_area_um2 / um2_per_px2)
    else:
        min_area = st.slider("Min Cell Area (px²)", 10, 2000, 50, 10)

    min_circ      = st.slider("Min Circularity",    0.0, 1.0, 0.1, 0.01)
    max_circ      = st.slider("Max Circularity",    0.0, 1.0, 1.0, 0.01)
    min_intensity = st.slider("Min Mean Intensity", 0,   255,  15,   1)
    max_intensity = st.slider("Max Mean Intensity", 0,   255, 255,   1,
                              help="Lower this to exclude very bright artefacts "
                                   "such as scale bars, labels, or saturated pixels.")
    if max_intensity < min_intensity:
        st.warning("⚠️ Max Mean Intensity is below Min Mean Intensity — no cells will match. "
                   "Adjust one of the sliders.")

    st.markdown("**Thresholding**")
    thresh_method = st.radio("Method", ["Otsu (auto)", "Manual"], index=0)
    manual_thresh = None
    if thresh_method == "Manual":
        manual_thresh = st.slider("Manual Threshold", 0, 255, 30, 1)

    st.markdown("**Display**")
    st.caption("Outline colours are fixed: 🟢 Green channel · 🔴 Red channel")


# ─────────────────────────────────────────────────
#  MAIN UI
# ─────────────────────────────────────────────────
st.markdown('<div class="main-header">🔬 Cell Viability Analyser</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a <b>green channel</b> image (live cells) and a '
    '<b>red channel</b> image (dead cells) to analyse viability.</div>',
    unsafe_allow_html=True
)

col_g, col_r = st.columns(2)
with col_g:
    st.markdown("#### 🟢 Green Channel — Live Cells")
    uploaded_green = st.file_uploader(
        "Upload green channel image",
        type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
        key="green_upload",
        label_visibility="collapsed",
    )
    if uploaded_green:
        st.success(f"✅ Loaded: {uploaded_green.name}")

with col_r:
    st.markdown("#### 🔴 Red Channel — Dead Cells")
    uploaded_red = st.file_uploader(
        "Upload red channel image",
        type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
        key="red_upload",
        label_visibility="collapsed",
    )
    if uploaded_red:
        st.success(f"✅ Loaded: {uploaded_red.name}")

if uploaded_green is None and uploaded_red is None:
    st.markdown("---")
    st.markdown(
        "#### How to use\n"
        "1. Upload the **green channel** image (live/viable cells) on the left.\n"
        "2. Upload the **red channel** image (dead cells) on the right.\n"
        "3. You can upload just one channel if needed.\n"
        "4. Adjust segmentation parameters in the sidebar.\n"
        "5. Hover over any cell in the result image to see its measurements in a tooltip.\n"
        "6. Download tables as CSV or the annotated image using the buttons below."
    )
    st.stop()

tm = 'otsu' if thresh_method == "Otsu (auto)" else 'manual'

# ─────────────────────────────────────────────────
#  LOADING ANIMATION + ANALYSIS
# ─────────────────────────────────────────────────
green_count = red_count = 0
green_results = red_results = None

CELL_LOADER_HTML = """
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:2rem 1rem;gap:1.2rem;">
  <div style="position:relative;width:220px;height:220px;">
    <svg width="220" height="220" viewBox="0 0 220 220" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="cyto" cx="45%" cy="40%" r="55%">
          <stop offset="0%" stop-color="#d4f0e2"/>
          <stop offset="100%" stop-color="#a8dfc0"/>
        </radialGradient>
        <radialGradient id="nuc" cx="40%" cy="38%" r="55%">
          <stop offset="0%" stop-color="#7bc9a0"/>
          <stop offset="100%" stop-color="#1a7a4a"/>
        </radialGradient>
        <radialGradient id="cyto2" cx="45%" cy="40%" r="55%">
          <stop offset="0%" stop-color="#ffd6d6"/>
          <stop offset="100%" stop-color="#f5a0a0"/>
        </radialGradient>
        <radialGradient id="nuc2" cx="40%" cy="38%" r="55%">
          <stop offset="0%" stop-color="#f07070"/>
          <stop offset="100%" stop-color="#c03030"/>
        </radialGradient>
      </defs>
      <style>
        @keyframes floatA{{0%,100%{{transform:translateY(0px) rotate(0deg)}}50%{{transform:translateY(-6px) rotate(1.5deg)}}}}
        @keyframes floatB{{0%,100%{{transform:translateY(0px) rotate(0deg)}}50%{{transform:translateY(-5px) rotate(-1.2deg)}}}}
        @keyframes scan{{0%{{transform:translateX(0px);opacity:.4}}85%{{transform:translateX(210px);opacity:.4}}86%{{opacity:0}}100%{{transform:translateX(210px);opacity:0}}}}
        @keyframes ping{{0%{{opacity:0}}20%{{opacity:.9}}100%{{opacity:0}}}}
        .fa{{animation:floatA 4s ease-in-out infinite}}
        .fb{{animation:floatB 4.5s ease-in-out infinite}}
        .sc{{animation:scan 2.8s linear infinite}}
        .p1{{animation:ping 2.8s 1.1s ease-out infinite}}
        .p2{{animation:ping 2.8s 2.0s ease-out infinite}}
      </style>
      <g class="fa">
        <ellipse cx="72" cy="110" rx="58" ry="64" fill="url(#cyto)" stroke="#1a7a4a" stroke-width="2.5" opacity=".95"/>
        <ellipse cx="70" cy="112" rx="22" ry="22" fill="url(#nuc)" opacity=".9"/>
        <circle cx="66" cy="108" r="7" fill="#0f5e36" opacity=".85"/>
        <ellipse cx="44" cy="90" rx="7" ry="4" fill="#2ecc71" opacity=".5" transform="rotate(-25,44,90)"/>
        <ellipse cx="96" cy="95" rx="6" ry="3.5" fill="#2ecc71" opacity=".5" transform="rotate(15,96,95)"/>
        <ellipse cx="50" cy="138" rx="7" ry="3.5" fill="#2ecc71" opacity=".45" transform="rotate(10,50,138)"/>
        <ellipse cx="93" cy="135" rx="5" ry="3" fill="#2ecc71" opacity=".45" transform="rotate(-20,93,135)"/>
      </g>
      <g class="fb">
        <path d="M178,62 C202,55 220,80 218,108 C216,136 204,162 182,170 C160,178 138,165 132,142 C126,119 132,88 148,72 C158,62 168,65 178,62 Z" fill="url(#cyto2)" stroke="#c03030" stroke-width="2" opacity=".93"/>
        <circle cx="218" cy="95" r="8" fill="#f5a0a0" stroke="#c03030" stroke-width="1.5" opacity=".8"/>
        <circle cx="208" cy="155" r="6" fill="#f5a0a0" stroke="#c03030" stroke-width="1.5" opacity=".75"/>
        <ellipse cx="172" cy="112" rx="15" ry="13" fill="url(#nuc2)" opacity=".85"/>
        <ellipse cx="156" cy="120" rx="8" ry="7" fill="url(#nuc2)" opacity=".7"/>
        <circle cx="170" cy="110" r="4" fill="#8b0000" opacity=".7"/>
        <circle cx="158" cy="121" r="3" fill="#8b0000" opacity=".65"/>
      </g>
      <line x1="10" y1="0" x2="10" y2="220" stroke="#1a7a4a" stroke-width="1.5" opacity=".35" class="sc"/>
      <circle cx="72" cy="110" r="5" fill="none" stroke="#1a7a4a" stroke-width="1.5" opacity="0" class="p1"/>
      <circle cx="175" cy="114" r="5" fill="none" stroke="#c03030" stroke-width="1.5" opacity="0" class="p2"/>
    </svg>
  </div>
  <div style="font-size:14px;color:#555;letter-spacing:.04em;">Analysing cells — please wait…</div>
  <div style="width:200px;height:4px;background:#e0e0e0;border-radius:2px;overflow:hidden;">
    <div style="height:100%;background:#1a7a4a;border-radius:2px;animation:progress 3s ease-in-out infinite;"></div>
  </div>
  <style>@keyframes progress{{0%{{width:0%}}80%{{width:90%}}100%{{width:100%}}}}</style>
</div>
"""

loader_placeholder = st.empty()
loader_placeholder.markdown(CELL_LOADER_HTML, unsafe_allow_html=True)

with st.spinner(""):
    if uploaded_green:
        g_img    = load_image(uploaded_green)
        g_ch     = extract_channel(g_img, 'green')
        g_labels, g_thresh, g_valid = segment_cells(
            g_ch, min_area, min_circ, max_circ, min_intensity, max_intensity,
            threshold_method=tm, manual_threshold=manual_thresh
        )
        g_df, g_area_col = build_results_df(g_valid, use_microns, scale_px_per_um)
        g_sum_df  = build_channel_summary(g_df, g_area_col, g_ch, g_valid, g_labels,
                                          g_img.shape, use_microns, scale_px_per_um)
        g_overlay = build_overlay(g_img, g_labels, g_valid, 'green')
        green_count   = len(g_df)
        green_results = (g_img, g_ch, g_labels, g_thresh, g_valid,
                         g_df, g_area_col, g_sum_df, g_overlay)

    if uploaded_red:
        r_img    = load_image(uploaded_red)
        r_ch     = extract_channel(r_img, 'red')
        r_labels, r_thresh, r_valid = segment_cells(
            r_ch, min_area, min_circ, max_circ, min_intensity, max_intensity,
            threshold_method=tm, manual_threshold=manual_thresh
        )
        r_df, r_area_col = build_results_df(r_valid, use_microns, scale_px_per_um)
        r_sum_df  = build_channel_summary(r_df, r_area_col, r_ch, r_valid, r_labels,
                                          r_img.shape, use_microns, scale_px_per_um)
        r_overlay = build_overlay(r_img, r_labels, r_valid, 'red')
        red_count   = len(r_df)
        red_results = (r_img, r_ch, r_labels, r_thresh, r_valid,
                       r_df, r_area_col, r_sum_df, r_overlay)

loader_placeholder.empty()

# ─────────────────────────────────────────────────
#  TOP METRIC ROW
# ─────────────────────────────────────────────────
st.markdown("---")
total_cells  = green_count + red_count
survival_pct = round(100.0 * green_count / total_cells, 2) if total_cells > 0 else 0.0
dead_pct     = round(100.0 * red_count   / total_cells, 2) if total_cells > 0 else 0.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🦠 Total Cells",  total_cells)
m2.metric("🟢 Live (Green)", green_count)
m3.metric("🔴 Dead (Red)",   red_count)
m4.metric("✅ Survival %",   f"{survival_pct}%")
m5.metric("☠️ Death %",      f"{dead_pct}%")

# ─────────────────────────────────────────────────
#  GREEN CHANNEL SECTION
# ─────────────────────────────────────────────────
if green_results:
    g_img, g_ch, g_labels, g_thresh, g_valid, g_df, g_area_col, g_sum_df, g_overlay = green_results
    st.markdown("---")
    show_channel_section(
        title_html='<div class="section-title-green">🟢 Green Channel — Live Cells</div>',
        ch_name="Green",
        img_array=g_img,
        overlay=g_overlay,
        valid_cells=g_valid,
        labels=g_labels,
        threshold_used=g_thresh,
        results_df=g_df,
        area_col=g_area_col,
        summary_df=g_sum_df,
        use_microns=use_microns,
        scale_px_per_um=scale_px_per_um,
        cell_channel=g_ch,
        file_prefix='green_cells',
    )

# ─────────────────────────────────────────────────
#  RED CHANNEL SECTION
# ─────────────────────────────────────────────────
if red_results:
    r_img, r_ch, r_labels, r_thresh, r_valid, r_df, r_area_col, r_sum_df, r_overlay = red_results
    st.markdown("---")
    show_channel_section(
        title_html='<div class="section-title-red">🔴 Red Channel — Dead Cells</div>',
        ch_name="Red",
        img_array=r_img,
        overlay=r_overlay,
        valid_cells=r_valid,
        labels=r_labels,
        threshold_used=r_thresh,
        results_df=r_df,
        area_col=r_area_col,
        summary_df=r_sum_df,
        use_microns=use_microns,
        scale_px_per_um=scale_px_per_um,
        cell_channel=r_ch,
        file_prefix='red_cells',
    )

# ─────────────────────────────────────────────────
#  COMBINED VIABILITY SUMMARY
# ─────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">📋 Combined Viability Summary</div>', unsafe_allow_html=True)

viability_data = {
    "Total Cells (Green + Red)":  total_cells,
    "Live Cells (Green Channel)": green_count,
    "Dead Cells (Red Channel)":   red_count,
    "Survival Percentage (%)":    survival_pct,
    "Death Percentage (%)":       dead_pct,
}

if green_results and not g_sum_df.empty:
    area_label = [c for c in g_sum_df.columns if "Area" in c]
    if area_label:
        viability_data["Avg Live Cell Area"]        = round(float(g_sum_df[area_label[0]].iloc[0]), 4)
    viability_data["Avg Live Cell Intensity"]       = round(float(g_sum_df["Average Cell Intensity"].iloc[0]), 2)
    viability_data["Avg Live Cell Circularity"]     = round(float(g_sum_df["Average Circularity"].iloc[0]), 4)
    viability_data["Live Cell Spatial Dist."]       = round(float(g_sum_df["Spatial Distribution (0–1)"].iloc[0]), 4)

if red_results and not r_sum_df.empty:
    area_label = [c for c in r_sum_df.columns if "Area" in c]
    if area_label:
        viability_data["Avg Dead Cell Area"]        = round(float(r_sum_df[area_label[0]].iloc[0]), 4)
    viability_data["Avg Dead Cell Intensity"]       = round(float(r_sum_df["Average Cell Intensity"].iloc[0]), 2)
    viability_data["Avg Dead Cell Circularity"]     = round(float(r_sum_df["Average Circularity"].iloc[0]), 4)
    viability_data["Dead Cell Spatial Dist."]       = round(float(r_sum_df["Spatial Distribution (0–1)"].iloc[0]), 4)

viability_df = pd.DataFrame(list(viability_data.items()), columns=["Metric", "Value"])
st.dataframe(viability_df.set_index("Metric"), use_container_width=True)

st.download_button(
    "🖨️ Download Combined Viability Summary CSV",
    data=viability_df.to_csv(index=False).encode('utf-8'),
    file_name="viability_summary.csv",
    mime="text/csv",
    key="dl_viability",
)

if total_cells > 0:
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    bars = ax3.bar(
        ["Live (Green)", "Dead (Red)"],
        [survival_pct, dead_pct],
        color=['#2ecc71', '#e74c3c'],
        width=0.4, edgecolor='white', linewidth=1.2
    )
    for bar, val in zip(bars, [survival_pct, dead_pct]):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax3.set_ylabel("Percentage (%)")
    ax3.set_title("Cell Viability", fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 115)
    ax3.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=False)
    plt.close(fig3)

st.markdown("""
> **Survival %** = Live cells (green) ÷ Total cells × 100  
> **Spatial Distribution (0–1)**: `0` = clustered · `1` = evenly spread (Clark-Evans index)  
> **Hover** over any cell in the result image to see its individual measurements instantly.
""")
