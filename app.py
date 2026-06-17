import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries
import io
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
    """Extract named channel as float32 2-D array."""
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
                  min_intensity, threshold_method='otsu', manual_threshold=None):
    thresh = manual_threshold if (threshold_method == 'manual' and manual_threshold is not None) \
             else threshold_otsu(cell_channel)
    binary = cell_channel > thresh
    binary = remove_small_objects(binary, max_size=max(5, min_area // 2 - 1))
    labels = label(binary)
    valid = []
    for region in regionprops(labels, intensity_image=cell_channel):
        area  = region.area
        perim = region.perimeter
        circ  = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
        mi    = get_mi(region)
        if area >= min_area and min_circ <= circ <= max_circ and mi >= min_intensity:
            valid.append((region, circ))
    return labels, thresh, valid


def build_overlay(img_array, labels, valid_cells, outline_color):
    """Draw outlines on a display-safe RGB copy of the image."""
    if img_array.ndim == 2:
        disp = np.stack([img_array] * 3, axis=-1)
    else:
        disp = img_array[:, :, :3].copy()
    if disp.dtype != np.uint8:
        pmax = disp.max()
        disp = ((disp / pmax) * 255).astype(np.uint8) if pmax > 0 else disp.astype(np.uint8)

    color_map = {
        'yellow':  [255, 255,   0],
        'cyan':    [  0, 255, 255],
        'magenta': [255,   0, 255],
        'white':   [255, 255, 255],
        'red':     [255,  50,  50],
        'green':   [ 50, 255,  50],
    }
    color = color_map.get(outline_color, [255, 255, 0])
    for region, _ in valid_cells:
        mask = np.zeros(labels.shape, dtype=bool)
        mask[region.coords[:, 0], region.coords[:, 1]] = True
        outline = find_boundaries(mask, mode='outer')
        disp[outline] = color
    return disp


def build_interactive_overlay(overlay_img, valid_cells, results_df, area_col,
                              ch_name, use_microns, scale_px_per_um):
    """
    Build a Plotly figure showing the annotated overlay image where hovering
    over a detected cell pops up a tooltip with that cell's measurements
    (Cell #, Area, Mean Intensity, Circularity, Centroid).
    """
    h, w = overlay_img.shape[0], overlay_img.shape[1]

    fig = go.Figure()

    # Background image (the outlined overlay) — pinned to pixel coordinates
    fig.add_layout_image(
        dict(
            source=Image.fromarray(overlay_img),
            xref="x", yref="y",
            x=0, y=0,
            sizex=w, sizey=h,
            sizing="stretch",
            layer="below",
        )
    )

    area_unit = "µm²" if (use_microns and scale_px_per_um) else "px²"
    # Plotly marker `size` is in screen pixels, not data units, so scale the
    # hover-target radius relative to the image's longest side rather than
    # using raw pixel coordinates (keeps hover targets sensible for both
    # small thumbnails and large microscopy images).
    longest_side = max(h, w)
    render_px    = 480  # approx rendered height set below; used for scaling

    if not results_df.empty:
        for (region, circ), (_, row) in zip(valid_cells, results_df.iterrows()):
            cy, cx = region.centroid  # row, col -> y, x
            area_val = row[area_col]
            mi       = row["Mean Intensity"]
            hover_text = (
                f"<b>Cell #{int(row['Cell #'])}</b><br>"
                f"Channel: {ch_name}<br>"
                f"Area: {area_val:,.2f} {area_unit}<br>"
                f"Mean Intensity: {mi:.2f}<br>"
                f"Circularity: {circ:.4f}<br>"
                f"Centroid: ({cx:.1f}, {cy:.1f})"
            )
            # Hover-target radius in *data* pixels, converted to an
            # approximate on-screen marker size so it roughly tracks the
            # cell's real footprint after the image is scaled to fit the plot.
            cell_radius_px = max(4.0, np.sqrt(region.area / np.pi))
            marker_px      = np.clip(cell_radius_px * (render_px / longest_side) * 2,
                                     10, 40)
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers",
                marker=dict(
                    size=marker_px,
                    color="rgba(255,255,0,0.01)",   # near-invisible fill, but still hoverable
                    line=dict(width=1.5, color="rgba(255,255,0,0.35)"),  # faint ring so hover targets are discoverable
                ),
                hovertemplate=hover_text + "<extra></extra>",
                hoverlabel=dict(
                    bgcolor="#1a1a1a",
                    bordercolor="#ffd400",
                    font=dict(color="#ffffff", size=13, family="Arial"),
                ),
                showlegend=False,
            ))

    fig.update_xaxes(
        visible=False, range=[0, w],
        constrain="domain",
    )
    fig.update_yaxes(
        visible=False, range=[h, 0],   # invert y so image isn't flipped
        scaleanchor="x", scaleratio=1,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=480,
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font=dict(color="white"),
        hoverlabel=dict(
            bgcolor="#1a1a1a",
            bordercolor="#ffd400",
            font=dict(color="#ffffff", size=13, family="Arial"),
        ),
        hovermode="closest",
        title=dict(
            text=f"Detected = {len(results_df)}  |  Channel = {ch_name}  (hover over a cell for details)",
            font=dict(color="white", size=13),
        ),
    )
    return fig


def spatial_distribution_score(centroids, img_shape):
    n = len(centroids)
    if n < 2:
        return 0.0
    pts = np.array(centroids)
    if n == 2:
        d = np.linalg.norm(pts[0] - pts[1])
        return round(min(1.0, d / np.hypot(img_shape[0], img_shape[1]) * 2), 4)
    area    = img_shape[0] * img_shape[1]
    density = n / area
    expected_nn = 1.0 / (2 * np.sqrt(density))
    nn_dists = []
    for i, p in enumerate(pts):
        others = np.concatenate([pts[:i], pts[i+1:]], axis=0)
        nn_dists.append(np.min(np.linalg.norm(others - p, axis=1)))
    R = np.mean(nn_dists) / expected_nn
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


def build_channel_summary(df, area_col, cell_channel, valid_cells, img_shape,
                          use_microns=False, scale_px_per_um=None):
    if df.empty:
        return pd.DataFrame()
    total_px      = img_shape[0] * img_shape[1]
    area_px_vals  = df[area_col] * (scale_px_per_um ** 2) if (use_microns and scale_px_per_um) else df[area_col]
    pct_area      = round(100.0 * area_px_vals.sum() / total_px, 4)
    cell_mask     = np.zeros(img_shape[:2], dtype=bool)
    for region, _ in valid_cells:
        cell_mask[region.coords[:, 0], region.coords[:, 1]] = True
    bg_intensity  = float(cell_channel[~cell_mask].mean())
    centroids     = [(r.centroid[0], r.centroid[1]) for r, _ in valid_cells]
    spatial       = spatial_distribution_score(centroids, img_shape)
    area_label    = "Average Area (µm²)" if (use_microns and scale_px_per_um) else "Average Area (px²)"
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


def show_channel_section(title_html, ch_name, img_array, overlay, valid_cells, labels,
                         threshold_used, results_df, area_col, summary_df,
                         use_microns, scale_px_per_um, cell_channel, file_prefix):
    """Render annotated image + tables for one channel (overlay pre-built)."""
    st.markdown(title_html, unsafe_allow_html=True)

    n = len(results_df)


    # ── annotated image (static, for the download button) ──
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

    # ── interactive view: original (static) + detected (hover for details) ──
    ic1, ic2 = st.columns(2)
    with ic1:
        fig_orig, ax_orig = plt.subplots(figsize=(7, 6))
        fig_orig.patch.set_facecolor('#111')
        ax_orig.imshow(orig, cmap='gray' if img_array.ndim == 2 else None)
        ax_orig.set_title("Original", color='white', fontsize=12)
        ax_orig.axis('off')
        st.pyplot(fig_orig, use_container_width=True)
        plt.close(fig_orig)
    with ic2:
        st.caption("🖱️ Hover over a cell in the image below to see its details")
        interactive_fig = build_interactive_overlay(
            overlay, valid_cells, results_df, area_col,
            ch_name, use_microns, scale_px_per_um
        )
        st.plotly_chart(interactive_fig, use_container_width=True,
                        key=f"plotly_overlay_{file_prefix}")

    st.download_button(
        f"⬇️ Download Annotated Image (PNG)",
        data=fig_to_bytes(fig),
        file_name=f"{file_prefix}_annotated.png",
        mime="image/png",
        key=f"dl_img_{file_prefix}",
    )
    plt.close(fig)

    # ── measurements table ──
    st.markdown(f"**Individual Cell Measurements — {ch_name} channel**")
    if results_df.empty:
        st.warning("No cells detected. Try adjusting sidebar parameters.")
    else:
        area_unit = "µm²" if (use_microns and scale_px_per_um) else "px²"
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

    # ── channel summary ──
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

    # ── histogram ──
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

# ── Two upload boxes ──────────────────────────────
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

# ── Guard — need at least one image ──────────────
if uploaded_green is None and uploaded_red is None:
    st.markdown("---")
    st.markdown(
        "#### How to use\n"
        "1. Upload the **green channel** image (live/viable cells) on the left.\n"
        "2. Upload the **red channel** image (dead cells) on the right.\n"
        "3. You can upload just one channel if needed.\n"
        "4. Adjust segmentation parameters in the sidebar.\n"
        "5. Results for each channel appear below, followed by a combined viability summary."
    )
    st.stop()

tm = 'otsu' if thresh_method == "Otsu (auto)" else 'manual'

# ─────────────────────────────────────────────────
#  ANALYSE BOTH CHANNELS
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
        @keyframes floatA{0%,100%{transform:translateY(0px) rotate(0deg)}50%{transform:translateY(-6px) rotate(1.5deg)}}
        @keyframes floatB{0%,100%{transform:translateY(0px) rotate(0deg)}50%{transform:translateY(-5px) rotate(-1.2deg)}}
        @keyframes scan{0%{transform:translateX(0px);opacity:.4}85%{transform:translateX(210px);opacity:.4}86%{opacity:0}100%{transform:translateX(210px);opacity:0}}
        @keyframes ping{0%{opacity:0}20%{opacity:.9}100%{opacity:0}}
        .fa{animation:floatA 4s ease-in-out infinite}
        .fb{animation:floatB 4.5s ease-in-out infinite}
        .sc{animation:scan 2.8s linear infinite}
        .p1{animation:ping 2.8s 1.1s ease-out infinite}
        .p2{animation:ping 2.8s 2.0s ease-out infinite}
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
  <style>@keyframes progress{0%{width:0%}80%{width:90%}100%{width:100%}}</style>
</div>
"""

loader_placeholder = st.empty()
loader_placeholder.markdown(CELL_LOADER_HTML, unsafe_allow_html=True)

# ALL heavy work inside — loader stays visible until every computation is done
with st.spinner(""):

    if uploaded_green:
        g_img    = load_image(uploaded_green)
        g_ch     = extract_channel(g_img, 'green')
        g_labels, g_thresh, g_valid = segment_cells(
            g_ch, min_area, min_circ, max_circ, min_intensity,
            threshold_method=tm, manual_threshold=manual_thresh
        )
        g_df, g_area_col = build_results_df(g_valid, use_microns, scale_px_per_um)
        g_sum_df  = build_channel_summary(g_df, g_area_col, g_ch, g_valid,
                                          g_img.shape, use_microns, scale_px_per_um)
        g_overlay = build_overlay(g_img, g_labels, g_valid, 'green')
        green_count   = len(g_df)
        green_results = (g_img, g_ch, g_labels, g_thresh, g_valid,
                         g_df, g_area_col, g_sum_df, g_overlay)

    if uploaded_red:
        r_img    = load_image(uploaded_red)
        r_ch     = extract_channel(r_img, 'red')
        r_labels, r_thresh, r_valid = segment_cells(
            r_ch, min_area, min_circ, max_circ, min_intensity,
            threshold_method=tm, manual_threshold=manual_thresh
        )
        r_df, r_area_col = build_results_df(r_valid, use_microns, scale_px_per_um)
        r_sum_df  = build_channel_summary(r_df, r_area_col, r_ch, r_valid,
                                          r_img.shape, use_microns, scale_px_per_um)
        r_overlay = build_overlay(r_img, r_labels, r_valid, 'red')
        red_count   = len(r_df)
        red_results = (r_img, r_ch, r_labels, r_thresh, r_valid,
                       r_df, r_area_col, r_sum_df, r_overlay)

# Loader clears only after ALL computation — segmentation + outlines + summaries — is complete
loader_placeholder.empty()

# ─────────────────────────────────────────────────
#  TOP METRIC ROW
# ─────────────────────────────────────────────────
st.markdown("---")
total_cells    = green_count + red_count
survival_pct   = round(100.0 * green_count / total_cells, 2) if total_cells > 0 else 0.0
dead_pct       = round(100.0 * red_count   / total_cells, 2) if total_cells > 0 else 0.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🦠 Total Cells",        total_cells)
m2.metric("🟢 Live (Green)",       green_count)
m3.metric("🔴 Dead (Red)",         red_count)
m4.metric("✅ Survival %",         f"{survival_pct}%")
m5.metric("☠️ Death %",            f"{dead_pct}%")


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
    "Total Cells (Green + Red)":   total_cells,
    "Live Cells (Green Channel)":  green_count,
    "Dead Cells (Red Channel)":    red_count,
    "Survival Percentage (%)":     survival_pct,
    "Death Percentage (%)":        dead_pct,
}

# Add per-channel averages if available
if green_results and not g_sum_df.empty:
    area_label = [c for c in g_sum_df.columns if "Area" in c]
    if area_label:
        viability_data["Avg Live Cell Area"] = round(float(g_sum_df[area_label[0]].iloc[0]), 4)
    viability_data["Avg Live Cell Intensity"]    = round(float(g_sum_df["Average Cell Intensity"].iloc[0]), 2)
    viability_data["Avg Live Cell Circularity"]  = round(float(g_sum_df["Average Circularity"].iloc[0]), 4)
    viability_data["Live Cell Spatial Dist."]    = round(float(g_sum_df["Spatial Distribution (0–1)"].iloc[0]), 4)

if red_results and not r_sum_df.empty:
    area_label = [c for c in r_sum_df.columns if "Area" in c]
    if area_label:
        viability_data["Avg Dead Cell Area"] = round(float(r_sum_df[area_label[0]].iloc[0]), 4)
    viability_data["Avg Dead Cell Intensity"]    = round(float(r_sum_df["Average Cell Intensity"].iloc[0]), 2)
    viability_data["Avg Dead Cell Circularity"]  = round(float(r_sum_df["Average Circularity"].iloc[0]), 4)
    viability_data["Dead Cell Spatial Dist."]    = round(float(r_sum_df["Spatial Distribution (0–1)"].iloc[0]), 4)

viability_df = pd.DataFrame(list(viability_data.items()), columns=["Metric", "Value"])
st.dataframe(viability_df.set_index("Metric"), use_container_width=True)

st.download_button(
    "🖨️ Download Combined Viability Summary CSV",
    data=viability_df.to_csv(index=False).encode('utf-8'),
    file_name="viability_summary.csv",
    mime="text/csv",
    key="dl_viability",
)

# ── Viability bar chart ───────────────────────────
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
""")
