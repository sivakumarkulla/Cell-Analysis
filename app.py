import streamlit as st
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
import io
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Cell Analysis Tool",
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
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────

def load_image(uploaded_file):
    return np.array(Image.open(uploaded_file))


def detect_channel(img_array):
    """Return best channel name + per-channel mean intensities."""
    if img_array.ndim == 2:
        return 'grayscale', {}

    nc = img_array.shape[2]
    names = ['red', 'green', 'blue', 'alpha'][:nc]
    ch = {}
    for i, name in enumerate(names):
        arr = img_array[:, :, i].astype(np.float32)
        nz  = arr[arr > 0]
        ch[name] = float(nz.mean()) if len(nz) else 0.0

    active = {k: v for k, v in ch.items() if k != 'alpha'}
    if not any(v > 0 for v in active.values()):
        return 'unknown', ch

    values = sorted(active.values(), reverse=True)
    best   = max(active, key=lambda k: active[k])

    if len(values) > 1 and values[1] > 0 and values[0] / values[1] < 1.5:
        return 'unknown', ch

    return best, ch


def extract_channel(img_array, channel):
    if img_array.ndim == 2:
        return img_array.astype(np.float32)
    cmap = {'red': 0, 'green': 1, 'blue': 2}
    if channel in cmap and img_array.shape[2] > cmap[channel]:
        return img_array[:, :, cmap[channel]].astype(np.float32)
    nc = min(img_array.shape[2], 3)
    return img_array[:, :, :nc].mean(axis=2).astype(np.float32)


def get_region_mean_intensity(region):
    """Compatible accessor for skimage ≥0.26 and older."""
    try:
        return region.intensity_mean
    except AttributeError:
        return region.mean_intensity


def segment_cells(cell_channel, min_area, min_circ, max_circ,
                  min_intensity, threshold_method='otsu', manual_threshold=None):
    thresh = manual_threshold if (threshold_method == 'manual' and manual_threshold is not None) \
             else threshold_otsu(cell_channel)

    binary = cell_channel > thresh
    # remove_small_objects max_size = min_area-1 keeps objects >= min_area
    binary = remove_small_objects(binary, max_size=max(5, min_area // 2 - 1))
    labels = label(binary)

    valid = []
    for region in regionprops(labels, intensity_image=cell_channel):
        area  = region.area
        perim = region.perimeter
        circ  = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
        mi    = get_region_mean_intensity(region)

        if (area >= min_area
                and min_circ <= circ <= max_circ
                and mi >= min_intensity):
            valid.append((region, circ))

    return labels, thresh, valid


def build_overlay(img_array, labels, valid_cells, outline_color):
    if img_array.ndim == 2:
        disp = np.stack([img_array] * 3, axis=-1)
    else:
        disp = img_array[:, :, :3].copy()

    if disp.dtype != np.uint8:
        pmax = disp.max()
        disp = ((disp / pmax) * 255).astype(np.uint8) if pmax > 0 else disp.astype(np.uint8)

    color_map = {'yellow': [255,255,0], 'cyan': [0,255,255],
                 'magenta': [255,0,255], 'white': [255,255,255]}
    color = color_map.get(outline_color, [255,255,0])

    for region, _ in valid_cells:
        mask         = np.zeros(labels.shape, dtype=bool)
        mask[region.coords[:,0], region.coords[:,1]] = True
        outline      = find_boundaries(mask, mode='outer')
        disp[outline] = color

    return disp


def spatial_distribution_score(centroids, img_shape):
    """Clark-Evans index normalised to [0, 1]."""
    n = len(centroids)
    if n < 2:
        return 0.0

    pts = np.array(centroids)

    if n == 2:
        d = np.linalg.norm(pts[0] - pts[1])
        diag = np.hypot(img_shape[0], img_shape[1])
        return round(min(1.0, d / (diag / 2)), 4)

    area    = img_shape[0] * img_shape[1]
    density = n / area
    expected_nn = 1.0 / (2 * np.sqrt(density))

    nn_dists = []
    for i, p in enumerate(pts):
        others = np.concatenate([pts[:i], pts[i+1:]], axis=0)
        nn_dists.append(np.min(np.linalg.norm(others - p, axis=1)))

    R = np.mean(nn_dists) / expected_nn          # ~0 clustered → ~2.15 dispersed
    return round(float(np.clip(R / 2.15, 0, 1)), 4)


def build_results_df(valid_cells, use_microns=False, scale_px_per_um=None):
    px_to_um2 = (1.0 / scale_px_per_um**2) if (use_microns and scale_px_per_um) else None
    area_col  = "Area (µm²)" if px_to_um2 else "Area (px²)"
    rows = []
    for idx, (region, circ) in enumerate(valid_cells, 1):
        area_val = region.area * px_to_um2 if px_to_um2 else int(region.area)
        rows.append({
            "Cell #":         idx,
            area_col:         round(float(area_val), 4) if px_to_um2 else int(area_val),
            "Mean Intensity": round(float(get_region_mean_intensity(region)), 2),
            "Circularity":    round(circ, 4),
            "Centroid X":     round(region.centroid[1], 2),
            "Centroid Y":     round(region.centroid[0], 2),
        })
    return pd.DataFrame(rows), area_col


def build_summary(df, area_col, cell_channel, valid_cells, img_shape, use_microns=False, scale_px_per_um=None):
    if df.empty:
        return pd.DataFrame()

    total_px       = img_shape[0] * img_shape[1]
    # % area always computed in pixels regardless of display unit
    area_px_values = df[area_col] * (scale_px_per_um**2) if (use_microns and scale_px_per_um) else df[area_col]
    cell_area_tot  = area_px_values.sum()
    pct_area       = round(100.0 * cell_area_tot / total_px, 4)

    cell_mask = np.zeros(img_shape[:2], dtype=bool)
    for region, _ in valid_cells:
        cell_mask[region.coords[:,0], region.coords[:,1]] = True
    bg_intensity = float(cell_channel[~cell_mask].mean())

    centroids = [(r.centroid[0], r.centroid[1]) for r, _ in valid_cells]
    spatial   = spatial_distribution_score(centroids, img_shape)

    area_label = f"Average Area (µm²)" if (use_microns and scale_px_per_um) else "Average Area (px²)"

    return pd.DataFrame([{
        "Total Cells Detected":        len(df),
        area_label:                    round(float(df[area_col].mean()), 4),
        "Average Circularity":         round(df["Circularity"].mean(), 4),
        "Average Cell Intensity":      round(df["Mean Intensity"].mean(), 2),
        "Average Background Intensity":round(bg_intensity, 2),
        "% Area Occupied by Cells":    pct_area,
        "Spatial Distribution (0–1)":  spatial,
    }])


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf.read()


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
            min_value=0.01,
            max_value=1000.0,
            value=1.0,
            step=0.1,
            help="Enter the pixel-to-micron scale for your microscope/objective. "
                 "e.g. if 1 µm = 4.5 pixels, enter 4.5"
        )
        st.caption(f"1 px² = {1/scale_px_per_um**2:.4f} µm²")

    st.markdown("**Segmentation Filters**")
    # Min area slider — label and range switch with unit toggle
    if use_microns and scale_px_per_um:
        um2_per_px2   = 1.0 / scale_px_per_um ** 2
        min_area_um2  = st.slider(
            "Min Cell Area (µm²)",
            min_value=round(10  * um2_per_px2, 4),
            max_value=round(2000 * um2_per_px2, 2),
            value=round(50  * um2_per_px2, 4),
            step=round(10  * um2_per_px2, 4),
            format="%.4f",
        )
        st.caption(f"= {min_area_um2 / um2_per_px2:.0f} px²")
        min_area = int(min_area_um2 / um2_per_px2)   # convert back to px for segmentation
    else:
        min_area = st.slider("Min Cell Area (px²)", 10, 2000, 50, 10)

    min_circ      = st.slider("Min Circularity",    0.0, 1.0, 0.1, 0.01)
    max_circ      = st.slider("Max Circularity",    0.0, 1.0, 1.0, 0.01)
    min_intensity = st.slider("Min Mean Intensity", 0,   255,  15,  1)

    st.markdown("**Thresholding**")
    thresh_method = st.radio("Method", ["Otsu (auto)", "Manual"], index=0)
    manual_thresh = None
    if thresh_method == "Manual":
        manual_thresh = st.slider("Manual Threshold", 0, 255, 30, 1)

    st.markdown("**Display**")
    outline_color = st.selectbox("Outline Colour", ["yellow","cyan","magenta","white"])


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────
st.markdown('<div class="main-header">🔬 Microscopic Cell Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a fluorescence or brightfield microscopy image to detect, measure, and summarise cells.</div>', unsafe_allow_html=True)

c1, c2 = st.columns([3, 1])
with c1:
    uploaded = st.file_uploader(
        "Upload image",
        type=["tif","tiff","png","jpg","jpeg","bmp"],
        label_visibility="collapsed",
    )
with c2:
    st.info("**Formats**\nTIF · PNG · JPG · BMP\nRGB · RGBA · Grayscale")

if uploaded is None:
    st.markdown("---")
    st.markdown(
        "#### How to use\n"
        "1. Upload a microscopy image above.\n"
        "2. The app auto-detects the active fluorescence channel.\n"
        "3. If ambiguous, choose Green / Red / Grayscale from the dropdown.\n"
        "4. Tune parameters in the sidebar if needed.\n"
        "5. Download annotated image or CSV reports with the buttons below results."
    )
    st.stop()


# ─── Load + channel selection ───────────────────
img_array    = load_image(uploaded)
auto_channel, ch_means = detect_channel(img_array)
ndim_str     = f"shape {img_array.shape} · dtype {img_array.dtype}"

if auto_channel == 'unknown' or (img_array.ndim > 2 and auto_channel not in ('red','green','blue','grayscale')):
    st.warning(f"⚠️ Channel auto-detection inconclusive ({ndim_str}). Please select below.")
    opts = ["Green","Red","Blue","Grayscale (average)"]
    pick = st.selectbox("Select channel for cell detection", opts)
    channel = 'gray' if 'Grayscale' in pick else pick.lower()
else:
    channel = auto_channel
    st.success(f"✅ Auto-detected channel: **{channel.capitalize()}** — {ndim_str}")


# ─── Analyse ─────────────────────────────────────
with st.spinner("Analysing cells…"):
    cell_channel = extract_channel(img_array, channel)
    tm  = 'otsu' if thresh_method == "Otsu (auto)" else 'manual'
    labels, threshold_used, valid_cells = segment_cells(
        cell_channel, min_area, min_circ, max_circ, min_intensity,
        threshold_method=tm, manual_threshold=manual_thresh
    )
    results_df, area_col = build_results_df(valid_cells, use_microns, scale_px_per_um)
    summary_df  = build_summary(results_df, area_col, cell_channel, valid_cells, img_array.shape, use_microns, scale_px_per_um)
    overlay_img = build_overlay(img_array, labels, valid_cells, outline_color)


# ─── Quick metric row ────────────────────────────
st.markdown("---")
m1,m2,m3,m4,m5 = st.columns(5)
n = len(results_df)
area_unit = "µm²" if (use_microns and scale_px_per_um) else "px²"
m1.metric("🦠 Cells",                    n)
m2.metric(f"📐 Avg Area ({area_unit})",  f"{results_df[area_col].mean():.2f}"    if n else "—")
m3.metric("⭕ Avg Circularity",          f"{results_df['Circularity'].mean():.3f}"   if n else "—")
m4.metric("💡 Avg Intensity",            f"{results_df['Mean Intensity'].mean():.1f}" if n else "—")
m5.metric("🗺️ Spatial Score",           f"{summary_df['Spatial Distribution (0–1)'].iloc[0]:.3f}" if n else "—")


# ─── Annotated image ─────────────────────────────
st.markdown('<div class="section-title">📸 Annotated Image — Detected Cell Outlines</div>', unsafe_allow_html=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#111')

orig_disp = img_array[:,:,:3] if img_array.ndim==3 else img_array
axes[0].imshow(orig_disp, cmap='gray' if img_array.ndim==2 else None)
axes[0].set_title("Original Image", color='white', fontsize=13)
axes[0].axis('off')

axes[1].imshow(overlay_img)
axes[1].set_title(
    f"Detected Cells = {n}  |  Threshold = {threshold_used:.1f}  |  Channel = {channel.capitalize()}",
    color='white', fontsize=13
)
axes[1].axis('off')
plt.tight_layout(pad=0.5)

ic, dc = st.columns([4,1])
with ic:
    st.pyplot(fig, use_container_width=True)
with dc:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.download_button("⬇️ Download Image", data=fig_to_bytes(fig),
                       file_name="annotated_cells.png", mime="image/png")
plt.close(fig)


# ─── Cell measurements table ─────────────────────
st.markdown('<div class="section-title">📊 Individual Cell Measurements</div>', unsafe_allow_html=True)

if results_df.empty:
    st.warning("No cells detected. Try reducing Min Area or Min Intensity in the sidebar.")
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
        height=min(420, 40 + 36*len(results_df)),
    )
    st.download_button(
        "🖨️ Download Cell Measurements CSV",
        data=df_to_csv_bytes(results_df),
        file_name="cell_measurements.csv",
        mime="text/csv",
    )


# ─── Summary table ───────────────────────────────
st.markdown('<div class="section-title">📋 Summary Statistics</div>', unsafe_allow_html=True)

if not summary_df.empty:
    st.dataframe(
        summary_df.T.rename(columns={0:"Value"}),
        use_container_width=True,
    )
    st.download_button(
        "🖨️ Download Summary CSV",
        data=df_to_csv_bytes(summary_df),
        file_name="cell_summary.csv",
        mime="text/csv",
    )
    st.markdown("""
    > **Spatial Distribution (0–1)**: Clark-Evans nearest-neighbour index.  
    > `0` = cells clustered at one spot · `1` = cells evenly spread across the surface.
    """)


# ─── Intensity histogram ─────────────────────────
with st.expander("📈 Channel Intensity Histogram"):
    fig2, ax = plt.subplots(figsize=(8,3))
    ax.hist(cell_channel.ravel(), bins=128, color='#1a7a4a', alpha=0.85)
    ax.axvline(threshold_used, color='red', linestyle='--', lw=1.5,
               label=f"Threshold = {threshold_used:.1f}")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Count")
    ax.set_title(f"Intensity Distribution — {channel.capitalize()} channel")
    ax.legend()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
