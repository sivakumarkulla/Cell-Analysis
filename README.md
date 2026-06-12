# 🔬 Microscopic Cell Analyser

A Streamlit web app for automated detection and measurement of fluorescent cells in microscopy images.

## Features

- **Auto channel detection** — detects Green / Red / Blue / Grayscale automatically; shows a dropdown when ambiguous
- **Cell segmentation** — Otsu thresholding (or manual) + connected-component labelling via scikit-image
- **Per-cell measurements**: Area, Mean Intensity, Circularity, Centroid X/Y
- **Summary statistics**: Cell count, averages, % surface coverage, background intensity, spatial distribution score
- **Annotated image** with coloured outlines around each detected cell
- **CSV exports** for both the measurements table and summary table
- **Histogram** of channel intensity with threshold overlay
- Works with `.tif`, `.tiff`, `.png`, `.jpg`, `.bmp` — 8-bit or 16-bit, RGB / RGBA / Grayscale

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Parameters (sidebar)

| Parameter | Default | Description |
|---|---|---|
| Min Cell Area | 50 px² | Discard objects smaller than this |
| Min / Max Circularity | 0.1 – 1.0 | 1 = perfect circle |
| Min Mean Intensity | 15 | Discard dim objects |
| Threshold Method | Otsu | Auto or manual pixel threshold |
| Outline Colour | yellow | Colour of drawn cell outlines |

## Spatial Distribution Score

Uses the **Clark-Evans nearest-neighbour index** normalised to [0, 1]:

- **0** — cells clustered at a single point
- **1** — cells evenly / uniformly distributed across the image surface

## Supported Image Types

| Type | Channels | Notes |
|---|---|---|
| TIF / TIFF | RGB, RGBA, Grayscale | 8-bit and 16-bit supported |
| PNG | RGB, RGBA, Grayscale | — |
| JPG / JPEG | RGB | — |
| BMP | RGB | — |
