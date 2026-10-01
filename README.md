# Medical Image Enhancement & Segmentation

A mini project for Image and Video Processing (IVP) that builds a complete pipeline to enhance and segment brain MRI scans using classical image processing techniques — no deep learning, just contrast enhancement, denoising, and multiple segmentation algorithms, wrapped in an interactive Streamlit app.

**Live demo:** https://ivp-mri-pipeline-qbwcsbmnowtewnawvjmtss.streamlit.app/

> ⚠️ **This is a student academic project, not a medical device.** It cannot diagnose anything and should never be used to make real health decisions.

---

## What it does

1. **Enhancement** — denoises the image (median filtering) and boosts contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization), compared against plain histogram equalization.
2. **Segmentation** — isolates the region of interest using three methods of increasing sophistication: global thresholding, Otsu's method, and region growing with a robust, artifact-resistant seed selection.
3. **Auto-tuning** — automatically searches CLAHE and region-growing parameters to find settings that maximize contrast/localize the region without over-distorting the image.
4. **Plain-language explanation** — translates the pipeline's pixel-level output into a non-technical summary of what was processed and found (explicitly non-diagnostic).

## Key finding

Classical, purely intensity-based segmentation (global threshold, Otsu) struggles to separate a tumor from healthy brain tissue because both can have overlapping brightness levels — these methods end up separating brain tissue from background instead. Region growing, by expanding only through locally similar pixels from a well-chosen seed, produces much smaller, more anatomically plausible regions. This motivates why real clinical tools use learning-based segmentation (e.g. U-Net) rather than classical thresholding alone.

## Tech stack

- Python 3.x
- OpenCV (`opencv-python-headless`)
- NumPy
- Streamlit (interactive web app)
- Pillow

## Project structure

ivp-mri-pipeline/
├── app.py # Streamlit app — full interactive pipeline
├── requirements.txt # Python dependencies
└── README.md

## Running locally

```bash
git clone https://github.com/YOUR-USERNAME/ivp-mri-pipeline.git
cd ivp-mri-pipeline
pip install -r requirements.txt
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`. Upload a brain MRI image (JPG/PNG) to run it through the pipeline.
