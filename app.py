"""
Interactive demo v4: upload a brain MRI image, auto-tune or manually adjust
pipeline parameters, download outputs, and get a plain-language explanation
of what the processing actually did.

Run with:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import deque
import io

st.set_page_config(page_title="MRI Enhancement & Segmentation", layout="wide")
st.title("Medical Image Enhancement & Segmentation")
st.caption("Upload a brain MRI image to run it through the full pipeline. Adjust parameters in the sidebar, or let the app auto-tune them for you.")

st.warning(
    "⚠️ **This is a student mini-project, not a medical device.** "
    "It uses basic classical image processing (no AI/clinical validation) and "
    "cannot diagnose anything or assess urgency. Nothing shown here should be "
    "used to make any health decision — only a certified radiologist or doctor "
    "can interpret a real scan."
)

with st.expander("ℹ️ New here? Click for a simple explanation of what this app does"):
    st.markdown("""
    This tool takes a brain MRI scan and does two things to it:

    **1. Makes it easier to see.** Raw MRI scans often look washed-out or grainy.
    We clean up the noise and boost the contrast so details are more visible —
    similar to adjusting brightness/contrast on a photo, but smarter about it.

    **2. Tries to automatically outline the interesting region (like a tumor).**
    We try three different automatic methods and compare how well each one does.
    Spoiler: the simplest methods aren't very good at this — and that's actually
    an interesting finding, not a failure. Scroll down to see why.
    """)

# ---------------- Pipeline functions ----------------

def denoise(img):
    return cv2.medianBlur(img, 3)

def apply_clahe(img, clip_limit, tile_size):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)

def apply_hist_eq(img):
    return cv2.equalizeHist(img)

def global_threshold(img, fixed_value):
    _, mask = cv2.threshold(img, fixed_value, 255, cv2.THRESH_BINARY)
    return mask

def otsu_threshold(img):
    otsu_val, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, otsu_val

def get_robust_seed(img, border_margin=15, top_percentile=97):
    h, w = img.shape
    inner = img[border_margin:h-border_margin, border_margin:w-border_margin]
    smoothed = cv2.GaussianBlur(inner, (5, 5), 0)
    thresh_val = np.percentile(smoothed, top_percentile)
    bright_mask = (smoothed >= thresh_val).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
    if num_labels <= 1:
        return (w // 2, h // 2)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cx, cy = centroids[largest_label]
    return (int(cx) + border_margin, int(cy) + border_margin)

def region_growing(img, seed, intensity_thresh, max_pixels_ratio=0.35):
    h, w = img.shape
    visited = np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)
    max_pixels = int(h * w * max_pixels_ratio)
    seed_val = int(img[seed[1], seed[0]])
    q = deque([seed])
    visited[seed[1], seed[0]] = True
    count = 0
    while q and count < max_pixels:
        x, y = q.popleft()
        mask[y, x] = 255
        count += 1
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                if abs(int(img[ny, nx]) - seed_val) <= intensity_thresh:
                    visited[ny, nx] = True
                    q.append((nx, ny))
    return mask

def psnr(original, processed):
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

def to_png_bytes(img_array):
    success, buf = cv2.imencode(".png", img_array)
    return io.BytesIO(buf.tobytes())

# ---------------- Auto-tuning ----------------

def auto_tune_clahe(original, denoised, min_psnr=17.0):
    """
    Searches a small grid of CLAHE settings and picks the one that gives the
    highest contrast gain while keeping PSNR above a floor (so it doesn't
    distort the image too much). This is the same trade-off you evaluated
    manually in Day 2 — just automated.
    """
    clip_candidates = [1.5, 2.0, 2.5, 3.0, 3.5]
    tile_candidates = [4, 8, 16]

    best = None
    for clip in clip_candidates:
        for tile in tile_candidates:
            result = apply_clahe(denoised, clip, tile)
            contrast = np.std(result)
            quality = psnr(original, result)
            if quality >= min_psnr:
                score = contrast  # maximize contrast subject to the PSNR floor
                if best is None or score > best["score"]:
                    best = {"clip": clip, "tile": tile, "contrast": contrast, "psnr": quality, "score": score}

    if best is None:
        # fallback: no candidate met the PSNR floor, just pick the gentlest setting
        return {"clip": 1.5, "tile": 8, "contrast": np.std(apply_clahe(denoised, 1.5, 8)),
                "psnr": psnr(original, apply_clahe(denoised, 1.5, 8))}
    return best

def auto_tune_region_growing(img, seed, target_min=0.005, target_max=0.05):
    """
    Searches intensity thresholds and picks the smallest one that produces a
    foreground ratio inside a 'plausible lesion size' band (0.5%-5% of the
    image, based on what your Day 4 results showed was realistic). Falls back
    to the threshold that gets closest to the band if none land inside it.
    """
    candidates = list(range(5, 51, 5))
    in_band = []
    closest = None
    closest_dist = None

    for t in candidates:
        mask = region_growing(img, seed, t)
        ratio = np.sum(mask == 255) / mask.size
        if target_min <= ratio <= target_max:
            in_band.append({"thresh": t, "ratio": ratio})
        dist = min(abs(ratio - target_min), abs(ratio - target_max)) if not (target_min <= ratio <= target_max) else 0
        if closest is None or dist < closest_dist:
            closest = {"thresh": t, "ratio": ratio}
            closest_dist = dist

    if in_band:
        return in_band[0]  # smallest threshold that lands in the plausible band
    return closest  # best available if nothing landed in-band

def generate_plain_explanation(contrast_before, contrast_after, region_ratio, otsu_ratio):
    contrast_gain_pct = ((contrast_after - contrast_before) / contrast_before) * 100
    lines = []
    lines.append(
        f"**Image clarity:** The scan's contrast was improved by about "
        f"{contrast_gain_pct:.0f}% during processing, which generally makes "
        f"fine details easier to see than in the original upload."
    )
    if region_ratio < 0.005:
        lines.append(
            "**Automated region detection:** The region-growing method did not "
            "find a clearly bounded bright area to highlight in this image. This "
            "can happen for several reasons — including that there may simply be "
            "no prominent bright region, or that image quality/positioning affected "
            "the algorithm. It is **not** a reliable way to confirm that nothing "
            "is present."
        )
    elif region_ratio < 0.05:
        lines.append(
            f"**Automated region detection:** A small, localized region "
            f"(roughly {region_ratio*100:.1f}% of the image) was automatically "
            f"highlighted by the algorithm. This simply means that area had a "
            f"cluster of pixels brighter than its surroundings — the algorithm "
            f"has no medical knowledge and cannot tell you whether this is "
            f"normal anatomy, an artifact, or something that would need a "
            f"doctor's attention."
        )
    else:
        lines.append(
            f"**Automated region detection:** A relatively large area "
            f"(roughly {region_ratio*100:.1f}% of the image) was highlighted. "
            f"As discussed in this project's own findings, simple algorithms "
            f"like this one often highlight broad tissue regions rather than "
            f"a specific area of concern, so a large highlighted area is "
            f"**not** itself meaningful without expert review."
        )
    lines.append(
        "**Bottom line:** This tool shows you *how* an image was processed and "
        "*what pixels* an algorithm flagged — it does not and cannot tell you "
        "what is actually happening in your body, and it cannot assess whether "
        "anything requires urgent attention. If this scan is real, please show "
        "the original to a radiologist or your doctor."
    )
    return "\n\n".join(lines)

# ---------------- Sidebar controls ----------------

st.sidebar.header("Pipeline Parameters")
auto_tune = st.sidebar.checkbox("🪄 Auto-tune parameters for this image", value=False)

st.sidebar.subheader("Enhancement (CLAHE)")
clip_limit = st.sidebar.slider("Clip limit", 1.0, 5.0, 2.0, 0.1, disabled=auto_tune)
tile_size = st.sidebar.slider("Tile grid size", 4, 16, 8, 1, disabled=auto_tune)

st.sidebar.subheader("Global Threshold")
global_thresh_val = st.sidebar.slider("Threshold value", 0, 255, 127, 1)

st.sidebar.subheader("Region Growing")
intensity_thresh = st.sidebar.slider("Intensity similarity threshold", 5, 50, 20, 1, disabled=auto_tune)

# ---------------- UI ----------------

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("L")
    img = np.array(pil_img)

    with st.spinner("Processing..."):
        denoised = denoise(img)
        hist_eq = apply_hist_eq(denoised)

        if auto_tune:
            clahe_result = auto_tune_clahe(img, denoised)
            clip_limit, tile_size = clahe_result["clip"], clahe_result["tile"]
        clahe_img = apply_clahe(denoised, clip_limit, tile_size)

        global_mask = global_threshold(clahe_img, global_thresh_val)
        otsu_mask, otsu_val = otsu_threshold(clahe_img)
        seed = get_robust_seed(clahe_img)

        if auto_tune:
            rg_result = auto_tune_region_growing(clahe_img, seed)
            intensity_thresh = rg_result["thresh"]
        region_mask = region_growing(clahe_img, seed, intensity_thresh)

    if auto_tune:
        st.success(
            f"🪄 Auto-tuned parameters for this image — "
            f"CLAHE clip limit: {clip_limit}, tile size: {tile_size} | "
            f"Region growing intensity threshold: {intensity_thresh}"
        )

    st.subheader("1. Enhancement")
    col1, col2, col3 = st.columns(3)
    col1.image(img, caption="Original", use_container_width=True)
    col2.image(hist_eq, caption=f"Histogram Eq. (std={np.std(hist_eq):.1f}, PSNR={psnr(img, hist_eq):.1f})", use_container_width=True)
    col3.image(clahe_img, caption=f"CLAHE (std={np.std(clahe_img):.1f}, PSNR={psnr(img, clahe_img):.1f})", use_container_width=True)
    col3.download_button("Download CLAHE image", to_png_bytes(clahe_img), "clahe_enhanced.png", "image/png")

    with st.expander("ℹ️ What am I looking at? (Enhancement)"):
        st.markdown("""
        - **Original**: the scan exactly as uploaded — often dim or low-contrast.
        - **Histogram Equalization**: an older, simpler contrast-boosting method.
          It stretches brightness across the whole image at once — works, but can
          overdo it and make things look unnatural.
        - **CLAHE**: a smarter version that boosts contrast in small local patches
          instead of the whole image at once, so it doesn't overexpose bright
          areas or crush dark ones. Generally the better result of the two.
        """)

    st.subheader("2. Segmentation (on CLAHE-enhanced image)")
    col4, col5, col6 = st.columns(3)
    global_ratio = np.sum(global_mask == 255) / global_mask.size
    otsu_ratio = np.sum(otsu_mask == 255) / otsu_mask.size
    region_ratio = np.sum(region_mask == 255) / region_mask.size

    col4.image(global_mask, caption=f"Global Threshold (fg={global_ratio:.3f})", use_container_width=True)
    col4.download_button("Download mask", to_png_bytes(global_mask), "global_mask.png", "image/png")

    col5.image(otsu_mask, caption=f"Otsu (threshold={otsu_val:.0f}, fg={otsu_ratio:.3f})", use_container_width=True)
    col5.download_button("Download mask", to_png_bytes(otsu_mask), "otsu_mask.png", "image/png")

    col6.image(region_mask, caption=f"Region Growing (seed={seed}, fg={region_ratio:.4f})", use_container_width=True)
    col6.download_button("Download mask", to_png_bytes(region_mask), "region_mask.png", "image/png")

    with st.expander("ℹ️ What am I looking at? (Segmentation)"):
        st.markdown("""
        Each method tries to draw a white outline around the "important" region
        (ideally, just the tumor) and leave everything else black.

        - **Global Threshold**: picks one fixed brightness cutoff for the whole
          image. Simple, but not adaptive — usually grabs too much or too little.
        - **Otsu**: automatically picks the best cutoff for each image, but it's
          still just one brightness line — it tends to separate the whole brain
          from the black background, not the tumor from healthy brain tissue.
        - **Region Growing**: starts from one bright "seed" point and expands
          outward only through similar-brightness neighbors — like a paint
          bucket tool that stops at edges. This is why it tends to isolate a
          smaller, more localized blob — closer to what a real tumor looks like.

        **Foreground ratio** = what % of the image got marked white. A tumor is
        small, so a *lower* number (not higher) is usually the better sign here.
        """)

    st.subheader("Summary")
    st.markdown(f"""
    | Metric | Value |
    |---|---|
    | Original contrast (std) | {np.std(img):.2f} |
    | CLAHE contrast (std) | {np.std(clahe_img):.2f} |
    | Global threshold foreground ratio | {global_ratio:.4f} |
    | Otsu foreground ratio | {otsu_ratio:.4f} |
    | Region growing foreground ratio | {region_ratio:.4f} |
    """)

    st.subheader("Full Pipeline Panel")
    panel = np.hstack([img, clahe_img, global_mask, otsu_mask, region_mask])
    st.image(panel, caption="Original | CLAHE | Global | Otsu | Region Growing", use_container_width=True)
    st.download_button("Download full panel", to_png_bytes(panel), "full_pipeline_panel.png", "image/png")

    st.subheader("What does this mean for you? (Plain-language explanation)")
    st.error(
        "This explanation is generated from basic pixel statistics only. "
        "It is **not a diagnosis** and cannot assess urgency."
    )
    st.markdown(generate_plain_explanation(
        np.std(img), np.std(clahe_img), region_ratio, otsu_ratio
    ))
else:
    st.info("Upload an image to see it processed through the pipeline.")