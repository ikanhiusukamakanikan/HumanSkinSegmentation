from __future__ import annotations

import io
import math
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Ellipse
from PIL import Image
from sklearn.mixture import GaussianMixture


DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/"
    "Skin_NonSkin.txt"
)
LOCAL_DATASET_PATH = Path("data") / "Skin_NonSkin.txt"
APP_TITLE = "Human skin segmentation with the GMM-EM algorithm"


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
)


@st.cache_data(show_spinner=False)
def download_dataset() -> bytes:
    with urllib.request.urlopen(DATASET_URL, timeout=45) as response:
        return response.read()


@st.cache_data(show_spinner=False)
def parse_dataset(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data), header=None, sep=r"\s+", engine="python")
    if df.shape[1] < 4:
        raise ValueError("Dataset harus berisi empat kolom: B G R label.")

    df = df.iloc[:, :4].copy()
    df.columns = ["B", "G", "R", "skin"]
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna().astype({"B": "uint8", "G": "uint8", "R": "uint8", "skin": "int16"})
    labels = set(df["skin"].unique().tolist())
    if not {1, 2}.issubset(labels):
        raise ValueError("Label dataset harus memakai 1 untuk kulit dan 2 untuk non-kulit.")

    return df


@st.cache_data(show_spinner=False)
def dataset_to_cbcr(df: pd.DataFrame) -> pd.DataFrame:
    b = df["B"].astype(np.float64).to_numpy()
    g = df["G"].astype(np.float64).to_numpy()
    r = df["R"].astype(np.float64).to_numpy()

    cb = np.round(128 - 0.168736 * r - 0.331364 * g + 0.5 * b)
    cr = np.round(128 + 0.5 * r - 0.418688 * g - 0.081312 * b)

    out = pd.DataFrame(
        {
            "Cb": np.clip(cb, 0, 255).astype(np.float32),
            "Cr": np.clip(cr, 0, 255).astype(np.float32),
            "skin": df["skin"].astype(np.int16).to_numpy(),
        }
    )
    return out


def sample_rows(values: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or len(values) <= max_rows:
        return values

    rng = np.random.default_rng(seed)
    indexes = rng.choice(len(values), size=max_rows, replace=False)
    return values[indexes]


@st.cache_resource(show_spinner=False)
def fit_gmm_models(
    skin_data: np.ndarray,
    non_skin_data: np.ndarray,
    n_components: int,
    random_state: int,
) -> tuple[GaussianMixture, GaussianMixture]:
    skin_model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        reg_covar=1e-6,
        max_iter=200,
    ).fit(skin_data)

    non_skin_model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        reg_covar=1e-6,
        max_iter=200,
    ).fit(non_skin_data)

    return skin_model, non_skin_model


def pil_to_rgb_array(image: Image.Image, max_side: int) -> np.ndarray:
    image = image.convert("RGB")
    width, height = image.size
    longest = max(width, height)

    if longest > max_side:
        scale = max_side / longest
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return np.asarray(image, dtype=np.uint8)


def rgb_to_cbcr_features(rgb: np.ndarray) -> np.ndarray:
    rgb_float = rgb.astype(np.float64)
    r = rgb_float[..., 0]
    g = rgb_float[..., 1]
    b = rgb_float[..., 2]

    cb = np.round(128 - 0.168736 * r - 0.331364 * g + 0.5 * b)
    cr = np.round(128 + 0.5 * r - 0.418688 * g - 0.081312 * b)

    features = np.column_stack(
        [
            np.clip(cb.ravel(), 0, 255),
            np.clip(cr.ravel(), 0, 255),
        ]
    )
    return features.astype(np.float32)


def clean_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8 > 127


def segment_skin(
    rgb: np.ndarray,
    skin_model: GaussianMixture,
    non_skin_model: GaussianMixture,
    threshold: float,
    kernel_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = rgb_to_cbcr_features(rgb)
    skin_score = skin_model.score_samples(features)
    non_skin_score = non_skin_model.score_samples(features)
    score_delta = skin_score - non_skin_score

    mask = (score_delta > threshold).reshape(rgb.shape[:2])
    mask = clean_mask(mask, kernel_size)

    segmented = rgb.copy()
    segmented[~mask] = 0

    overlay = rgb.astype(np.float32)
    highlight = np.array([24, 190, 120], dtype=np.float32)
    overlay[mask] = 0.55 * overlay[mask] + 0.45 * highlight
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    mask_image = (mask.astype(np.uint8) * 255)
    return mask, mask_image, segmented, overlay


def add_gmm_ellipses(
    ax: plt.Axes,
    model: GaussianMixture,
    edge_color: str,
    label_prefix: str,
) -> None:
    for index, (mean, covariance) in enumerate(zip(model.means_, model.covariances_)):
        values, vectors = np.linalg.eigh(covariance[:2, :2])
        order = values.argsort()[::-1]
        values = np.maximum(values[order], 1e-8)
        vectors = vectors[:, order]

        angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
        width, height = 2.0 * math.sqrt(2.0) * np.sqrt(values)
        ellipse = Ellipse(
            xy=mean[:2],
            width=width,
            height=height,
            angle=angle,
            facecolor=edge_color,
            edgecolor=edge_color,
            linewidth=2,
            alpha=0.18,
            label=f"{label_prefix} {index + 1}" if index == 0 else None,
        )
        ax.add_patch(ellipse)


def plot_cbcr_model(
    cbcr_df: pd.DataFrame,
    skin_model: GaussianMixture,
    non_skin_model: GaussianMixture,
    max_points: int,
    seed: int,
) -> plt.Figure:
    skin_values = cbcr_df.loc[cbcr_df["skin"] == 1, ["Cb", "Cr"]].to_numpy()
    non_skin_values = cbcr_df.loc[cbcr_df["skin"] == 2, ["Cb", "Cr"]].to_numpy()
    skin_plot = sample_rows(skin_values, max_points, seed)
    non_skin_plot = sample_rows(non_skin_values, max_points, seed + 1)

    fig, ax = plt.subplots(figsize=(8.5, 6.3))
    ax.scatter(
        non_skin_plot[:, 0],
        non_skin_plot[:, 1],
        s=8,
        c="#53616d",
        alpha=0.18,
        linewidths=0,
        label="Non-kulit",
    )
    ax.scatter(
        skin_plot[:, 0],
        skin_plot[:, 1],
        s=9,
        c="#d85c45",
        alpha=0.35,
        linewidths=0,
        label="Kulit",
    )
    add_gmm_ellipses(ax, non_skin_model, "#2d6cdf", "GMM non-kulit")
    add_gmm_ellipses(ax, skin_model, "#ef8b2c", "GMM kulit")

    ax.set_title("Distribusi Cb-Cr dan komponen GMM")
    ax.set_xlabel("Cb")
    ax.set_ylabel("Cr")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.grid(True, color="#e3e7eb", linewidth=0.8)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def read_dataset_from_source(source: str, uploaded_file: st.runtime.uploaded_file_manager.UploadedFile | None) -> bytes | None:
    if source == "Upload file":
        if uploaded_file is None:
            return None
        return uploaded_file.getvalue()

    if source == "File lokal":
        if not LOCAL_DATASET_PATH.exists():
            return None
        return LOCAL_DATASET_PATH.read_bytes()

    return download_dataset()


st.title(APP_TITLE)
st.caption("Berbasis resep buku: GMM kulit dan non-kulit pada ruang warna Cb-Cr.")

st.sidebar.header("Dataset")
source_options = ["Upload file", "Unduh UCI", "File lokal"]
default_source = 2 if LOCAL_DATASET_PATH.exists() else 0
dataset_source = st.sidebar.radio("Sumber data latih", source_options, index=default_source)

uploaded_dataset = None
if dataset_source == "Upload file":
    uploaded_dataset = st.sidebar.file_uploader(
        "Skin_NonSkin.txt",
        type=["txt", "csv", "data"],
    )

dataset_bytes = None
dataset_error = None
try:
    dataset_bytes = read_dataset_from_source(dataset_source, uploaded_dataset)
except (urllib.error.URLError, TimeoutError, OSError) as exc:
    dataset_error = f"Gagal memuat dataset: {exc}"

if dataset_error:
    st.sidebar.error(dataset_error)

if dataset_bytes is None:
    st.info("Muat dataset Skin_NonSkin.txt untuk melatih model.")
    st.stop()

try:
    dataset_df = parse_dataset(dataset_bytes)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if dataset_source == "Unduh UCI":
    if st.sidebar.button("Simpan ke data/"):
        LOCAL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_DATASET_PATH.write_bytes(dataset_bytes)
        st.sidebar.success(f"Tersimpan: {LOCAL_DATASET_PATH}")

cbcr_df = dataset_to_cbcr(dataset_df)
counts = dataset_df["skin"].value_counts()
skin_count = int(counts.get(1, 0))
non_skin_count = int(counts.get(2, 0))

st.sidebar.header("Model")
max_count = max(skin_count, non_skin_count)
sample_limit = st.sidebar.slider(
    "Maks sample per kelas",
    min_value=1_000,
    max_value=max(1_000, min(max_count, 200_000)),
    value=min(50_000, max(1_000, max_count)),
    step=1_000,
)
n_components = st.sidebar.slider(
    "Komponen GMM",
    min_value=1,
    max_value=max(1, min(8, skin_count, non_skin_count)),
    value=min(4, skin_count, non_skin_count),
)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.header("Segmentasi")
threshold = st.sidebar.slider(
    "Ambang score kulit",
    min_value=-25.0,
    max_value=25.0,
    value=0.0,
    step=0.5,
)
kernel_size = st.sidebar.select_slider(
    "Pembersihan mask",
    options=[1, 3, 5, 7, 9],
    value=3,
)
max_side = st.sidebar.slider(
    "Maks sisi gambar",
    min_value=256,
    max_value=1800,
    value=900,
    step=128,
)

skin_values = cbcr_df.loc[cbcr_df["skin"] == 1, ["Cb", "Cr"]].to_numpy(dtype=np.float32)
non_skin_values = cbcr_df.loc[cbcr_df["skin"] == 2, ["Cb", "Cr"]].to_numpy(dtype=np.float32)
skin_train = sample_rows(skin_values, sample_limit, int(random_state))
non_skin_train = sample_rows(non_skin_values, sample_limit, int(random_state) + 1)

with st.spinner("Melatih dua Gaussian Mixture Model..."):
    skin_gmm, non_skin_gmm = fit_gmm_models(
        skin_train,
        non_skin_train,
        int(n_components),
        int(random_state),
    )

metric_cols = st.columns(4)
metric_cols[0].metric("Sample kulit", f"{skin_count:,}")
metric_cols[1].metric("Sample non-kulit", f"{non_skin_count:,}")
metric_cols[2].metric("Latih kulit", f"{len(skin_train):,}")
metric_cols[3].metric("Latih non-kulit", f"{len(non_skin_train):,}")

segment_tab, model_tab = st.tabs(["Segmentasi gambar", "Model Cb-Cr"])

with segment_tab:
    image_file = st.file_uploader(
        "Gambar uji",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    if image_file is None:
        st.info("Upload gambar untuk menjalankan segmentasi.")
    else:
        image = Image.open(image_file)
        rgb = pil_to_rgb_array(image, int(max_side))
        mask, mask_image, segmented, overlay = segment_skin(
            rgb,
            skin_gmm,
            non_skin_gmm,
            float(threshold),
            int(kernel_size),
        )

        coverage = float(mask.mean() * 100)
        st.metric("Area terdeteksi kulit", f"{coverage:.2f}%")

        img_cols = st.columns(4)
        img_cols[0].image(rgb, caption="Original")
        img_cols[1].image(mask_image, caption="Mask")
        img_cols[2].image(segmented, caption="Segmentasi")
        img_cols[3].image(overlay, caption="Overlay")

with model_tab:
    max_points = st.slider(
        "Titik plot per kelas",
        min_value=500,
        max_value=20_000,
        value=4_000,
        step=500,
    )
    fig = plot_cbcr_model(
        cbcr_df,
        skin_gmm,
        non_skin_gmm,
        int(max_points),
        int(random_state),
    )
    st.pyplot(fig, clear_figure=True)
