"""Streamlit interactive viewer for Brain Hemorrhage AI inference."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("BRAIN_HEMORRHAGE_API_URL", "http://127.0.0.1:8000")

LABEL_COLORS = {
    1: "#ff0000",
    2: "#ffff00",
    3: "#00ffff",
    4: "#00ff00",
    5: "#ff00ff",
}
LABEL_NAMES = {1: "EDH", 2: "SDH", 3: "SAH", 4: "IPH", 5: "IVH"}
LABEL_FULL_NAMES = {
    1: "epidural hemorrhage (EDH)",
    2: "subdural hemorrhage (SDH)",
    3: "subarachnoid hemorrhage (SAH)",
    4: "intraparenchymal hemorrhage (IPH)",
    5: "intraventricular hemorrhage (IVH)",
}


def _format_subtype_list(class_ids: list[int], full_names: bool = True) -> str:
    """Format detected class IDs as readable English."""
    if not class_ids:
        return "no hemorrhage subtypes"
    names = [
        LABEL_FULL_NAMES.get(int(class_id), LABEL_NAMES.get(int(class_id), str(class_id)))
        if full_names
        else LABEL_NAMES.get(int(class_id), str(class_id))
        for class_id in class_ids
    ]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _build_summary_paragraph(
    summary: dict[str, Any],
    confidence: dict[str, Any],
    volume_df: pd.DataFrame,
) -> str:
    """Build a one-paragraph clinical-style summary in plain English."""
    classes_present = summary.get("classes_present", [])
    ich_present = len(classes_present) > 0
    total_volume = float(summary.get("total_volume_ml", 0.0))
    study_confidence = float(confidence.get("study_confidence", 0.0))
    processing_time = float(summary.get("processing_time_sec", 0.0))
    spacing = summary.get("spacing", [0, 0, 0])
    shape = summary.get("shape", [0, 0, 0])
    scan_name = summary.get("scan_name", "this scan")

    sx, sy, sz = (float(spacing[0]), float(spacing[1]), float(spacing[2])) if len(spacing) >= 3 else (0.0, 0.0, 0.0)
    dx, dy, dz = (int(shape[0]), int(shape[1]), int(shape[2])) if len(shape) >= 3 else (0, 0, 0)

    if ich_present:
        finding = (
            f"Intracranial hemorrhage is **present** in {scan_name}. "
            f"The model detected {_format_subtype_list(classes_present)} "
            f"with an estimated total volume of **{total_volume:.2f} mL**."
        )
        subtype_rows = volume_df[
            (volume_df["subtype"] != "TOTAL") & (volume_df["volume_ml"].astype(float) > 0)
        ]
        if not subtype_rows.empty:
            parts = [
                f"{row['subtype']}: {float(row['volume_ml']):.2f} mL"
                for _, row in subtype_rows.iterrows()
            ]
            finding += f" Breakdown by subtype: {', '.join(parts)}."
    else:
        finding = (
            f"No intracranial hemorrhage was detected in {scan_name}. "
            f"Estimated hemorrhage volume is **{total_volume:.2f} mL**."
        )

    return (
        f"{finding} "
        f"The model's confidence in this result is **{study_confidence:.0%}**. "
        f"The CT volume is {dx} × {dy} × {dz} voxels "
        f"(in-plane spacing {sx:.2f} mm, slice thickness {sz:.2f} mm). "
        f"Analysis completed in {processing_time:.1f} seconds."
    )


def _confidence_table(confidence: dict[str, Any]) -> pd.DataFrame:
    """Present confidence metrics as a simple table."""
    rows = [
        {"Metric": "Overall study confidence", "Value": f"{float(confidence.get('study_confidence', 0)):.1%}"},
        {"Metric": "Mean softmax confidence", "Value": f"{float(confidence.get('mean_softmax_confidence', 0)):.1%}"},
    ]
    per_class = confidence.get("per_class_confidence", {})
    for label_str, value in sorted(per_class.items(), key=lambda item: int(item[0])):
        label = int(label_str)
        rows.append(
            {
                "Metric": f"{LABEL_NAMES.get(label, label)} confidence",
                "Value": f"{float(value):.1%}",
            }
        )
    return pd.DataFrame(rows)


def _volume_table_display(volume_df: pd.DataFrame) -> pd.DataFrame:
    """Show only clinically relevant volume rows."""
    display = volume_df.copy()
    display = display[display["subtype"] != "TOTAL"]
    display = display[display["volume_ml"].astype(float) > 0]
    if display.empty:
        return volume_df[volume_df["subtype"] == "TOTAL"]
    return display.rename(
        columns={
            "subtype": "Subtype",
            "volume_ml": "Volume (mL)",
            "percent_of_total_hemorrhage": "% of total",
        }
    )


def _label_cmap() -> mcolors.ListedColormap:
    return mcolors.ListedColormap(["#00000000"] + [LABEL_COLORS[i] for i in range(1, 6)])


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))


def _labels_in_slice(label_slice: np.ndarray) -> list[int]:
    return sorted(int(value) for value in np.unique(label_slice) if int(value) in LABEL_NAMES)


def _add_subtype_legend(ax: plt.Axes, labels_present: list[int]) -> None:
    if not labels_present:
        return
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=LABEL_COLORS[label],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=10,
            label=LABEL_NAMES[label],
        )
        for label in labels_present
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.92, title="Subtype")


def _draw_class_contours(ax: plt.Axes, label_slice: np.ndarray, labels: list[int], linewidth: float = 1.8) -> None:
    """Outline each hemorrhage region so small predictions remain visible."""
    for label in labels:
        class_mask = (label_slice == label).astype(np.float32)
        if class_mask.sum() == 0:
            continue
        color = LABEL_COLORS[label]
        ax.contour(class_mask.T, levels=[0.5], colors=[color], linewidths=linewidth, origin="lower")
        ax.contour(
            class_mask.T,
            levels=[0.5],
            colors=["white"],
            linewidths=max(0.8, linewidth * 0.45),
            linestyles="--",
            origin="lower",
        )


def _brain_window_slice(ct_volume: np.ndarray, slice_index: int, clip_min: float = -40.0, clip_max: float = 120.0) -> np.ndarray:
    ct_slice = ct_volume[:, :, slice_index]
    clipped = np.clip(ct_slice, clip_min, clip_max)
    return ((clipped - clip_min) / (clip_max - clip_min)).astype(np.float32)


def _ct_figure(ct_slice: np.ndarray, title: str) -> plt.Figure:
    """Original CT only — clinical grayscale, clearly separated from prediction panels."""
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#ececec")
    ax.set_facecolor("#000000")
    ax.imshow(ct_slice.T, cmap="bone", origin="lower", vmin=0.0, vmax=1.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#2c3e50")
        spine.set_linewidth(2.5)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1a1a1a", pad=10)
    ax.axis("off")
    fig.text(0.5, 0.02, "No segmentation overlay", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def _annotated_overlay_figure(ct_slice: np.ndarray, label_slice: np.ndarray, title: str) -> plt.Figure:
    """CT with colored hemorrhage regions, fill, and contour outlines."""
    labels_present = _labels_in_slice(label_slice)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#ececec")
    ax.imshow(ct_slice.T, cmap="gray", origin="lower", vmin=0.0, vmax=1.0, alpha=0.95)

    rgba = np.zeros((*label_slice.shape, 4), dtype=np.float32)
    for label in labels_present:
        mask = label_slice == label
        rgb = _hex_to_rgb(LABEL_COLORS[label])
        rgba[mask, 0] = rgb[0]
        rgba[mask, 1] = rgb[1]
        rgba[mask, 2] = rgb[2]
        rgba[mask, 3] = 0.72

    ax.imshow(rgba.transpose(1, 0, 2), origin="lower")
    _draw_class_contours(ax, label_slice, labels_present, linewidth=2.2)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1a1a1a", pad=10)
    _add_subtype_legend(ax, labels_present)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _prediction_mask_figure(label_slice: np.ndarray, title: str) -> plt.Figure:
    """Prediction-only map on black — makes hemorrhage areas easy to spot."""
    labels_present = _labels_in_slice(label_slice)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#111111")
    ax.set_facecolor("#000000")

    color_image = np.zeros((*label_slice.shape, 3), dtype=np.float32)
    for label in labels_present:
        mask = label_slice == label
        color_image[mask] = _hex_to_rgb(LABEL_COLORS[label])

    ax.imshow(color_image.transpose(1, 0, 2), origin="lower")
    _draw_class_contours(ax, label_slice, labels_present, linewidth=2.0)
    ax.set_title(title, fontsize=12, fontweight="bold", color="white", pad=10)
    _add_subtype_legend(ax, labels_present)
    ax.axis("off")
    if not labels_present:
        ax.text(
            0.5,
            0.5,
            "No hemorrhage\non this slice",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#aaaaaa",
            fontsize=11,
        )
    fig.tight_layout()
    return fig


def _side_by_side_figure(ct_slice: np.ndarray, label_slice: np.ndarray) -> plt.Figure:
    """Wide comparison: original CT vs annotated prediction."""
    labels_present = _labels_in_slice(label_slice)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#f8f8f8")

    axes[0].set_facecolor("#000000")
    axes[0].imshow(ct_slice.T, cmap="bone", origin="lower", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original CT", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].set_facecolor("#000000")
    axes[1].imshow(ct_slice.T, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    rgba = np.zeros((*label_slice.shape, 4), dtype=np.float32)
    for label in labels_present:
        mask = label_slice == label
        rgb = _hex_to_rgb(LABEL_COLORS[label])
        rgba[mask, :3] = rgb
        rgba[mask, 3] = 0.75
    axes[1].imshow(rgba.transpose(1, 0, 2), origin="lower")
    _draw_class_contours(axes[1], label_slice, labels_present, linewidth=2.4)
    axes[1].set_title("Predicted hemorrhage (marked)", fontsize=12, fontweight="bold")
    _add_subtype_legend(axes[1], labels_present)
    axes[1].axis("off")

    fig.suptitle("Original vs predicted", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def _overlay_figure(ct_slice: np.ndarray, label_slice: np.ndarray, title: str) -> plt.Figure:
    """Ground-truth overlay using the same annotation style as predictions."""
    return _annotated_overlay_figure(ct_slice, label_slice, title)


def _label_slice_figure(label_slice: np.ndarray, title: str) -> plt.Figure:
    return _prediction_mask_figure(label_slice, title)


def _fetch_file_bytes(download_url: str) -> bytes:
    response = requests.get(f"{API_URL}{download_url}", timeout=120)
    response.raise_for_status()
    return response.content


def _load_nifti_volume_from_bytes(data: bytes) -> np.ndarray:
    """Load a NIfTI volume from in-memory bytes (nibabel requires a file path)."""
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as handle:
        handle.write(data)
        temp_path = Path(handle.name)
    try:
        return np.asarray(nib.load(str(temp_path)).get_fdata())
    finally:
        temp_path.unlink(missing_ok=True)


def _cache_volumes(payload: dict[str, Any], ct_bytes: bytes, mask_bytes: bytes | None) -> None:
    """Load and cache volumes for the slice viewer (once per prediction job)."""
    job_id = payload["job_id"]
    if st.session_state.get("volumes_job_id") == job_id:
        return

    st.session_state["ct_volume"] = _load_nifti_volume_from_bytes(ct_bytes).astype(np.float32)
    pred_bytes = _fetch_file_bytes(payload["download_urls"]["prediction_mask.nii.gz"])
    st.session_state["pred_volume"] = _load_nifti_volume_from_bytes(pred_bytes).astype(np.int64)
    st.session_state["mask_volume"] = (
        _load_nifti_volume_from_bytes(mask_bytes).astype(np.int64) if mask_bytes else None
    )
    st.session_state["overlay_bytes"] = _fetch_file_bytes(payload["download_urls"]["overlay.png"])
    st.session_state["pred_mask_bytes"] = pred_bytes
    st.session_state["volumes_job_id"] = job_id


def main() -> None:
    st.set_page_config(page_title="Brain Hemorrhage AI", layout="wide")
    st.title("Brain Hemorrhage AI")
    st.caption("Upload a CT scan, run inference, and review hemorrhage segmentation results.")

    uploaded_ct = st.file_uploader("Upload CT (.nii.gz)", type=["nii", "gz"])
    uploaded_mask = st.file_uploader("Optional ground-truth mask (.nii.gz)", type=["nii", "gz"])

    if st.button("Run prediction", type="primary", disabled=uploaded_ct is None):
        with st.spinner("Running inference..."):
            files = {"file": (uploaded_ct.name, uploaded_ct.getvalue(), "application/gzip")}
            if uploaded_mask is not None:
                files["mask"] = (uploaded_mask.name, uploaded_mask.getvalue(), "application/gzip")
            response = requests.post(f"{API_URL}/predict", files=files, timeout=600)
            response.raise_for_status()
            payload = response.json()
        st.session_state["prediction"] = payload
        st.session_state["ct_bytes"] = uploaded_ct.getvalue()
        st.session_state["mask_bytes"] = uploaded_mask.getvalue() if uploaded_mask is not None else None
        st.success("Prediction complete.")

    payload = st.session_state.get("prediction")
    if not payload:
        st.info("Upload a CT scan and click **Run prediction** to begin.")
        return

    summary = payload["prediction_summary"]
    confidence = payload["confidence"]
    volume_df = pd.DataFrame(payload["volume_report"])
    classes_present = summary.get("classes_present", [])
    ich_present = len(classes_present) > 0

    st.subheader("Clinical Summary")
    st.markdown(_build_summary_paragraph(summary, confidence, volume_df))

    st.subheader("Key findings")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ICH present", "Yes" if ich_present else "No")
    col2.metric("Total volume (mL)", f"{summary.get('total_volume_ml', 0):.2f}")
    col3.metric("Confidence", f"{confidence.get('study_confidence', 0):.0%}")
    col4.metric("Analysis time", f"{summary.get('processing_time_sec', 0):.1f} s")

    _cache_volumes(
        payload,
        ct_bytes=st.session_state["ct_bytes"],
        mask_bytes=st.session_state.get("mask_bytes"),
    )

    ct_volume = st.session_state["ct_volume"]
    pred_volume = st.session_state["pred_volume"]
    gt_volume = st.session_state.get("mask_volume")
    mask_bytes = st.session_state["pred_mask_bytes"]
    overlay_bytes = st.session_state["overlay_bytes"]

    depth = ct_volume.shape[2]
    default_slice = int(summary.get("largest_slice", depth // 2))
    slice_index = st.slider("Axial slice", min_value=0, max_value=depth - 1, value=min(default_slice, depth - 1))

    st.subheader("Slice Viewer")
    st.caption(
        "Compare the original CT (left) with the annotated prediction (center). "
        "Colored regions show detected hemorrhage; dashed white outlines mark boundaries."
    )
    ct_slice = _brain_window_slice(ct_volume, slice_index)
    pred_slice = pred_volume[:, :, slice_index].astype(np.int64)

    st.pyplot(_side_by_side_figure(ct_slice, pred_slice), clear_figure=True, use_container_width=True)

    detail_columns = st.columns(4 if gt_volume is not None else 3)
    detail_columns[0].pyplot(_ct_figure(ct_slice, "Original CT"), clear_figure=True)
    detail_columns[1].pyplot(
        _annotated_overlay_figure(ct_slice, pred_slice, "CT + predicted hemorrhage"),
        clear_figure=True,
    )
    detail_columns[2].pyplot(_prediction_mask_figure(pred_slice, "Prediction map"), clear_figure=True)
    if gt_volume is not None:
        gt_slice = gt_volume[:, :, slice_index].astype(np.int64)
        detail_columns[3].pyplot(_overlay_figure(ct_slice, gt_slice, "Ground truth"), clear_figure=True)

    middle_slice = depth // 2
    if slice_index != middle_slice:
        st.caption(f"Middle slice index: {middle_slice}")

    st.image(overlay_bytes, caption="Generated overlay (largest hemorrhage slice)", use_container_width=True)

    st.subheader("Hemorrhage volumes")
    st.dataframe(_volume_table_display(volume_df), use_container_width=True, hide_index=True)

    with st.expander("Confidence details"):
        st.dataframe(_confidence_table(confidence), use_container_width=True, hide_index=True)

    with st.expander("Full clinical report"):
        st.markdown(payload["prediction_report_md"])

    st.subheader("Downloads")
    download_cols = st.columns(3)
    download_items = [
        ("prediction_mask.nii.gz", mask_bytes),
        ("overlay.png", overlay_bytes),
        ("prediction_summary.json", json.dumps(summary, indent=2).encode("utf-8")),
        ("volume_report.csv", volume_df.to_csv(index=False).encode("utf-8")),
        ("prediction_report.md", payload["prediction_report_md"].encode("utf-8")),
    ]
    for index, (name, content) in enumerate(download_items):
        download_cols[index % 3].download_button(
            label=f"Download {name}",
            data=content,
            file_name=name,
            mime="application/octet-stream",
        )


if __name__ == "__main__":
    main()
