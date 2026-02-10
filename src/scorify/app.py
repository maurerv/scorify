import io
import zipfile

import pandas as pd
import streamlit as st

from scorify.parsing import export_star, load_star_files
from scorify.plots import (
    correlation_scatter,
    orientation_heatmap,
    particle_count_bar,
    psi_histogram,
    score_boxplot,
    spatial_scatter_3d,
    threshold_survival,
)

st.set_page_config(page_title="Scorify", page_icon="◈", layout="wide")

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

with st.sidebar:
    st.title("◈ Scorify")
    uploaded = st.file_uploader(
        "files",
        type=["star"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"upload_{st.session_state.upload_key}",
    )
    if uploaded:
        if st.button("Clear All", use_container_width=True):
            st.session_state.upload_key += 1
            st.rerun()

if not uploaded:
    st.markdown("## Upload star files to get started")
    st.caption("Drag & drop `.star` files in the sidebar.")
    st.stop()


@st.cache_data(show_spinner="Parsing star files…")
def _load(files):
    return load_star_files(files)


df = _load(uploaded)

if df.empty:
    st.error("No data found in uploaded files.")
    st.stop()

has_score = "score" in df.columns
all_tomos = sorted(df["tomogram"].unique())

with st.sidebar:
    if has_score:
        score_min, score_max = float(df["score"].min()), float(df["score"].max())
        threshold = st.slider(
            "Score threshold",
            score_min,
            score_max,
            score_min,
            step=0.001,
            format="%.4f",
        )
    else:
        threshold = None

    export_slot = st.container()

    selected_tomos = (
        st.pills(
            "Tomograms",
            all_tomos,
            default=all_tomos,
            selection_mode="multi",
        )
        or []
    )

tomo_filtered = df[df["tomogram"].isin(selected_tomos)]
if threshold is not None:
    filtered = tomo_filtered[tomo_filtered["score"] >= threshold]
else:
    filtered = tomo_filtered

if filtered.empty:
    st.warning("No particles match current filters.")
    st.stop()

with export_slot:
    per_tomo = st.toggle("Per tomogram", value=False)
    if per_tomo:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for tomo in selected_tomos:
                sub = filtered[filtered["tomogram"] == tomo]
                if not sub.empty:
                    zf.writestr(f"{tomo}.star", export_star(sub))
        st.download_button(
            "Export .star",
            data=buf.getvalue(),
            file_name="filtered_per_tomogram.zip",
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.download_button(
            "Export .star",
            data=export_star(filtered),
            file_name="filtered.star",
            mime="application/octet-stream",
            use_container_width=True,
        )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Particles", f"{len(tomo_filtered):,}")
c2.metric("Tomograms", f"{filtered['tomogram'].nunique()}")
if has_score:
    c3.metric("Mean Score", f"{filtered['score'].mean():.4f}")
    c4.metric("Above Threshold", f"{len(filtered):,}")
else:
    c3.metric("Mean Score", "N/A")
    c4.metric("Above Threshold", "N/A")

tab_overview, tab_scores, tab_counts, tab_angles, tab_spatial, tab_corr, tab_thresh = (
    st.tabs(
        [
            "Overview",
            "Score Comparison",
            "Particle Counts",
            "Angular Analysis",
            "Spatial",
            "Correlations",
            "Threshold Explorer",
        ]
    )
)

with tab_overview:
    if has_score:
        summary = (
            filtered.groupby("tomogram")["score"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .rename(columns={"count": "particles"})
            .sort_index()
        )
        st.dataframe(summary, width="stretch", height=min(800, len(summary) * 40 + 60))
    else:
        summary = filtered.groupby("tomogram").size().rename("particles").to_frame()
        st.dataframe(summary, width="stretch", height=min(800, len(summary) * 40 + 60))

with tab_scores:
    if has_score:
        st.plotly_chart(score_boxplot(filtered, selected_tomos), width="stretch")
    else:
        st.info("No score column found in the data.")

with tab_counts:
    st.plotly_chart(particle_count_bar(filtered, selected_tomos), width="stretch")

with tab_angles:
    has_rot_tilt = "rot" in filtered.columns and "tilt" in filtered.columns
    has_psi = "psi" in filtered.columns
    if has_rot_tilt:
        st.plotly_chart(orientation_heatmap(filtered), width="stretch")
    if has_psi:
        st.plotly_chart(psi_histogram(filtered), width="stretch")
    if not has_rot_tilt and not has_psi:
        st.info("No Euler angle columns found in the data.")

with tab_spatial:
    if len(selected_tomos) == 1:
        st.plotly_chart(
            spatial_scatter_3d(filtered, selected_tomos[0]), width="stretch"
        )
    else:
        tomo_choice = st.selectbox("Select tomogram for 3D view", selected_tomos)
        st.plotly_chart(spatial_scatter_3d(filtered, tomo_choice), width="stretch")

with tab_corr:
    # Build per-tomogram aggregate table
    _agg_parts = {"particles": filtered.groupby("tomogram").size()}
    if has_score:
        _score_agg = filtered.groupby("tomogram")["score"].agg(
            ["mean", "median", "std", "min", "max"]
        )
        _score_agg.columns = [f"score_{c}" for c in _score_agg.columns]
        for c in _score_agg.columns:
            _agg_parts[c] = _score_agg[c]
    for pos in ("x", "y", "z"):
        if pos in filtered.columns:
            _agg_parts[f"{pos}_mean"] = filtered.groupby("tomogram")[pos].mean()
            _agg_parts[f"{pos}_std"] = filtered.groupby("tomogram")[pos].std()
    _agg = pd.DataFrame(_agg_parts)

    _metrics = list(_agg.columns)
    _fit_options = {"None": None, "Linear": 1, "Quadratic": 2}
    col_x, col_y, col_c, col_fit = st.columns([2, 2, 2, 2])
    with col_x:
        x_metric = st.selectbox("X axis", _metrics, index=0)
    with col_y:
        y_idx = min(1, len(_metrics) - 1) if has_score else 0
        y_metric = st.selectbox("Y axis", _metrics, index=y_idx)
    with col_c:
        color_metric = st.selectbox("Color by", ["None"] + _metrics)
    with col_fit:
        fit_choice = st.selectbox("Fit", list(_fit_options.keys()))

    st.plotly_chart(
        correlation_scatter(
            _agg,
            x_metric,
            y_metric,
            color_col=color_metric if color_metric != "None" else None,
            poly_degree=_fit_options[fit_choice],
        ),
        width="stretch",
    )

with tab_thresh:
    if has_score:
        st.plotly_chart(
            threshold_survival(tomo_filtered, selected_tomos), width="stretch"
        )
    else:
        st.info("No score column found in the data.")
