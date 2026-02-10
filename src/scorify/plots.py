import numpy as np
import pandas as pd
import plotly.graph_objects as go

_PALETTE = [
    "#60a5fa",
    "#6ee7b7",
    "#fbbf24",
    "#f87171",
    "#38bdf8",
    "#fb923c",
    "#a3e635",
    "#e879f9",
    "#67e8f9",
    "#fda4af",
]

_DEFAULTS = dict(
    margin=dict(l=40, r=20, t=40, b=40),
)


def _layout(**kw) -> dict:
    return {**_DEFAULTS, **kw}


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def score_boxplot(df: pd.DataFrame, tomograms: list[str]) -> go.Figure:
    means = df[df["tomogram"].isin(tomograms)].groupby("tomogram")["score"].mean()
    ordered = means.reindex(tomograms).sort_values(ascending=True).index.tolist()

    fig = go.Figure()
    for i, tomo in enumerate(ordered):
        sub = df[df["tomogram"] == tomo]
        fig.add_trace(
            go.Box(
                x=sub["score"],
                name=tomo,
                marker_color=_color(i),
                line_color=_color(i),
                hoverinfo="x+name",
            )
        )
    fig.update_layout(
        **_layout(
            xaxis_title="Score",
            title="Score Comparison",
            xaxis=dict(tickangle=0),
            yaxis=dict(tickangle=0),
            showlegend=False,
            height=max(300, len(tomograms) * 24 + 80),
            bargroupgap=0.05,
            hoverlabel=dict(align="left"),
            hovermode="closest",
        )
    )
    fig.update_yaxes(automargin=True)
    return fig


def particle_count_bar(df: pd.DataFrame, tomograms: list[str]) -> go.Figure:
    counts = (
        df[df["tomogram"].isin(tomograms)]
        .groupby("tomogram")
        .size()
        .reindex(tomograms, fill_value=0)
        .sort_values(ascending=True)
    )
    fig = go.Figure(
        go.Bar(
            y=counts.index,
            x=counts.values,
            orientation="h",
            marker_color=[_color(i) for i in range(len(counts))],
        )
    )
    fig.update_layout(
        **_layout(
            xaxis_title="Particle Count",
            title="Particles per Tomogram",
            showlegend=False,
            height=max(300, len(tomograms) * 24 + 80),
            bargap=0.15,
        )
    )
    fig.update_yaxes(automargin=True)
    return fig


def orientation_heatmap(df: pd.DataFrame) -> go.Figure:
    """2D histogram of rotation (azimuth) vs tilt — shows orientation coverage."""
    fig = go.Figure(
        go.Histogram2d(
            x=df["rot"],
            y=df["tilt"],
            nbinsx=72,
            nbinsy=36,
            colorscale="Viridis",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        **_layout(
            title="Orientation Distribution",
            xaxis_title="Rotation (\u00b0)",
            yaxis_title="Tilt (\u00b0)",
            height=500,
        )
    )
    return fig


def psi_histogram(df: pd.DataFrame) -> go.Figure:
    """In-plane rotation distribution."""
    fig = go.Figure(
        go.Histogram(
            x=df["psi"],
            nbinsx=72,
            marker_color=_color(0),
            opacity=0.8,
        )
    )
    fig.update_layout(
        **_layout(
            title="In-plane Rotation (\u03c8)",
            xaxis_title="Psi (\u00b0)",
            yaxis_title="Count",
        )
    )
    return fig


def spatial_scatter_3d(df: pd.DataFrame, tomogram: str) -> go.Figure:
    sub = df[df["tomogram"] == tomogram]
    has_pos = all(c in sub.columns for c in ("x", "y", "z"))
    if not has_pos or sub.empty:
        fig = go.Figure()
        fig.update_layout(**_layout(title=f"No position data for {tomogram}"))
        return fig

    fig = go.Figure(
        go.Scatter3d(
            x=sub["x"],
            y=sub["y"],
            z=sub["z"],
            mode="markers",
            marker=dict(
                size=2.5,
                color=sub["score"],
                colorscale="Viridis",
                colorbar=dict(title="Score"),
                opacity=0.8,
            ),
            text=sub["score"].round(4).astype(str),
            hovertemplate="x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<br>score: %{text}<extra></extra>",
        )
    )
    _axis = dict(showticklabels=False, showgrid=False, showbackground=False)
    fig.update_layout(
        **_layout(
            title=f"Particle Positions \u2014 {tomogram}",
            height=700,
            scene=dict(
                xaxis=dict(title="X", **_axis),
                yaxis=dict(title="Y", **_axis),
                zaxis=dict(title="Z", **_axis),
                bgcolor="rgba(0,0,0,0)",
                camera=dict(
                    eye=dict(x=0, y=0, z=1.8),
                    up=dict(x=0, y=1, z=0),
                ),
            ),
        )
    )
    return fig


def correlation_scatter(
    agg: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    poly_degree: int | None = None,
) -> go.Figure:
    """Scatter plot of per-tomogram aggregate metrics.

    Parameters
    ----------
    agg : DataFrame with tomogram as index and aggregate columns.
    x_col, y_col : columns to plot on each axis.
    color_col : optional column for marker color scale.
    poly_degree : polynomial degree for fit line (1=linear, 2=quadratic, None=off).
    """
    fig = go.Figure()
    marker = dict(size=8, opacity=0.85)
    if color_col and color_col in agg.columns:
        marker.update(
            color=agg[color_col],
            colorscale="Viridis",
            colorbar=dict(title=color_col),
        )
    else:
        marker["color"] = _color(0)

    fig.add_trace(
        go.Scatter(
            x=agg[x_col],
            y=agg[y_col],
            mode="markers",
            marker=marker,
            text=agg.index,
            hovertemplate="%{text}<br>"
            + f"{x_col}: "
            + "%{x:.4g}<br>"
            + f"{y_col}: "
            + "%{y:.4g}<extra></extra>",
            showlegend=False,
        )
    )

    annotation_text = None
    if poly_degree is not None and len(agg) > poly_degree:
        mask = agg[[x_col, y_col]].dropna().index
        xv = agg.loc[mask, x_col].values.astype(float)
        yv = agg.loc[mask, y_col].values.astype(float)
        if len(xv) > poly_degree and np.std(xv) > 0:
            coeffs = np.polyfit(xv, yv, poly_degree)
            y_pred = np.polyval(coeffs, xv)
            ss_res = np.sum((yv - y_pred) ** 2)
            ss_tot = np.sum((yv - np.mean(yv)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            x_fit = np.linspace(xv.min(), xv.max(), 200)
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=np.polyval(coeffs, x_fit),
                    mode="lines",
                    line=dict(color=_color(1), dash="dash", width=2),
                    showlegend=False,
                )
            )
            annotation_text = f"R\u00b2 = {r2:.3f}"

    fig.update_layout(
        **_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            title=f"{y_col} vs {x_col}",
            showlegend=False,
            height=550,
        )
    )
    if annotation_text:
        fig.add_annotation(
            text=annotation_text,
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            showarrow=False,
            font=dict(size=14),
        )
    return fig


def threshold_survival(df: pd.DataFrame, tomograms: list[str]) -> go.Figure:
    fig = go.Figure()
    if "score" not in df.columns:
        fig.update_layout(**_layout(title="No score data"))
        return fig

    lo, hi = float(df["score"].min()), float(df["score"].max())
    thresholds = np.linspace(lo, hi, 200)

    for i, tomo in enumerate(tomograms):
        scores = df.loc[df["tomogram"] == tomo, "score"].values
        counts = np.array([(scores >= t).sum() for t in thresholds])
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=counts,
                mode="lines",
                name=tomo,
                line=dict(color=_color(i), width=2),
            )
        )

    fig.update_layout(
        **_layout(
            xaxis_title="Score Threshold",
            yaxis_title="Particles Remaining",
            title="Threshold Survival Curve",
            height=650,
            margin=dict(l=40, r=20, t=40, b=140),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.28,
                xanchor="center",
                x=0.5,
            ),
        )
    )
    return fig
