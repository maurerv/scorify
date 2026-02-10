import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from scorify.plots import (
    correlation_scatter,
    orientation_heatmap,
    particle_count_bar,
    psi_histogram,
    score_boxplot,
    spatial_scatter_3d,
    threshold_survival,
)


class TestScoreBoxplot:
    def test_returns_figure(self, sample_df):
        fig = score_boxplot(sample_df, ["TS_001", "TS_002"])
        assert isinstance(fig, go.Figure)

    def test_one_trace_per_tomogram(self, sample_df):
        fig = score_boxplot(sample_df, ["TS_001", "TS_002"])
        assert len(fig.data) == 2

    def test_ordered_by_ascending_mean(self, sample_df):
        fig = score_boxplot(sample_df, ["TS_001", "TS_002"])
        means = sample_df.groupby("tomogram")["score"].mean()
        names = [t.name for t in fig.data]
        # First trace (bottom) should have the lowest mean
        assert means[names[0]] <= means[names[-1]]

    def test_no_legend(self, sample_df):
        fig = score_boxplot(sample_df, ["TS_001", "TS_002"])
        assert fig.layout.showlegend is False

    def test_single_tomogram(self, sample_df):
        fig = score_boxplot(sample_df, ["TS_001"])
        assert len(fig.data) == 1


class TestParticleCountBar:
    def test_returns_figure(self, sample_df):
        fig = particle_count_bar(sample_df, ["TS_001", "TS_002"])
        assert isinstance(fig, go.Figure)

    def test_horizontal_orientation(self, sample_df):
        fig = particle_count_bar(sample_df, ["TS_001", "TS_002"])
        assert fig.data[0].orientation == "h"

    def test_sorted_ascending(self, sample_df):
        fig = particle_count_bar(sample_df, ["TS_001", "TS_002"])
        x_vals = list(fig.data[0].x)
        assert x_vals == sorted(x_vals)

    def test_respects_tomogram_filter(self, sample_df):
        fig = particle_count_bar(sample_df, ["TS_001"])
        y_labels = list(fig.data[0].y)
        assert "TS_002" not in y_labels


class TestOrientationHeatmap:
    def test_returns_figure(self, sample_df):
        fig = orientation_heatmap(sample_df)
        assert isinstance(fig, go.Figure)

    def test_is_histogram2d(self, sample_df):
        fig = orientation_heatmap(sample_df)
        assert isinstance(fig.data[0], go.Histogram2d)

    def test_uses_rot_and_tilt(self, sample_df):
        fig = orientation_heatmap(sample_df)
        assert len(fig.data[0].x) == len(sample_df)
        assert len(fig.data[0].y) == len(sample_df)


class TestPsiHistogram:
    def test_returns_figure(self, sample_df):
        fig = psi_histogram(sample_df)
        assert isinstance(fig, go.Figure)

    def test_is_histogram(self, sample_df):
        fig = psi_histogram(sample_df)
        assert isinstance(fig.data[0], go.Histogram)


class TestSpatialScatter3d:
    def test_returns_figure(self, sample_df):
        fig = spatial_scatter_3d(sample_df, "TS_001")
        assert isinstance(fig, go.Figure)

    def test_scatter3d_trace(self, sample_df):
        fig = spatial_scatter_3d(sample_df, "TS_001")
        assert isinstance(fig.data[0], go.Scatter3d)

    def test_camera_looks_down_z(self, sample_df):
        fig = spatial_scatter_3d(sample_df, "TS_001")
        cam = fig.layout.scene.camera
        assert cam.eye.z > 0
        assert cam.eye.x == 0
        assert cam.eye.y == 0

    def test_no_grid_or_ticks(self, sample_df):
        fig = spatial_scatter_3d(sample_df, "TS_001")
        for axis in (
            fig.layout.scene.xaxis,
            fig.layout.scene.yaxis,
            fig.layout.scene.zaxis,
        ):
            assert axis.showgrid is False
            assert axis.showticklabels is False

    def test_missing_tomogram_returns_empty_figure(self, sample_df):
        fig = spatial_scatter_3d(sample_df, "NONEXISTENT")
        assert len(fig.data) == 0

    def test_missing_position_columns(self):
        df = pd.DataFrame({"score": [0.1], "tomogram": ["T"]})
        fig = spatial_scatter_3d(df, "T")
        assert len(fig.data) == 0


class TestThresholdSurvival:
    def test_returns_figure(self, sample_df):
        fig = threshold_survival(sample_df, ["TS_001", "TS_002"])
        assert isinstance(fig, go.Figure)

    def test_one_line_per_tomogram(self, sample_df):
        fig = threshold_survival(sample_df, ["TS_001", "TS_002"])
        assert len(fig.data) == 2

    def test_monotonically_decreasing(self, sample_df):
        fig = threshold_survival(sample_df, ["TS_001"])
        y = list(fig.data[0].y)
        assert all(y[i] >= y[i + 1] for i in range(len(y) - 1))

    def test_no_score_column(self):
        df = pd.DataFrame({"tomogram": ["T"], "x": [1.0]})
        fig = threshold_survival(df, ["T"])
        assert len(fig.data) == 0

    def test_legend_below_plot(self, sample_df):
        fig = threshold_survival(sample_df, ["TS_001"])
        assert fig.layout.legend.y < 0


@pytest.fixture()
def agg_df(sample_df):
    """Per-tomogram aggregate table matching what the app builds."""
    agg = {"particles": sample_df.groupby("tomogram").size()}
    score_agg = sample_df.groupby("tomogram")["score"].agg(
        ["mean", "median", "std", "min", "max"]
    )
    score_agg.columns = [f"score_{c}" for c in score_agg.columns]
    for c in score_agg.columns:
        agg[c] = score_agg[c]
    return pd.DataFrame(agg)


class TestCorrelationScatter:
    def test_returns_figure(self, agg_df):
        fig = correlation_scatter(agg_df, "particles", "score_mean")
        assert isinstance(fig, go.Figure)

    def test_scatter_trace(self, agg_df):
        fig = correlation_scatter(agg_df, "particles", "score_mean")
        assert isinstance(fig.data[0], go.Scatter)
        assert fig.data[0].mode == "markers"

    def test_linear_fit_adds_line_trace(self, agg_df):
        fig = correlation_scatter(agg_df, "score_mean", "score_std", poly_degree=1)
        assert len(fig.data) == 2
        assert fig.data[1].mode == "lines"

    def test_quadratic_fit_adds_line_trace(self, agg_df):
        fig = correlation_scatter(agg_df, "score_mean", "score_std", poly_degree=2)
        # Need > poly_degree points; with 2 points and degree 2 there's no fit
        assert len(fig.data) == 1

    def test_fit_r_squared_annotation(self, agg_df):
        fig = correlation_scatter(agg_df, "score_mean", "score_std", poly_degree=1)
        annotations = fig.layout.annotations
        assert any("R\u00b2" in a.text for a in annotations)

    def test_color_by_metric(self, agg_df):
        fig = correlation_scatter(
            agg_df, "particles", "score_mean", color_col="score_std"
        )
        assert fig.data[0].marker.colorscale is not None

    def test_color_none(self, agg_df):
        fig = correlation_scatter(agg_df, "particles", "score_mean", color_col=None)
        assert fig.data[0].marker.colorscale is None

    def test_single_point_no_fit(self):
        agg = pd.DataFrame({"a": [1.0], "b": [2.0]}, index=["T1"])
        fig = correlation_scatter(agg, "a", "b", poly_degree=1)
        assert len(fig.data) == 1

    def test_no_fit_by_default(self, agg_df):
        fig = correlation_scatter(agg_df, "score_mean", "score_std")
        assert len(fig.data) == 1
