from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator


FAMILY_LABELS = {
    "single_diode_exponential": "Single Shockley Diode",
    "double_diode_exponential": "Double Shockley Diode",
    "experimental": r"PWL $i(v)$",
}

ERROR_OUTPUTS = {
    "single_diode_exponential": "single_diode_error_vs_iter_summary_20260302_p90.png",
    "double_diode_exponential": "double_diode_error_vs_iter_summary_20260302_p90.png",
    "experimental": "experimental__error_vs_iter_summary_20260302_p90.png",
}

VOLTOL_OUTPUTS = {
    "single_diode_exponential": "single_p90_vs_rel_tol_all_hidden.png",
    "double_diode_exponential": "double_diode_p90_vs_rel_tol_all_hidden.png",
    "experimental": "experimental_p90_vs_rel_tol_all_hidden.png",
}

STATIC_FIGURES = (
    "figures/bidirectionalampschematic_cropped.pdf",
    "figures/double_shockley_iv.pdf",
    "figures/mosfet_id_vds.pdf",
    "figures/non_linear_circuit_simple_cropped-1.pdf",
    "figures/shockley_iv.pdf",
    "figures/experiments/circuits2_cropped.pdf",
)


def regenerate_mnist_assets(repo_root: Path) -> list[Path]:
    """Regenerate the two MNIST panels directly from bundled numerical data."""

    output_root = repo_root / "outputs" / "paper"
    records: list[dict[str, Any]] = []
    generated = _generate_mnist_assets(repo_root, output_root, records)
    manifest_path = output_root / "mnist_asset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "asset_count": len(records),
                "matplotlib_version": matplotlib.__version__,
                "protocol": "data/paper/mnist/training_protocol.json",
                "assets": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return generated


def regenerate_paper_assets(repo_root: Path) -> list[Path]:
    """Regenerate all data-driven paper assets and stage source artwork.

    The output tree mirrors ``paper/reference`` so paths can be copied directly
    into a manuscript checkout. Reference images are never used as plotting
    inputs.
    """

    output_root = repo_root / "outputs" / "paper"
    generated: list[Path] = []
    records: list[dict[str, Any]] = []

    for family in FAMILY_LABELS:
        source = (
            repo_root
            / "data"
            / "error_vs_iter"
            / "summaries"
            / family
            / "error_vs_iter_summary_20260302.json"
        )
        target = (
            output_root
            / "figures"
            / "experiments"
            / "digits_error_vs_iter"
            / ERROR_OUTPUTS[family]
        )
        _plot_error_vs_iterations(source, target, family)
        _record(records, repo_root, target, [source], "matplotlib:error-vs-iterations")
        generated.append(target)

        source = repo_root / "data" / "paper" / "vol_tol" / f"{family}.json"
        target = (
            output_root
            / "figures"
            / "experiments"
            / "digits_error_vs_vol_tol"
            / VOLTOL_OUTPUTS[family]
        )
        _plot_voltage_tolerance(source, target)
        _record(records, repo_root, target, [source], "matplotlib:voltage-tolerance")
        generated.append(target)

        source = (
            repo_root
            / "data"
            / "timing"
            / "figure_inputs"
            / family
            / "combined_latest_by_hidden.csv"
        )
        target = (
            output_root
            / "figures"
            / "experiments"
            / "digits_timing_cpu"
            / f"{family}_cpu_spice_vs_coordinate_descent_loglog.png"
        )
        _plot_timing(source, target, family)
        _record(records, repo_root, target, [source], "matplotlib:timing")
        generated.append(target)

    source = repo_root / "data" / "paper" / "overrelaxation" / "error_vs_iter_p90_summary.json"
    target = (
        output_root
        / "figures"
        / "digits_overrelaxation"
        / "p90_vs_iterations_overrelax.png"
    )
    _plot_overrelaxation(source, target)
    _record(records, repo_root, target, [source], "matplotlib:overrelaxation")
    generated.append(target)

    for depth in (1, 2, 3):
        source = (
            repo_root
            / "data"
            / "paper"
            / "conditioning"
            / f"conditioning_vs_runtime_hidden{depth}.csv"
        )
        target = (
            output_root
            / "figures"
            / "supplementary"
            / "conditioning_vs_runtime_experimental"
            / f"raw_inv_neg_log_rho_side_by_side_hidden{depth}.png"
        )
        _plot_conditioning(source, target, depth)
        _record(records, repo_root, target, [source], "matplotlib:conditioning")
        generated.append(target)

    generated.extend(_generate_mnist_assets(repo_root, output_root, records))

    source = (
        repo_root
        / "data"
        / "paper"
        / "component_timing"
        / "component_timing_comparison.csv"
    )
    target = (
        output_root
        / "figures"
        / "supplementary"
        / "supplementary_timing"
        / "double_diode_hidden3_width128_component_timing_comparison_bar.pdf"
    )
    _plot_component_timing(source, target)
    _record(records, repo_root, target, [source], "matplotlib:component-timing")
    generated.append(target)

    reference_root = repo_root / "paper" / "reference"
    for relative in STATIC_FIGURES:
        source = reference_root / relative
        target = output_root / relative
        _copy(source, target)
        _record(records, repo_root, target, [source], "source-artwork:copy")
        generated.append(target)

    table_sources = {
        "table1_accuracy_longtable.tex": (
            repo_root / "data" / "timing" / "tables" / "table1_accuracy_longtable.tex"
        ),
        "table2_runtime_decomposition_longtable.tex": (
            repo_root
            / "data"
            / "timing"
            / "tables"
            / "table2_runtime_decomposition_longtable.tex"
        ),
        "table_timing_run_accuracies_input_gain_longtable.tex": (
            repo_root
            / "data"
            / "timing"
            / "tables"
            / "table_timing_run_accuracies_input_gain_longtable.tex"
        ),
    }
    for name, source in table_sources.items():
        target = output_root / "tables" / name
        _copy(source, target)
        _record(records, repo_root, target, [source], "curated-table:copy")
        generated.append(target)

    source = (
        repo_root
        / "data"
        / "paper"
        / "accuracy_ladder"
        / "accuracy_ladder_timing_minimal_table.csv"
    )
    target = output_root / "tables" / "table_accuracy_ladder_timing.tex"
    _write_accuracy_ladder_table(source, target)
    _record(records, repo_root, target, [source], "table:accuracy-ladder")
    generated.append(target)

    manifest_path = output_root / "asset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "asset_count": len(records),
                "matplotlib_version": matplotlib.__version__,
                "assets": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return generated


def _generate_mnist_assets(
    repo_root: Path,
    output_root: Path,
    records: list[dict[str, Any]],
) -> list[Path]:
    accuracy_source = (
        repo_root
        / "data"
        / "paper"
        / "mnist"
        / "mean_test_accuracy_selected_runs_with_perfect_diode.json"
    )
    accuracy_target = (
        output_root
        / "figures"
        / "supplementary"
        / "mnist"
        / "mean_test_accuracy_selected_runs_with_perfect_diode.png"
    )
    _plot_mnist_accuracy(accuracy_source, accuracy_target)
    _record(
        records,
        repo_root,
        accuracy_target,
        [accuracy_source],
        "matplotlib:mnist-accuracy",
    )

    pca_dir = repo_root / "data" / "paper" / "pca_sweep"
    pca_sources = [
        pca_dir / "pca_sweep_iter4.npz",
        pca_dir / "pca_sweep_iter4_flat_spice_layers.npz",
        pca_dir / "run_info.json",
    ]
    pca_target = (
        output_root
        / "figures"
        / "supplementary"
        / "mnist"
        / "pca_sweep_rel_l1_blue_dots_x1e3_log_scale.png"
    )
    _plot_pca_error(*pca_sources, pca_target)
    _record(records, repo_root, pca_target, pca_sources, "matplotlib:pca-relative-error")
    return [accuracy_target, pca_target]


def _plot_error_vs_iterations(source: Path, target: Path, family: str) -> None:
    payload = _read_json(source)
    hidden = payload.get("hidden")
    if not isinstance(hidden, dict) or not hidden:
        raise ValueError(
            "Expected error summary JSON to contain a non-empty 'hidden' object. "
            f"Provided value: source={source}, hidden={hidden!r}."
        )
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    colors = plt.get_cmap("tab10").colors
    for index, hidden_key in enumerate(sorted(hidden, key=_hidden_sort_key)):
        points = []
        for row in hidden[hidden_key]:
            iteration = int(row.get("iterations", row.get("iteration", -1)))
            if 4 <= iteration <= 128 and "p90" in row:
                points.append((iteration, max(float(row["p90"]), 1e-16)))
        points.sort()
        if not points:
            continue
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=colors[index % len(colors)],
            marker="o",
            linewidth=2.0,
            markersize=6,
            label=_hidden_label(hidden_key),
        )
    ticks = [4, 8, 16, 32, 64, 128]
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(4, 128)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{int(value)}" if value in ticks else "")
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Number of iterations", fontsize=15)
    ax.set_ylabel("P90 relative error", fontsize=15)
    ax.set_title(FAMILY_LABELS[family], fontsize=16, pad=10)
    ax.tick_params(axis="both", which="both", labelsize=13)
    ax.grid(True, which="major", linestyle=":", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()
    _save_figure(fig, target, dpi=220)


def _plot_voltage_tolerance(source: Path, target: Path) -> None:
    payload = _read_json(source)
    hidden = payload.get("hidden")
    if not isinstance(hidden, dict) or not hidden:
        raise ValueError(
            "Expected voltage-tolerance JSON to contain a non-empty 'hidden' object. "
            f"Provided value: source={source}, hidden={hidden!r}."
        )
    fig, ax_error = plt.subplots(figsize=(7.4, 5.4))
    ax_sweeps = ax_error.twinx()
    colors = plt.get_cmap("tab10").colors
    hidden_handles = []
    hidden_labels = []
    for index, hidden_key in enumerate(sorted(hidden, key=_hidden_sort_key)):
        row = hidden[hidden_key]
        triples = sorted(
            zip(row["rel_tol"], row["p90"], row["avg_outer_iterations"]),
            key=lambda values: float(values[0]),
        )
        tolerances = [float(values[0]) for values in triples]
        errors = [float(values[1]) for values in triples]
        sweeps = [float(values[2]) for values in triples]
        color = colors[index % len(colors)]
        line = ax_error.plot(
            tolerances,
            errors,
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=color,
        )[0]
        ax_sweeps.plot(
            tolerances,
            sweeps,
            marker="s",
            markersize=4.5,
            linewidth=1.5,
            linestyle="--",
            color=color,
            alpha=0.95,
        )
        hidden_handles.append(line)
        hidden_labels.append(str(row.get("label", _hidden_label(hidden_key))))

    ax_error.set_xscale("log")
    ax_error.invert_xaxis()
    ax_error.set_yscale("log")
    ax_error.set_ylim(*[float(value) for value in payload.get("p90_ylim", [8e-6, 2e-3])])
    ax_sweeps.set_ylim(*[float(value) for value in payload.get("avg_ylim", [5.0, 31.0])])
    ax_error.set_xlabel("Relative voltage tolerance", fontsize=15)
    ax_error.set_ylabel("P90 relative error", fontsize=15)
    ax_sweeps.set_ylabel("Average outer sweeps", fontsize=15)
    ax_error.tick_params(axis="both", which="both", labelsize=13)
    ax_sweeps.tick_params(axis="y", labelsize=13)
    ax_error.set_title(str(payload.get("title", "Voltage tolerance")), fontsize=16, pad=10)
    ax_error.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    ax_error.annotate(
        "P90 error",
        xy=(0.015, 0.88),
        xytext=(0.18, 0.88),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=11,
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        va="center",
    )
    ax_sweeps.annotate(
        "Avg. sweeps",
        xy=(0.985, 0.80),
        xytext=(0.70, 0.80),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=11,
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        va="center",
    )

    legend_top = fig.legend(
        hidden_handles,
        hidden_labels,
        title="Configuration",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(hidden_labels),
        fontsize=12,
        title_fontsize=13,
    )
    fig.add_artist(legend_top)
    metric_handles = [
        Line2D([0], [0], color="black", marker="o", linewidth=1.8, label="P90 error"),
        Line2D(
            [0],
            [0],
            color="black",
            marker="s",
            linewidth=1.5,
            linestyle="--",
            label="Avg. outer sweeps",
        ),
    ]
    fig.legend(
        handles=metric_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.88))
    _save_figure(fig, target, dpi=300)


def _plot_timing(source: Path, target: Path, family: str) -> None:
    rows = _read_csv(source)
    required = {"depth", "width", "coord_user_time_seconds", "spice_total_seconds"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(
            f"Expected timing CSV columns {sorted(required)}. "
            f"Provided value: source={source}, missing={sorted(missing)}."
        )
    depths = sorted({int(row["depth"]) for row in rows})
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    for index, depth in enumerate(depths):
        color = cmap(index % 10)
        subset = sorted(
            (row for row in rows if int(row["depth"]) == depth),
            key=lambda row: int(row["width"]),
        )
        coord = [
            (int(row["width"]), _positive_float(row.get("coord_user_time_seconds")))
            for row in subset
        ]
        spice = [
            (int(row["width"]), _positive_float(row.get("spice_total_seconds")))
            for row in subset
        ]
        coord = [point for point in coord if point[1] is not None]
        spice = [point for point in spice if point[1] is not None]
        if coord:
            ax.plot(
                [point[0] for point in coord],
                [point[1] for point in coord],
                marker="o",
                linestyle="-",
                color=color,
                linewidth=2,
                markersize=6,
            )
        if spice:
            ax.plot(
                [point[0] for point in spice],
                [point[1] for point in spice],
                marker="s",
                linestyle="--",
                color=color,
                linewidth=2,
                markersize=6,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hidden Layer Size", fontsize=18)
    ax.set_ylabel("Time (seconds)", fontsize=18)
    ax.set_title(FAMILY_LABELS[family].replace("$", ""), fontsize=20)
    ax.tick_params(axis="both", which="both", labelsize=15)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    depth_handles = [
        Line2D(
            [0],
            [0],
            color=cmap(index % 10),
            linestyle="-",
            marker="o",
            linewidth=2,
            markersize=6,
            label=f"Hidden {depth}",
        )
        for index, depth in enumerate(depths)
    ]
    method_handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o", linewidth=2, label="CD"),
        Line2D(
            [0], [0], color="black", linestyle="--", marker="s", linewidth=2, label="SPICE"
        ),
    ]
    ax.legend(handles=depth_handles + method_handles, loc="best", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, target, dpi=300)


def _plot_overrelaxation(source: Path, target: Path) -> None:
    payload = _read_json(source)
    series = payload.get("series")
    if not isinstance(series, dict) or not series:
        raise ValueError(
            "Expected overrelaxation JSON to contain a non-empty 'series' object. "
            f"Provided value: source={source}, series={series!r}."
        )
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    colors = plt.get_cmap("tab10").colors
    ticks = sorted(
        {
            int(point["iter"])
            for points in series.values()
            for point in points
            if 4 <= int(point.get("iter", -1)) <= 128
        }
    )
    positions = {iteration: index for index, iteration in enumerate(ticks)}
    for index, (omega, points) in enumerate(sorted(series.items(), key=lambda item: float(item[0]))):
        kept = [point for point in points if int(point.get("iter", -1)) in positions]
        ax.plot(
            [positions[int(point["iter"])] for point in kept],
            [max(float(point["p90"]), 1e-16) for point in kept],
            color=colors[index % len(colors)],
            marker="o",
            linewidth=2.0,
            markersize=6,
            label=f"omega = {omega}",
        )
    ax.set_title("Overrelaxation sweep Double Diode Exponential", fontsize=16, pad=10)
    ax.set_xlabel("Number of iterations", fontsize=13)
    ax.set_ylabel("P90 relative error", fontsize=13)
    ax.set_yscale("log")
    ax.set_xlim(min(positions.values()), max(positions.values()))
    ax.set_xticks(list(positions.values()), [str(tick) for tick in ticks])
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.18)
    ax.legend(frameon=False, fontsize=10, title="Overrelaxation")
    fig.tight_layout()
    _save_figure(fig, target, dpi=240)


def _plot_conditioning(source: Path, target: Path, depth: int) -> None:
    rows = _read_csv(source)
    family_styles = (
        ("single_diode_exponential", "Single Shockley Diode", "#1f77b4", "o", "-"),
        ("double_diode_exponential", "Double Shockley Diode", "#d62728", "D", "-."),
        ("experimental", "PWL I-V", "#2ca02c", "s", "--"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))
    all_widths = sorted({int(row["width"]) for row in rows})
    handles = []
    for key, label, color, marker, linestyle in family_styles:
        subset = sorted(
            (row for row in rows if row["nonlinearity"] == key),
            key=lambda row: int(row["width"]),
        )
        widths = [int(row["width"]) for row in subset]
        sweeps = [float(row["avg_outer_iterations"]) for row in subset]
        proxy = [float(row["inv_neg_log_rho"]) for row in subset]
        line = axes[0].plot(
            widths,
            sweeps,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=6,
            color=color,
            label=label,
        )[0]
        axes[1].plot(
            widths,
            proxy,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=6,
            color=color,
        )
        handles.append(line)
    axes[0].set_title("Average outer sweeps", fontsize=15)
    axes[0].set_xlabel("Hidden Width", fontsize=13)
    axes[0].set_ylabel("Average outer sweeps", fontsize=13)
    axes[1].set_title(r"$1/(-\log \rho(T))$", fontsize=15)
    axes[1].set_xlabel("Hidden Width", fontsize=13)
    axes[1].set_ylabel(r"$1/(-\log \rho(T))$", fontsize=13)
    for axis in axes:
        axis.set_xticks(all_widths)
        axis.tick_params(axis="both", labelsize=11)
        axis.grid(True, alpha=0.3)
    plural = "Layer" if depth == 1 else "Layers"
    fig.suptitle(f"{depth} Hidden {plural}", fontsize=17, y=0.985)
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.925),
        fontsize=11,
    )
    fig.subplots_adjust(top=0.68, left=0.08, right=0.98, bottom=0.14, wspace=0.16)
    _save_figure(fig, target, dpi=220)


def _plot_mnist_accuracy(source: Path, target: Path) -> None:
    payload = _read_json(source)
    series = payload.get("series")
    if not isinstance(series, list) or not series:
        raise ValueError(
            "Expected MNIST accuracy JSON to contain a non-empty 'series' list. "
            f"Provided value: source={source}, series={series!r}."
        )
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    colors = ("tab:blue", "tab:orange", "tab:green")
    minimum = math.inf
    maximum = -math.inf
    all_steps = []
    for index, item in enumerate(series):
        steps = np.asarray(item["steps"], dtype=int)
        mean = np.asarray(item["mean_accuracy_percent"], dtype=float)
        low = np.asarray(item["min_accuracy_percent"], dtype=float)
        high = np.asarray(item["max_accuracy_percent"], dtype=float)
        if not (steps.size == mean.size == low.size == high.size):
            raise ValueError(
                "Expected each MNIST series array to have equal length. "
                f"Provided value: label={item.get('label')!r}."
            )
        color = colors[index % len(colors)]
        ax.fill_between(steps, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(
            steps,
            mean,
            color=color,
            linewidth=2.0,
            label=f"{item['label']} (n={int(item['run_count'])})",
        )
        minimum = min(minimum, float(np.nanmin(low)))
        maximum = max(maximum, float(np.nanmax(high)))
        all_steps.append(steps)
    concatenated_steps = np.concatenate(all_steps)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Test accuracy (%)", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    ax.set_xlim(int(concatenated_steps.min()), int(concatenated_steps.max()))
    padding = max(0.2, 0.05 * (maximum - minimum))
    ax.set_ylim(minimum - padding, maximum + padding)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    _save_figure(fig, target, dpi=300, bbox_inches="tight")


def _plot_pca_error(cd_path: Path, spice_path: Path, run_info_path: Path, target: Path) -> None:
    with np.load(cd_path, allow_pickle=False) as archive:
        cd = {name: archive[name] for name in archive.files}
    with np.load(spice_path, allow_pickle=False) as archive:
        spice = {name: archive[name] for name in archive.files}
    info = _read_json(run_info_path)
    steps = int(info.get("pca_steps", 30))
    sigma = float(info.get("pca_n_sigma", 3.0))
    sample_count = int(np.asarray(cd["Layer_1"]).shape[0])
    if steps * steps != sample_count:
        raise ValueError(
            "Expected pca_steps squared to equal the PCA sample count. "
            f"Provided value: pca_steps={steps}, sample_count={sample_count}."
        )
    coordinates = np.linspace(-sigma, sigma, steps)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    x_values = grid_x.ravel()
    y_values = grid_y.ravel()
    layers = ("Layer_1", "Layer_2")
    titles = ("Hidden Layer", "Output Layer")
    norm = LogNorm(vmin=0.01, vmax=100.0)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), squeeze=False)
    mappable = None
    for axis, layer, title in zip(axes[0], layers, titles):
        if layer not in cd or layer not in spice:
            raise KeyError(
                f"Expected PCA layer {layer!r} in both NPZ files. "
                f"Provided value: cd={sorted(cd)}, spice={sorted(spice)}."
            )
        cd_values = np.asarray(cd[layer]).reshape(sample_count, -1)
        spice_values = np.asarray(spice[layer]).reshape(sample_count, -1)
        if cd_values.shape != spice_values.shape:
            raise ValueError(
                "Expected CD and SPICE PCA arrays to have identical shape. "
                f"Provided value: layer={layer}, cd={cd_values.shape}, spice={spice_values.shape}."
            )
        mae = np.mean(np.abs(cd_values - spice_values), axis=1)
        reference = np.mean(np.abs(spice_values), axis=1)
        scaled_error = mae / (reference + 1e-12) * 1e3
        mappable = axis.scatter(
            x_values,
            y_values,
            c=np.clip(scaled_error, 1e-12, None),
            cmap="Blues",
            norm=norm,
            s=26.0,
            marker="o",
            linewidths=0.0,
        )
        axis.grid(True, color="#d7e6d1", linewidth=0.8, alpha=0.9)
        axis.set_title(title, fontsize=16)
        axis.set_xlabel("PCA dim 1 (sigma)", fontsize=15)
        axis.set_ylabel("PCA dim 2 (sigma)", fontsize=15)
        axis.set_xticks([-sigma, 0.0, sigma])
        axis.set_yticks([-sigma, 0.0, sigma])
        axis.tick_params(axis="both", labelsize=13)
        axis.set_aspect("equal")
        axis.set_facecolor("#fbfcf8")
    fig.subplots_adjust(bottom=0.30, top=0.80, wspace=0.20)
    if mappable is not None:
        color_axis = fig.add_axes([0.22, 0.08, 0.56, 0.035])
        colorbar = fig.colorbar(mappable, cax=color_axis, orientation="horizontal")
        colorbar.set_ticks([0.01, 0.1, 1.0, 10.0, 100.0])
        colorbar.set_ticklabels(["0.01", "0.1", "1", "10", "100"])
        colorbar.set_label("Relative L1 error (x1e-3, log scale)", fontsize=15)
        colorbar.ax.tick_params(labelsize=13)
    fig.suptitle("MNIST PCA sweep", fontsize=18)
    _save_figure(fig, target, dpi=300, bbox_inches="tight")


def _plot_component_timing(source: Path, target: Path) -> None:
    rows = _read_csv(source)
    components = [row["component"] for row in rows]
    values32 = np.asarray([float(row["float32_seconds"]) for row in rows])
    values64 = np.asarray([float(row["float64_seconds"]) for row in rows])
    y_values = np.arange(len(components))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.barh(y_values - height / 2, values32, height=height, color="#c44e52", label="float32")
    ax.barh(y_values + height / 2, values64, height=height, color="#4c72b0", label="float64")
    ax.set_yticks(y_values)
    ax.set_yticklabels(components, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Wall-clock time (s)", fontsize=13)
    ax.set_title("Float32 vs float64 local-update timing", fontsize=15)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    _save_figure(fig, target, bbox_inches="tight")


def _write_accuracy_ladder_table(source: Path, target: Path) -> None:
    rows = _read_csv(source)
    names = {
        "Single diode": "Single Shockley",
        "Double diode": "Double Shockley",
        "PWL I-V": r"PWL \(i(v)\)",
    }
    body = []
    previous = None
    for row in rows:
        family = row["nonlinearity"]
        if previous is not None and family != previous:
            body.append(r"\addlinespace")
        body.append(
            f"{names.get(family, family)} & {row['point']} & {float(row['accuracy_pct']):.1f} & "
            f"{float(row['cd_seconds']):.1f} & {float(row['spice_sim_seconds']):.1f} & "
            f"{float(row['speedup']):.1f}$\\times$ & {_latex_scientific(float(row['p90_relative_error']))}\\\\"
        )
        previous = family
    text = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\begin{tabular}{llrrrrr}\n"
        "\\toprule\n"
        "Nonlinearity & Point & Acc. [\\%] & CD [s] & SPICE [s] & Speed-up & P90 rel. err.\\\\\n"
        "\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Accuracy-ladder timing comparison for fixed two-hidden-layer, width-128 Digits networks. "
        "SPICE reports simulation time only, excluding netlist generation.}\n"
        "\\label{tab:accuracy_ladder_timing}\n"
        "\\end{table}\n"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _latex_scientific(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected figure input JSON to exist. Provided value: {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected figure input CSV to exist. Provided value: {path}.")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Expected figure input CSV to contain data rows. Provided value: {path}.")
    return rows


def _copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Expected source paper asset to exist. Provided value: {source}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _save_figure(
    figure,
    target: Path,
    *,
    dpi: int | None = None,
    bbox_inches: str | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {}
    if dpi is not None:
        options["dpi"] = dpi
    if bbox_inches is not None:
        options["bbox_inches"] = bbox_inches
    if target.suffix.lower() == ".pdf":
        options["metadata"] = {"CreationDate": None, "ModDate": None}
    figure.savefig(target, **options)
    plt.close(figure)


def _positive_float(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if result > 0 else None


def _hidden_sort_key(value: str) -> tuple[int, str]:
    digits = "".join(character for character in value if character.isdigit())
    return (int(digits), value) if digits else (10_000, value)


def _hidden_label(value: str) -> str:
    digits = "".join(character for character in value if character.isdigit())
    return f"Hidden {digits}" if digits else value.replace("_", " ").title()


def _record(
    records: list[dict[str, Any]],
    repo_root: Path,
    output: Path,
    inputs: list[Path],
    method: str,
) -> None:
    records.append(
        {
            "output": _relative(output, repo_root / "outputs" / "paper"),
            "inputs": [_relative(path, repo_root) for path in inputs],
            "method": method,
            "sha256": _sha256(output),
        }
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
