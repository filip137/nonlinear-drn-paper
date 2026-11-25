"""Train the same moons network in Python and ngspice and replay saved states."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.linear_model import LogisticRegression

import model
import spice
import activation


DEFAULT_OUT = Path(__file__).resolve().parents[2] / "outputs" / "moons_bridges"


def train_case(case, out, binary, kind):
    data = model.cohort()
    order = model.schedule(data.train, 3)
    enabled = case != "null"
    fine = case == "fine"
    if fine:
        order = order[:3]
    directory = out / case
    print(f"{case}: running {len(order)} presentations", flush=True)
    test = np.empty(0, dtype=int) if fine else data.test
    weights, readback, live_read, seconds = spice.train(
        binary, directory / "transient", data, order, teaching=enabled,
        maxstep_us=.125 if fine else .5, test=test, kind=kind)
    py_weights = model.python_train(data, order, enabled=enabled, kind=kind)
    np.savez_compressed(directory/"training.npz", order=order, weights=weights,
                        py_weights=py_weights, readback=readback, seconds=seconds,
                        test_order=test, kind=kind, **live_read)
    print(f"{case}: {len(weights)-1} physical updates saved in {seconds:.1f} s", flush=True)


def load_case(out, name):
    with np.load(out / name / "training.npz") as saved:
        return dict(saved)


def python_read(inputs, weights, kind):
    v = model.solve(inputs, weights, kind=kind)
    return v[:, -2] - v[:, -1]


def rms(a):
    return float(np.sqrt(np.mean(a*a)))


def plot(out, data, weights, live_pred, metrics, checkpoints, kind):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 4.1), layout="constrained")
    FigureCanvasAgg(fig)
    boundary, curve = fig.subplots(1, 2)
    grid = np.linspace(model.CM-.17, model.CM+.17, 90)
    xx, yy = np.meshgrid(grid, grid)
    vv = python_read(np.column_stack((xx.ravel(), yy.ravel())), weights, kind).reshape(xx.shape)
    boundary.contourf(xx, yy, vv, [-1, 0, 1], colors=["#e0eff7", "#fce9d9"])
    boundary.contour(xx, yy, vv, [0], colors="#555555", linewidths=1)
    ix = data.test
    for label, marker, color in ((-1,"o","#22628c"),(1,"^","#ad5015")):
        pts = data.inputs[ix[data.labels[ix]==label]]
        boundary.scatter(*pts.T, s=17, marker=marker, c=color, label=f"Class {label:+d}")
    missed = data.inputs[ix[live_pred!=data.labels[ix]]]
    boundary.scatter(*missed.T, s=51, facecolors="none", edgecolors="#222", linewidths=.9,
                     label="Live read error")
    boundary.set(xlabel="Input x (V)", ylabel="Input y (V)", title="Held-out moons",
                 aspect="equal")
    boundary.legend(frameon=True, fontsize=8)
    for prefix, style, color in (("spice","o-","#22628c"),("python","x--","#ad5015")):
        margins = [metrics[f"{prefix}_{j}"]["test"]["signed_margin_mV"] for j in checkpoints]
        curve.plot(np.arange(len(checkpoints)), margins, style, label=prefix.capitalize(),
                   color=color, lw=1.8)
    curve.axhline(0, color="#777", lw=.6)
    curve.set(xticks=np.arange(len(checkpoints)), xlabel="Completed training epochs",
              ylabel="Mean signed output voltage (mV)", title="Held-out output margin")
    curve.legend(frameon=False)
    fig.savefig(out/"moons.png", dpi=160)


def summarize(out, binary, kind):
    data = model.cohort()
    trained = load_case(out, "learn")
    null = load_case(out, "null")
    fine = load_case(out, "fine")
    reference = model.schedule(data.train, 3)
    for case, length in ((trained,len(reference)), (null,len(reference)), (fine,3)):
        np.testing.assert_array_equal(case["order"], reference[:length])
        if str(case.get("kind", "diode")) != kind:
            raise ValueError("Training artifacts use a different activation")
    initial, final = trained["weights"][[0,-1]]
    predictions = {}
    metrics = {}
    checkpoints = [0,64,128,192]
    for j in checkpoints:
        predictions[f"spice_{j}"] = spice.read(binary, out/"learn"/f"dc_{j}", data.inputs, trained["weights"][j], kind=kind)
        predictions[f"python_{j}"] = python_read(data.inputs, trained["py_weights"][j], kind)
    predictions["null_spice"] = spice.read(binary, out/"null"/"dc_final", data.inputs, null["weights"][-1], kind=kind)
    predictions["null_python"] = python_read(data.inputs, null["py_weights"][-1], kind)
    for name,v in predictions.items():
        metrics[name] = {split: model.score(v[indices], data.labels[indices])
                         for split, indices in (("train",data.train),("test",data.test))}
    magnitude = trained["weights"][1]-initial
    approx = trained["py_weights"][1]-initial
    sp_final = predictions["spice_192"]
    py_final = predictions["python_192"]
    test = data.test
    read_error = sp_final-python_read(data.inputs, final, kind)
    yhat = model.predict(sp_final[test])
    for case in (trained, null):
        np.testing.assert_array_equal(case["test_order"], test)
    live = trained["test_output"]
    live_pred = model.predict(live)
    volts = (data.inputs[:,0],data.inputs[:,1])
    linear = LogisticRegression(C=1e4).fit(data.inputs[data.train], data.labels[data.train])
    result = dict(activation=kind, metrics=metrics,
        live_test={name: model.score(case["test_output"],data.labels[test])
                   for name,case in (("learn",trained),("null",null))},
        live_test_correct=int(np.sum(live_pred==data.labels[test])),
        live_vs_frozen_test_decisions_agree=int(np.sum(live_pred==yhat)),
        live_vs_frozen_test_output_rms_difference_uV=1e6*rms(live-sp_final[test]),
        live_test_max_weight_excursion_mV=1e3*float(np.max(np.abs(trained["test_weights"]-final))),
        train_examples=len(data.train), test_examples=len(test), trained_presentations=len(reference),
        final_test_correct=int(np.sum(yhat==data.labels[test])),
        final_test_confusion_actual_by_predicted=[[int(np.sum((data.labels[test]==a)&(yhat==b)))
                                                  for b in (-1,1)] for a in (-1,1)],
        linear_logistic_accuracy={split:float(linear.score(data.inputs[ix],data.labels[ix]))
                                  for split,ix in (("train",data.train),("test",test))},
        first_update_cosine=float(np.dot(magnitude,approx)/(np.linalg.norm(magnitude)*np.linalg.norm(approx))),
        first_update_rms_difference_uV=1e6*rms(magnitude-approx),
        final_weight_rms_difference_mV=1e3*rms(final-trained["py_weights"][-1]),
        trained_weights_min_max_V=[float(trained["weights"].min()), float(trained["weights"].max())],
        final_max_abs_weight_change_mV=1e3*float(np.max(np.abs(final-initial))),
        null_final_max_abs_weight_change_mV=1e3*float(np.max(np.abs(null["weights"][-1]-initial))),
        final_same_weight_dc_rms_difference_uV=1e6*rms(read_error[test]),
        final_independent_test_output_rms_difference_mV=1e3*rms((sp_final-py_final)[test]),
        final_independent_test_decisions_agree=int(np.sum(yhat==model.predict(py_final[test]))),
        fine_step_3_presentations_max_weight_difference_uV=1e6*float(np.max(np.abs(fine["weights"]-trained["weights"][:4]))),
        final_test_min_abs_margin_uV=1e6*float(np.min(np.abs(sp_final[test]))),
        final_test_median_abs_margin_mV=1e3*float(np.median(np.abs(sp_final[test]))))
    if kind != "diode":
        grid = np.linspace(.73, 1.07, 25)
        points = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
        measured = spice.read(binary, out/"learn"/"dc_grid", points, final, kind=kind)
        plane = activation.affine_basis(points)
        fitted = plane @ np.linalg.lstsq(plane, measured, rcond=None)[0]
        result["grid_residual_from_best_plane_rms_mV"] = 1000*rms(measured-fitted)
        result["grid_output_standard_deviation_mV"] = 1000*float(measured.std())
        if kind == "bump":
            probes = np.vstack((activation.CENTERS, points))
            currents = spice.read_cells(binary, out/"cell_transfer", probes)
            theory = activation.collector_currents(probes)
            result["cell_peak_current_uA"] = float(currents[0,0])
            result["cell_max_theory_error_nA"] = 1000*float(np.abs(currents-theory).max())
            result["cell_centers_select_own_cell"] = bool(np.array_equal(
                currents[:len(activation.CENTERS)].argmax(axis=1), np.arange(len(activation.CENTERS))))
        fresh_inputs, fresh_labels = model.fresh_test(data)
        fresh_output = spice.read(binary, out/"learn"/"dc_fresh", fresh_inputs, final, kind=kind)
        result["fresh_test"] = dict(size=len(fresh_labels), seed=2026,
            spice=model.score(fresh_output, fresh_labels),
            python=model.score(python_read(fresh_inputs, trained["py_weights"][-1], kind), fresh_labels),
            logistic_accuracy=float(linear.score(fresh_inputs, fresh_labels)))
    (out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    with (out/"trained_weights.csv").open("w",newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["edge","node_a","node_b","spice_weight_V","python_weight_V"])
        for k, (a,b) in enumerate(model.LAYOUTS[kind].edges):
            writer.writerow([k,a,b,final[k],trained["py_weights"][-1,k]])
    with (out/"predictions.csv").open("w",newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id","split","x_V","y_V","label",*[f"{name}_V" for name in predictions],
                         "live_test_V","live_null_test_V"])
        in_train = set(data.train)
        live_values = dict(zip(test, zip(live, null["test_output"])))
        for i,label in enumerate(data.labels):
            writer.writerow([i,"train" if i in in_train else "test",volts[0][i],volts[1][i],label,
                             *[v[i] for v in predictions.values()],*live_values.get(i, ("",""))])
    plot(out, data, final, live_pred, metrics, checkpoints, kind)
    print(json.dumps(result, indent=2), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--case",choices=["learn","null","fine","report","all"],default="all")
    cli.add_argument("--activation", choices=model.KINDS, default="diode")
    cli.add_argument("--out",type=Path)
    cli.add_argument("--ngspice",default=shutil.which("ngspice"))
    args = cli.parse_args()
    out = args.out or (DEFAULT_OUT if args.activation == "diode" else
                       DEFAULT_OUT.with_name(f"moons_bridges_{args.activation}"))
    if not args.ngspice:
        cli.error("Supply an ngspice executable using --ngspice or PATH")
    for case in ("learn","null","fine") if args.case=="all" else (args.case,):
        if case != "report":
            train_case(case, out, args.ngspice, args.activation)
    if args.case in ("report","all"):
        summarize(out, args.ngspice, args.activation)


if __name__ == "__main__":
    main()
