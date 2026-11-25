"""Margin-gated diode network on the original two-moons training set."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

import model
import spice

KIND = "dense-bank"
MARGIN = model.DENSE_MARGIN
STORE_PF = model.DENSE_STORE_PF
EPOCHS = model.DENSE_EPOCHS
VALID_SEED = 913
TEST_SEED = 20707
TEST_SIZE = 1024
LIVE_COUNT = 128
REPEAT_SEED = 100
OUT = Path(__file__).resolve().parents[2] / "outputs" / "moons_accuracy_final"


def points(seed, n, bounds):
    xy, labels = make_moons(n_samples=n, noise=.1, random_state=seed)
    return model.encode(xy, *bounds), 2*labels-1


def cohort(dataset_seed=12):
    old = model.cohort(dataset_seed)
    x, labels = points(TEST_SEED, TEST_SIZE, old.raw_bounds)
    data = model.Cohort(np.vstack((old.inputs, x)), np.concatenate((old.labels, labels)),
                        old.train, old.test, old.raw_bounds)
    return data, np.arange(len(old.inputs), len(data.inputs))


def simulate(case, out, binary):
    data, test = cohort(REPEAT_SEED if case == "repeat" else 12)
    order = model.schedule(data.train, EPOCHS)
    step = .5
    live = test[:LIVE_COUNT] if case == "learn" else []
    if case == "fine":
        order = order[:20]
        step = .125
    margin = None if case == "ungated" else MARGIN
    teach = case != "null"
    print(f"{case} started, {len(order)} training and {len(live)} live test presentations", flush=True)
    hist, rb, extra, seconds = spice.train(binary, out/case/"transient", data, order, teaching=teach,
                                            maxstep_us=step, test=live, kind=KIND,
                                            store_pf=STORE_PF, margin=margin)
    py = model.python_train(data, order, enabled=teach, kind=KIND, margin=margin, store_pf=STORE_PF)
    np.savez_compressed(out/case/"training.npz", order=order, weights=hist, py_weights=py,
                        readback=rb, live_order=live, seconds=seconds, **extra)
    print(f"{case} finished in {seconds:.1f}s, {extra['write_enabled'].sum()} write windows", flush=True)


def python_read(x, weights, kind=KIND):
    nodes = model.solve(x, weights, kind=kind)
    return nodes[:, -2]-nodes[:, -1]


def validation(out):
    data = model.cohort()
    val, y = points(VALID_SEED, 384, data.raw_bounds)
    rows = []
    order = model.schedule(data.train, EPOCHS)
    for pf in (.33, 1.):
        for threshold in (None, 10e-6, 30e-6, 100e-6, 300e-6, .001):
            h = model.python_train(data, order, kind=KIND, margin=threshold, store_pf=pf)
            for epoch, w in enumerate(h[len(data.train)::len(data.train)], 1):
                rows.append((pf, threshold, epoch,
                             int(np.sum(model.predict(python_read(val,w)) == y)), len(y)))
    out.mkdir(parents=True, exist_ok=True)
    with (out/"validation.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["store_pF","threshold_V","epoch","correct","examples"])
        writer.writerows(rows)


def metrics(v, labels):
    predicted = model.predict(v)
    return dict(correct=int(np.sum(predicted == labels)), count=len(labels),
                accuracy=float(np.mean(predicted == labels)),
                mean_signed_margin_uV=1e6*float(np.mean(v*labels)),
                median_abs_margin_uV=1e6*float(np.median(np.abs(v))),
                confusion_actual_by_predicted=[[int(np.sum((labels==a)&(predicted==b)))
                                                for b in (-1,1)] for a in (-1,1)])


def figure(out, x, y, output, final_weights, epochs):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8.5,3.8), layout="constrained")
    FigureCanvasAgg(fig)
    view, curve = fig.subplots(1, 2)
    axis = np.linspace(.73,1.07,80)
    xx, yy = np.meshgrid(axis,axis)
    surface = python_read(np.column_stack((xx.ravel(),yy.ravel())), final_weights).reshape(xx.shape)
    view.contourf(xx,yy,surface,[-1,0,1],colors=["#dcebf3","#f9e9d9"])
    view.contour(xx,yy,surface,[0],colors="#243640",linewidths=1)
    for label,marker,color in ((-1,"o","#22628c"),(1,"^","#ad5015")):
        view.scatter(*x[y==label].T, s=6, marker=marker, c=color, alpha=.65)
    missed = x[model.predict(output)!=y]
    view.scatter(*missed.T,s=37,facecolors="none",edgecolors="black",label="Error")
    view.set(aspect="equal",title="Locked test set",xlabel="Input x (V)",ylabel="Input y (V)")
    if len(missed):
        view.legend(frameon=False,fontsize=8)
    curve.plot(range(len(epochs)), 100*np.array(epochs), "o-", color="#22628c")
    curve.axhline(95, ls="--", lw=1, color="#777")
    curve.set(xlabel="Epoch",ylabel="Accuracy (%)",title="Validation in ngspice",ylim=(45,102))
    fig.savefig(out/"moons.png", dpi=160)


def report(out, binary):
    data, test = cohort()
    val, targets = points(VALID_SEED, 384, data.raw_bounds)
    truth = data.labels[test]
    schedule = model.schedule(data.train, EPOCHS)
    result = {"kind":KIND, "epochs":EPOCHS, "seed_test":TEST_SEED,
              "training_examples":len(data.train), "store_pF":STORE_PF,
              "threshold_uV":1e6*MARGIN, "validation_seed":VALID_SEED, "cases":{}}
    with np.load(out/"fine"/"training.npz") as chk:
        fine_weights, fine_gates = chk["weights"], chk["write_enabled"]
    for case in ("learn","ungated","null"):
        with np.load(out/case/"training.npz") as saved:
            np.testing.assert_array_equal(saved["order"], schedule)
            hist, py = saved["weights"], saved["py_weights"]
            if hist.shape != (len(schedule)+1, len(model.LAYOUTS[KIND].edges)):
                raise ValueError("Incomplete bridge history")
            gates = saved["write_enabled"]
            final = hist[-1]
            all_x = np.vstack((data.inputs[data.train], val, data.inputs[data.test], data.inputs[test]))
            volts = spice.read(binary, out/case/"dc_final", all_x, final, kind=KIND, store_pf=STORE_PF)
            train_v, val_v, previous_v, locked = np.split(volts, np.cumsum([len(data.train),len(val),len(data.test)]))
            entry = dict(train=metrics(train_v,data.labels[data.train]), validation=metrics(val_v,targets),
                         original_test=metrics(previous_v,data.labels[data.test]), test=metrics(locked,truth),
                         writes_by_epoch=gates.reshape(EPOCHS,-1).sum(axis=1).tolist())
            result["cases"][case] = entry
            if case != "learn":
                continue
            np.testing.assert_array_equal(saved["live_order"], test[:LIVE_COUNT])
            if saved["test_write_enabled"].any():
                raise ValueError("A write window opened during live testing")
            live = saved["test_output"]
            entry["live_test"] = metrics(live,truth[:LIVE_COUNT])
            entry["live_vs_dc_decisions_agree"] = int(np.sum(model.predict(live)==model.predict(locked[:LIVE_COUNT])))
            entry["live_vs_dc_rms_uV"] = 1e6*float(np.sqrt(np.mean((live-locked[:LIVE_COUNT])**2)))
            entry["python_test"] = metrics(python_read(data.inputs[test],py[-1]),truth)
            same_weights = python_read(data.inputs[test],final)
            entry["same_weight_dc_vs_python_rms_uV"] = 1e6*float(np.sqrt(np.mean((locked-same_weights)**2)))
            entry["test_accuracy_by_read_offset_uV"] = {
                str(offset): float(np.mean(model.predict(locked+offset*1e-6)==truth))
                for offset in (-30,-10,0,10,30)}
            old_x, old_y = model.fresh_test(data)
            old_v = spice.read(binary,out/case/"dc_previous_512",old_x,final,kind=KIND,store_pf=STORE_PF)
            entry["previous_512"] = metrics(old_v,old_y)
            entry["gate_mismatches_vs_free_read"] = int(np.sum(gates != (data.labels[schedule]*saved["readback"][:,0]<MARGIN)))
            entry["fine_gate_differences"] = int(np.sum(fine_gates!=gates[:len(fine_gates)]))
            entry["fine_prefix_max_weight_difference_uV"] = 1e6*float(np.max(np.abs(fine_weights-hist[:len(fine_weights)])))
            entry["weight_range_V"] = [float(hist.min()),float(hist.max())]
            entry["validation_by_epoch"] = []
            for epoch, weights in enumerate(hist[::len(data.train)]):
                v = spice.read(binary, out/case/f"dc_epoch_{epoch}",val,weights,kind=KIND,store_pf=STORE_PF)
                entry["validation_by_epoch"].append(metrics(v,targets)["accuracy"])
            figure(out, data.inputs[test], truth, locked, final, entry["validation_by_epoch"])
            with (out/"trained_weights.csv").open("w",newline="") as f:
                writer = csv.writer(f,lineterminator="\n")
                writer.writerow(["edge","node_a","node_b","spice_V","python_V"])
                for i, (a,b) in enumerate(model.LAYOUTS[KIND].edges):
                    writer.writerow([i,a,b,final[i],py[-1,i]])
            np.savez_compressed(out/"test_readout.npz",inputs=data.inputs[test],labels=truth,
                                dc_V=locked,live_V=live,live_indices=np.arange(len(live)))
    baseline = LogisticRegression(C=1e4).fit(data.inputs[data.train],data.labels[data.train])
    result["linear_logistic_test_accuracy"] = float(baseline.score(data.inputs[test],truth))
    earlier_weights = Path(__file__).resolve().parent/"reference"/"diode-bank"/"trained_weights.csv"
    with earlier_weights.open() as f:
        old_weights = np.array([float(row["spice_weight_V"]) for row in csv.DictReader(f)])
    old_output = spice.read(binary,out/"previous_grid_dc",data.inputs[test],old_weights,kind="diode-bank")
    result["previous_six_cell_spice_test"] = metrics(old_output,truth)
    repeated, test = cohort(REPEAT_SEED)
    with np.load(out/"repeat"/"training.npz") as saved:
        np.testing.assert_array_equal(saved["order"], model.schedule(repeated.train,EPOCHS))
        repeat_v = spice.read(binary,out/"repeat"/"dc_final",repeated.inputs[test],
                              saved["weights"][-1],kind=KIND,store_pf=STORE_PF)
        result["independent_training_repeat"] = dict(dataset_seed=REPEAT_SEED,
            spice_test=metrics(repeat_v,repeated.labels[test]),
            python_test=metrics(python_read(repeated.inputs[test],saved["py_weights"][-1]),repeated.labels[test]))
    (out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


def execute(case_name, out, binary):
    if case_name == "validation":
        validation(out)
        return
    cases = ("learn","ungated","null","fine","repeat","report") if case_name == "all" else [case_name]
    for case in cases:
        if case == "report":
            report(out, binary)
        else:
            simulate(case, out, binary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("learn","ungated","null","fine","repeat","report","validation","all"), default="all")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--ngspice", default=shutil.which("ngspice"))
    args = parser.parse_args()
    if not args.ngspice and args.case != "validation":
        parser.error("provide --ngspice or put ngspice on PATH")
    execute(args.case, args.out, args.ngspice)


if __name__ == "__main__":
    main()
