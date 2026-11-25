"""Generate ngspice decks, keep all bridge weight capacitors live during training."""
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

import activation
from model import ANCHOR, CM, DIODE_IS_UA, HIDDEN, LAYOUTS, POL, TARGET, TEACH_US, W0

HERE = Path(__file__).resolve().parent
PERIOD_US = 62.


def network(frozen=None, *, kind="diode"):
    parts = [
        ".param lam=0 junction_ratio=1 sink_ratio=1 rmid=1Meg cstore=1p rleak=1e16",
        ".include bridge.inc", "Vdd dd 0 1.8", "Vbp vbp 0 1.30", "Vpr pr 0 .2",
        f"Vpark park 0 {W0}", f"Vbias bias 0 {CM+.13}", ".temp 27",
        f".model activation D(Is={DIODE_IS_UA}u N=1)",
    ]
    if kind == "diode":
        for k, (h, ref, pol) in enumerate(zip(HIDDEN, ANCHOR, POL)):
            a, b = (h, f"a{k}") if pol>0 else (f"a{k}", h)
            parts += [f"Vanchor{k} a{k} 0 {ref}", f"Dhid{k} {a} {b} activation", f"Ch{k} {h} 0 .5p"]
    else:
        parts.extend(activation.devices(kind))
    parts += ["Cp pos 0 .5p", "Cn neg 0 .5p"]
    for k, (a, b) in enumerate(LAYOUTS[kind].edges):
        parts.append(f"Xe{k} {a} {b} w{k} m{k} m{k} dd vbp pr park en enb dump dumpb bridge")
        parts.append(f".ic v(w{k})={W0}" if frozen is None else f"Vw{k} w{k} 0 {frozen[k]:.15g}")
    return parts


def run(binary, folder, parts, trace, limit=1800):
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE/"bridge.inc", folder/"bridge.inc")
    (folder/"network.cir").write_text("\n".join(parts+[".end", ""]))
    path = folder/"trace.raw"
    path.unlink(missing_ok=True)
    begin = time.monotonic()
    with (folder/"ngspice.log").open("w") as log:
        p = subprocess.run([binary, "-b", "-n", "-r", path.name, "network.cir"],
                           cwd=folder, stdout=log, stderr=subprocess.STDOUT, timeout=limit)
    report = (folder/"ngspice.log").read_text()
    if p.returncode or "aborted" in report.lower() or "timestep too small" in report.lower():
        raise RuntimeError(report[-2100:])
    with path.open("rb") as f:
        header = []
        while (line:=f.readline()) != b"Binary:\n":
            if not line:
                raise ValueError("Missing binary data")
            header.append(line.decode().strip())
        names = [row.split()[1].lower() for row in header[header.index("Variables:")+1:]]
        points = int(next(s.split(":")[1] for s in header if s.startswith("No. Points:")))
        array = np.fromfile(f, dtype=np.float64)
    if array.size != points*len(names) or points<1 or not np.isfinite(array).all():
        raise ValueError("Invalid SPICE result")
    cols = dict(zip(names, array.reshape(points,-1).T))
    for key in trace:
        if key not in cols:
            raise ValueError(f"Missing {key}; available {names}")
    return cols, time.monotonic()-begin


def train(binary, folder, data, order, teaching=True, maxstep_us=.5, test=(), *, kind="diode"):
    if not len(order):
        raise ValueError("Training requires at least one sample")
    parts = [f"Two moons, {kind} activation with nominal varactor bridges", *network(kind=kind)]
    presentations = np.concatenate((order, np.asarray(test, dtype=int)))
    stop = len(order)*PERIOD_US
    for k, name in enumerate(("x", "y", "target")):
        seq = (data.inputs[presentations,k] if k<2 else
               np.concatenate((data.labels[order]*TARGET, np.zeros(len(test)))))
        old = CM if k<2 else 0.
        lines = [f"V{name} {name} 0 PWL(0 {old:.12g}"]
        for j, value in enumerate(seq):
            t = j*PERIOD_US
            lines.append(f"+ {t+1:g}u {old:.12g} {t+5:g}u {value:.12g}")
            old = value
        parts += [*lines, "+ )"]
    parts += [
        f"Vtraining training 0 PWL(0 1 {stop-.5:g}u 1 {stop:g}u 0)",
        "Venclk enclk 0 PULSE(0 1.8 14u .02u .02u 24.98u 62u)",
        "Ben en 0 V=v(enclk)*v(training)",
        # Disconnect park before en rises; restore it after en falls.
        "Vdumpclk dumpclk 0 PULSE(1.8 0 13.95u .02u .02u 25.10u 62u)",
        "Bdump dump 0 V=1.8+(v(dumpclk)-1.8)*v(training)",
        "Eenb enb dd en 0 -1", "Edumpb dumpb dd dump 0 -1",
        "Vteach teach 0 PULSE(0 1 17u 4u 4u 21u 62u)",
        f"Bteach pos neg I={{{TEACH_US if teaching else 0}u*v(training)*v(teach)*(v(pos,neg)-v(target))}}",
        ".options method=gear reltol=2e-5 abstol=1e-15 vntol=2e-8 chgtol=1e-16 gmin=1e-18 itl4=150",
    ]
    weights = [f"v(w{k})" for k in range(len(LAYOUTS[kind].edges))]
    parts += [".save time v(pos) v(neg) "+" ".join(weights),
              f".tran {maxstep_us}u {len(presentations)*PERIOD_US:g}u 0 {maxstep_us}u"]
    raw, elapsed = run(binary, folder, parts, ["time","v(pos)","v(neg)",*weights])
    ts = raw["time"]*1e6
    if abs(ts[-1]-len(presentations)*PERIOD_US)>1e-5:
        raise ValueError("Incomplete training transient")
    ends = np.arange(len(order))*PERIOD_US + 60
    hist = np.array([np.interp(ends, ts, raw[k]) for k in weights]).T
    output = raw["v(pos)"]-raw["v(neg)"]
    readback = np.column_stack([np.interp(ends-48,ts,output), np.interp(ends-22,ts,output),
                                np.interp(ends,ts,output)])
    test_ends = stop + np.arange(len(test))*PERIOD_US + 60
    test_read = dict(test_output=np.interp(test_ends, ts, output),
                     test_weights=np.array([np.interp(test_ends, ts, raw[k]) for k in weights]).T)
    # Netlists, numeric readbacks and logs suffice to reproduce the large binary trace.
    (folder/"trace.raw").unlink()
    return np.vstack((np.full(len(weights), W0), hist)), readback, test_read, elapsed


def dc_sweep(inputs, saved):
    parts = ["Vindex idx 0 0"]
    for k,name in enumerate(("x","y")):
        pts = [f"{j},{v:.12g}" for j,v in enumerate(inputs[:,k])]
        parts += [f"B{name} {name} 0 V=pwl(v(idx),", "+ "+",\n+ ".join(pts)+")"]
    return parts + [".options reltol=1e-7 abstol=1e-15 gmin=1e-18", ".save "+" ".join(saved),
              f".dc Vindex 0 {len(inputs)-1} 1"]


def read(binary, folder, inputs, weights, *, kind="diode"):
    parts = ["Teacher-disconnected DC inference on saved bridge capacitor voltages", *network(weights, kind=kind),
             "Ven en 0 0", "Venb enb 0 1.8", "Vdump dump 0 1.8", "Vdumpb dumpb 0 0"]
    parts.extend(dc_sweep(inputs, ["v(pos)", "v(neg)"]))
    raw, _ = run(binary, folder, parts, ["v(pos)","v(neg)"])
    if len(raw["v(pos)"]) != len(inputs):
        raise ValueError("Incomplete DC readback")
    return raw["v(pos)"]-raw["v(neg)"]


def read_cells(binary, folder, inputs):
    parts = ["BJT hidden-cell transfer at fixed collector voltage", "Vdd dd 0 1.8", ".temp 27",
             *activation.devices("bump")]
    saved = []
    for k, h in enumerate(activation.HIDDEN):
        parts.append(f"Vhold{k} {h} 0 .9")
        saved.append(f"i(vhold{k})")
    parts.extend(dc_sweep(inputs, saved))
    raw, _ = run(binary, folder, parts, saved)
    return -1e6*np.column_stack([raw[name] for name in saved])
