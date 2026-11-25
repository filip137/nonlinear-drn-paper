"""Matched varactor-bridge network, quasi-static Kirchhoff / junction-charge model."""
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import activation

CM = .9
W0 = 1.55
TARGET = .14                      # teaching voltage between the two outputs
TEACH_US = 120.
SENSE = .5                       # 2 uA through the nominal diode-connected NMOS
CSTORE_PF = 1.
VT = .025865                     # 27 C
DIODE_IS_UA = 1.
HIDDEN = ("xl", "xr", "yl", "yr")
FREE = HIDDEN + ("pos", "neg")
FIXED = ("x", "y", "bias")
NODES = FIXED + FREE
EDGES = tuple(zip(("x", "x", "y", "y"), HIDDEN)) + tuple(
    (source, output) for output in ("pos", "neg") for source in HIDDEN + FIXED
)
POL = np.array([1., -1., 1., -1.])
ANCHOR = CM + np.array([-.045, .045, -.045, .045])
KINDS = ("diode", "bump", "affine")


@dataclass(frozen=True)
class Layout:
    edges: tuple
    nodes: tuple
    a: np.ndarray
    b: np.ndarray
    incidence: np.ndarray


def make_layout(hidden, edges):
    nodes = FIXED + hidden + ("pos", "neg")
    a = np.array([nodes.index(start) for start, _ in edges])
    b = np.array([nodes.index(end) for _, end in edges])
    return Layout(edges, nodes, a, b, np.eye(len(nodes))[a] - np.eye(len(nodes))[b])


DIODE_LAYOUT = make_layout(HIDDEN, EDGES)
CELL_LAYOUT = make_layout(activation.HIDDEN, activation.EDGES)
LAYOUTS = {"diode": DIODE_LAYOUT, "bump": CELL_LAYOUT, "affine": CELL_LAYOUT}
A, B, INC = DIODE_LAYOUT.a, DIODE_LAYOUT.b, DIODE_LAYOUT.incidence


@dataclass
class Cohort:
    inputs: np.ndarray
    labels: np.ndarray
    train: np.ndarray
    test: np.ndarray
    raw_bounds: tuple = ()


def encode(xy, lo, hi):
    return CM + .17 * np.clip(2*(xy-lo)/(hi-lo)-1, -1, 1)


def cohort():
    xy, labels = make_moons(n_samples=192, noise=.10, random_state=12)
    train, test = train_test_split(np.arange(len(xy)), train_size=64, stratify=labels,
                                   random_state=31)
    lo, hi = xy[train].min(axis=0), xy[train].max(axis=0)
    return Cohort(encode(xy, lo, hi), 2*labels-1, train, test, (lo, hi))


def fresh_test(data):
    xy, labels = make_moons(n_samples=512, noise=.10, random_state=2026)
    return encode(xy, *data.raw_bounds), 2*labels-1


def schedule(train, epochs):
    rng = np.random.default_rng(71)
    return np.concatenate([rng.permutation(train) for _ in range(epochs)])


def solve(inputs, w, label=0, teaching=0., *, kind="diode"):
    """Rows are independent inputs, all conductances in uS, all currents in uA."""
    inputs = np.atleast_2d(inputs)
    layout = LAYOUTS[kind]
    fixed = np.column_stack((inputs, np.full(len(inputs), CM+.13)))
    # Parallel complementary read pair in triode, plus two 1 Mohm midpoint resistors.
    g = 100*(w-1.) + .5
    lap = layout.incidence.T @ (g[:, None]*layout.incidence)
    differential = np.zeros(len(layout.nodes)-len(FIXED))
    differential[-2:] = (1., -1.)
    mat = lap[3:, 3:] + teaching*np.outer(differential, differential)
    rhs = -fixed @ lap[:3, 3:] + teaching*label*TARGET*differential
    if kind != "diode":
        currents = (activation.collector_currents(inputs) if kind == "bump"
                    else activation.affine_basis(inputs) @ activation.AFFINE)
        rhs[:, :-2] -= currents
        return np.column_stack((fixed, np.linalg.solve(mat, rhs.T).T))
    v = np.full((len(inputs), len(FREE)), CM)
    for _ in range(55):
        exp = np.exp(POL*(v[:, :4]-ANCHOR)/VT)
        residual = v @ mat.T - rhs
        residual[:, :4] += POL*DIODE_IS_UA*(exp-1)
        if np.max(np.abs(residual)) < 1e-10:
            break
        jac = np.broadcast_to(mat, (len(inputs), *mat.shape)).copy()
        jac[:, np.arange(4), np.arange(4)] += DIODE_IS_UA*exp/VT
        dv = np.linalg.solve(jac, residual[..., None])[..., 0]
        v -= dv / np.maximum(1, np.max(np.abs(dv), axis=1, keepdims=True)/.05)
    else:
        raise RuntimeError("Kirchhoff solve failed to settle")
    return np.column_stack((fixed, v))


def bridge_charge(nodes, *, kind="diode"):
    """Midpoint-subtracted nonlinear junction charge, pC. Cj0=1 pF, Vj=.7 V."""
    def q(v):
        return 1.4*np.expm1(.5*np.log1p((v-SENSE)/.7))
    layout = LAYOUTS[kind]
    a, b = nodes[..., layout.a], nodes[..., layout.b]
    return q(a) + q(b) - 2*q((a+b)/2)


def python_train(data, order, enabled=True, *, kind="diode"):
    w = np.full(len(LAYOUTS[kind].edges), W0)
    history = [w.copy()]
    for i in order:
        # The 2 fF write bus was parked at W0. Charge shares at write enable.
        armed = (CSTORE_PF*w + .002*W0)/(CSTORE_PF+.002)
        free = solve(data.inputs[i], armed, kind=kind)[0]
        start = bridge_charge(free, kind=kind)
        beta = TEACH_US if enabled else 0.
        new = armed.copy()
        # Endpoint charge and conductance settle self-consistently with a live store.
        for _ in range(12):
            nudged = solve(data.inputs[i], new, data.labels[i], beta, kind=kind)[0]
            next_w = armed + (bridge_charge(nudged, kind=kind)-start)/(CSTORE_PF+.002)
            if np.max(np.abs(next_w-new)) < 1e-12:
                break
            new = next_w
        else:
            raise RuntimeError("Nudged weight iteration failed")
        w = next_w
        history.append(w.copy())
    return np.array(history)


def predict(volts):
    # Ties at the numerical floor always predict -1.
    return np.where(volts>1e-9, 1, -1)


def score(volts, labels):
    return dict(accuracy=float(np.mean(predict(volts)==labels)),
                signed_margin_mV=1000*float(np.mean(labels*volts)),
                target_rmse_mV=1000*float(np.sqrt(np.mean((volts-labels*TARGET)**2))))


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--epochs", type=int, default=3)
    cli.add_argument("--activation", choices=KINDS, default="diode")
    args = cli.parse_args()
    if args.epochs < 1:
        cli.error("--epochs must be positive")
    data = cohort()
    trajectory = python_train(data, schedule(data.train, args.epochs), kind=args.activation)
    for epoch, weights in enumerate(trajectory[::len(data.train)]):
        nodes = solve(data.inputs, weights, kind=args.activation)
        outputs = nodes[:, -2]-nodes[:, -1]
        print(f"epoch {epoch}")
        for split, ix in (("train", data.train), ("test", data.test)):
            s = score(outputs[ix], data.labels[ix])
            print(f"  {split:5} accuracy {s['accuracy']:.2%}, "
                  f"mean signed margin {s['signed_margin_mV']:.3f} mV")
