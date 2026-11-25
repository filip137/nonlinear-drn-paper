"""Six localized cells with transistor or diode-resistor implementations."""
import numpy as np
from scipy.special import wrightomega


def bridge_edges(hidden):
    return tuple(("bias", h) for h in hidden) + tuple(
        (h, output) for output in ("pos", "neg") for h in hidden)


# Equal-width bins across the encoder range. All cells share five voltage references.
X_CENTERS = .73 + .34 * (np.arange(3) + .5) / 3
Y_CENTERS = .73 + .34 * (np.arange(2) + .5) / 2
CENTERS = np.array([(x, y) for x in X_CENTERS for y in Y_CENTERS])
HIDDEN = tuple(f"h{k}" for k in range(len(CENTERS)))
EDGES = bridge_edges(HIDDEN)
GRID_X = .73 + .34*(np.arange(4)+.5)/4
GRID_Y = .73 + .34*(np.arange(3)+.5)/3
GRID_CENTERS = np.array([(x,y) for x in GRID_X for y in GRID_Y])
GRID_HIDDEN = tuple(f"h{k}" for k in range(len(GRID_CENTERS)))
GRID_EDGES = bridge_edges(GRID_HIDDEN)
GRID_GAIN = 2.4
GRID_REF = .90
GRID_TAIL_OHMS = 2500.
BASE = .74
OFFSET = .040
TAIL_UA = 48.
BETA = 1000.
THERMAL_V = 8.617333262145e-5 * (273.15 + 27.)
PN_IS_UA = 1e-6
STEERING_REF = .94
TAIL_OHMS = 10_000.


def collector_currents(inputs):
    """Collector currents in µA when all five matched NPNs are forward active."""
    distance = (np.atleast_2d(inputs)[:, None, :] - CENTERS) / THERMAL_V
    competition = 2 * np.exp(-OFFSET / THERMAL_V) * np.cosh(distance).sum(axis=2)
    return TAIL_UA * (BETA / (BETA + 1)) / (1 + competition)


def affine_basis(inputs):
    return np.column_stack((np.ones(len(inputs)), inputs - .9))


def affine_coefficients():
    # Approximate each cell over a uniform voltage grid, independent of moons data.
    grid = np.linspace(.73, 1.07, 41)
    points = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    return np.linalg.lstsq(affine_basis(points), collector_currents(points), rcond=None)[0]


AFFINE = affine_coefficients()


def steering_sum(inputs, *, centers=CENTERS, gain=1., reference=STEERING_REF):
    """Sum exp(V_anode / VT) for the four input-driven steering diodes."""
    distance = gain*(inputs[:, None, :] - centers) / THERMAL_V
    return 2*np.exp(reference/THERMAL_V)*np.cosh(distance).sum(axis=2)


def diode_bank_load(hidden, steering, tail_ohms=TAIL_OHMS):
    """Hidden-node current in µA and its slope in µS after eliminating the cathode."""
    vt = THERMAL_V
    r = tail_ohms*1e-6  # V per µA
    sense = np.exp(hidden/vt)
    total = steering+sense
    # Solve u/r = Is [sum_j exp((a_j-u)/VT) - 5] for common cathode voltage u.
    offset = 5*r*PN_IS_UA
    cathode = vt*wrightomega(np.log(r*PN_IS_UA*total/vt)+offset/vt)-offset
    scale = PN_IS_UA*np.exp(-cathode/vt)
    forward = scale*sense
    cathode_slope = sense/(total + vt/(r*scale))
    return forward-PN_IS_UA, forward/vt*(1-cathode_slope)


def devices(kind):
    dense = kind == "dense-bank"
    if dense:
        hidden, xs, ys, drive_gain = GRID_HIDDEN, GRID_X, GRID_Y, GRID_GAIN
    else:
        hidden, xs, ys, drive_gain = HIDDEN, X_CENTERS, Y_CENTERS, 1.
    parts = [f"Ch{k} {h} 0 .5p" for k, h in enumerate(hidden)]
    if kind == "affine":
        for k, h in enumerate(HIDDEN):
            base, gx, gy = AFFINE[:, k]
            parts.append(f"Bcell{k} {h} 0 I=1u*({base:.15g}+({gx:.15g})*(v(x)-.9)"
                         f"+({gy:.15g})*(v(y)-.9))")
        return parts
    bank = kind in ("diode-bank","dense-bank")
    if bank:
        parts += [f"Vshift bshift 0 {GRID_REF if dense else STEERING_REF}", f".model cellpn D(Is={PN_IS_UA}u N=1)"]
    else:
        parts += [f"Vbase bcenter 0 {BASE}", f"Vshift bshift 0 {BASE-OFFSET}",
                  f".model cellnpn NPN(Is=1e-15 Bf={BETA} Br=1)"]
    for axis, centers in (("x", xs), ("y", ys)):
        for i, center in enumerate(centers):
            parts.append(f"V{axis}c{i} {axis}c{i} 0 {center:.15g}")
            for sign, gain in (("p", drive_gain), ("m", -drive_gain)):
                parts.append(f"E{axis}{sign}{i} {axis}{sign}{i} bshift {axis} {axis}c{i} {gain}")
    for k, h in enumerate(hidden):
        i, j = divmod(k, len(ys))
        common = f"k{k}" if bank else f"em{k}"
        if bank:
            parts += [f"Rtail{k} {common} 0 {GRID_TAIL_OHMS if dense else TAIL_OHMS}", f"Dout{k} {h} {common} cellpn"]
        else:
            parts += [f"Itail{k} {common} 0 {TAIL_UA}u", f"Qout{k} {h} bcenter {common} cellnpn"]
        for base in (f"xp{i}", f"xm{i}", f"yp{j}", f"ym{j}"):
            parts.append(f"D{k}{base} {base} {common} cellpn" if bank
                         else f"Q{k}{base} dd {base} {common} cellnpn")
    return parts
