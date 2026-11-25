# Diode activations for the varactor-bridge network

The compact circuit has four hidden nodes and 12 trainable varactor bridges. Each raw input drives two hidden nodes through one bridge apiece. Each hidden node connects to both outputs through two more bridges. A single shunt diode on each hidden node provides the nonlinearity. The two outputs receive all their input through these four nodes.

The nominal study uses the original 64 training points and three epochs. The full ngspice transient then presents 128 held-out inputs with the teacher and write switches disabled. A second transient repeats the clocks and inputs with teaching disabled throughout. A three-presentation timestep check and fixed-weight DC replay on 512 additional moons points complete the comparison. Both the diode stage and its linear control use the same protocol.

## Fixed-rail diode

Two diodes point from their hidden nodes to the shared 0.525 V rail. The other two point from the shared 1.275 V rail into their hidden nodes. A positive excursion progressively turns one orientation on, and a negative excursion turns the other on. The current entering a shunt in Python follows the Shockley equation with the same polarity, saturation current (1 pA), ideality factor (1) and temperature (27 °C) as the PN diode in ngspice. Each hidden node also has the prototype's 0.5 pF node capacitance. The 14-MOSFET varactor writer and read port on each edge are the same as earlier.

At the 0.9 V common-mode voltage the shunt carries 2 µA and its small-signal conductance is 77.3 µS. The control replaces each diode with a 12.93 kΩ resistor to a shifted reference that matches both current and slope at 0.9 V. On the final free training reads, the first x-driven shunt spanned 2.62–318.84 µS. The devices operate well into their nonlinear range even though the trained differential output is close to planar. Across a 25 × 25 input sweep its deviation from the best-fitting plane was 0.019 mV RMS on an output with 0.923 mV standard deviation.

## Localized response using five diodes and a resistor

This option uses the same six regular grid locations and 18 trainable bridges as the transistor-cell experiment. Each hidden node connects to the anode of one sensing diode. Four other diode anodes receive the shared voltages 0.94 V ± (x−cx) and 0.94 V ± (y−cy). All five cathodes meet at one node with a 10 kΩ resistor to ground.

Near a cell's center, the sensing diode conducts and pulls the hidden voltage down. Far from the center, one of the steering diodes raises the shared cathode voltage, reducing current through the sensing diode. This makes the cell respond to a neighborhood in both input coordinates. Six cells use 30 PN diodes and six resistors. The ten shared signed input drives use ideal linear sources in SPICE, as they did in the transistor-cell circuit. The resistor sets the current as the cathode voltage changes. The Python implementation solves that cathode voltage self-consistently with the hidden node. The operating bias was selected using the 64 training points.

## Results

| Hidden activation | Trainable bridges | Held-out live reads | Additional 512-point DC replay | Mean signed live margin |
|---|---:|---:|---:|---:|
| Four single-diode shunts | 12 | 105/128 (82.0%) | 434/512 (84.8%) | 0.755 mV |
| Four matched linear shunts | 12 | 105/128 (82.0%) | 433/512 (84.6%) | 0.661 mV |
| Six five-diode cells | 18 | 106/128 (82.8%) | 453/512 (88.5%) | 0.188 mV |
| Previous six five-transistor cells | 18 | 112/128 (87.5%) | 463/512 (90.4%) | 0.645 mV |

Across the input grid the five-diode network departed from its best-fitting output plane by 0.077 mV RMS, with an output standard deviation of 0.213 mV. At a fixed 0.9 V hidden-node voltage, the corresponding cell drew the most current at each of the six centers. The SPICE transfer sweep agreed with the analytic load to within 0.001 nA. Both diode networks scored 64/128 with teaching disabled. The independently trained Python model reproduced all 128 live decisions for each diode network and overestimated the mean signed margin in both cases.

On these two cohorts the single-shunt circuit offers the lowest component count and the largest margin of the three nonlinear circuits shown. The diode bank gives a more localized response and higher accuracy on the 512-point replay, at the cost of a smaller margin. This is a nominal matched-device comparison with a single training split. The 512-point cohort was also used for the earlier transistor comparison.

[Fixed-rail results](reference/shunt/results.json) · [Linear control](reference/shunt-linear/results.json) · [Diode bank results](reference/diode-bank/results.json) · [Diode bank boundary](reference/diode-bank/moons.png)

## Reproduce

Run from this directory with ngspice 41 and the repository's Python environment.

```sh
python model.py --activation shunt
python run.py --activation shunt --ngspice /path/to/ngspice
python run.py --activation shunt-linear --ngspice /path/to/ngspice
python run.py --activation diode-bank --ngspice /path/to/ngspice
```

The generated decks, traces and case logs are in the worktree's `outputs/moons_bridges_<activation>/` directories.
