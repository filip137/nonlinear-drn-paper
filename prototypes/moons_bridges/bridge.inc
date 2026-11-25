* Continuously enabled varactor difference-of-squares writer.
* A,B: actual network nodes. W: stored gate voltage. MID: local midpoint.
* REF is wired to MID except in the fixed-reference diagnostic.
* VBP, PR, PARK and complementary LEARN clocks can be shared.
.model junction D(Is=1e-19 Cjo=1p Vj=.7 M=.5)
.model mn nmos(level=1 vto=.4 kp=200u gamma=0 lambda={lam} tox=10n cgso=.15n cgdo=.15n)
.model mp pmos(level=1 vto=-.4 kp=100u gamma=0 lambda={lam} tox=10n cgso=.15n cgdo=.15n)
.model readn nmos(level=1 vto=.4 kp=100u gamma=0 lambda=0 tox=20n cgso=.15n cgdo=.15n)
.model readp pmos(level=1 vto=-.4 kp=100u gamma=0 lambda=0 tox=20n cgso=.15n cgdo=.15n)

.subckt tg a b en enb dd
Mn a en b 0 mn w=0.6u l=0.4u
Mp a enb b dd mp w=0.6u l=0.4u
.ends

.subckt bridge a b w mid ref dd vbp pr park en enb dump dumpb
* A 2*Rmid shunt across the read port also generates the reference voltage.
Ra a mid {rmid}
Rb b mid {rmid}
Da s a junction
Db s b junction {junction_ratio}
Dref r ref junction 2

* Diode-connected NMOS transistors sum junction displacement and bias current.
* Is and Ir are copied with opposite signs into the same output branch.
Mbs s vbp dd dd mp w=4u l=1u
Mbr r vbp dd dd mp w=4u l=1u
Ms s s 0 0 mn w=2u l=1u
Mr r r 0 0 mn w=2u l=1u
Mscopy p s 0 0 mn w=2u l=1u
Mpdiode p p dd dd mp w=4u l=1u
Msource bus p dd dd mp w=4u l=1u
Msink bus r 0 0 mn w={2u*sink_ratio} l=1u

* Park the output bus between nudges. Isolate W before removing the nudge.
Cbus bus 0 2f
Xwrite bus w en enb dd tg
Xpark bus park dump dumpb dd tg
Cw w 0 {cstore}
Rleak w 0 {rleak}
* High-impedance conductance control on W. ISENSE is a zero-volt ammeter.
Visense a ard 0
Mreadn ard w b 0 readn w=1u l=1u
Mreadp ard pr b dd readp w=1u l=1u
.ends
