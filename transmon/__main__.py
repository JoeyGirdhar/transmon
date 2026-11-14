"""
Quick summary of a transmon from the command line.

    python -m transmon --EJ 15 --EC 0.3
    python -m transmon --Rn 10 --C 80        (from circuit parameters)
"""

import argparse

from .circuit import EC_from_capacitance, EJ_from_resistance
from .transmon import Transmon


def pretty_freq(f_ghz):
    for scale, unit in [(1, "GHz"), (1e-3, "MHz"), (1e-6, "kHz")]:
        if abs(f_ghz) >= scale:
            return f"{f_ghz / scale:.3g} {unit}"
    return f"{f_ghz * 1e9:.3g} Hz"


def main():
    parser = argparse.ArgumentParser(prog="python -m transmon")
    parser.add_argument("--EJ", type=float, help="EJ/h in GHz")
    parser.add_argument("--EC", type=float, help="EC/h in GHz")
    parser.add_argument("--Rn", type=float, help="junction resistance in kOhm, instead of --EJ")
    parser.add_argument("--C", type=float, help="total capacitance in fF, instead of --EC")
    parser.add_argument("--ng", type=float, default=0.5, help="offset charge (default 0.5)")
    args = parser.parse_args()

    EJ = args.EJ if args.EJ is not None else (args.Rn and EJ_from_resistance(args.Rn))
    EC = args.EC if args.EC is not None else (args.C and EC_from_capacitance(args.C))
    if not EJ or not EC:
        parser.error("need EJ (or Rn) and EC (or C)")

    qubit = Transmon(EJ, EC, ng=args.ng)
    E = qubit.levels(3)
    E = E - E[0]
    g = qubit.n_matrix(3)
    # how much f01 moves between ng = 0 and ng = 1/2
    wobble = abs(qubit.charge_dispersion(1) - qubit.charge_dispersion(0))

    print(f"EJ = {EJ:.3f} GHz   EC = {EC:.4f} GHz   EJ/EC = {qubit.ratio:.1f}")
    print(f"f01 = {E[1]:.4f} GHz   f12 = {E[2] - E[1]:.4f} GHz")
    print(f"anharmonicity = {pretty_freq(qubit.anharmonicity)}"
          f"   (alpha/EC = {qubit.anharmonicity / EC:.3f})")
    print(f"charge dispersion of f01 = {pretty_freq(wobble)} peak to peak")
    print(f"<1|n|0> = {g[1, 0]:.3f}   <2|n|1> = {g[2, 1]:.3f}")


if __name__ == "__main__":
    main()
