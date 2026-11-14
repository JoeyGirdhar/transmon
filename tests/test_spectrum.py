import numpy as np
import pytest

from transmon import Transmon
from transmon.analytic import (asymptotic_dispersion, asymptotic_levels,
                               asymptotic_n_element, mathieu_levels)
from transmon.circuit import (EC_from_capacitance, EJ_from_critical_current,
                              capacitance_from_EC, flux_tuned_EJ)

EC = 0.3


@pytest.mark.parametrize("ratio", [0.5, 1, 5, 10, 20, 50, 100])
@pytest.mark.parametrize("ng", [0.0, 0.5])
def test_charge_basis_matches_mathieu(ratio, ng):
    qubit = Transmon(ratio * EC, EC, ng=ng)
    exact = mathieu_levels(ratio * EC, EC, ng, count=6)
    assert np.allclose(qubit.levels(6), exact, atol=1e-9)


def test_spectrum_is_periodic_and_symmetric_in_ng():
    qubit = Transmon(5 * EC, EC)
    for ng in [0.1, 0.33, 0.8]:
        assert np.allclose(qubit.levels(4, ng=ng), qubit.levels(4, ng=ng + 1))
        assert np.allclose(qubit.levels(4, ng=ng), qubit.levels(4, ng=-ng))


def test_converged_in_ncut():
    small = Transmon(50 * EC, EC, ncut=12).levels(6)
    big = Transmon(50 * EC, EC, ncut=40).levels(6)
    assert np.allclose(small, big, atol=1e-10)


def test_levels_approach_perturbation_theory():
    EJ = 100 * EC
    E = Transmon(EJ, EC).levels(3)
    approx = asymptotic_levels(EJ, EC, np.arange(3))
    assert np.allclose(E - E[0], approx - approx[0], rtol=0.01)


def test_anharmonicity_approaches_eq_2_12():
    # alpha -> -EC and alpha_r -> -(8 EJ/EC)^(-1/2), but slowly: the
    # corrections only fall off like (EJ/EC)^(-1/2).
    misses = []
    for ratio in [50, 100, 300]:
        qubit = Transmon(ratio * EC, EC)
        assert qubit.anharmonicity / EC == pytest.approx(-1, abs=0.15)
        alpha_r = qubit.anharmonicity / qubit.f01
        misses.append(abs(alpha_r / -(8 * ratio) ** -0.5 - 1))
    assert misses[0] > misses[1] > misses[2]
    assert misses[2] < 0.1


def test_anharmonicity_changes_sign_near_ratio_9():
    # "For EJ/EC ~ 9, it changes sign" (Sec. II C)
    assert Transmon(8 * EC, EC).anharmonicity > 0
    assert Transmon(10 * EC, EC).anharmonicity < 0


def test_dispersion_formula_reproduces_numbers_in_paper():
    # Below eq. (2.8): max |dE01/dng| = pi * eps_1 is 7.3e-2, 1.5e-4 and
    # 8.9e-8 EC for EJ/EC = 20, 50, 100.
    for ratio, quoted in [(20, 7.3e-2), (50, 1.5e-4), (100, 8.9e-8)]:
        slope = np.pi * abs(asymptotic_dispersion(ratio * EC, EC, 1)) / EC
        assert slope == pytest.approx(quoted, rel=0.03)


@pytest.mark.parametrize("m", [0, 1])
def test_dispersion_approaches_asymptotic_form(m):
    for ratio in [50, 80]:
        exact = Transmon(ratio * EC, EC).charge_dispersion(m)
        approx = asymptotic_dispersion(ratio * EC, EC, m)
        assert np.sign(exact) == np.sign(approx)
        assert exact == pytest.approx(approx, rel=0.25)


def test_the_transmon_tradeoff():
    # The whole point of the paper: going from EJ/EC = 20 to 60 kills the
    # charge dispersion by three orders of magnitude while the relative
    # anharmonicity drops by less than a factor of two.
    a, b = Transmon(20 * EC, EC), Transmon(60 * EC, EC)
    assert a.charge_dispersion(1) / b.charge_dispersion(1) > 1000
    alpha_r = [q.anharmonicity / q.f01 for q in (a, b)]
    assert alpha_r[0] / alpha_r[1] < 2


def test_neighbor_matrix_elements():
    EJ = 300 * EC
    g = Transmon(EJ, EC).n_matrix(5)
    for j in range(3):
        assert g[j + 1, j] == pytest.approx(asymptotic_n_element(EJ, EC, j), rel=0.05)
    # eq. (3.5): couplings that skip a level are tiny next to the ~1
    # nearest-neighbor ones
    g = Transmon(50 * EC, EC, ng=0.25).n_matrix(4)
    assert abs(g[2, 0]) < 1e-3 and abs(g[3, 1]) < 1e-3


def test_circuit_conversions():
    assert EC_from_capacitance(80) == pytest.approx(0.2421, rel=1e-3)
    assert capacitance_from_EC(EC_from_capacitance(65)) == pytest.approx(65)
    assert EJ_from_critical_current(1) == pytest.approx(0.4967, rel=1e-3)


def test_flux_tuning():
    assert flux_tuned_EJ(20, 0) == pytest.approx(20)
    assert flux_tuned_EJ(20, 0.5) == pytest.approx(0, abs=1e-12)
    assert flux_tuned_EJ(20, 0.5, d=0.2) == pytest.approx(4)
    assert flux_tuned_EJ(20, 0.3) == pytest.approx(flux_tuned_EJ(20, 1.3))
