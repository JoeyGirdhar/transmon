import numpy as np
import pytest

from transmon import Transmon
from transmon.dynamics import (DrivenTransmon, calibrate_pi_pulse, drag_pulse,
                               gaussian, leakage, x_gate_error)


@pytest.fixture(scope="module")
def qubit():
    return DrivenTransmon(Transmon(15, 0.3), levels=4)


def test_weak_resonant_drive_gives_full_rabi_flop(qubit):
    rabi = 0.002                       # 2 MHz, way below |alpha|
    t_pi = 1 / (2 * rabi)
    pops = qubit.square_pulse_populations(rabi, np.array([t_pi / 2, t_pi]))
    assert pops[0, 1] == pytest.approx(0.5, abs=1e-3)
    assert pops[1, 1] == pytest.approx(1, abs=1e-3)


def test_propagator_matches_constant_drive(qubit):
    T = 30.0
    U = qubit.propagator(lambda t: 0.01 * np.ones_like(t), T)
    pops = qubit.square_pulse_populations(0.01, np.array([T]))[0]
    assert np.allclose(np.abs(U[:, 0]) ** 2, pops, atol=1e-6)


def test_propagator_is_unitary(qubit):
    U = qubit.propagator(drag_pulse(8, qubit.alpha), 8)
    assert np.allclose(U.conj().T @ U, np.eye(4), atol=1e-12)


def test_gaussian_is_a_pi_pulse_shape():
    Ox, _ = gaussian(20)
    t = np.linspace(0, 20, 5001)
    assert np.trapezoid(Ox(t), t) == pytest.approx(0.5, rel=1e-4)
    assert Ox(0.0) == pytest.approx(0, abs=1e-12)


def test_drag_suppresses_leakage(qubit):
    T = 10
    plain = qubit.propagator(drag_pulse(T, qubit.alpha, beta=0), T)
    drag = qubit.propagator(drag_pulse(T, qubit.alpha, beta=1), T)
    assert leakage(drag) < leakage(plain) / 100


def test_calibrated_drag_beats_gaussian(qubit):
    T = 12
    plain = calibrate_pi_pulse(qubit, T, beta=0)
    drag = calibrate_pi_pulse(qubit, T, beta=1)
    assert plain["error"] < 1e-3
    assert drag["error"] < plain["error"] / 100


def test_gate_error_of_ideal_gates():
    X = np.eye(4, dtype=complex)
    X[:2, :2] = [[0, 1], [1, 0]]
    assert x_gate_error(X) == pytest.approx(0, abs=1e-12)
    assert x_gate_error(np.eye(4)) == pytest.approx(2 / 3)
