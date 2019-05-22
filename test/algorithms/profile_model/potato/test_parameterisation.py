from __future__ import division
from __future__ import print_function
import os
import pytest
from scitbx import matrix
from dials.algorithms.profile_model.potato.parameterisation import (
    Simple1MosaicityParameterisation,
    Simple6MosaicityParameterisation,
    Angular2MosaicityParameterisation,
    Angular4MosaicityParameterisation,
    WavelengthSpreadParameterisation,
    ModelState,
    ReflectionModelState,
)
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex


def test_Simple1MosaicityParameterisation():

    p = Simple1MosaicityParameterisation(params=(1e-3,))

    assert p.is_angular() == False
    assert p.num_parameters() == 1
    assert p.parameters() == (1e-3,)
    p.set_parameters((2e-3,))
    assert p.parameters() == (2e-3,)
    assert p.sigma()[0] == pytest.approx(p.parameters()[0] ** 2)
    assert p.sigma()[1] == 0
    assert p.sigma()[2] == 0
    assert p.sigma()[3] == 0
    assert p.sigma()[4] == pytest.approx(p.parameters()[0] ** 2)
    assert p.sigma()[5] == 0
    assert p.sigma()[6] == 0
    assert p.sigma()[7] == 0
    assert p.sigma()[8] == pytest.approx(p.parameters()[0] ** 2)
    d = p.first_derivatives()
    assert len(d) == 1
    assert d[0][0] == pytest.approx(2 * p.parameters()[0])
    assert d[0][1] == 0
    assert d[0][2] == 0
    assert d[0][3] == 0
    assert d[0][4] == pytest.approx(2 * p.parameters()[0])
    assert d[0][5] == 0
    assert d[0][6] == 0
    assert d[0][7] == 0
    assert d[0][8] == pytest.approx(2 * p.parameters()[0])


def test_Simple6MosaicityParameterisation():

    params = (1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3)

    p = Simple6MosaicityParameterisation(params=params)

    assert p.is_angular() == False
    assert p.num_parameters() == 6
    assert p.parameters()[0] == pytest.approx(params[0])
    assert p.parameters()[1] == pytest.approx(params[1])
    assert p.parameters()[2] == pytest.approx(params[2])
    assert p.parameters()[3] == pytest.approx(params[3])
    assert p.parameters()[4] == pytest.approx(params[4])
    assert p.parameters()[5] == pytest.approx(params[5])

    params = (2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3)
    p.set_parameters(params)
    assert p.parameters()[0] == pytest.approx(params[0])
    assert p.parameters()[1] == pytest.approx(params[1])
    assert p.parameters()[2] == pytest.approx(params[2])
    assert p.parameters()[3] == pytest.approx(params[3])
    assert p.parameters()[4] == pytest.approx(params[4])
    assert p.parameters()[5] == pytest.approx(params[5])

    b1, b2, b3, b4, b5, b6 = params
    assert p.sigma()[0] == pytest.approx(b1 ** 2)
    assert p.sigma()[1] == pytest.approx(b1 * b2)
    assert p.sigma()[2] == pytest.approx(b1 * b4)
    assert p.sigma()[3] == pytest.approx(b1 * b2)
    assert p.sigma()[4] == pytest.approx(b2 ** 2 + b3 * b3)
    assert p.sigma()[5] == pytest.approx(b2 * b4 + b3 * b5)
    assert p.sigma()[6] == pytest.approx(b1 * b4)
    assert p.sigma()[7] == pytest.approx(b2 * b4 + b3 * b5)
    assert p.sigma()[8] == pytest.approx(b4 ** 2 + b5 ** 2 + b6 ** 2)

    dSdb = [
        (2 * b1, b2, b4, b2, 0, 0, b4, 0, 0),
        (0, b1, 0, b1, 2 * b2, b4, 0, b4, 0),
        (0, 0, 0, 0, 2 * b3, b5, 0, b5, 0),
        (0, 0, b1, 0, 0, b2, b1, b2, 2 * b4),
        (0, 0, 0, 0, 0, b3, 0, b3, 2 * b5),
        (0, 0, 0, 0, 0, 0, 0, 0, 2 * b6),
    ]

    d = p.first_derivatives()
    assert len(d) == 6
    for a, b in zip(dSdb, p.first_derivatives()):
        for i in range(9):
            assert b[i] == pytest.approx(a[i])


def test_WavelengthSpreadParameterisation():

    params = (1e-3,)

    p = WavelengthSpreadParameterisation(params=params)
    assert p.num_parameters() == 1
    assert p.parameters()[0] == pytest.approx(params[0])
    params = (2e-3,)
    p.set_parameters(params)
    assert p.parameters()[0] == pytest.approx(params[0])
    assert p.sigma() == pytest.approx(params[0] ** 2)
    assert p.first_derivatives()[0] == pytest.approx(2 * params[0])


def test_Angular2MosaicityParameterisation():
    params = (1e-3, 2e-3)

    p = Angular2MosaicityParameterisation(params=params)

    assert p.is_angular() == True
    assert p.num_parameters() == 2
    assert p.parameters()[0] == pytest.approx(params[0])
    assert p.parameters()[1] == pytest.approx(params[1])

    params = (2e-3, 3e-3)
    p.set_parameters(params)
    assert p.parameters()[0] == pytest.approx(params[0])
    assert p.parameters()[1] == pytest.approx(params[1])

    b1, b2 = params
    assert p.sigma()[0] == pytest.approx(b1 ** 2)
    assert p.sigma()[1] == pytest.approx(0)
    assert p.sigma()[2] == pytest.approx(0)
    assert p.sigma()[3] == pytest.approx(0)
    assert p.sigma()[4] == pytest.approx(b1 ** 2)
    assert p.sigma()[5] == pytest.approx(0)
    assert p.sigma()[6] == pytest.approx(0)
    assert p.sigma()[7] == pytest.approx(0)
    assert p.sigma()[8] == pytest.approx(b2 ** 2)

    dSdb = [(2 * b1, 0, 0, 0, 2 * b1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 2 * b2)]

    d = p.first_derivatives()
    assert len(d) == 2
    for a, b in zip(dSdb, p.first_derivatives()):
        for i in range(9):
            assert b[i] == pytest.approx(a[i])


def test_Angular4MosaicityParameterisation():
    params = (1e-3, 2e-3, 3e-3, 4e-3)

    p = Angular4MosaicityParameterisation(params=params)

    assert p.is_angular() == True
    assert p.num_parameters() == 4
    assert p.parameters()[0] == pytest.approx(params[0])
    assert p.parameters()[1] == pytest.approx(params[1])
    assert p.parameters()[2] == pytest.approx(params[2])
    assert p.parameters()[3] == pytest.approx(params[3])

    params = (2e-3, 3e-3, 4e-3, 5e-3)
    p.set_parameters(params)
    assert p.parameters()[0] == pytest.approx(params[0])
    assert p.parameters()[1] == pytest.approx(params[1])
    assert p.parameters()[2] == pytest.approx(params[2])
    assert p.parameters()[3] == pytest.approx(params[3])

    b1, b2, b3, b4 = params
    assert p.sigma()[0] == pytest.approx(b1 ** 2)
    assert p.sigma()[1] == pytest.approx(b1 * b2)
    assert p.sigma()[2] == pytest.approx(0)
    assert p.sigma()[3] == pytest.approx(b1 * b2)
    assert p.sigma()[4] == pytest.approx(b2 ** 2 + b3 * b3)
    assert p.sigma()[5] == pytest.approx(0)
    assert p.sigma()[6] == pytest.approx(0)
    assert p.sigma()[7] == pytest.approx(0)
    assert p.sigma()[8] == pytest.approx(b4 ** 2)

    dSdb = [
        (2 * b1, b2, 0, b2, 0, 0, 0, 0, 0),
        (0, b1, 0, b1, 2 * b2, 0, 0, 0, 0),
        (0, 0, 0, 0, 2 * b3, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 2 * b4),
    ]

    d = p.first_derivatives()
    assert len(d) == 4
    for a, b in zip(dSdb, p.first_derivatives()):
        for i in range(9):
            assert b[i] == pytest.approx(a[i])


def check_model_state_with_fixed(
    experiment,
    mosaicity_parameterisation,
    wavelength_parameterisation,
    fix_mosaic_spread=False,
    fix_wavelength_spread=False,
    fix_unit_cell=False,
    fix_orientation=False,
):

    state = ModelState(
        experiment,
        mosaicity_parameterisation,
        wavelength_parameterisation,
        fix_mosaic_spread=fix_mosaic_spread,
        fix_wavelength_spread=fix_wavelength_spread,
        fix_unit_cell=fix_unit_cell,
        fix_orientation=fix_orientation,
    )

    assert state.is_orientation_fixed() == fix_orientation
    assert state.is_unit_cell_fixed() == fix_unit_cell
    assert state.is_mosaic_spread_fixed() == fix_mosaic_spread
    assert state.is_wavelength_spread_fixed() == fix_wavelength_spread

    U = state.get_U()
    B = state.get_B()
    A = state.get_A()
    M = state.get_M()
    L = state.get_L()

    assert len(U) == 9
    assert len(B) == 9
    assert len(A) == 9
    assert len(M) == 9
    if wavelength_parameterisation is not None:
        assert len(L) == 1
    else:
        assert len(L) == 0

    U_params = state.get_U_params()
    B_params = state.get_B_params()
    M_params = state.get_M_params()
    L_params = state.get_L_params()

    assert len(U_params) == 3
    assert len(B_params) == 2
    assert len(M_params) == mosaicity_parameterisation.num_parameters()
    if wavelength_parameterisation is not None:
        assert len(L_params) == 1
    else:
        assert len(L_params) == 0

    dU = state.get_dU_dp()
    dB = state.get_dB_dp()
    dM = state.get_dM_dp()
    dL = state.get_dL_dp()

    assert len(dU) == 3
    assert len(dB) == 2
    assert len(dM) == mosaicity_parameterisation.num_parameters()
    if wavelength_parameterisation is not None:
        assert len(dL) == 1
    else:
        assert len(dL) == 0

    params = state.get_active_parameters()

    expected_len = 0
    if not fix_mosaic_spread:
        expected_len += mosaicity_parameterisation.num_parameters()
    if not fix_wavelength_spread:
        if wavelength_parameterisation is not None:
            expected_len += wavelength_parameterisation.num_parameters()
    if not fix_unit_cell:
        expected_len += 2
    if not fix_orientation:
        expected_len += 3

    assert len(params) == expected_len
    new_params = params
    state.set_active_parameters(new_params)


def test_ModelState(dials_regression):

    experiments = ExperimentListFactory.from_json_file(
        os.path.join(dials_regression, "potato_test_data", "experiments.json")
    )
    experiments[0].scan.set_oscillation((0, 0.01), deg=True)

    S1 = Simple1MosaicityParameterisation()
    S6 = Simple6MosaicityParameterisation()
    W = WavelengthSpreadParameterisation()

    with pytest.raises(AssertionError):
        check_model_state_with_fixed(experiments[0], S1, None, fix_mosaic_spread=True)
    with pytest.raises(AssertionError):
        check_model_state_with_fixed(experiments[0], S1, None, fix_unit_cell=True)
    with pytest.raises(AssertionError):
        check_model_state_with_fixed(experiments[0], S1, None, fix_orientation=True)
    check_model_state_with_fixed(experiments[0], S1, None, fix_wavelength_spread=True)
    check_model_state_with_fixed(experiments[0], S1, W, fix_mosaic_spread=True)
    check_model_state_with_fixed(experiments[0], S1, W, fix_wavelength_spread=True)
    check_model_state_with_fixed(experiments[0], S1, W, fix_unit_cell=True)
    check_model_state_with_fixed(experiments[0], S1, W, fix_orientation=True)

    with pytest.raises(AssertionError):
        check_model_state_with_fixed(experiments[0], S6, None, fix_mosaic_spread=True)
    with pytest.raises(AssertionError):
        check_model_state_with_fixed(experiments[0], S6, None, fix_unit_cell=True)
    with pytest.raises(AssertionError):
        check_model_state_with_fixed(experiments[0], S6, None, fix_orientation=True)
    check_model_state_with_fixed(experiments[0], S6, None, fix_wavelength_spread=True)
    check_model_state_with_fixed(experiments[0], S6, W, fix_mosaic_spread=True)
    check_model_state_with_fixed(experiments[0], S6, W, fix_wavelength_spread=True)
    check_model_state_with_fixed(experiments[0], S6, W, fix_unit_cell=True)
    check_model_state_with_fixed(experiments[0], S6, W, fix_orientation=True)


def check_reflection_model_state_with_fixed(
    experiment,
    mosaicity_parameterisation,
    wavelength_parameterisation,
    fix_mosaic_spread=False,
    fix_wavelength_spread=False,
    fix_unit_cell=False,
    fix_orientation=False,
):

    state = ModelState(
        experiment,
        mosaicity_parameterisation,
        wavelength_parameterisation,
        fix_mosaic_spread=fix_mosaic_spread,
        fix_wavelength_spread=fix_wavelength_spread,
        fix_unit_cell=fix_unit_cell,
        fix_orientation=fix_orientation,
    )

    model = ReflectionModelState(
        state, matrix.col(experiment.beam.get_s0()), matrix.col((1, 1, 1))
    )

    assert model.get_sigma() == mosaicity_parameterisation.sigma()
    assert model.get_r() == state.get_A() * matrix.col((1, 1, 1))

    if wavelength_parameterisation is not None:
        assert model.get_sigma_lambda() == wavelength_parameterisation.sigma()
    else:
        assert model.get_sigma_lambda() == 0

    dS_dp = model.get_dS_dp()
    dr_dp = model.get_dr_dp()
    dL_dp = model.get_dL_dp()

    assert len(dS_dp) == len(state.get_labels())
    assert len(dr_dp) == len(state.get_labels())
    assert len(dL_dp) == len(state.get_labels())

    if not fix_wavelength_spread:
        assert dr_dp[-1] == (0, 0, 0)
        assert dS_dp[-1] == (0, 0, 0, 0, 0, 0, 0, 0, 0)
        dr_dp = dr_dp[:-1]
        dS_dp = dS_dp[:-1]
        dL_dp = dL_dp[:-1]
    if not fix_mosaic_spread:
        num_params = mosaicity_parameterisation.num_parameters()
        for i in range(num_params):
            assert dr_dp[-(i + 1)] == (0, 0, 0)
            assert dS_dp[-(i + 1)] == state.get_dM_dp()[-(i + 1)]
            assert dL_dp[-1] == 0
        dr_dp = dr_dp[:-num_params]
        dS_dp = dS_dp[:-num_params]
        dL_dp = dL_dp[:-num_params]
    if not fix_orientation:
        num_params = state.num_U_params()
        for i in range(num_params):
            assert dS_dp[-(i + 1)] == (0, 0, 0, 0, 0, 0, 0, 0, 0)
            assert dL_dp[-(i + 1)] == 0
        dr_dp = dr_dp[:-num_params]
        dS_dp = dS_dp[:-num_params]
        dL_dp = dL_dp[:-num_params]
    if not fix_unit_cell:
        num_params = state.num_B_params()
        for i in range(num_params):
            assert dS_dp[-(i + 1)] == (0, 0, 0, 0, 0, 0, 0, 0, 0)
            assert dL_dp[-(i + 1)] == 0
        dr_dp = dr_dp[:-num_params]
        dS_dp = dS_dp[:-num_params]
        dL_dp = dL_dp[:-num_params]


def test_ReflectionModelState(dials_regression):

    experiments = ExperimentListFactory.from_json_file(
        os.path.join(dials_regression, "potato_test_data", "experiments.json")
    )
    experiments[0].scan.set_oscillation((0, 0.01), deg=True)

    S1 = Simple1MosaicityParameterisation()
    S6 = Simple6MosaicityParameterisation()
    W = WavelengthSpreadParameterisation()

    check_reflection_model_state_with_fixed(
        experiments[0], S1, None, fix_wavelength_spread=True
    )
    check_reflection_model_state_with_fixed(
        experiments[0], S1, W, fix_mosaic_spread=True
    )
    check_reflection_model_state_with_fixed(
        experiments[0], S1, W, fix_wavelength_spread=True
    )
    check_reflection_model_state_with_fixed(experiments[0], S1, W, fix_unit_cell=True)
    check_reflection_model_state_with_fixed(experiments[0], S1, W, fix_orientation=True)

    check_reflection_model_state_with_fixed(
        experiments[0], S6, None, fix_wavelength_spread=True
    )
    check_reflection_model_state_with_fixed(
        experiments[0], S6, W, fix_mosaic_spread=True
    )
    check_reflection_model_state_with_fixed(
        experiments[0], S6, W, fix_wavelength_spread=True
    )
    check_reflection_model_state_with_fixed(experiments[0], S6, W, fix_unit_cell=True)
    check_reflection_model_state_with_fixed(experiments[0], S6, W, fix_orientation=True)
