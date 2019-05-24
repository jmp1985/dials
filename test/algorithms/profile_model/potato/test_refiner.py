from __future__ import division
from __future__ import print_function
from collections import namedtuple
from os.path import join
from random import uniform, randint
import pytest
from dxtbx.model.experiment_list import ExperimentListFactory
from scitbx import matrix
from dials.array_family import flex
from dials.algorithms.refinement.parameterisation.crystal_parameters import (
    CrystalUnitCellParameterisation,
    CrystalOrientationParameterisation,
)
from dials.algorithms.profile_model.potato.parameterisation import (
    Simple1MosaicityParameterisation,
    Simple6MosaicityParameterisation,
    Angular2MosaicityParameterisation,
    Angular4MosaicityParameterisation,
    WavelengthSpreadParameterisation,
    ModelState,
    ReflectionModelState,
)
from dials.algorithms.profile_model.potato.model import (
    compute_change_of_basis_operation,
)
from dials.algorithms.profile_model.potato.refiner import (
    ConditionalDistribution,
    rotate_vec3_double,
    rotate_mat3_double,
    ReflectionLikelihood,
)

def first_derivative(func, x, h):
    return (-func(x + 2 * h) + 8 * func(x + h) - 8 * func(x - h) + func(x - 2 * h)) / (
        12 * h
    )

def generate_data(experiments, reflections):

    from random import seed

    seed(0)

    index = randint(0, len(reflections))

    h = reflections[index]["miller_index"]

    s0 = matrix.col(experiments[0].beam.get_s0())

    U_param = CrystalOrientationParameterisation(experiments[0].crystal)
    B_param = CrystalUnitCellParameterisation(experiments[0].crystal)

    U = matrix.sqr(experiments[0].crystal.get_U())
    B = matrix.sqr(experiments[0].crystal.get_B())
    r = U * B * matrix.col(h)
    s2 = s0 + r
    mobs = (
        s2 + matrix.col((uniform(0, 1e-3), uniform(0, 1e-3), uniform(0, 1e-3)))
    ).normalize() * s0.length()
    mobs = matrix.col((uniform(0, 1e-3), uniform(0, 1e-3)))
    sp = s2.normalize()*s0.length()

    b1, b2, b3, b4, b5, b6 = (
        uniform(1e-3, 3e-3),
        uniform(0.0, 1e-3),
        uniform(1e-3, 3e-3),
        uniform(0.0, 1e-3),
        uniform(0.0, 1e-3),
        uniform(1e-3, 3e-3),
    )

    S_param = (b1, b2, b3, b4, b5, b6)
    L_param = (uniform(1e-3, 2e-3),)
    ctot = randint(100, 1000)

    T = matrix.sqr((uniform(1e-3, 2e-3), 0, uniform(1e-6, 2e-6), uniform(1e-3, 2e-3)))
    Sobs = T * T.transpose()

    params = [S_param, U_param, B_param, L_param]

    return params, s0, sp, h, ctot, mobs, Sobs

@pytest.fixture
def testdata(dials_regression):

    TestData = namedtuple(
        "TestData", ["experiment", "models", "s0", "sp", "h", "ctot", "mobs", "Sobs"]
    )

    experiments = ExperimentListFactory.from_json_file(
        join(dials_regression, "potato_test_data", "experiments.json")
    )
    experiments[0].scan.set_oscillation((0, 0.01), deg=True)
    reflections = flex.reflection_table.from_predictions_multi(experiments)

    models, s0, sp, h, ctot, mobs, Sobs = generate_data(experiments, reflections)

    return TestData(
        experiment=experiments[0],
        models=models,
        s0=s0,
        sp=sp,
        h=h,
        ctot=ctot,
        mobs=mobs,
        Sobs=Sobs,
    )

def test_ConditionalDistribution(testdata):
    def check(
        mosaicity_parameterisation,
        wavelength_parameterisation,
        fix_mosaic_spread=False,
        fix_wavelength_spread=False,
        fix_unit_cell=False,
        fix_orientation=False,
    ):
        experiment = testdata.experiment
        models = testdata.models
        s0 = testdata.s0
        sp = testdata.sp
        h = testdata.h
        ctot = testdata.ctot
        mobs = testdata.mobs
        Sobs = testdata.Sobs

        U_params = models[1].get_param_vals()
        B_params = models[2].get_param_vals()
        M_params = flex.double(models[0][: mosaicity_parameterisation.num_parameters()])
        L_params = flex.double(models[3])

        state = ModelState(
            experiment,
            mosaicity_parameterisation,
            wavelength_parameterisation,
            fix_mosaic_spread=fix_mosaic_spread,
            fix_wavelength_spread=fix_wavelength_spread,
            fix_unit_cell=fix_unit_cell,
            fix_orientation=fix_orientation,
        )
        state.set_U_params(U_params)
        state.set_B_params(B_params)
        state.set_M_params(M_params)
        state.set_L_params(L_params)

        model = ReflectionModelState(state, s0, h)
       
        def get_conditional(model):
            # Compute the change of basis
            R = compute_change_of_basis_operation(s0, sp)

            # The s2 vector
            r = model.get_r()
            s2 = s0 + r

            # Rotate the mean vector
            mu = R * s2

            # Rotate the covariance matrix
            S = R * model.get_sigma() * R.transpose()

            # Rotate the first derivative matrices
            dS = rotate_mat3_double(R, model.get_dS_dp())

            # Rotate the first derivative of s2
            dmu = rotate_vec3_double(R, model.get_dr_dp())

            # Construct the conditional distribution
            conditional = ConditionalDistribution(
                s0, mu, dmu, S, dS
            )
            return conditional

        conditional = get_conditional(model)
            
        step = 1e-6

        dm_dp = conditional.first_derivatives_of_mean()
        dS_dp = conditional.first_derivatives_of_sigma()

        parameters = state.get_active_parameters()
        
        def compute_sigma(parameters):
            state.set_active_parameters(parameters)
            model = ReflectionModelState(state, s0, h)
            conditional = get_conditional(model)
            return conditional.sigma()

        def compute_mean(parameters):
            state.set_active_parameters(parameters)
            model = ReflectionModelState(state, s0, h)
            conditional = get_conditional(model)
            return conditional.mean()

        dm_num = []
        for i in range(len(parameters)):

            def f(x):
                p = [pp for pp in parameters]
                p[i] = x
                return compute_mean(p)

            dm_num.append(first_derivative(f, parameters[i], step))

        for n, c in zip(dm_num, dm_dp):
            assert all(abs(nn - cc) < 1e-7 for nn, cc in zip(n, c))

        ds_num = []
        for i in range(len(parameters)):

            def f(x):
                p = [pp for pp in parameters]
                p[i] = x
                return compute_sigma(p)

            ds_num.append(first_derivative(f, parameters[i], step))

        for n, c in zip(ds_num, dS_dp):
            assert all(abs(nn - cc) < 1e-7 for nn, cc in zip(n, c))

    S1 = Simple1MosaicityParameterisation()
    S6 = Simple6MosaicityParameterisation()

    check(S1, None, fix_wavelength_spread=True)
    check(S1, None, fix_wavelength_spread=True, fix_mosaic_spread=True)
    check(S1, None, fix_wavelength_spread=True, fix_unit_cell=True)
    check(S1, None, fix_wavelength_spread=True, fix_orientation=True)

    check(S6, None, fix_wavelength_spread=True)
    check(S6, None, fix_wavelength_spread=True, fix_mosaic_spread=True)
    check(S6, None, fix_wavelength_spread=True, fix_unit_cell=True)
    check(S6, None, fix_wavelength_spread=True, fix_orientation=True)

def test_rotate_vec3_double():

    vectors = flex.vec3_double([matrix.col((1, 1, 1)).normalize()])

    R = compute_change_of_basis_operation(matrix.col((0, 0, 1)), matrix.col(vectors[0]))

    rotated = rotate_vec3_double(R, vectors)

    assert rotated[0] == pytest.approx((0, 0, 1))


def test_rotate_mat3_double():

    A = matrix.diag((1, 1, 1))
    R = compute_change_of_basis_operation(matrix.col((0, 0, 1)), matrix.col((1, 1, 1)))
    A = R.transpose() * A * R
    matrices = flex.mat3_double([A])

    R = R.transpose()

    rotated = rotate_mat3_double(R, matrices)

    assert rotated[0] == pytest.approx((1, 0, 0, 0, 1, 0, 0, 0, 1))


def test_ReflectionLikelihood(testdata):
    
    def check(
        mosaicity_parameterisation,
        wavelength_parameterisation,
        fix_mosaic_spread=False,
        fix_wavelength_spread=False,
        fix_unit_cell=False,
        fix_orientation=False,
    ):
        experiment = testdata.experiment
        models = testdata.models
        s0 = testdata.s0
        sp = testdata.sp
        h = testdata.h
        ctot = testdata.ctot
        mobs = testdata.mobs
        Sobs = testdata.Sobs

        U_params = models[1].get_param_vals()
        B_params = models[2].get_param_vals()
        M_params = flex.double(models[0][: mosaicity_parameterisation.num_parameters()])
        L_params = flex.double(models[3])

        state = ModelState(
            experiment,
            mosaicity_parameterisation,
            wavelength_parameterisation,
            fix_mosaic_spread=fix_mosaic_spread,
            fix_wavelength_spread=fix_wavelength_spread,
            fix_unit_cell=fix_unit_cell,
            fix_orientation=fix_orientation,
        )
        state.set_U_params(U_params)
        state.set_B_params(B_params)
        state.set_M_params(M_params)
        state.set_L_params(L_params)

        def get_reflection_likelihood(state):
            return ReflectionLikelihood(
                state,
                s0,
                sp,
                h,
                ctot,
                mobs,
                Sobs)

        likelihood = get_reflection_likelihood(state)
            
        step = 1e-6

        dL_dp = likelihood.first_derivatives()

        parameters = state.get_active_parameters()
        
        assert len(dL_dp) == len(parameters)
        def compute_likelihood(parameters):
            state.set_active_parameters(parameters)
            likelihood = get_reflection_likelihood(state)
            return likelihood.log_likelihood()

        dL_num = []
        for i in range(len(parameters)):

            def f(x):
                p = [pp for pp in parameters]
                p[i] = x
                return compute_likelihood(p)

            dL_num.append(first_derivative(f, parameters[i], step))

        assert len(dL_num) == len(parameters)
        for n, c in zip(dL_num, dL_dp):
            assert n == pytest.approx(c)

    S1 = Simple1MosaicityParameterisation()
    S6 = Simple6MosaicityParameterisation()

    check(S1, None, fix_wavelength_spread=True)
    check(S1, None, fix_wavelength_spread=True, fix_mosaic_spread=True)
    check(S1, None, fix_wavelength_spread=True, fix_unit_cell=True)
    check(S1, None, fix_wavelength_spread=True, fix_orientation=True)

    check(S6, None, fix_wavelength_spread=True)
    check(S6, None, fix_wavelength_spread=True, fix_mosaic_spread=True)
    check(S6, None, fix_wavelength_spread=True, fix_unit_cell=True)
    check(S6, None, fix_wavelength_spread=True, fix_orientation=True)




def test_Refiner():
    pass


def test_RefinerData():
    pass
