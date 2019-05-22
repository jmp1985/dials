from __future__ import division
from __future__ import print_function
import numpy.random
from scitbx import matrix
from math import sqrt, pi, sin, cos, log, exp, ceil
from dials_scratch.jmp.potato.util.simplex import SimpleSimplex
from dials_scratch.jmp.potato.util.generate_simple import generate_from_reflections2
from dials_scratch.jmp.potato.util.generate_simple import (
    generate_from_reflections_binned,
)
from dials_scratch.jmp.potato.model import compute_change_of_basis_operation
from dials_scratch.jmp.potato.refiner import Refiner
from dials_scratch.jmp.potato.refiner import RefinerData
from dials_scratch.jmp.potato.parameterisation import ModelState
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex
from numpy.random import choice as sample


import logging
from dials.util import log

logger = logging.getLogger("dials.test")
log.config(info=None, debug=None)


def generate_observations2(experiments, reflections, sigma):

    A = matrix.sqr(experiments[0].crystal.get_A())
    s0 = matrix.col(experiments[0].beam.get_s0())

    s2_obs = flex.vec3_double()
    for i in range(len(reflections)):

        h = matrix.col(reflections[i]["miller_index"])

        r = A * h
        s2 = s0 + r

        s2_obs.append(s2)

    reflections["s2"] = s2_obs
    return reflections


def tst_ideal():

    numpy.random.seed(100)

    # Ensure we have a data block
    experiments = ExperimentListFactory.from_json_file("experiments.json")
    experiments[0].scan.set_oscillation((0, 1.0), deg=True)
    experiments[0].beam.set_s0((0, 0, -1))

    s0 = matrix.col(experiments[0].beam.get_s0())

    # The predicted reflections
    reflections = flex.reflection_table.from_predictions_multi(experiments, padding=4)
    print(len(reflections))

    sigma = matrix.sqr((1e-6, 0, 0, 0, 2e-6, 0, 0, 0, 3e-6))

    reflections = generate_observations2(experiments, reflections, sigma)

    sp_list, h_list, ctot_list, xbar_list, Sobs_list = generate_from_reflections2(
        experiments[0].crystal.get_A(), s0, sigma, reflections
    )

    index = sample(range(len(sp_list)), 200)

    def select_sample(d, index):
        return [d[i] for i in index]

    sp_list = select_sample(sp_list, index)
    h_list = select_sample(h_list, index)
    ctot_list = select_sample(ctot_list, index)
    xbar_list = select_sample(xbar_list, index)
    Sobs_list = select_sample(Sobs_list, index)

    print("Using %d reflections: " % len(sp_list))

    values = flex.double((sqrt(1.1e-6), 0, sqrt(2.1e-6), 0, 0, sqrt(3.1e-6)))
    offset = flex.double([sqrt(1e-7) for v in values])

    # parameterisation = SimpleMosaicityParameterisation((1,0,1,0,0,1))
    state = ModelState(
        experiments[0],
        fix_orientation=False,
        fix_unit_cell=False,
        fix_rlp_mosaicity=False,
        fix_wavelength_spread=True,
        fix_angular_mosaicity=True,
    )
    state.set_M_params(flex.double((1, 0, 1, 0, 0, 1)))
    # state.set_L_params(flex.double((1,)))
    # state.set_W_params(flex.double((1,0,1)))
    Sobs_list = flex.double(Sobs_list)
    data = RefinerData(s0, sp_list, h_list, ctot_list, xbar_list, Sobs_list)
    refiner = Refiner(state, data)
    ml = refiner.refine()
    params = refiner.parameters

    M = matrix.sqr(
        (params[0], 0, 0, params[1], params[2], 0, params[3], params[4], params[5])
    )

    sigma = M * M.transpose()
    print(sigma)

    expected = matrix.sqr(
        (
            1.0030467686e-06,
            -1.98473936999e-09,
            -8.60673302905e-10,
            -1.98473936999e-09,
            2.00630994244e-06,
            -1.64963854836e-08,
            -8.60673302905e-10,
            -1.64963854836e-08,
            2.97450815302e-06,
        )
    )

    assert all(1e6 * abs(a - b) < 1e-7 for a, b in zip(sigma, expected))

    print("OK")


def tst_binned():

    numpy.random.seed(100)

    # Ensure we have a data block
    experiments = ExperimentListFactory.from_json_file("experiments.json")
    experiments[0].scan.set_oscillation((0, 1.0), deg=True)
    experiments[0].beam.set_s0((0, 0, -1))

    s0 = matrix.col(experiments[0].beam.get_s0())

    # The predicted reflections
    reflections = flex.reflection_table.from_predictions_multi(experiments, padding=4)
    print(len(reflections))

    sigma = matrix.sqr((1e-6, 0, 0, 0, 2e-6, 0, 0, 0, 3e-6))

    reflections = generate_observations2(experiments, reflections, sigma)

    s2_list, ctot_list, xbar_list, Sobs_list = generate_from_reflections_binned(
        s0, sigma, reflections
    )

    index = sample(range(len(s2_list)), 200)

    def select_sample(d, index):
        return [d[i] for i in index]

    s2_list = select_sample(s2_list, index)
    ctot_list = select_sample(ctot_list, index)
    xbar_list = select_sample(xbar_list, index)
    Sobs_list = select_sample(Sobs_list, index)

    print("Using %d reflections: " % len(s2_list))

    values = flex.double((sqrt(1e-6), 0, sqrt(2e-6), 0, 0, sqrt(3e-6)))
    offset = flex.double([sqrt(1e-7) for v in values])

    parameterisation = SimpleMosaicityParameterisation((1, 0, 1, 0, 0, 1))
    Sobs_list = flex.double(Sobs_list)
    data = ProfileRefinerData(s0, s2_list, ctot_list, xbar_list, Sobs_list)
    refiner = ProfileRefiner(parameterisation, data)
    ml = refiner.refine()
    params = refiner.parameters

    M = matrix.sqr(
        (params[0], 0, 0, params[1], params[2], 0, params[3], params[4], params[5])
    )

    sigma = M * M.transpose()
    print(sigma)

    expected = matrix.sqr(
        (
            1.07025484551e-06,
            1.30518861783e-09,
            -1.72635922351e-09,
            1.30518861783e-09,
            2.10252906788e-06,
            -1.64646310672e-08,
            -1.72635922351e-09,
            -1.64646310672e-08,
            3.12149393966e-06,
        )
    )
    assert all(1e6 * abs(a - b) < 1e-7 for a, b in zip(sigma, expected))

    print("OK")


if __name__ == "__main__":
    tst_ideal()
    # tst_binned()
