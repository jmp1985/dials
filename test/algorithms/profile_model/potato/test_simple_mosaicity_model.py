from __future__ import division
from __future__ import print_function
from scitbx import matrix
from dials.algorithms.profile_model.potato.model import Simple6MosaicityModel
from dials.algorithms.profile_model.potato.model import Simple6MosaicityParameterisation


def tst_simple_mosaicity_model():


    sigma = matrix.sqr((1, 0, 0, 0, 2, 0, 0, 0, 3))

    model = Simple6MosaicityParameterisation(sigma)

    parameterisation = model.parameterisation()

    params = parameterisation.parameters()

    expected = (1.0, 0.0, 1.4142135623730951, 0.0, 0.0, 1.7320508075688772)

    assert all(abs(a - b) < 1e-7 for a, b in zip(params, expected))

    model.compose(parameterisation)

    sigma2 = model.sigma()

    assert all(abs(a - b) < 1e-7 for a, b in zip(sigma, sigma2))

