from __future__ import division
from __future__ import print_function
import pytest
from scitbx import matrix
from dials.array_family import flex
from dials.algorithms.profile_model.potato.model import (
    compute_change_of_basis_operation,
)
from dials.algorithms.profile_model.potato.refiner import (
    rotate_vec3_double,
    rotate_mat3_double,
)


def test_ConditionalDistribution_mean():
    pass


def test_ConditionalDistribution_sigma():
    pass


def test_ConditionalDistribution_first_derivatives_of_mean():
    pass


def test_ConditionalDistribution_first_derivatives_of_sigma():
    pass


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


def test_ReflectionLikelihood_log_likelihood():
    pass


def test_ReflectionLikelihood_first_derivatives():
    pass


def test_ReflectionLikelihood_fisher_information():
    pass


def test_Refiner():
    pass


def test_RefinerData():
    pass
