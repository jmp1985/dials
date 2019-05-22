from __future__ import division
import scitbx.linalg
from dials_scratch.jmp.potato.parameterisation import Simple1MosaicityParameterisation
from dials_scratch.jmp.potato.parameterisation import Simple6MosaicityParameterisation
from dials_scratch.jmp.potato.parameterisation import Angular2MosaicityParameterisation
from dials_scratch.jmp.potato.parameterisation import Angular4MosaicityParameterisation
from scitbx.linalg import eigensystem, l_l_transpose_cholesky_decomposition_in_place
from dials_scratch.jmp.potato import PredictorAngular
from dials_scratch.jmp.potato import PredictorSimple
from dials_scratch.jmp.potato import BBoxCalculatorAngular
from dials_scratch.jmp.potato import BBoxCalculatorSimple
from dials_scratch.jmp.potato import MaskCalculatorAngular
from dials_scratch.jmp.potato import MaskCalculatorSimple
from dials.array_family import flex
from scitbx import matrix
from math import exp, sqrt


class ProfileModelBase(object):
    """
  Class to store profile model

  """

    def __init__(self, params):
        """
    Initialise the class

    """
        self.params = params

    def sigma(self):
        """
    Get the sigma

    """
        return self.parameterisation().sigma()

    def update_model_state_parameters(self, state):
        """
    Update the model state with the parameters

    """
        state.set_M_params(self.params)

    def update_model(self, state):
        """
    Update the model

    """

        # Compute the eigen decomposition of the covariance matrix and check
        # largest eigen value
        eigen_decomposition = eigensystem.real_symmetric(
            state.get_M().as_flex_double_matrix()
        )
        L = eigen_decomposition.values()
        if L[0] > 1e-5:
            raise RuntimeError("Mosaicity matrix is unphysically large")

        self.params = state.get_M_params()


class SimpleProfileModelBase(ProfileModelBase):
    """
  Base class for simple profile models

  """

    def predict_reflections(self, experiments, miller_indices, probability=0.9973):
        """
    Predict the reflections

    """
        predictor = PredictorSimple(experiments[0], self.sigma(), probability)
        return predictor.predict(miller_indices)

    def compute_bbox(self, experiments, reflections, probability=0.9973):
        """
    Compute the bounding box

    """
        calculator = BBoxCalculatorSimple(experiments[0], self.sigma(), probability, 4)
        calculator.compute(reflections)

    def compute_mask(self, experiments, reflections, probability=0.9973):
        """
    Compute the mask

    """
        calculator = MaskCalculatorSimple(experiments[0], self.sigma(), probability)
        calculator.compute(reflections)

    def sigma_for_reflection(self, s0, r):
        """
    Get sigma for a reflections

    """
        return self.sigma()

    def compute_partiality(self, experiments, reflections):
        """
    Compute the partiality

    """
        s0 = matrix.col(experiments[0].beam.get_s0())
        num = reflections.get_flags(reflections.flags.indexed).count(True)

        # Compute the marginal variance for the 000 reflection
        S00 = experiments[0].crystal.mosaicity.sigma()[8]

        partiality = flex.double(len(reflections))
        partiality_variance = flex.double(len(reflections))
        for k in range(len(reflections)):
            s1 = matrix.col(reflections[k]["s1"])
            s2 = matrix.col(reflections[k]["s2"])
            sbox = reflections[k]["shoebox"]

            r = s2 - s0
            sigma = experiments[0].crystal.mosaicity.sigma()
            R = compute_change_of_basis_operation(s0, s2)
            S = R * (sigma) * R.transpose()
            mu = R * s2
            assert abs(1 - mu.normalize().dot(matrix.col((0, 0, 1)))) < 1e-7

            S11 = matrix.sqr((S[0], S[1], S[3], S[4]))
            S12 = matrix.col((S[2], S[5]))
            S21 = matrix.col((S[6], S[7])).transpose()
            S22 = S[8]

            mu1 = matrix.col((mu[0], mu[1]))
            mu2 = mu[2]
            eps = s0.length() - mu2
            var_eps = S22 / num  # FIXME Approximation

            partiality[k] = exp(-0.5 * eps * (1 / S22) * eps) * sqrt(S00 / S22)
            partiality_variance[k] = (
                var_eps * (eps ** 2 / (S00 * S22)) * exp(eps ** 2 / S22)
            )

        reflections["partiality"] = partiality
        reflections["partiality.inv.variance"] = partiality_variance

    @classmethod
    def from_params(Class, params):
        """
    Create the class from some parameters

    """
        return Class(params)


class Simple1ProfileModel(SimpleProfileModelBase):
    """
  Simple 1 profile model class

  """

    def parameterisation(self):
        """
    Get the parameterisation

    """
        return Simple1MosaicityParameterisation(self.params)

    @classmethod
    def from_sigma_d(Class, sigma_d):
        """
    Create the profile model from sigma_d estimate

    """
        return Class.from_params(flex.double((sigma_d,)))

    @classmethod
    def from_sigma(Class, sigma):
        """
    Construct the profile model from the sigma

    """

        # Construct triangular matrix
        LL = flex.double()
        for j in range(3):
            for i in range(j + 1):
                LL.append(sigma[j * 3 + i])

        # Do the cholesky decomposition
        ll = l_l_transpose_cholesky_decomposition_in_place(LL)

        assert abs(LL[1] - 0) < TINY
        assert abs(LL[2] - LL[0]) < TINY
        assert abs(LL[3] - 0) < TINY
        assert abs(LL[4] - 0) < TINY
        assert abs(LL[5] - LL[0]) < TINY

        # Setup the parameters
        return Class.from_params(flex.double((LL[0],)))


class Simple6ProfileModel(SimpleProfileModelBase):
    """
  Class to store profile model

  """

    def parameterisation(self):
        """
    Get the parameterisation

    """
        return Simple6MosaicityParameterisation(self.params)

    @classmethod
    def from_sigma_d(Class, sigma_d):
        """
    Create the profile model from sigma_d estimate

    """
        return Class.from_params(flex.double((sigma_d, 0, sigma_d, 0, 0, sigma_d)))

    @classmethod
    def from_sigma(Class, sigma):
        """
    Construct the profile model from the sigma

    """

        # Construct triangular matrix
        LL = flex.double()
        for j in range(3):
            for i in range(j + 1):
                LL.append(sigma[j * 3 + i])

        # Do the cholesky decomposition
        ll = l_l_transpose_cholesky_decomposition_in_place(LL)

        # Setup the parameters
        return Class.from_params(
            flex.double((LL[0], LL[1], LL[2], LL[3], LL[4], LL[5]))
        )


class AngularProfileModelBase(ProfileModelBase):
    """
  Class to store profile model

  """

    def sigma_for_reflection(self, s0, r):
        """
    Sigma for a reflection

    """
        Q = compute_change_of_basis_operation(s0, r)
        return Q.transpose() * self.sigma() * Q

    def predict_reflections(self, experiments, miller_indices, probability=0.9973):
        """
    Predict the reflections

    """
        predictor = PredictorAngular(experiments[0], self.sigma(), probability)
        return predictor.predict(miller_indices)

    def compute_bbox(self, experiments, reflections, probability=0.9973):
        """
    Compute the bounding box

    """
        calculator = BBoxCalculatorAngular(experiments[0], self.sigma(), probability, 4)
        calculator.compute(reflections)

    def compute_mask(self, experiments, reflections, probability=0.9973):
        """
    Compute the mask

    """
        calculator = MaskCalculatorAngular(experiments[0], self.sigma(), probability)
        calculator.compute(reflections)

    def compute_partiality(self, experiments, reflections):
        """
    Compute the partiality

    """
        s0 = matrix.col(experiments[0].beam.get_s0())
        num = reflections.get_flags(reflections.flags.indexed).count(True)
        partiality = flex.double(len(reflections))
        partiality_variance = flex.double(len(reflections))
        for k in range(len(reflections)):
            s1 = matrix.col(reflections[k]["s1"])
            s2 = matrix.col(reflections[k]["s2"])
            sbox = reflections[k]["shoebox"]

            r = s2 - s0
            sigma = experiments[0].crystal.mosaicity.sigma()
            R = compute_change_of_basis_operation(s0, s2)
            Q = compute_change_of_basis_operation(s0, r)
            S = R * (Q.transpose() * sigma * Q) * R.transpose()
            mu = R * s2
            assert abs(1 - mu.normalize().dot(matrix.col((0, 0, 1)))) < 1e-7

            S11 = matrix.sqr((S[0], S[1], S[3], S[4]))
            S12 = matrix.col((S[2], S[5]))
            S21 = matrix.col((S[6], S[7])).transpose()
            S22 = S[8]

            mu1 = matrix.col((mu[0], mu[1]))
            mu2 = mu[2]
            eps = s0.length() - mu2
            var_eps = S22 / num  # FIXME Approximation

            S00 = S22  # FIXME
            partiality[k] = exp(-0.5 * eps * (1 / S22) * eps) * sqrt(S00 / S22)
            partiality_variance[k] = (
                var_eps * (eps ** 2 / (S00 * S22)) * exp(eps ** 2 / S22)
            )

        reflections["partiality"] = partiality
        reflections["partiality.inv.variance"] = partiality_variance

    @classmethod
    def from_params(Class, params):
        """
    Create the class from some parameters

    """
        return Class(params)


class Angular2ProfileModel(AngularProfileModelBase):
    """
  Class to store profile model

  """

    def parameterisation(self):
        """
    Get the parameterisation

    """
        return Angular2MosaicityParameterisation(self.params)

    @classmethod
    def from_sigma_d(Class, sigma_d):
        """
    Create the profile model from sigma_d estimate

    """
        return Class.from_params(flex.double((sigma_d, sigma_d)))

    @classmethod
    def from_sigma(Class, sigma):
        """
    Construct the profile model from the sigma

    """

        # Construct triangular matrix
        LL = flex.double()
        for j in range(3):
            for i in range(j + 1):
                LL.append(sigma[j * 3 + i])

        # Do the cholesky decomposition
        ll = l_l_transpose_cholesky_decomposition_in_place(LL)

        # Check the sigma is as we expect
        TINY = 1e-10
        assert abs(LL[1] - 0) < TINY
        assert abs(LL[2] - LL[0]) < TINY
        assert abs(LL[3] - 0) < TINY
        assert abs(LL[4] - 0) < TINY

        # Setup the parameters
        return Class.from_params(flex.double((LL[0], LL[5])))


class Angular4ProfileModel(AngularProfileModelBase):
    """
  Class to store profile model

  """

    def parameterisation(self):
        """
    Get the parameterisation

    """
        return Angular4MosaicityParameterisation(self.params)

    @classmethod
    def from_sigma_d(Class, sigma_d):
        """
    Create the profile model from sigma_d estimate

    """
        return Class.from_params(flex.double((sigma_d, 0, sigma_d, sigma_d)))

    @classmethod
    def from_sigma(Class, sigma):
        """
    Construct the profile model from the sigma

    """

        # Construct triangular matrix
        LL = flex.double()
        for j in range(3):
            for i in range(j + 1):
                LL.append(sigma[j * 3 + i])

        # Do the cholesky decomposition
        ll = l_l_transpose_cholesky_decomposition_in_place(LL)

        # Check the sigma is as we expect
        TINY = 1e-10
        assert abs(LL[3] - 0) < TINY
        assert abs(LL[4] - 0) < TINY

        # Setup the parameters
        return Class.from_params(flex.double((LL[0], LL[1], LL[2], LL[5])))


class ProfileModelFactory(object):
    """
  Class to create profile models

  """

    @classmethod
    def from_sigma_d(Class, params, sigma_d):
        """
    Construct a profile model from an initial sigma estimate

    """
        if params.profile.rlp_mosaicity.model == "simple1":
            return Simple1ProfileModel.from_sigma_d(sigma_d)
        elif params.profile.rlp_mosaicity.model == "simple6":
            return Simple6ProfileModel.from_sigma_d(sigma_d)
        elif params.profile.rlp_mosaicity.model == "angular2":
            return Angular2ProfileModel.from_sigma_d(sigma_d)
        elif params.profile.rlp_mosaicity.model == "angular4":
            return Angular4ProfileModel.from_sigma_d(sigma_d)

        raise RuntimeError(
            "Unknown profile model: %s" % params.profile.rlp_mosaicity.model
        )


def compute_change_of_basis_operation(s0, s2):
    """
  Compute the change of basis operation that puts the s2 vector along the z axis

  """
    e1 = s2.cross(s0).normalize()
    e2 = s2.cross(e1).normalize()
    e3 = s2.normalize()
    R = matrix.sqr(e1.elems + e2.elems + e3.elems)
    return R


def compute_change_of_basis_operation2(s0, r):
    """
  Compute the change of basis operation that puts the r vector along the z axis

  """
    e1 = r.cross(s0).normalize()
    e2 = r.cross(e1).normalize()
    e3 = r.normalize()
    R = matrix.sqr(e1.elems + e2.elems + e3.elems)
    return R
