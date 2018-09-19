""" model_design.py


"""

# Authorship ----------------------------------------------------------------------------------------------------------#
__author__      = "Geoffrey Hyde Garrett"
__copyright__   = None
__credits__     = None
__license__     = "MIT"
__version__     = "1.0.0"
__maintainer__  = "Geoffrey Hyde Garrett"
__email__       = "g.h.garrett13@gmail.com"
__status__      = "Pre-alpha"

# Imports -------------------------------------------------------------------------------------------------------------#
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify


class NonLinearDesignModel(object):
    def __init__(self, basis_functions):
        """
        TODO: (*) Complete non-linear design model.
        :param basis_functions: (list)
        """
        self._basis_functions = basis_functions
        raise NotImplementedError("TODO: Complete non-linear design model.")


class LinearDesignModel(object):
    def __init__(self, basis_functions):
        """
        :param basis_functions: (list)
        """
        self._basis_functions = basis_functions

    @property
    def basis_functions(self):
        return self._basis_functions

    def __add__(self, other):
        return LinearDesignModel(self.basis_functions + other.basis_functions)

    def __radd__(self, other):
        return LinearDesignModel(other.basis_functions + self.basis_functions)

    def __repr__(self):
        vec_param = '  '.join(['B' + str(i) for i in range(len(self._basis_functions))])
        vec_basis = '  '.join(self._basis_functions)
        return 'DesignModel(y = [{}]^T [{}])'.format(vec_param, vec_basis)

    def __str__(self):
        return 'y = ' + ' + '.join(['B_{}'.format(i) + '*' + j for
                                    i, j in enumerate(self._basis_functions)]) + ''

    def __latex__(self):
        return '$' + self.__str__().replace('*', '\cdot{}').replace('pi', '\pi').replace('B', '\\beta') + '$'

    def information_matrix(self, x):
        information_matrix = np.array([])
        for basis in self.basis_functions:
            if len(information_matrix) == 0:
                if (basis == '1') or (basis == 'x^0'):
                    information_matrix = np.full((len(x), 1), 1)
                else:
                    information_matrix = lambdify(sp.Symbol('x'), basis, 'numpy')(x)
            else:
                information_matrix = np.c_[information_matrix, lambdify(sp.Symbol('x'), basis, 'numpy')(x)]
        return information_matrix

