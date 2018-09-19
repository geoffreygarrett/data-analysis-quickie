""" data_analysis.py
This script provides data analysis tools such as least-squares-solution for linear design models. Functionality will be
extended to non-linear design models
"""

# Authorship -----------------------------------------------------------------------------#
__author__      = "Geoffrey Hyde Garrett"
__copyright__   = None
__credits__     = None
__license__     = "MIT"
__version__     = "1.0.0"
__maintainer__  = "Geoffrey Hyde Garrett"
__email__       = "g.h.garrett13@gmail.com"
__status__      = "Pre-alpha"

# Imports  -------------------------------------------------------------------------------#
from datpy.helper_functions import kth_order_moment_about_mean
from datpy.helper_functions import unweighted_least_squares
from datpy.helper_functions import weighted_least_squares
from datpy.model_design import LinearDesignModel
import matplotlib.pyplot as plt
import numpy as np
import statistics
from matplotlib.pyplot import figure
import matplotlib.mlab as mlab


class DataAnalysis2D(object):

    @staticmethod
    def skewness(var, bins=None):
        """
        TODO: Complete skewness doc-strings.
        :param var:
        :param bins:
        :return:
        """
        return kth_order_moment_about_mean(var, 3, bins=bins) / np.power(np.std(var), 3)

    @staticmethod
    def kurtosis_old(var, bins=None):
        """
        TODO: Complete kurtosis_old doc-strings.
        :param var:
        :param bins:
        :return:
        """
        return kth_order_moment_about_mean(var, 4, bins=bins) / np.power(np.std(var), 4)

    @staticmethod
    def kurtosis_new(var, bins=None):
        """
        TODO: Complete kurtosis_new doc-strings.
        :param var:
        :param bins:
        :return:
        """
        return kth_order_moment_about_mean(var, 4, bins=bins) / np.power(np.std(var), 4) - 3

    def __init__(self, design_model: LinearDesignModel, x: np.ndarray, y: np.ndarray, Pyy: np.ndarray = None):
        """
        :param design_model: Design model to be analysed (LinearDesignModel/NonLinearDesignModel)
        :param x: Independent variable associated with observations (np.ndarray)
        :param y: Vector of observations (np.ndarray)
        """
        self._x = x
        self._y = y
        self._Pyy = Pyy
        self._design_model = design_model

    @property
    def Pyy(self):
        try:
            return self._Py
        except AttributeError:
            raise AttributeError("Matrix Py has not been provided, please set using Py setter.")

    @Pyy.setter
    def Pyy(self, arg):
        self._Pyy = arg

    def unweighted_least_squares(self):
        """
        :return: Unweighted least squares solution of the vector of parameters (np.ndarray)
        """
        return unweighted_least_squares(self._design_model.information_matrix(self._x), self._y)

    def unweighted_prediction(self, x):
        """
        :param x: Independent variable associated with observations (np.ndarray)
        :return: Unweighted predictions using the design model (np.ndarray)
        """
        return np.matmul(self._design_model.information_matrix(x), self.unweighted_least_squares())

    def unweighted_residuals(self):
        """
        :return: Unweighted residuals between observations and unweighted prediction (np.ndarray)
        """
        return self._y - self.unweighted_prediction(self._x)

    def weighted_least_squares(self):
        """
        :return: Unweighted least squares solution of the vector of parameters (np.ndarray)
        """
        return weighted_least_squares(self._design_model.information_matrix(self._x), self._y, Pyy=self.Pyy)

    def weighted_prediction(self, x):
        """
        :param x: Independent variable associated with observations (np.ndarray)
        :return: Weighted predictions using the design model (np.ndarray)
        """
        return np.matmul(self.weighted_least_squares(), self._design_model.information_matrix(x))

    def weighted_residuals(self):
        """
        :return: Weighted residuals between observations and weighted prediction (np.ndarray)
        """
        return self._y - self.weighted_prediction(self._x)

    def plot_model(self, plot_type='matplotlib', show_save=('show'), title=None, ylabel=None, xlabel=None, name=None,
                   type='unweighted', legend=True):
        """
        TODO: (*) Finish doc-strings for plot_model.
        TODO: (*)
        :param plot_type:
        :param show_save:
        :param title:
        :param ylabel:
        :param xlabel:
        :param name:
        :param type:
        :return:
        """
        if plot_type is 'matplotlib':
            figure(num=None, figsize=(8, 5), dpi=300, facecolor='w', edgecolor='k')
            plt.plot(self._x, self._y, label='Raw data', linewidth=1)
            if title:
                plt.title(title)
            plt.grid()
            smoothed_x = np.linspace(self._x[0], self._x[-1], 1000)
            if type is 'weighted':
                predicted_y = np.matmul(self._design_model.information_matrix(smoothed_x),
                                        self.weighted_least_squares())
            elif type is 'unweighted':
                predicted_y = np.matmul(self._design_model.information_matrix(smoothed_x),
                                        self.unweighted_least_squares())
            else:
                raise SystemError("{} type not recognised, please use <weighted> or <unweighted>.".format(type))
            plt.plot(
                smoothed_x, predicted_y,
                label=self._design_model.__latex__() + ' ' + str(type),
                linestyle='-',
                linewidth=2,
                color='red')
            if legend:
                plt.legend()
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            if 'show' in show_save:
                plt.show()
            if 'save' in show_save:
                plt.savefig(str(name) + '.png', bbox_inches='tight')

    def plot_residual_hist(self, plot_type='matplotlib', show_save=('show'), title=None, ylabel=None, xlabel=None,
                           name=None, bins=30, type='unweighted'):
        '''
        TODO: (*) Finish doc-strings for plot_residual_hist.
        :param plot_type:
        :param show_save:
        :param title:
        :param ylabel:
        :param xlabel:
        :param name:
        :param bins:
        :return:
        '''
        if type is 'unweighted':
            residuals = self.unweighted_residuals()
        elif type is 'weighted':
            residuals = self.weighted_residuals()
        else:
            raise SystemError("{} type argument not recognised. Please use weighted or unweighted".format(type))
        mu = statistics.mean(residuals)
        sigma = statistics.stdev(residuals)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        if plot_type is 'matplotlib':
            figure(num=None, figsize=(8, 5), dpi=300, facecolor='w', edgecolor='k')
            if title:
                plt.title(title)
            plt.grid()
            plt.hist(residuals,
                     bins=bins,
                     density=True,
                     label='Normalised histogram ({} bins)'.format(bins),
                     edgecolor='black',
                     linewidth=0.8)
            plt.plot(x, mlab.normpdf(x, mu, sigma),
                     label='Normal Distribution ($\sigma={0:.3g}'.format(sigma) + ',\;\mu={0:.3g}$)'.format(mu),
                     color='red', linewidth='2')

            # DEVELOPMENT FOR ANOMALIES
            # p, ed = np.histogram(residuals, bins=30, density=True)
            # mid = [np.mean([ed[i], ed[i + 1]]) for i in range(len(ed) - 1)]
            # plt.scatter(mid[0], p[0], s=400, facecolors='none', edgecolors='g', linewidths=3, label='Note A')

            plt.legend()
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            if 'show' in show_save:
                plt.show()
            if 'save' in show_save:
                plt.savefig(str(name) + '.png', bbox_inches='tight')
