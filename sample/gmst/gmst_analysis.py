from sample.gmst.gmst_handler import *
from datpy.model_design import LinearDesignModel
from datpy.data_analysis import DataAnalysis2D
import scipy.fftpack
from scipy.signal import argrelextrema
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GMSL_HDR = 50
GSML_COLUMNS = ['altt', 'mfc', 'year', 'n_obs', 'n_wobs', 'gmsl', 'gmsl_std', 'smoothed', 'gmsl_gia', 'gmsl_gia_std',
                'gmsl_gia_smthed_20', 'gmsl_gia_smthed_sig']
GSML_FILENAME = 'gmst_dataset/GMSL_TPJAOS_4.2_199209_201805.txt'


def txt_to_array(relative_path, skip_header):
    return np.genfromtxt(relative_path, skip_header=skip_header)


def array_to_dataframe(array, columns):
    return pd.DataFrame(array, columns=columns)


MOD = 0
SMALL_SIZE = 8 + MOD
MEDIUM_SIZE = 10 + MOD
BIGGER_SIZE = 12 + MOD

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

if __name__ == '__main__':
    data_array = txt_to_array(GSML_FILENAME, skip_header=GMSL_HDR)
    data_frame = array_to_dataframe(data_array, columns=GSML_COLUMNS)
    refined_data_frame = data_frame[['year', 'gmsl']]

    ####################################################################################################################
    # MODEL 1
    ####################################################################################################################
    # model 2 involving a trend, bias and signal
    model_1 = LinearDesignModel(basis_functions=['x^0', 'x'])

    # instantiate a data analysis model with the given data and linear model
    model_1_analysis = DataAnalysis2D(model_1, refined_data_frame['year'].values, refined_data_frame['gmsl'].values)

    # plot the trend + bias model for the data set
    model_1_analysis.plot_model(
        ylabel='GMSL variation (mm) w.r.t. 20 year mean reference',
        xlabel='Year',
        show_save=('save'),
        name='gmst_plots/model1_plot')

    # plot the residual probability density function
    model_1_analysis.plot_residual_hist(
        ylabel='Probability density f($\epsilon$)',
        xlabel='Residuals ($\epsilon$) [mm]',
        show_save=('save'),
        name='gmst_plots/model1_residuals')

    # print the least square solution
    print('Vector of parameters: ', model_1_analysis.unweighted_least_squares())

    # print the skewness of the residual distribution
    print('Skewness: ', model_1_analysis.skewness(model_1_analysis.unweighted_residuals(), 30))

    # print the kurtosis of the residual distribution
    print('Kurtosis: ', model_1_analysis.kurtosis_new(model_1_analysis.unweighted_residuals(), 30))

    ####################################################################################################################
    # MODEL 2
    ####################################################################################################################
    # model 2 involving a trend, bias and signal
    model_2 = LinearDesignModel(basis_functions=['x^0', 'x', 'cos(2*pi*x)', 'sin(2*pi*x)'])

    # instantiate a data analysis model with the given data and linear model
    model_2_analysis = DataAnalysis2D(model_2, refined_data_frame['year'].values, refined_data_frame['gmsl'].values)

    # plot the trend + bias model for the data set
    model_2_analysis.plot_model(
        ylabel='GMSL variation (mm) w.r.t. 20 year mean reference',
        xlabel='Year',
        show_save=('save'),
        name='gmst_plots/model2_plot'
    )

    # plot the residual probability density function
    model_2_analysis.plot_residual_hist(

        ylabel='Probability density f($\epsilon$)',
        xlabel='Residuals ($\epsilon$) [mm]',
        show_save=('save'),
        name='gmst_plots/model2_residuals'
    )

    # print the least square solution
    print('Vector of parameters: ', model_2_analysis.unweighted_least_squares())

    # print the skewness of the residual distribution
    print('Skewness: ', model_2_analysis.skewness(model_2_analysis.unweighted_residuals(), 30))

    # print the kurtosis of the residual distribution
    print('Kurtosis: ', model_2_analysis.kurtosis_new(model_2_analysis.unweighted_residuals(), 30))

    ####################################################################################################################
    # MODEL 3
    ####################################################################################################################
    # model 3 involving a trend, bias and signal
    model_3 = LinearDesignModel(basis_functions=['x^0', 'x', 'x^2', 'cos(2*pi*x)', 'sin(2*pi*x)'])

    # instantiate a data analysis model with the given data and linear model
    model_3_analysis = DataAnalysis2D(model_3, refined_data_frame['year'].values, refined_data_frame['gmsl'].values)

    # plot the trend + bias model for the data set
    model_3_analysis.plot_model(
        ylabel='GMSL variation (mm) w.r.t. 20 year mean reference',
        xlabel='Year',
        show_save=('save'),
        name='gmst_plots/model3_plot'
    )

    # plot the residual probability density function
    model_3_analysis.plot_residual_hist(
        ylabel='Probability density f($\epsilon$)',
        xlabel='Residuals ($\epsilon$) [mm]',
        show_save=('save'),
        name='gmst_plots/model3_residuals'
    )

    # print the least square solution
    print('Vector of parameters: ', model_3_analysis.unweighted_least_squares())

    # print the skewness of the residual distribution
    print('Skewness: ', model_3_analysis.skewness(model_3_analysis.unweighted_residuals(), 30))

    # print the kurtosis of the residual distribution
    print('Kurtosis: ', model_3_analysis.kurtosis_new(model_3_analysis.unweighted_residuals(), 30))

    ####################################################################################################################
    # FOURIER ANALYSIS
    ####################################################################################################################
    x = refined_data_frame['year'].values
    y = refined_data_frame['gmsl'].values

    # Number of sample points
    N = len(x)

    # sample spacing
    T = (y[-1]-y[0]) / N

    # fast fourier transform
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)

    refined = 2.0 / N * np.abs(yf[:N // 2])
    idx = argrelextrema(refined, np.greater)

    figure(num=None, figsize=(8, 5), dpi=300, facecolor='w', edgecolor='k')
    plt.grid()
    plt.semilogy(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.semilogy(xf[idx][refined[idx] >= 1.0], refined[idx][refined[idx] >= 1.0], linewidth=0, marker='o')
    plt.ylabel('Amplitude [mm]')
    plt.xlabel('Frequency (per year)')
    plt.savefig('gmst_plots/fourier_analysis.png')
    # plt.show()

    ####################################################################################################################
    # MODEL 4
    ####################################################################################################################
    _basis_functions = ['x^0', 'x', 'x^2', 'cos(2*pi*x)', 'sin(2*pi*x)']
    for freq in xf[idx][refined[idx] >= 1.0]:
        _basis_functions += ['cos(2*pi*x*{})'.format(str(freq)), 'sin(2*pi*x*{})'.format(str(freq))]

    # model 4 involving a trend, bias and signal
    model_4 = LinearDesignModel(basis_functions=_basis_functions)

    # instantiate a data analysis model with the given data and linear model
    model_4_analysis = DataAnalysis2D(model_4, refined_data_frame['year'].values, refined_data_frame['gmsl'].values)

    # plot the trend + bias model for the data set
    model_4_analysis.plot_model(
        ylabel='GMSL variation (mm) w.r.t. 20 year mean reference',
        xlabel='Year',
        show_save=('save'),
        name='gmst_plots/model4_plot',
        legend=False
    )

    # plot the residual probability density function
    model_4_analysis.plot_residual_hist(
        ylabel='Probability density f($\epsilon$)',
        xlabel='Residuals ($\epsilon$) [mm]',
        show_save=('save'),
        name='gmst_plots/model4_residuals',
    )

    # print the least square solution
    print('Vector of parameters: ', model_4_analysis.unweighted_least_squares())

    # print the skewness of the residual distribution
    print('Skewness: ', model_4_analysis.skewness(model_4_analysis.unweighted_residuals(), 30))

    # print the kurtosis of the residual distribution
    print('Kurtosis: ', model_4_analysis.kurtosis_new(model_4_analysis.unweighted_residuals(), 30))
