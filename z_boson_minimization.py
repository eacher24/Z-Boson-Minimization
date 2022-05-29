# -*- coding: utf-8 -*-
"""
------------Z BOSON: FINAL ASSIGNMENT-----------
PHYS20161 - Assignment 2 - Z boson interactions
------------------------------------------------
This python script reads csv files, performs different rounds of validations
and then

1.)Sorts the data into numerical order and removes any zeros.
2.)Stacks the two data files to produce one array of data.
3.)Goes through another round of filtering extreme values.
4.)Fits the observed data onto a predicted model.
5.)Minimises the chi squared between the two fits and performs a second
minimisation with more outliers removed and a better starting guess for mass
and width.

From this, the data containing the mass, m_z, the width, Γ_z, the reduced
chi squared and the lifetime can be obtained and printed.

If an error is found in the data input/validation stages, the programme comes
to a hault and a statement is printed to the console.

Last Updated: 20/12/2021
Author: Elise Acher 

"""

import statistics
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize


def open_dataset1():
    """
     Open_dataset1() Function.

    ----------
        •Used to open the first datafile, which must be in the form of csv.
        •Once data is opened, it is passed through a numerical filter, whereby
        rows containing nan's are deleted.

    Parameters
    ----------
        • dataset1 : 2-dimensional array.

    Returns
    -------
        • dataset1 to the main function to be used later in the code.

    """
    try:
        read_dataset1 = np.genfromtxt(
            'z_boson_data_1.csv', delimiter=',', comments='%', autostrip=True)
        validate_dataset1 = read_dataset1[~np.isnan(read_dataset1).any(axis=1)]
        remove_negative = validate_dataset1[validate_dataset1[:, 1]> 0]

    except TypeError:
        print('Input Data is not the correct type.')

    return remove_negative



def open_dataset2():
    """
    Open_dataset2() Function.

    ------
        •Used to open the first datafile, which must be in the form of csv.
        •Once data is opened, it is passed through a numerical filter, whereby
        rows containing nan's are deleted.

    Parameters
    ----------
        • dataset1 : 2-dimensional array.

    Returns
    -------
        • dataset2 to the main function to be used later in the code.

    """
    try:
        read_dataset2 = np.genfromtxt(
            'z_boson_data_2.csv', delimiter=',', comments='%', autostrip=True)
        validate_dataset2 = read_dataset2[~np.isnan(read_dataset2).any(axis=1)]
        remove_negative = validate_dataset2[validate_dataset2[:, 1]> 0]

    except TypeError:
        print('Input Data is not the correct type.')

    return remove_negative







def merge_input_data(data1, data2):
    """
    merge_input_data(data1, data2) Function.

    ---------
        • Performs a vertical stack of the two input datasets, to produce one
        2-D array comprising all of the useful input data

    Parameters
    ----------
        • data1 : 2-dimensional array.
        • data2 : 2-dimensional array.

    Returns
    -------
        • merge :  2-dimensional array
        or
        • ValueError : Print statement notifying user that input data cannot
    be accepted.

    """
    try:
        merge = np.vstack((data1, data2))
    except ValueError:
        print("Programme cannot merge two datasets of unequal colomns.")

    return merge


# Performs Two Tests on the merged datasets


def x_sect_filter(merged_datasets):
    """
    x_filter_filter(merge) Function.

    ----------
        • First stage of data filtering, comprisises of checking if the range
    of input values of the x section coloumn lies within (+-) 3 standard
    deviations of the mean of x section values.
        • If the data does not satify these conditions, the entire row it
    belongs to is deleted.

    Parameters
    ----------
    merge : 2-dimensional array.

    Returns
    -------
    filtered_x_sect : 2-dimensional array.

    """

    #Determine statistical results to validate data with
    x_sect_stdeva = statistics.stdev(merged_datasets[:, 1])
    x_sect_mean = np.mean(merged_datasets[:, 1])

    filter2 = merged_datasets[np.abs(x_sect_mean -
                                     merged_datasets[:, 1]) < x_sect_stdeva ]

    return filter2



def uncert_filter(x_sect_filter):
    """
    uncert_filter(merge) Function.

    ----------
        • Second stage of data filtering, comprisises of checking if the range
    of input values of the uncertainity coloumn lies within (+-) 3 standard
    deviations of the mean of uncertainity values.
        • If the data does not satify these conditions, the entire row it
    belongs to is deleted.

    Parameters
    ----------
    updated_array : 2-dimensional array.

    Returns
    -------
    filtered_uncert : 2-dimensional array.

    """


    #Determine statistical results to validate data with
    uncert_stdeva = statistics.stdev(x_sect_filter[:, 2])
    uncert_mean = np.mean(x_sect_filter[:, 2])

    filter2 = x_sect_filter[np.abs(uncert_mean -
                                   x_sect_filter[:, 2]) < 2*uncert_stdeva ]
    final_sort = np.argsort(filter2[:, 0])
    filter2 = filter2[final_sort,:]

    return filter2


def predicted_x_section(filtered_inputdata, mass, width):
    """
    predicted_x_section() Function.

    ----------
        • Using the energy values column of the fully filtered input data, a
    prediction of the results can be made using an desired input mass and
    width.

    Parameters
    ----------
        • filtered_inputdata : 2-dimensional array.
        • mass : float.
        • width: float.

    Returns
    -------
    x_section : 2-dimensional array.

    """
    energy = filtered_inputdata[:, 0]
    width_ee = 0.08391
    conversion_factor = ((0.3894)*(10**6))

    num = ((np.pi)*12) * (energy**2) * (width_ee**2)
    den = (((energy**4)*(mass**2)) - (2*(energy**2)*(mass**4)) + (mass**6) +
           ((mass**4)*(width**2)))
    x_section = (num/den)*(conversion_factor)


    return x_section



def chi_squared(parameter_guesses, filtered_inputdata):
    """
    chi_squared() Function.

    ----------
        • Initiates a chi squared test on the observed and predicted data.
        • This function is then minimised to obtain a minimised chi square.

    Parameters
    ----------
        • parameter_guesses : 2 initial floats called from predicted function.
        • filtered_inputdata : 2-dimensional array.

    Returns
    -------
        • chi_sum : Sum of all the chi squares.

    """
    mass, width = parameter_guesses
    observed = filtered_inputdata[:, 1]
    predicted = predicted_x_section(filtered_inputdata, mass, width)
    uncert = filtered_inputdata[:, 2]


    chi_before_sum = (((observed - predicted)/uncert)**2)
    chi_sum = sum(chi_before_sum)



    return chi_sum

def predicted_x_section_2(filtered_inputdata, mass, width):
    """
    predicted_x_section() Function.

    ----------
        • Formulates a better predicted fit, based on the first minimization.
        • This is used to improve the guesses in fmin for the second round of
    optimzation.

    Parameters
    ----------
        • filtered_inputdata : 2-dimensional array.
        • mass : float.
        • width : float.

    Returns
    -------
        • minimized_x_section : 2-dimensional array.

    """
    energy = filtered_inputdata[:, 0]
    width_ee = 0.08391
    conversion_factor = ((0.3894)*(10**6))

    num = ((np.pi)*12) * (energy**2) * (width_ee**2)
    den = (((energy**4)*(mass**2)) - (2*(energy**2)*(mass**4)) + (mass**6) +
           ((mass**4)*(width**2)))
    minimized_x_section = (num/den)*(conversion_factor)


    return minimized_x_section

def chi_squared_2(parameter_guesses, filtered_inputdata):
    """

    chi_squared_2() Function.

    ----------
        • Initiates a 2nd chi squared test on the improved guesses between
    observed and predicted data.
        • This function is then minimised a second time to obtain an optimally
    minimised chi square value.

    Parameters
    ----------
        • parameter_guesses : 2 initial floats called from predicted function.
        • filtered_inputdata : 2-dimensional array.

    Returns
    -------
        • chi_sum_2 : Sum of all the chi squares.


    """
    mass, width = parameter_guesses
    observed = filtered_inputdata[:, 1]
    predicted = predicted_x_section_2(filtered_inputdata, mass, width)
    uncert = filtered_inputdata[:, 2]


    chi_before_sum = (((observed - predicted)/uncert)**2)
    chi_sum_2 = sum(chi_before_sum)



    return chi_sum_2

def reduced_chi_squared(minimize_2, filtered_inputdata):
    """


    Parameters
    ----------
    minimize_2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    minimization_data = minimize_2
    (mass_and_width, chi_square_value) = minimization_data[:2]
    reduced_chi = (minimization_data[1])/ (len((filtered_inputdata)[:, 0])
                                           - len(mass_and_width))

    print('Mass = {:0.4g}GeV/c^2'.format(mass_and_width[0]))
    print('Width = {:0.4g} GeV'.format(mass_and_width[1]))
    print('Reduced Chi Squared = {:0.3f}'
          .format(reduced_chi))

    return reduced_chi, mass_and_width

def lifetime(minimize_2):
    """
    lifetime() Function.

    ----------
        • Unpacks a tuple from the the second minimization.
        • Calculates the lifetime of the Z boson.

    Parameters
    ----------
        • minimize_2 : tuple.

    Returns
    -------
        • lifetime : float.

    """
    minimization_data = minimize_2
    (mass_and_width, chi_square_value) = minimization_data[:2]
    width = mass_and_width[1]
    lifetime = scipy.constants.hbar/width

    return lifetime

# ------------------------ Graphs Section ----------------------------

def combined_data_graph(predicted_minimized, filtered_inputdata):
    """
    Combined_data_graph() Function.

    ----------
        • Combines the predicted and observed fit onto one graph.

    Parameters
    -------
        • predicted_minimized : 2-dimensional array.
        • filtered_inputdata :  2-dimensional array.

    Returns
    -------
    None.

    """
    energy = filtered_inputdata[:, 0]
    x_section = predicted_minimized


    fig = plt.figure(figsize=((7,9)))
    ax = fig.add_subplot(211)
    ax.set_title('Cross Section against Energy:',
                 fontsize = 20, fontfamily = 'Times New Roman')
    ax.set_xlabel('Energy, E [GeV]', fontsize = 16, fontfamily = 'Times New Roman')
    ax.set_ylabel('Cross Section, $\sigma$ [nb]', fontsize = 16,
                  fontfamily = 'Times New Roman')
    ax.errorbar(filtered_inputdata[:, 0], filtered_inputdata[:, 1],
                filtered_inputdata[:, 2], fmt='o', markersize = 3, zorder = 1)

    ax.plot(energy, x_section, linewidth = 2, color = 'purple', zorder = 2)
    ax.legend(['predicted data', 'observed data'])



    # fig = plt.figure(figsize=((7,5)))
    residuals_y =  filtered_inputdata[:, 1] - x_section
    residuals_x = filtered_inputdata[:, 0]
    ax_2 = fig.add_subplot(414)
    ax_2.scatter(residuals_x, residuals_y, color = 'green', s = 4)
    ax_2.set_title('Residual Data',
                 fontsize = 12, fontfamily = 'Times New Roman')
    ax_2.set_xlabel('Energy, E [GeV]', fontsize = 11,
                    fontfamily = 'Times New Roman')
    ax_2.set_ylabel('Residual difference [nb]', fontsize = 11,
                  fontfamily = 'Times New Roman')
    ax_2.axhline(y=0.0, color = 'r', linestyle = '-')
    ax_2.errorbar(residuals_x, residuals_y, filtered_inputdata[:, 2], fmt='o',
                  markersize = 3, zorder = 1)
    plt.savefig('Summary_of_Data.png', dpi = 300)
    plt.show()

def contour_plot(minimize_2, data):
    """
    Contour_plot() Function.

    ----------
         • Produce 2 mesh arrays from the data that can be used to produce a
         contour plot around the central values for the optimised mass and
         width obtained previously.

    Parameters
    ----------
        • minimize_2 : 2-dimensional array .
        • data : 2-dimensional array .

    Returns
    -------
    None.

    """
    minimization_data = minimize_2
    (mass_and_width, chi_square_value) = minimization_data[:2]
    x = np.linspace(mass_and_width[0] - 0.05, mass_and_width[0] + 0.05, 120)
    y = np.linspace(mass_and_width[1] - 0.02, mass_and_width[1] + 0.02, 100)
    z = np.zeros((len(y), len(x)))

    for i in range(len(x)):
        for j in range(len(y)):
            z[j, i] = chi_squared_2([x[i], y[j]], data)

    x, y = np.meshgrid(x,y)

    fig = plt.figure(figsize=(4,4))
    fig = fig.add_subplot(111)
    # cs_filled = plt.contourf(x,y,z, levels = [chi_square_value
    #                                           + 1, chi_square_value
    #                                           + 2, chi_square_value + 3])
    cs = plt.contour(x,y,z, levels = [chi_square_value
                                      + 1, chi_square_value
                                      + 2.6, chi_square_value + 6])
    plt.scatter(mass_and_width[0], mass_and_width[1])

    lines = []
    for line in cs.collections[0].get_paths():
        lines.append(line.vertices)

    fig.clabel(cs, fontsize = 10, colors = 'black')

    fig.set_title('Mass versus Width Contour Plot', fontsize = 14,
                  font = 'Times New Roman')
    fig.set_xlabel('Mass, $m_z$ $[GeV/c^2]$')
    fig.set_ylabel(r'Width, $\Gamma_z$ $GeV$' )
    plt.savefig('Contour_Plot.png', dpi = 300)
    plt.show()

    return lines

#--------------------------------- Main Section ---------------------------

def __main__():
    """
    __main__() Function.

    ----------
        • Executes the main function.

    Returns
    -------
        • mass: float.
        • width : float.
        • Reduced Chi Squared : float.
        • Lifetime : float.
        • Graphs.

    """


    #Read in and Validate
    dataset1_input = open_dataset1()
    dataset2_input = open_dataset2()

    #Merge Datasets
    merged_datasets = merge_input_data(dataset1_input, dataset2_input)

    #Perform first round of filters to remove extreme values
    x_sect_filtered = x_sect_filter(merged_datasets)
    filtered_inputdata = uncert_filter(x_sect_filtered)

    #Set up predicted and observed data for comparison
    predicted = predicted_x_section(filtered_inputdata, 90, 3)



    #First Minimisation
    minimize = optimize.fmin(chi_squared, (90, 3), args=(filtered_inputdata,))

    #Second Minimization
    predicted_minimized = predicted_x_section_2(filtered_inputdata,
                                                minimize[0],
                                                minimize[1])

    #Get rid of more outliers
    filtered_inputdata = filtered_inputdata[np.abs(filtered_inputdata[:, 1]
                                                   - predicted_minimized)
                                            < 3 * filtered_inputdata[:, 2]]


    #Use better/filtered predicted data for second optimization
    minimize_2 = optimize.fmin(chi_squared_2, (minimize[0], minimize[1]),
                               args=(filtered_inputdata,), full_output= True)

    #Calculation of other variables
    reduced_chi_squared_value = reduced_chi_squared(minimize_2,
                                                    filtered_inputdata)
    lifetime_value = lifetime(minimize_2)
    print('Lifetime = {:3.2E} seconds '.format(lifetime_value))


    predicted = predicted_x_section(filtered_inputdata, minimize[0],
                                    minimize[1])
    combined_data_graph(predicted, filtered_inputdata)

    lines = contour_plot(minimize_2, filtered_inputdata)

    # uncertainities = uncert_mass_width(lines, minimize_2)


    return 0

__main__()
