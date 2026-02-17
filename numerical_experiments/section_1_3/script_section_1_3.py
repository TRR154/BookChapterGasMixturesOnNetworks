"""

This script computes pressure law plots for the experiment in Section 1.3

Investigates: the pressure difference for different densities.

Run from the MAIN folder via:
python -m numerical_experiments.section_1_3.script_section_1_3
NOTE:
1. Do NOT forget -m 
2. Leave out .py in the end 


Results: plots of pressure laws 
    - Stored in: graphics/pressure_plots/
"""

import matplotlib.pyplot as plt

from pressure_laws_comparison import plot_gerg_approximation,plot_pressure_mixtures


def main():

    #Fig 1.1
    plot_pressure_mixtures()
    plt.close()


    #Fig 1.1 black and white
    plot_pressure_mixtures(black_white=True)
    plt.close()

    #Fig 1.2
    plot_gerg_approximation(relative_error=True)
    plt.close()

    #Fig 1.3
    plot_gerg_approximation(relative_error=False)
    plt.close()


    #Fig 1.3 in black and white
    plot_gerg_approximation(relative_error=False,black_white=True)
    plt.close()


if __name__ == "__main__":
    main()
    