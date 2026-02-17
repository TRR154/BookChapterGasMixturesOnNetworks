"""
This script computes and stores numerical solutions of the one-velocity model on Gaslib40-3 
with different pressure laws for the experiment in Section 1.6.4.

Investigates: the differences of one-velocity solutions wth different pressure laws on one Gaslib40-3. 


Run from the main folder via:
python -m numerical_experiments.section_1_6_4.script_section_1_6_4
NOTE:
1. Do NOT forget -m 
2. Leave out .py in the end 

Results: 

   solutions of 1v and 2v solutions on gaslib40-3
    - Stored in: save_solutions/solutions/network_1v_timedep/massinflow/1v_gaslib40...
                 save_solutions/solutions/network_1v_timedep/pressureinflow/1v_gaslb40...

    plots of differences for solutions with different pressure laws
    - Stored in: graphics/networks/network_1v_timedep/massflowinflow/gaslib40_removed_edit/....
                 graphics/networks/pressure_difference_1v/massflowinflow
"""

import matplotlib.pyplot as plt

from create_modified_gaslib_network_plot import plot_modified_gaslib40_net

from network_1v_timedep_dxdtform_massflowinflow import compute_gaslib40_modified

from network_1v_images import plot_all_pressure_law_difference


def main():

    # Creating Fig 1.10
    plot_modified_gaslib40_net()




    #compute the gaslib40-3 data
    #NOTE: This takes ~1h
    #Not needed if data already computed
    model = "speed_of_sound"
    algebraic = 1.0
    compute_gaslib40_modified(model=model,algebraic=algebraic)
    plt.close()

    model = "virial_expansion"
    algebraic = 1.0
    compute_gaslib40_modified(model=model,algebraic=algebraic)
    plt.close()

    model = "virial_expansion_mix"
    algebraic = 1.0
    compute_gaslib40_modified(model=model,algebraic=algebraic)
    plt.close()

    model = "gerg"
    algebraic = 1.0
    compute_gaslib40_modified(model=model,algebraic=algebraic)
    plt.close()

    model = "gerg_fit"
    algebraic = 1.0
    compute_gaslib40_modified(model=model,algebraic=algebraic)
    plt.close()

    # Creating Fig 1.11 Fig 1.12 Fig 1.13  
    #NOTE: This takes ~1h
    plot_all_pressure_law_difference()


if __name__ == "__main__":
    main()
    