"""

This script computes and stores numerical solutions of the simplified and non-simplified one-velocity model on Gaslib40-3 
for the experiment in Section 1.6.5.

Investigates: the differences of simplified and non-simplified one-velocity model on Gaslib40-3. 

Run from the main folder via:
python -m numerical_experiments.section_1_6_5.script_section_1_6_5
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

from network_1v_timedep_dxdtform_massflowinflow import compute_gaslib40_modified

from network_1v_images import plot_all_algebraic_vs_speed_of_sound


def main():



    #compute the gaslib40-3 data
    #NOTE: This takes ~1h
    #Not needed if data already computed
    model = "speed_of_sound"
    algebraic = 1.0
    compute_gaslib40_modified(model=model,algebraic=algebraic)
    plt.close()

    model = "speed_of_sound"
    algebraic = 0.0
    compute_gaslib40_modified(model=model,algebraic=algebraic)
    plt.close()

    # Creating Fig 1.14 
    #NOTE: This takes ~1h
    plot_all_algebraic_vs_speed_of_sound()

    #Plots:
    #graphics/networks/network_1v_timedep/massflowinflow/gaslib40_removed_edit/....
    #graphics/networks/pressure_difference_1v/massflowinflow

if __name__ == "__main__":
    main()
    