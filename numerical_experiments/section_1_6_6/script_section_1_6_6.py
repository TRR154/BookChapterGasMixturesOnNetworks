"""

This script computes and stores numerical solutions of the one and two-velocity model on Gaslib40-3 
for the experiment in Section 1.6.6.

Investigates: the differences of one and two-velocity solutions on one Gaslib40-3. 

Run from the main folder via:
python -m numerical_experiments.section_1_6_6.script_section_1_6_6
NOTE:
1. Do NOT forget -m 
2. Leave out .py in the end 

Results:
   solutions of 1v and 2v solutions on one pipe
    - Stored in: save_solutions/solutions/network_1v_timedep/massinflow/1v_one_pipe...
    - Stored in: save_solutions/solutions/network_2v_f_{f}_timedep/massinflow/1v_one_pipe...
    - Stored in: save_solutions/solutions/network_1v_timedep/pressureinflow/2v_one_pipe...
    - Stored in: save_solutions/solutions/network_2v_f_{f}_timedep/pressureinflow/2v_one_pipe...

    plots 1v vs 2v solutions on one pipe
    - Stored in: graphics/networks/1v_vs_2v/gaslib40_removed_edit
"""

import matplotlib.pyplot as plt

from network_1v_timedep_dxdtform_massflowinflow import compute_gaslib40_modified_smaller_disc

from network_2v_timedep_dxdtform_massflowinflow import compute_gaslib40_scenarions

from diff_1v_2v_network_massflowinflow import plot_all_1v_vs_2v


def main():



    #compute the gaslib40-3 1v solution
    #NOTE: This takes ~1h
    #Not needed if data already computed
    model = "speed_of_sound"
    algebraic = 1.0
    compute_gaslib40_modified_smaller_disc(model=model,algebraic=algebraic)
    plt.close()

    # computes the gaslib40-3 2v solutions
    #NOTE: This takes ~2h
    compute_gaslib40_scenarions()

    # Creating Fig 1.17 and Fig 1.18 
    network_type = "gaslib40"
    plot_all_1v_vs_2v(network_type=network_type)


if __name__ == "__main__":
    main()
    