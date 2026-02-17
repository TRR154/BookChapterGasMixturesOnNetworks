"""

This script computes and stores numerical solutions of the one and two-velocity model on one pipe 
for the experiment in Section 1.5.3.

Investigates: the differences of one and two-velocity solutions on one pipe. 


Run from the main folder via:
python -m numerical_experiments.section_1_5_3.script_section_1_5_3
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
    - Stored in: graphics/networks/1v_vs_2v/...
"""

import matplotlib.pyplot as plt

from network_1v_timedep_dxdtform_massflowinflow import one_pipe_computations as one_pipe_1v_massinflow
from network_1v_timedep_dxdtform_pressureinflow import one_pipe_computations as one_pipe_1v_pressinflow

from network_2v_timedep_dxdtform_massflowinflow import compute_one_pipe as one_pipe_2v_massinflow
from network_2v_timedep_dxdtform_pressureinflow import compute_one_pipe as one_pipe_2v_pressinflow


from diff_1v_2v_one_pipe_massflowinflow import plot_all_1v_vs_2v as plot_1v_vs_2v_mass
from diff_1v_2v_one_pipe_pressureinflow import plot_all_1v_vs_2v as plot_1v_vs_2v_press


def main():

    #Not needed if data already computed
    # 1v one pipe
    one_pipe_1v_massinflow()
    one_pipe_1v_pressinflow()

    #Not needed if data already computed
    # 2v one pipe
    one_pipe_2v_massinflow()
    one_pipe_2v_pressinflow()


    #Creates Fig.1.6 
    plot_1v_vs_2v_press()
    plt.close()
    #Creates Fig.1.6 
    plot_1v_vs_2v_mass()
    plt.close()


if __name__ == "__main__":
    main()
    