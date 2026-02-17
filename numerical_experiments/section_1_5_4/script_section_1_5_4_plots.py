"""
This script plots the results of the experiment in Section 1.5.4. 

Focus: Investigate the effect of changing the inner friction on numerical solutions of the 2v model for a single pipe. 

Requires: Solutions computed in script_section_1_5_4_computations

Output: plots corresponding to Fig. 1.8 and 1.9 in Section 1.5.4. 
    - Stored in: graphics/networks/onepipe_2v_nooutflow

To run this script, navigate to the main directory and run 
    
    python -m numerical_experiments.section_1_5_4.script_section_1_5_4_plots

NOTE: This is run as a Python module in order to properly access all subfolders in the directory. Therefore:
    - The `-m` flag must be included. 
    - `.py` must NOT be included at the end. 
"""

from onepipe_instationary import plot_vnorms_log_timescale, plot_vnorms_lin_timescale


def main():
    plot_vnorms_log_timescale(5)
    plot_vnorms_lin_timescale(3)

if __name__ == "__main__":
    main()
