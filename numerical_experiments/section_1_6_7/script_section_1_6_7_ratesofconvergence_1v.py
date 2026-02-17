"""
This script computes rates of convergence from the linear interpolant values for the one-velocity model for the experiment in Section 1.6.7

Requires: linear interpolant values created by script_section_1_6_7_linearinterpolants_1v.py

Results: relative errors and rates of convergence while scaling the dt and dx parameters for the one-velocity model. 
    - Prints values to the terminal. 

To run this script, navigate to the main directory and run 
    
    python -m numerical_experiments.section_1_6_7.script_section_1_6_7_ratesofconvergence_1v

NOTE: 
1. This is run as a Python module in order to properly access all subfolders in the directory. Therefore:
    - The `-m` flag must be included. 
    - `.py` must NOT be included at the end. 
"""
from linearinterpolants import norms_of_diffs_and_eoc

def main():
    max_exponent = 7
    space_norm_order = 1
    time_norm_order = 1
    offset = 2

    norms_of_diffs_and_eoc(max_exponent=max_exponent, space_norm_order=space_norm_order, time_norm_order=time_norm_order,offset=offset,two_velocity=False)

main()

