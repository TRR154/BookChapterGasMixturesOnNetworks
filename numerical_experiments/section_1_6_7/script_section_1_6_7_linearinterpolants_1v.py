"""
This script computes linear interpolant values from solutions to the one-velocity model for the experiment in Section 1.6.7.

Requires: numerical solutions created by script_section_1_6_7_computesolutions_1v.py

Results: values of linear interpolants generated from solutions of the one-velocity model computed on a fine space-time grid 
    - Stored in: save_solution/solutions/network_1v_timedep/compstudy/

To run this script, navigate to the main directory and run 
    
    python -m numerical_experiments.section_1_6_7.script_section_1_6_7_linearinterpolants_1v

NOTE: 
1. This is run as a Python module in order to properly access all subfolders in the directory. Therefore:
    - The `-m` flag must be included. 
    - `.py` must NOT be included at the end. 
2. The run time for this code is around TWO DAYS using university computing servers.  
"""
from linearinterpolants import linear_interpolant_values_new

def main():
    offset = 2
    for exponent_index in range(6):
        print("scale exponent = -"+str(exponent_index+offset))
        linear_interpolant_values_new(choice_of_network=2, scenario="time_dep", candidate_dx=1500, dt=720,\
                                      T=int(60*720/3), model="speed_of_sound", algebraic=1.0, exponent_groundtruth=7,\
                                        scale_exponent=exponent_index+offset,two_velocity=False)

main()
