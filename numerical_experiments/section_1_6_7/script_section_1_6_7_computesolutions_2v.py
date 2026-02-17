"""
This script computes and stores numerical solutions of the two-velocity model for the experiment in Section 1.6.7.

Requires: [none]

Results: numerical solutions of the two-velocity model. 
    - Stored in: save_solution/solutions/network_2v_f_1_timedep/massflowinflow

To run this script, navigate to the main directory and run 
    
    python -m numerical_experiments.section_1_6_7.script_section_1_6_7_computesolutions_2v

NOTE: 
1. This is run as a Python module in order to properly access all subfolders in the directory. Therefore:
    - The `-m` flag must be included. 
    - `.py` must NOT be included at the end. 
2. The run time for this code is around ONE WEEK using university computing servers.  
"""

from network_2v_timedep_dxdtform_massflowinflow import Network_2v_time
from pathlib import Path
import numpy as np


def main():
    model = "speed_of_sound"
    scenario = "time_dep"
    T = 60*60*4
    algebraic = 1.0 
    tol = 1e-7
    f = 1

    file_network = Path("network_data" ,"optimization_data", "network_files", "gaslib40_removed_edit.net")
    
    file_data = Path("network_data" ,"optimization_data", "solution_files","gaslib40_removed_edit.lsf")
    
    starting_offset = 2
    for i in range(6):
        
        actualexponent = -i-starting_offset
        print("")
        print("scaling exponent = "+str(actualexponent))
        scale = 1.5**actualexponent
        candidate_dx = 1500*scale
        candidate_dt = 720*scale
        dt = T/np.round(T/candidate_dt)
        
        network = Network_2v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=candidate_dx,
                                    dt=dt,T=T)
        
        network.solve(algebraic=algebraic,tol=tol,scenario=scenario,f=f)
        network.save_solution_network(algebraic=algebraic, scenario=scenario,f=f)

if __name__ == "__main__":
    main()

      