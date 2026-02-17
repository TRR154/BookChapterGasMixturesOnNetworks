"""
This script computes and stores numerical solutions for the experiment in Section 1.5.4.

Investigates: the effect of changing the inner friction on numerical solutions of the two-velocity model on a pipe. 

Requires: [none]

Output: numerical solutions.
    - Stored in: save_solution/solutions/onepipe_2v_nooutflow

To run this script, navigate to the main directory and run 
    
    python -m numerical_experiments.section_1_5_4.script_section_1_5_4_computations

NOTE: This is run as a Python module in order to properly access all subfolders in the directory. Therefore:
    - The `-m` flag must be included. 
    - `.py` must NOT be included at the end.  
"""

from onepipe_instationary import solutions


def main():
    for i in range(5):
        f = 10**(1-(i))
        solutions(f)

if __name__ == "__main__":
    main()

