#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is designed to investigate the effect of the inner friction parameter on the two-velocity model 
on instationary computations for a single pipe. This program contains code which:
1. Defines a solver function, including 
    - Initial and boundary data
    - An implementation of the box scheme for the two-velocity model
    - A Newton solver
2. Saves the results
3. Loads saved results and plots them 
"""

#Standard packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#Import jax and submodules for automatic differentiation and other tricks
import jax
from jax import numpy as jnp
from jax import grad

#For Newton solver
from copy import deepcopy

#For saving and retrieving files
from pathlib import Path
import pickle

#For reading in network
from network_2v_timedep_dxdtform_massflowinflow import Network_2v_time




matplotlib.rcParams.update({
    "text.usetex": True,              # use LaTeX for all text
    "pgf.texsystem": "pdflatex",      # choose LaTeX engine
    "font.family": "serif",           # use serif fonts
    "font.serif": ["Times New Roman"], # Times-like serif (matches newtxtext)
    "mathtext.fontset": "custom",     # use custom math font
    "mathtext.rm": "Times New Roman", # Times-like math (matches newtxmath)
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.sf": "Times New Roman",
    
    # ---------------------------
    # Optional: scale labels
    # ---------------------------
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    #"colorbar.labelsize": 10,
    "figure.titlesize": 20,
    "axes.titlesize":24
})


            
def solve(data,candidate_dx:float,dt:float,T:float,algebraic:float,tol:float,f:float):
    """
    Implements a solver using the box scheme for the two-velocity model. Other than 
    the direct implementation of initial and boundary conditions, this function is identical to that
    found in network_2v_timedep_massflowinflow.py

    Parameters:
        data: instance of Network_2v_time class, which contains all required physical information 
        candidate_dx: candidate size of space step (m) (here also actual size)
        dt: size of time step (s)
        T: size of time horizon (s)
        algebraic: Coefficient on nonlinear terms. Here fixed at 1.
        tol: Newton tolerance.
        f: Inner friction: couples the different terms together. 
    """

    algebraic = 1

    data.candidate_dx = candidate_dx
    data.dt = dt
    data.T = T
    timerange = [dt*j for j in range(data.N_time)]

    nrofpieces = np.ceil(data.pipe_length/candidate_dx)
    dx_list = data.pipe_length/nrofpieces
    data.N_x_all_list = np.array([int(n_x) + 1 for n_x in nrofpieces])
    N_x_all_list = data.N_x_all_list
    data.max_dx = np.max(dx_list)


    ##INITIAL CONDITIONS
    initial_vec = jnp.zeros(4*jnp.sum(data.N_x_all_list)) #number of variables * size of domain 

    rho_1_source = 10
    rho_2_source = 5
    v_source = 14

    for i,id in enumerate(data.pipe_id): 

        rho_1_h = rho_1_source*np.ones(data.N_x_all_list[i])
        rho_2_h = rho_2_source*np.ones(data.N_x_all_list[i])
        v_1_h = v_source*np.ones(data.N_x_all_list[i])
        v_2_h = -v_source*np.ones(data.N_x_all_list[i])

        #To reduce the effect of shocks, at each end for each constituent we interpolate
        #from the constant velocity value to 0 
        interpolation_domain = 2
        
        v_left = np.linspace(0,v_source,interpolation_domain)
        v_right = np.linspace(v_source,0,interpolation_domain)
        v_middle = v_source*np.ones(data.N_x_all_list[i] - 2*interpolation_domain)
        v_1_h = np.concatenate((v_left,v_middle,v_right))
        v_2_h = -np.concatenate((v_left,v_middle,v_right))
        
        initial_vec = initial_vec.at[4*np.sum(N_x_all_list[:i])+0*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+1*N_x_all_list[i]].set(rho_1_h)
        initial_vec = initial_vec.at[4*np.sum(N_x_all_list[:i])+1*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]].set(rho_2_h)
        initial_vec = initial_vec.at[4*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]].set(v_1_h)
        initial_vec = initial_vec.at[4*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+4*N_x_all_list[i]].set(v_2_h)
        
    ##BOUNDARY CONDITIONS
    bc_m1 = jnp.zeros((data.N_time,len(data.node_id)))
    bc_m2 = jnp.zeros((data.N_time,len(data.node_id)))
    for i,id in enumerate(data.node_id):
        mu_goal = 0.5
        def bc_m_values(t):
            return 0
        bc_m_vectorised = np.vectorize(bc_m_values)    
        def bc_mu_values(t):
            mu_value = mu_goal
            return mu_value      
        bc_mu_vectorised = np.vectorize(bc_mu_values)       
        bc_m1_values = bc_m_vectorised(timerange)*bc_mu_vectorised(timerange)
        bc_m2_values = bc_m_vectorised(timerange)*(np.ones(data.N_time)-bc_mu_vectorised(timerange))      
        bc_m1 = bc_m1.at[:,i].set(bc_m1_values) 
        bc_m2 = bc_m2.at[:,i].set(bc_m2_values) 
        
        
    data._precompute_pressure_law()
    p1 = data.p1_jax
    p2 = data.p2_jax
        
        
    def F(y,y_old,bc_m1,bc_m2):
        """
        Discrete system. 

        Args:
            y (jnp.ndarray): y^{n+1} new time step
            y_old (jnp.ndarray): y^n old time step
            bc_m1 (jnp.ndarray): array of mass flow boundary values for each node (0, if not bc node ),
            with size n_nodes*(N_time)
            bc_p1 (jnp.ndarray):  array of pressure boundary values for each node (0, if not bc node ),
            with size n_nodes*(N_time)
            bc_m2 (jnp.ndarray): array of mass flow boundary values for each node (0, if not bc node ),
            with size n_nodes*(N_time)
            bc_p2 (jnp.ndarray):  array of pressure for each node (0, if not bc node ),
            with size n_nodes*(N_time)
        Returns:
            jnp.ndarray: result
        """
        block_rho_1_list = []
        block_rho_2_list = []
        block_v_1_list = []
        block_v_2_list = []
        mass_cons_1_list = []
        mass_cons_2_list = []
        
        
        for i,id in enumerate(data.pipe_id):
            previous_steps=int(np.sum(N_x_all_list[:i]))

            rho_1_h = y[4*previous_steps:4*previous_steps+N_x_all_list[i]]
            rho_2_h = y[4*previous_steps+N_x_all_list[i]:4*previous_steps+2*N_x_all_list[i]]
            v_1_h = y[4*previous_steps+2*N_x_all_list[i]:4*previous_steps+3*N_x_all_list[i]]
            v_2_h = y[4*previous_steps+3*N_x_all_list[i]:4*previous_steps+4*N_x_all_list[i]]
    


            rho_1_h_old = y_old[4*previous_steps:4*previous_steps+N_x_all_list[i]]
            rho_2_h_old = y_old[4*previous_steps+N_x_all_list[i]:4*previous_steps+2*N_x_all_list[i]]
            v_1_h_old = y_old[4*previous_steps+2*N_x_all_list[i]:4*previous_steps+3*N_x_all_list[i]]
            v_2_h_old = y_old[4*previous_steps+3*N_x_all_list[i]:4*previous_steps+4*N_x_all_list[i]]
            
            # mass conservation CH4
            block_rho_1 = ((rho_1_h[1:]+rho_1_h[:-1])/2-(rho_1_h_old[1:]+rho_1_h_old[:-1])/2)\
                +dt*jnp.diff(rho_1_h*v_1_h)/dx_list[i] 
                
            # mass conservation H2
            block_rho_2 = ((rho_2_h[1:]+rho_2_h[:-1])/2-(rho_2_h_old[1:]+rho_2_h_old[:-1])/2) \
                + dt*jnp.diff(rho_2_h*v_2_h)/dx_list[i] 
                
            #momentum conservation CH4 
            block_v_1 = (((rho_1_h[1:]*v_1_h[1:]+rho_1_h[:-1]*v_1_h[:-1])/2\
                         -(rho_1_h_old[1:]*v_1_h_old[1:]+rho_1_h_old[:-1]*v_1_h_old[:-1])/2) \
                +dt*jnp.diff(algebraic*rho_1_h*v_1_h**2+p1(rho_1_h,rho_2_h))/dx_list[i]\
                +dt*1/2*((data.pipe_friction[i]/(2*data.pipe_diameter[i])*rho_1_h*jnp.abs(v_1_h)*v_1_h)[1:]\
                +f*rho_1_h[1:]*rho_2_h[1:]*(v_1_h[1:]-v_2_h[1:])\
                +(data.pipe_friction[i]/(2*data.pipe_diameter[i])*rho_1_h*jnp.abs(v_1_h)*v_1_h)[:-1]\
                +f*rho_1_h[:-1]*rho_2_h[:-1]*(v_1_h[:-1]-v_2_h[:-1])))
            
            #momentum conservation H2
            block_v_2 = (((rho_2_h[1:]*v_2_h[1:]+rho_2_h[:-1]*v_2_h[:-1])/2\
                          -(rho_2_h_old[1:]*v_2_h_old[1:]+rho_2_h_old[:-1]*v_2_h_old[:-1])/2)\
                +dt*jnp.diff(algebraic*rho_2_h*v_2_h**2+p2(rho_1_h,rho_2_h))/dx_list[i]\
                +dt*1/2*((data.pipe_friction[i]/(2*data.pipe_diameter[i])*rho_2_h*jnp.abs(v_2_h)*v_2_h)[1:]\
                +f*rho_1_h[1:]*rho_2_h[1:]*(v_2_h[1:]-v_1_h[1:])\
                +(data.pipe_friction[i]/(2*data.pipe_diameter[i])*rho_2_h*jnp.abs(v_2_h)*v_2_h)[:-1]\
                +f*rho_1_h[:-1]*rho_2_h[:-1]*(v_2_h[:-1]-v_1_h[:-1])))
            
            block_rho_1_list.append(block_rho_1)
            block_rho_2_list.append(block_rho_2)
            block_v_1_list.append(block_v_1)
            block_v_2_list.append(block_v_2)
                
        ## Add boundary conditions into the system    
        for i,id in enumerate(data.node_id):
        
            a_list = data.pipe_diameter**2*jnp.pi/4
        
            # pipes where out going node == current node
            out_pipes_index_list = np.where(np.array(data.pipe_out) == id)[0] 
            # pipes where in node == current node
            in_pipes_index_list = np.where(np.array(data.pipe_in) == id)[0] 
        
            # mass conservation on each inner junctions or outflow
            mass_cons_1 = 0
            mass_cons_2 = 0
        
            for in_pipe_index in in_pipes_index_list:
                rho_1_0_h = y[4*np.sum(N_x_all_list[:in_pipe_index])]
                rho_2_0_h = y[4*np.sum(N_x_all_list[:in_pipe_index])+N_x_all_list[in_pipe_index]]
                v_1_h = y[4*np.sum(N_x_all_list[:in_pipe_index])+2*N_x_all_list[in_pipe_index]]
                v_2_h = y[4*np.sum(N_x_all_list[:in_pipe_index])+3*N_x_all_list[in_pipe_index]]
                mass_cons_1 -= a_list[in_pipe_index]*rho_1_0_h*v_1_h
                mass_cons_2 -= a_list[in_pipe_index]*rho_2_0_h*v_2_h
    
            for out_pipe_index in out_pipes_index_list:
                rho_1_N1_h = y[4*np.sum(N_x_all_list[:out_pipe_index])+N_x_all_list[out_pipe_index]-1]
                rho_2_N1_h = y[4*np.sum(N_x_all_list[:out_pipe_index])+2*N_x_all_list[out_pipe_index]-1]
                v_1_N1_h = y[4*np.sum(N_x_all_list[:out_pipe_index])+3*N_x_all_list[out_pipe_index]-1]
                v_2_N1_h = y[4*np.sum(N_x_all_list[:out_pipe_index])+4*N_x_all_list[out_pipe_index]-1]
                mass_cons_1 += a_list[out_pipe_index]*rho_1_N1_h*v_1_N1_h
                mass_cons_2 += a_list[out_pipe_index]*rho_2_N1_h*v_2_N1_h
    
            mass_cons_1 += bc_m1[i]
            mass_cons_2 += bc_m2[i]
    
            mass_cons_1_list.append(jnp.array([mass_cons_1]))
            mass_cons_2_list.append(jnp.array([mass_cons_2]))

        result = jnp.concatenate(
            [ item for item in block_rho_1_list ]
           + [ item for item in block_rho_2_list ]
           + [ item for item in block_v_1_list ]
           + [ item for item in block_v_2_list ]
           + [ item for item in mass_cons_1_list ]
           + [ item for item in mass_cons_2_list ])

        return result
        


    #Newton iteration limit
    n_iter = 100
    data.conv = []
    data.u_sol = initial_vec
    u_new = deepcopy(initial_vec)

    data.u_store = np.zeros(((data.N_time),initial_vec.shape[0]))
    data.u_store[0] = initial_vec

    for i in range(data.N_time-1):

        print("".center(80,"-"))
        print(f"Time Step {i}")
        
        u_time_t = deepcopy(u_new)
        
        #Newton method implementation using jax for the Jacobian.
        F_step = lambda x: F(x,u_time_t,bc_m1[i,:],bc_m2[i,:])
        F_deriv = jax.jacfwd(F_step)
        
        for j in range(n_iter):
            rhs = F_step(u_new)                
            dk = jnp.linalg.solve(F_deriv(u_new),-rhs)

            u_new = u_new + dk

            if j == n_iter-1:
                raise Exception(f"Newton solver not converged after {n_iter} steps")
            rhs_norm = np.linalg.norm(dk)

            if rhs_norm < tol:
                break
            elif not np.isfinite(rhs_norm):
                raise Exception(f"Newton solver not converging - norm is NaN at {j} Newton steps")

        data.u_sol = u_new
        data.u_store[i+1] = u_new

        if rhs_norm < tol:
            data.conv.append(True)
        else:
            data.conv.append(False)
   
    if all(data.conv):
        data.conv = True
    else: 
        data.conv = False


# Functions for storing solutions, so that expensive computations do not need to be redone
def save_v_norms(v_norms:np.ndarray,f:float,T:int):
    file_path = Path(f"save_solution/solutions/onepipe_2v_nooutflow/f_{f}_T_{T}_v_norms.pkl")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(v_norms, file)
        
def save_solutions(u_store:np.ndarray,f:float,T:int):
    file_path = Path(f"save_solution/solutions/onepipe_2v_nooutflow/f_{f}_T_{T}_solutions.pkl")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(u_store, file)
    
        
def load_v_norms(f:float,T:int):
    file_path = Path(f"save_solution/solutions/onepipe_2v_nooutflow/f_{f}_T_{T}_v_norms.pkl")
    with open(file_path, "rb") as file:
        v_norms = pickle.load(file)
    return v_norms

def load_solutions(f:float,T:int):
    file_path = Path(f"save_solution/solutions/onepipe_2v_nooutflow/f_{f}_T_{T}_solutions.pkl")
    with open(file_path, "rb") as file:
        u_store = pickle.load(file)
    return u_store
    
        
        
def solutions(f:float):
    """
    This routine implements the solver. The only parameter is the inner friction since this is 
    the focus of the experiment.
    
    Parameters:
        f: Inner friction parameter.
    """

    print("f = "+str(f))
    
    #Set time and space step sizes and time horizon
    dx = 50
    dt = 12
    T = 60*60*dt
    T = int(T) #ensure that T is of int type

    #Read in network data and instantiate class, assuming the ideal gas law
    network_file = "one_pipe"
    file_network = Path("network_data" ,"optimization_data", "network_files", str(network_file)+".net")
    
    init_and_bdry_choice = "one_pipe"
    file_data = Path("network_data" ,"optimization_data", "solution_files", str(init_and_bdry_choice)+".lsf")
    
    data = Network_2v_time(file_network=file_network,file_data=file_data,model="speed_of_sound",candidate_dx=dx,dt=dt,T=T)
    
    data._calculate_friction_parameter()
    
    tol = 1e-7

    #Call solver 
    solve(data=data, candidate_dx=data.candidate_dx, dt=data.dt, T=data.T, algebraic=1.0, tol=tol, f=f)

    #Save solution
    u_store = data.u_store
    file_path = Path(f"save_solution/solutions/onepipe_2v_nooutflow/f_{f}_T_43200_simple_initialdata_solutions_tol_1e-07.pkl")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(u_store, file)
    


def plot_vnorms_log_timescale(exponent_range:int):
    """
    Plots velocity norms for several values of $f$ on a logarithmic time scale up to half the time horizon.
    
    Parameters:
        exponent_range: Defines the range of exponents i such that we plot results for inner friction values
            $f = 10^{1-i}$. 
    """

        
    #Define space and time parameters, same as above. This is necessary for instantiating the network
    #data again, which will provide us with additional necessary info. 
    dx = 50
    dt = 12
    T = 60*60*dt
    T = int(T)
        
    network_file = "one_pipe"
    file_network = Path("network_data" ,"optimization_data", "network_files", str(network_file)+".net")
    
    init_and_bdry_choice = "one_pipe"
    file_data = Path("network_data" ,"optimization_data", "solution_files", str(init_and_bdry_choice)+".lsf")
    
    data = Network_2v_time(file_network=file_network,file_data=file_data,model="speed_of_sound",candidate_dx=dx,dt=dt,T=T)    

    N_x_all_list = data.N_x_all_list
    N_time = int(data.N_time)
    
    fig,ax = plt.subplots(figsize=(10,6))
    timerange = np.linspace(0, N_time-1,N_time)*dt

    #Empty arrays to store several lines for plotting on one axis
    labels = []
    lines = []
    colors = matplotlib.color_sequences['Set1']

    v_norms_all_f = []
    for exponent in range(exponent_range):
        f = 10**(1-exponent)
        print("")
        print("f = "+str(f))
        file_path = Path(f"save_solution/solutions/onepipe_2v_nooutflow/f_{f}_T_43200_simple_initialdata_solutions_tol_1e-07.pkl")
        with open(file_path, "rb") as file:
            u_store_largertolerance = pickle.load(file)
            
            
        v_norms_l1_larger = np.empty(N_time)
        
        v1_time_largertolerance = u_store_largertolerance[:,2*N_x_all_list[0]:3*N_x_all_list[0]]
        v2_time_largertolerance = u_store_largertolerance[:,3*N_x_all_list[0]:4*N_x_all_list[0]]
        for timestep in range(N_time):
            v_norms_l1_larger[timestep] = (1/(data.pipe_length[0]*N_x_all_list[0]))*np.linalg.norm(v1_time_largertolerance[timestep,:]-v2_time_largertolerance[timestep,:], ord=1)

        v_norms_all_f.append(v_norms_l1_larger)
        line, = ax.plot(timerange, v_norms_l1_larger, marker = 'o', markersize=2, color=colors[exponent])
        # line, = ax.plot(timerange, v_norms_l1_larger,color=colors[exponent])

        ######################
        #For black and white plots

        
        markers = [None,None,None,None,None,None,None,'o']
        dashes = [(None,None),[5,5],[5,3,1,3],[5,2,5,2,5,10],[1,3],[5,3,1,2,1,10],[5,3,1,2,1,10],(None,None)]
        
        # line.set_color('k')
        if markers[exponent] != None:
            line.set_marker(markers[exponent])
        if dashes[exponent] != (None, None):
            line.set_dashes(dashes[exponent])
        
        ######################
        lines.append(line)
        labels.append(r'$f = $ '+str(f))
        
    ax.set_yscale('log')
    ax.set_ylim(1e-20,1e-2)
    ax.set_ylabel(r"$D(f;t)$", labelpad=10)
    ax.set_xscale('symlog', linthresh=12)
    ax.axvline(x=12, color='gray', linestyle=":")
    ax.set_xlim(-1,43200*10**0.11) 
    ax.set_xlabel("time in s")
    ax.legend(labels)
    # plt.show()
    
    file_path = Path("graphics/networks/onepipe_2v_nooutflow/all_f_ell1norm_symlogtime.pdf")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, bbox_inches = "tight",dpi=100)
    plt.close()
        
def plot_vnorms_lin_timescale(exponent_range:int):
    """
    Plots velocity norms for several values of $f$ on a linear time scale up to half the time horizon.
    
    Parameters:
        exponent_range: Defines the range of exponents i such that we plot results for inner friction values
            $f = 10^{1-i}$. 
    """
    
        
    #Define space and time parameters, same as above. This is necessary for instantiating the network
    #data again, which will provide us with additional necessary info. 
    dx = 50
    dt = 12
    T = 60*60*dt
    T = int(T)

    network_file = "one_pipe"
    file_network = Path("network_data" ,"optimization_data", "network_files", str(network_file)+".net")
    
    init_and_bdry_choice = "one_pipe"
    file_data = Path("network_data" ,"optimization_data", "solution_files", str(init_and_bdry_choice)+".lsf")
    
    data = Network_2v_time(file_network=file_network,file_data=file_data,model="speed_of_sound",candidate_dx=dx,dt=dt,T=T)    

    N_x_all_list = data.N_x_all_list
    N_time = int(data.N_time)
    
    fig,ax = plt.subplots(figsize=(10,6))
    timerange = np.linspace(0, N_time-1,N_time)*dt

    #Empty arrays to store several lines for plotting on one axis
    labels = []
    lines = []
    colors = matplotlib.color_sequences['Set1']

    v_norms_all_f = []
    vertical_offset_factors = [2, 3, 5]
    for exponent in range(exponent_range):
        f = 10**(1-exponent)
        print("")
        print("f = "+str(f))
        file_path = Path(f"save_solution/solutions/onepipe_2v_nooutflow/f_{f}_T_43200_simple_initialdata_solutions_tol_1e-07.pkl")
        with open(file_path, "rb") as file:
            u_store_largertolerance = pickle.load(file)
            
            
        v_norms_l1_larger = np.empty(N_time)
        
        v1_time_largertolerance = u_store_largertolerance[:,2*N_x_all_list[0]:3*N_x_all_list[0]]
        v2_time_largertolerance = u_store_largertolerance[:,3*N_x_all_list[0]:4*N_x_all_list[0]]
        for timestep in range(N_time):
            v_norms_l1_larger[timestep] = (1/(data.pipe_length[0]*N_x_all_list[0]))*np.linalg.norm(v1_time_largertolerance[timestep,:]-v2_time_largertolerance[timestep,:], ord=1)

        v_norms_all_f.append(v_norms_l1_larger)
        line, = ax.plot(timerange, v_norms_l1_larger, marker = 'o', markersize=2, color=colors[exponent])
        ax.text(timerange[int(np.size(timerange)/6)], v_norms_l1_larger[int(np.size(v_norms_l1_larger)/6)]*vertical_offset_factors[exponent], r'$f = $ '+str(f), fontsize=15, color=colors[exponent])

        # line, = ax.plot(timerange, v_norms_l1_larger)
        ######################
        #For black and white plots
     
        markers = [None,None,None,None,None,None,None,'o']
        dashes = [(None,None),[5,5],[5,3,1,3],[5,2,5,2,5,10],[1,3],[5,3,1,2,1,10],[5,3,1,2,1,10],(None,None)]
        
        # line.set_color('k')
        if markers[exponent] != None:
            line.set_marker(markers[exponent])
        if dashes[exponent] != (None, None):
            line.set_dashes(dashes[exponent])
        
        ######################
        lines.append(line)
        labels.append(r'$f = $ '+str(f))
        
        
    ax.set_yscale('log')
    ax.set_ylim(1e-20,1e-2)
    ax.set_ylabel(r"$D(f;t)$", labelpad=10)
    endtime = int(T/2)
    offset = 300
    ax.set_xlim(0-offset, endtime+offset)
    ax.set_xlabel("time in s")

    file_path = Path(f"graphics/networks/onepipe_2v_nooutflow/larger_f_ell1norm_lineartime_endtime_{endtime}.pdf")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, bbox_inches = "tight",dpi=100)
    plt.close()

    
    
    
    
    
    
    
    
