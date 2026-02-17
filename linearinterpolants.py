"""
This Python program contains code for converting numerical solutions on grid points to linear interpolants,
computing the interpolant values on a finer grid, and storing the results, as well as code for loading the 
stored interpolant values and using these to compute rates of convergence. This is applicable to both the 
one- and two-velocity models. 
"""
import numpy as np
import pickle

from pathlib import Path
from network_1v_timedep_dxdtform_massflowinflow import Network_1v_time
from network_2v_timedep_dxdtform_massflowinflow import Network_2v_time
import scipy
import sigfig

#################################
#LOAD SOLUTIONS
#################################


def load_solutions(choice_of_network:int,scenario:str,candidate_dx:float,dt:float,T:float,model:str,algebraic:float,two_velocity:bool=False,f:float=None):
    network_files = ["3mixT","gaslib40_edit","gaslib40_removed_edit","one_pipe","testm_new"]
    network_file = network_files[choice_of_network]
    
    # # init_and_bdry_data = ['3mixT_scen_1', 'gaslib40-gaslib40m', 'gaslib60-gaslib60m', 'gaslib60-gaslib80m', 'one_pipe', 'testm-testm', 'testm-testm_60_b', 'testm-testm_60_bar_mu_diff', 'testm-testm_60_fast_1_gas', 'testm-testm_70_b', 'testm-testm_70_b_mu_diff', 'testm-testm_90_b', 'testm-testm_new', 'testm-testm_z_1'] #full list, too long
    init_and_bdry_data = ['3mix_scen_1', 'gaslib40_edit','gaslib40_removed_edit','one_pipe','testm-testm_new']
    init_and_bdry_choice = init_and_bdry_data[choice_of_network]

    if two_velocity == False:
        if scenario == "time_dep":
            with open(f"save_solution/solutions/network_1v_timedep/massflowinflow/1v_{init_and_bdry_choice}_dx_{candidate_dx}_dt_{dt}_T_{T}_SCENARIO_{scenario}_pressure_{model}_algebraic_{algebraic}.pkl", "rb") as file:
                u_store = pickle.load(file)
        else:
            with open(f"save_solution/solutions/network_1v_timedep/1v_{init_and_bdry_choice}_dx_{candidate_dx}_SCENARIO_{scenario}_pressure_{model}_algebraic_{algebraic}.pkl", "rb") as file:
                u_sol = pickle.load(file)

        
    else:
        if f == None:
            raise Exception("Require inner friction for two-velocity solutions") 
        if scenario == "time_dep":
            file_path = Path(f"save_solution/solutions/network_2v_f_{f}_timedep/massflowinflow/2v_{init_and_bdry_choice}_dx_{candidate_dx}_dt_{dt}_T_{T}_SCENARIO_{scenario}_pressure_{model}_algebraic_{algebraic}.pkl")
            with open(file_path, "rb") as file:
                u_store = pickle.load(file)
        else:
            file_path = Path(f"save_solution/solutions/network_2v_f_{f}_timedep/2v_{init_and_bdry_choice}_dx_{candidate_dx}_SCENARIO_{scenario}_pressure_{model}_algebraic_{algebraic}.pkl")
            with open(file_path, "rb") as file:
                u_sol = pickle.load(file)
        
    return u_store



################
#INTERPOLATION
################




def linear_interpolant_pipe(N_x_all_list:np.ndarray, pipe_index:int, gridvalues:np.ndarray, interpolation_data:np.ndarray,two_velocity:bool=False):
    
    if two_velocity == False:
    
        rho1vals_pipe = interpolation_data[:,3*np.sum(N_x_all_list[:pipe_index]):3*np.sum(N_x_all_list[:pipe_index])+N_x_all_list[pipe_index]]
        rho2vals_pipe = interpolation_data[:,3*np.sum(N_x_all_list[:pipe_index])+N_x_all_list[pipe_index]:3*np.sum(N_x_all_list[:pipe_index])+2*N_x_all_list[pipe_index]]
        vvals_pipe = interpolation_data[:,3*np.sum(N_x_all_list[:pipe_index])+2*N_x_all_list[pipe_index]:3*np.sum(N_x_all_list[:pipe_index])+3*N_x_all_list[pipe_index]]
        
        
        rho1_interpolant = lambda x: scipy.interpolate.interpn(points=gridvalues, values=rho1vals_pipe, xi=[x])[0]
        rho2_interpolant = lambda x: scipy.interpolate.interpn(points=gridvalues, values=rho2vals_pipe, xi=[x])[0]
        v_interpolant = lambda x: scipy.interpolate.interpn(points=gridvalues, values=vvals_pipe, xi=[x])[0]
        
        return [rho1_interpolant, rho2_interpolant, v_interpolant]
        
    else:
        
        rho1vals_pipe = interpolation_data[:,4*np.sum(N_x_all_list[:pipe_index]):4*np.sum(N_x_all_list[:pipe_index])+N_x_all_list[pipe_index]]
        rho2vals_pipe = interpolation_data[:,4*np.sum(N_x_all_list[:pipe_index])+N_x_all_list[pipe_index]:4*np.sum(N_x_all_list[:pipe_index])+2*N_x_all_list[pipe_index]]
        v1vals_pipe = interpolation_data[:,4*np.sum(N_x_all_list[:pipe_index])+2*N_x_all_list[pipe_index]:4*np.sum(N_x_all_list[:pipe_index])+3*N_x_all_list[pipe_index]]
        v2vals_pipe = interpolation_data[:,4*np.sum(N_x_all_list[:pipe_index])+3*N_x_all_list[pipe_index]:4*np.sum(N_x_all_list[:pipe_index])+4*N_x_all_list[pipe_index]]
        
        rho1_interpolant = lambda x: scipy.interpolate.interpn(points=gridvalues, values=rho1vals_pipe, xi=[x])[0]
        rho2_interpolant = lambda x: scipy.interpolate.interpn(points=gridvalues, values=rho2vals_pipe, xi=[x])[0]
        v1_interpolant = lambda x: scipy.interpolate.interpn(points=gridvalues, values=v1vals_pipe, xi=[x])[0]
        v2_interpolant = lambda x: scipy.interpolate.interpn(points=gridvalues, values=v2vals_pipe, xi=[x])[0]
        
        return [rho1_interpolant, rho2_interpolant, v1_interpolant, v2_interpolant]








def linear_interpolant_values_new(choice_of_network:int,scenario:str,candidate_dx:float,dt:float,T:float,model:str,algebraic:float,exponent_groundtruth:int,scale_exponent:int,two_velocity:bool=False,f:float=None):
    print("time horizon = "+str(T))
    network_files = ["3mixT","gaslib40_edit","gaslib40_removed_edit","one_pipe","testm_new"]
    network_file = network_files[choice_of_network]
    file_network = Path("network_data" ,"optimization_data", "network_files", str(network_file)+".net")
    
    init_and_bdry_data = ['3mix_scen_1', 'gaslib40_edit','gaslib40_removed_edit','one_pipe','testm-testm_new']
    init_and_bdry_choice = init_and_bdry_data[choice_of_network]
    file_data = Path("network_data" ,"optimization_data", "solution_files", str(init_and_bdry_choice)+".lsf")
    
    exponent_groundtruth = exponent_groundtruth
    candidate_dt_groundtruth = (1.5**-exponent_groundtruth)*dt
    real_dt_groundtruth = T/round(T/candidate_dt_groundtruth)

    
    if two_velocity == False:
        variables = ['rho_1','rho_2', 'v']
        
        network_groundtruth = Network_1v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=(1.5**-exponent_groundtruth)*candidate_dx,
                                     dt=real_dt_groundtruth,T=T)
        
        time_linspace_groundtruth = np.linspace(0,network_groundtruth.T,round(network_groundtruth.T/real_dt_groundtruth + 1))
        
        scale = 1.5**(-scale_exponent)
        candidate_dt_interpolant = scale*dt
            
        real_dt_interpolant = T/round(T/candidate_dt_interpolant)
        
        network_interpolant = Network_1v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=scale*candidate_dx,
                                 dt=scale*dt,T=T)


        time_linspace_interpolant = np.linspace(0,network_interpolant.T,round(network_interpolant.T/real_dt_interpolant + 1))
        
        data_interpolant = load_solutions(choice_of_network=choice_of_network,scenario=scenario,candidate_dx=scale*candidate_dx,dt=real_dt_interpolant,T=T,model=model,algebraic=algebraic,two_velocity=two_velocity,f=f)
        
    else: 
        variables = ['rho_1','rho_2', 'v_1','v_2']
        network_groundtruth = Network_2v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=(1.5**-exponent_groundtruth)*candidate_dx,
                                    dt=real_dt_groundtruth,T=T)
        
        time_linspace_groundtruth = np.linspace(0,network_groundtruth.T,round(network_groundtruth.T/real_dt_groundtruth + 1))
        
        scale = 1.5**(-scale_exponent)
        candidate_dt_interpolant = scale*dt
            
        real_dt_interpolant = T/round(T/candidate_dt_interpolant)
        
        network_interpolant = Network_2v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=scale*candidate_dx,
                                dt=scale*dt,T=T)
        time_linspace_interpolant = np.linspace(0,network_interpolant.T,round(network_interpolant.T/real_dt_interpolant + 1))        
        data_interpolant = load_solutions(choice_of_network=choice_of_network,scenario=scenario,candidate_dx=scale*candidate_dx,dt=real_dt_interpolant,T=T,model=model,algebraic=algebraic,two_velocity=two_velocity,f=f)

    interpolant_values_network = [np.zeros((np.size(time_linspace_groundtruth), np.sum(network_groundtruth.N_x_all_list))) for var_index in range(len(variables))]
        
        

    

    for pipe_index in range(np.size(network_interpolant.N_x_all_list)):   
        if pipe_index % 10 == 0:
            print(f"Currently at pipe {pipe_index}") #This helps give an indication of the time the computation will take.
        pipe_linspace_interpolant = np.linspace(0, network_interpolant.pipe_length[pipe_index],network_interpolant.N_x_all_list[pipe_index])
        pipe_linspace_groundtruth = np.linspace(0, network_groundtruth.pipe_length[pipe_index],network_groundtruth.N_x_all_list[pipe_index])
        
        for time_index in range(np.size(time_linspace_groundtruth)):

            interpolant_functions_pipe = linear_interpolant_pipe(N_x_all_list=network_interpolant.N_x_all_list, pipe_index=pipe_index, gridvalues=(time_linspace_interpolant,pipe_linspace_interpolant), interpolation_data=data_interpolant,two_velocity=two_velocity)

            for var_index in range(len(variables)):    

                for space_index in range(np.size(pipe_linspace_groundtruth)):
                    interpolant_values_network[var_index][time_index,np.sum(network_groundtruth.N_x_all_list[:pipe_index])+space_index] = interpolant_functions_pipe[var_index]([time_linspace_groundtruth[time_index], pipe_linspace_groundtruth[space_index]])

                
            
    if two_velocity == False:
        file_path = "save_solution/solutions/network_1v_timedep/compstudy/"
        with open(Path(file_path+f"1v_gaslib_compstudy_groundtruthexp_{exponent_groundtruth}_scaleexp_{scale_exponent}_interpolant_values.pkl"), "wb") as file:
            pickle.dump(interpolant_values_network, file)
    else: 
        file_path = f"save_solution/solutions/network_2v_f_{f}_timedep/compstudy/"
        with open(Path(file_path+f"2v_gaslib_compstudy_groundtruthexp_{exponent_groundtruth}_scaleexp_{scale_exponent}_interpolant_values.pkl"), "wb") as file:
            pickle.dump(interpolant_values_network, file)
            
  



def linear_interpolant_norms(T:float,max_exponent:int,scale_exponent:int,space_norm_order:float,time_norm_order:float,two_velocity:bool=False,f:float=None):
    file_network = Path("network_data" ,"optimization_data", "network_files", "gaslib40_removed_edit.net")
    
    file_data = Path("network_data" ,"optimization_data", "solution_files", "gaslib40_removed_edit.lsf")
        
    network_groundtruth = Network_1v_time(file_network=file_network,file_data=file_data,model="speed_of_sound",candidate_dx=(1.5**-max_exponent)*1500,
                                  dt=(1.5**-max_exponent)*720,T=T)
    
    differences = []
    if two_velocity == False: #1v case
        variables = ['rho_1', 'rho_2', 'v']
        try:
            with open(Path(f"save_solution/solutions/network_1v_timedep/compstudy/1v_gaslib_compstudy_groundtruthexp_{max_exponent}_scaleexp_{scale_exponent}_interpolant_values.pkl"), "rb") as file:
                interpolant_values_sigma = pickle.load(file)
            with open(Path(f"save_solution/solutions/network_1v_timedep/compstudy/1v_gaslib_compstudy_groundtruthexp_{max_exponent}_scaleexp_{max_exponent}_interpolant_values.pkl"), "rb") as file:
                interpolant_values_sigma_max = pickle.load(file)
        except:
            raise Exception("Interpolant values need to be computed")         
    else: #2v case
        variables = ['rho_1', 'rho_2', 'v_1', 'v_2']
        try:
            with open(Path(f"save_solution/solutions/network_2v_f_{f}_timedep/compstudy/2v_gaslib_compstudy_groundtruthexp_{max_exponent}_scaleexp_{scale_exponent}_interpolant_values.pkl"), "rb") as file:
                interpolant_values_sigma = pickle.load(file)
            with open(Path(f"save_solution/solutions/network_2v_f_{f}_timedep/compstudy/2v_gaslib_compstudy_groundtruthexp_{max_exponent}_scaleexp_{max_exponent}_interpolant_values.pkl"), "rb") as file:
                interpolant_values_sigma_max = pickle.load(file)
        except:
            raise Exception("Interpolant values need to be computed")


    for var_index in range(len(variables)):
        differences.append(interpolant_values_sigma[var_index] - interpolant_values_sigma_max[var_index])

    nrpipes = np.size(network_groundtruth.N_x_all_list)
    nr_timegridpoints = np.shape(differences)[1]
    
    space_norms_perpipe_differences = [np.zeros((nr_timegridpoints,nrpipes)) for var_index in range(len(variables))]
    space_norms_perpipe_sigma_max = [np.zeros((nr_timegridpoints,nrpipes)) for var_index in range(len(variables))]

    space_norms_network_differences = [[] for var_index in range(len(variables))]
    space_norms_network_sigma_max = [[] for var_index in range(len(variables))]
    
        
    for time_index in range(nr_timegridpoints):    
        for pipe_index in range(nrpipes):
            for var_index in range(len(variables)):
                space_norms_perpipe_differences[var_index][time_index, pipe_index] = np.linalg.norm(differences[var_index][time_index,int(np.sum(network_groundtruth.N_x_all_list[:pipe_index])):int(np.sum(network_groundtruth.N_x_all_list[:pipe_index+1]))], ord=space_norm_order)
                space_norms_perpipe_sigma_max[var_index][time_index, pipe_index] = np.linalg.norm(interpolant_values_sigma_max[var_index][time_index,int(np.sum(network_groundtruth.N_x_all_list[:pipe_index])):int(np.sum(network_groundtruth.N_x_all_list[:pipe_index+1]))], ord=space_norm_order)


        for var_index in range(len(variables)):
            space_norms_network_differences[var_index].append(np.linalg.norm(space_norms_perpipe_differences[var_index][time_index, :], ord=space_norm_order)) 
            space_norms_network_sigma_max[var_index].append(np.linalg.norm(space_norms_perpipe_sigma_max[var_index][time_index, :], ord=space_norm_order))

    time_norm_space_norm_differences = np.array([np.linalg.norm(space_norms_network_differences[var_index],ord=time_norm_order) for var_index in range(len(variables))])
    time_norm_space_norm_sigma_max = np.array([np.linalg.norm(space_norms_network_sigma_max[var_index],ord=time_norm_order) for var_index in range(len(variables))])

    relative_norms = time_norm_space_norm_differences/time_norm_space_norm_sigma_max
    
    return relative_norms



def norms_of_diffs_and_eoc(max_exponent:int,space_norm_order:float,time_norm_order:float,offset:int,two_velocity:bool=False,f:float=None):
    print("")
    print("space norm order = "+str(space_norm_order))
    print("time norm order = "+str(time_norm_order))
    
    if two_velocity == False:
        normstring="One-velocity interpolation norms"
        variables = ['rho_1', 'rho_2', 'v']

    else:
        normstring="Two-velocity interpolation norms"
        variables = ['rho_1', 'rho_2', 'v_1', 'v_2']

    print("")
    print(normstring)
    norms = [[] for var_index in range(len(variables))]
    eoc = [[] for var_index in range(len(variables))]
        
    for scale_exponent in range(max_exponent-offset):
        print("scale exponent -"+str(scale_exponent+offset))
        norms_sigma = linear_interpolant_norms(T=int(60*720/3), max_exponent=max_exponent,scale_exponent=scale_exponent+offset, space_norm_order=space_norm_order, time_norm_order=time_norm_order,two_velocity=two_velocity,f=f)
        for var_index in range(len(variables)):
            norms[var_index].append(sigfig.round(float(norms_sigma[var_index]),sigfigs=3))
        if scale_exponent+offset <= max_exponent-2:
            norms_sigma_plus_1 = linear_interpolant_norms(T=int(60*720/3), max_exponent=max_exponent, scale_exponent=scale_exponent+1+offset, space_norm_order=space_norm_order, time_norm_order=time_norm_order,two_velocity=two_velocity,f=f)
            for var_index in range(len(variables)):
                eoc[var_index].append(sigfig.round(float(np.log2(norms_sigma[var_index]/norms_sigma_plus_1[var_index])/np.log2(1.5)), sigfigs=3))

                
    for var_index in range(len(variables)):
        print("")
        print("Norms of "+variables[var_index])
        print([np.format_float_scientific(value) for value in norms[var_index]])
        print("EOC of "+variables[var_index])
        print(eoc[var_index])
        
        

