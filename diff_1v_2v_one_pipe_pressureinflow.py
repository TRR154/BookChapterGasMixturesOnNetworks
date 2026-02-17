from network_2v_timedep_dxdtform_pressureinflow import Network_2v_time
from network_1v_timedep_dxdtform_pressureinflow import Network_1v_time
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import jax.numpy as jnp

import matplotlib

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
    "figure.titlesize": 20
})





def plot_1v_vs_2v(prop:str,prop_name:str,unit:str,algebraic:float,net_1:Network_1v_time,
                  net_2_list:list[Network_2v_time],f_list:list[float],save_name_fig:None):

    """
    plots the difference for one of the properties prop [rho_1,rho_2,pressure,m_1,m_2,v_1,v_2]
    Args:
        prop (str): prop
        prop_name (str): name of property
        unit (str): unit
        algebraic (float): constant in front of non linearity
        net_1 (Network_1v_time): 1v network
        net_2_list (list[Network_2v_time]): list of 2v networks
        f_list (list[float]): list of inner friction parameters
        save_name_fig (None): name of saving figure
        Defaults to None.
    """


    a_list = net_1.pipe_diameter**2*np.pi/4



    # create colormap
    for i,in_id in enumerate(net_1.pipe_in):

        p_net_1 = net_1.p_jax
        N_pipe = net_1.N_x_all_list[i]
        x = np.linspace(0,net_1.pipe_length[i],N_pipe)
        fig, ax = plt.subplots(constrained_layout=True)

        rho_1_net_1 = net_1.u_sol[i*3*N_pipe:(i*3+1)*N_pipe]
        rho_2_net_1 = net_1.u_sol[(i*3+1)*N_pipe:(i*3+2)*N_pipe]
        v_net_1 = net_1.u_sol[(i*3+2)*N_pipe:(i*3+3)*N_pipe]

        if prop == "rho_1": 
            value = rho_1_net_1
            ax.plot(x,value,label="1v",linewidth=2)

        elif prop == "rho_2": 
            value = rho_2_net_1
            ax.plot(x,value,label="1v",linewidth=2)


        elif prop == "v_1": 
            value = v_net_1
            ax.plot(x,value,label="1v",linewidth=2)

        elif prop == "v_2": 
            value = v_net_1
            ax.plot(x,value,label="1v",linewidth=2)

        elif prop == "pressure":
            value = p_net_1(rho_1_net_1,rho_2_net_1)*1e-5
            ax.plot(x,value,label="1v",linewidth=2)

        
        elif prop == "m_1":
            value = a_list[i]*rho_1_net_1*v_net_1
            ax.plot(x,value,label="1v",linewidth=2)

        elif prop == "m_2":
            value = a_list[i]*rho_2_net_1*v_net_1
            ax.plot(x,value,label="1v",linewidth=2)

        marker_list = ["x","o","D","^","s"]
        markevery = 2
        for k,net_2 in enumerate(net_2_list):
            rho_1_net_2 = net_2.u_sol[i*4*N_pipe:(i*4+1)*N_pipe]
            rho_2_net_2 = net_2.u_sol[(i*4+1)*N_pipe:(i*4+2)*N_pipe]
            v_1_net_2 = net_2.u_sol[(i*4+2)*N_pipe:(i*4+3)*N_pipe]
            v_2_net_2 = net_2.u_sol[(i*4+3)*N_pipe:(i*4+4)*N_pipe]
            p1_net_2 = net_2.p1_jax
            p2_net_2 = net_2.p2_jax

            if k < len(marker_list):
                marker = marker_list[k]
            else:
                marker = ""

            markevery +=1


            if prop == "rho_1": 
                ax.plot(x,rho_1_net_2,f"--{marker}",label=f"2v $f={f_list[k]}$",markevery=markevery,linewidth=2)

            elif prop == "rho_2": 
                ax.plot(x,rho_2_net_2,f"--{marker}",label=f"2v $f={f_list[k]}$",markevery=markevery,linewidth=2)

            elif prop == "v_1": 
                ax.plot(x,v_1_net_2,f"--{marker}",label=f"2v $f={f_list[k]}$",markevery=markevery,linewidth=2)

            elif prop == "v_2": 
                ax.plot(x,v_2_net_2,f"--{marker}",label=f"2v $f={f_list[k]}$",markevery=markevery,linewidth=2)
            
            elif prop == "pressure":
                value = p1_net_2(rho_1_net_2,rho_2_net_2)*1e-5+p2_net_2(rho_1_net_2,rho_2_net_2)*1e-5
                ax.plot(x,value,f"--{marker}",label=f"2v $f={f_list[k]}$",markevery=markevery,linewidth=2)

            elif prop == "m_1":
                value = a_list[i]*rho_1_net_2*v_1_net_2
                ax.plot(x,value,f"--{marker}",label=f"2v $f={f_list[k]}$",markevery=markevery,linewidth=2)

            elif prop == "m_2":
                value = a_list[i]*rho_2_net_2*v_2_net_2
                ax.plot(x,value,f"--{marker}",label=f"2v $f={f_list[k]}$",markevery=markevery,linewidth=2)

    ax.legend(loc="best")
    #ax.set_title(f"Difference of 1v vs 2v model for {prop_name} in {unit}")
    ax.set_xlabel(f"$x$ in m")
    ax.set_ylabel(f"{prop_name} in {unit}")
    if save_name_fig is not None:

        # Create a dummy figure just for the legend
        fig_legend = plt.figure(figsize=(4, 2))
        ax_legend = fig_legend.add_subplot(111)

        # Get handles and labels from your main axis
        handles, labels = ax.get_legend_handles_labels()

        # Create legend
        ax_legend.legend(
            handles,
            labels,
            ncol=len(handles),
            loc="center",
            frameon=False
        )

        # Remove axes
        ax_legend.axis("off")

        save_name = f"graphics/networks/1v_vs_2v/pressure_in_flow_{save_name_fig}_{prop}_{int(algebraic)}"

        # Save legend
        fig_legend.savefig(
            save_name+"_leg.pdf",
            dpi=300,
            bbox_inches="tight",
            transparent=True
        )

        plt.close(fig_legend)

        ax.get_legend().remove()

        fig.savefig(save_name+".pdf")
        plt.close(fig)


def plot_all_diff_network(net_1:Network_1v_time,net_2_list:list[Network_2v_time],
                          algebraic:float,f_list:list[float],save_name_fig:str=None):
    """
    plots all quantities for different pressures
    Args:
        net_1 (Network_1v_time): network 1 
        net_2_list (list[Network_2v_time]):list of networks 
        algebraic (float): constant in front of non linearity
        f_list (list[float]): list of inner friction parameters
        save_name_fig (str, optional): name under which figure is saved.
        The path to the figure is 
        graphics/networks/1v_vs_2v/{net_1.network_data}/pressure_in_flow_{save_name_fig}_\
            {in_node}_{out_node}_{prop}_{net_1.model}_{net_2.model}_{int(algebraic)}.pdf
        If None, then figure is not saved. Defaults to None.
    """
    
    plot_1v_vs_2v(prop="rho_1",prop_name="$\\rho_1$",unit="$\\frac{kg}{m^3}$",
                    algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,save_name_fig=save_name_fig,f_list=f_list)
    plot_1v_vs_2v(prop="rho_2",prop_name="$\\rho_2$",unit="$\\frac{kg}{m^3}$",
                    algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,save_name_fig=save_name_fig,f_list=f_list)
    plot_1v_vs_2v(prop="v_1",prop_name="$v_1$",unit="$\\frac{m}{s}$",
                    algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,save_name_fig=save_name_fig,f_list=f_list)
    plot_1v_vs_2v(prop="v_2",prop_name="$v_2$",unit="$\\frac{m}{s}$",
                    algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,save_name_fig=save_name_fig,f_list=f_list)
    plot_1v_vs_2v(prop="pressure",prop_name="$p$",unit="$bar$",
                    algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,save_name_fig=save_name_fig,f_list=f_list)
    plot_1v_vs_2v(prop="m_1",prop_name="$m_1$",unit="$\\frac{kg}{m^3s}$",
                    algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,save_name_fig=save_name_fig,f_list=f_list)
    plot_1v_vs_2v(prop="m_2",prop_name="$m_2$",unit="$\\frac{kg}{m^3s}$",
                    algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,save_name_fig=save_name_fig,f_list=f_list)
    #plot_1v_vs_2v(prop="h",prop_name="$h$",unit="bar",
    #               algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,save_name_fig=save_name_fig,f_list=f_list)


def plot_all_1v_vs_2v():
    """
    plots difference of the 1v vs 2v model for pressure inflow boundary conditions
    """

    file_network = Path("network_data" ,"optimization_data", "network_files", "one_pipe.net")
    file_data = Path("network_data", "optimization_data", "solution_files", "one_pipe.lsf")

    #####################################################################################
    # 1v sol
    
    scenarios = ["gaslib40_stationary", "time_dep"] 
    scenario = scenarios[0]
    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[0]
    dt = int(T/60)
    
    
    algebraic = 1.0 
    
    #Change to fit network
    candidate_dx = 200

    model ="speed_of_sound"
    algebraic = 1.0
    
    
    network_1v = Network_1v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=candidate_dx,
                              dt=dt,T=T)
    network_1v._precompute_pressure_law()
    network_1v.load_solution_network(algebraic=algebraic,scenario=scenario)
    

    #####################################################################################
    # 2v sol
    model ="speed_of_sound"
    algebraic = 1.0


    scenarios = ["gaslib11_stationary_1v", "time_dep"] 
    scenario = scenarios[0]
    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[0]
    dt = int(T/60)
    
    
    #Fixed computational parameters
    algebraic = 1.0 

    T = 60*60*24*150

    f_list = [0,0.01,0.1,0.2,0.5]
    network_2v_list = []
    for f in f_list:

        algebraic = 1.0
        network_2v = Network_2v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=candidate_dx,dt=dt,T=T)
        network_2v._precompute_pressure_law()
        network_2v.load_solution_network(algebraic=algebraic,scenario=scenario,f=f)
        network_2v_list.append(network_2v)



    plot_all_diff_network(net_1=network_1v,net_2_list=network_2v_list,f_list=f_list, algebraic=1.0,save_name_fig="diff_1v_2v_one_pipe")


if __name__ == "__main__":

    plot_all_1v_vs_2v()



