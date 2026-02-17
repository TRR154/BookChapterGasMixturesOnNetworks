import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

from pathlib import Path
from copy import deepcopy


from network_1v_timedep_dxdtform_massflowinflow import Network_1v_time as Network_1v_time_mass
from network_1v_timedep_dxdtform_pressureinflow import Network_1v_time as Network_1v_time_press


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



class PlotDiffClass():
    """
    This class plots the difference of networks, instantiated via Network_1v_time_mass or Network_1v_time_mass
    """



    @staticmethod
    def plot_sol_diff(prop:str,prop_name:str,unit:str,algebraic:float,
                      net_1:Network_1v_time_mass,net_2:Network_1v_time_mass,
                      mass_flow_bc:bool,save_name_fig:None,
                      relative_error:bool=False,save_legend:bool=False):
        """
        plots the property prop of the solution after computation of the solution,
        see Network_1v_time_mass.plot_sol_stationary for comparison

        Args:
            prop (str): property 
            prop_name (str): display name 
            unit (str): unit
            algebraic(float): constant in front of non linear term
            net_1 (Network_1v_time_mass): network
            net_2 (Network_1v_time_mass): network
            mass_flow_bc (bool): whether to use boundary conditions containing the mass inflow at 
            the boundary or the pressure_inflow
            save_name_fig (str, optional): name under which figure is saved.
            relative_error(bool, optional): whether to use a relative error
            save_legend (bool, optional): if legend should be saved. Defaults to False.

            The path to the figure is 
            graphics/networks/pressure_difference_1v/{bc_folder}/
            {save_name_fig}_{prop}_{net_1.model}_{net_2.model}_{int(algebraic)}_relative_error_{relative_error}.pdf
            If None, then figure is not saved. Defaults to None.
        """

        fig, ax = plt.subplots(constrained_layout=True)

        p_net_1 = net_1.p_jax
        p_net_2 = net_2.p_jax

        a_list = net_1.pipe_diameter**2*np.pi/4


        # create colormap
        prop_values = []
        N_x_all_list = net_1.N_x_all_list
        for i,in_id in enumerate(net_1.pipe_in):
            rho_1_net_1 = net_1.u_sol[3*np.sum(N_x_all_list[:i]):3*np.sum(N_x_all_list[:i])+N_x_all_list[i]]
            rho_2_net_1 = net_1.u_sol[3*np.sum(N_x_all_list[:i])+N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]]
            v_net_1 = net_1.u_sol[3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]]

            rho_1_net_2 = net_2.u_sol[3*np.sum(N_x_all_list[:i]):3*np.sum(N_x_all_list[:i])+N_x_all_list[i]]
            rho_2_net_2 = net_2.u_sol[3*np.sum(N_x_all_list[:i])+N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]]
            v_net_2 = net_2.u_sol[3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]]

            if prop == "rho_1": 
                value = rho_1_net_1 - rho_1_net_2
                if relative_error:
                    value /= rho_1_net_1

                prop_values.append(np.abs(value))

            elif prop == "rho_2": 
                value = rho_2_net_1 - rho_2_net_2
                if relative_error:
                    value /= rho_2_net_1
                prop_values.append(np.abs(value))

            elif prop == "v": 
                value = v_net_1-v_net_2
                if relative_error:
                    value /= v_net_1

                prop_values.append(np.abs(value))
            
            elif prop == "pressure":
                value = p_net_1(rho_1_net_1,rho_2_net_1)*1e-5-p_net_2(rho_1_net_2,rho_2_net_2)*1e-5
                if relative_error:
                    value /= p_net_1(rho_1_net_1,rho_2_net_1)*1e-5
                prop_values.append(np.abs(value))

            elif prop == "m_1":
                value = a_list[i]*(rho_1_net_1*v_net_1-rho_1_net_2*v_net_2)
                if relative_error:
                    value /= a_list[i]*(rho_1_net_1*v_net_1)

                prop_values.append(np.abs(value))

            elif prop == "m_2":
                value = a_list[i]*(rho_2_net_2*v_net_2-rho_2_net_2*v_net_2)
                if relative_error:
                    value /= a_list[i]*(rho_2_net_2*v_net_2)
                prop_values.append(np.abs(value))

            elif prop == "h":

                value = ((rho_1_net_1+rho_2_net_1)*v_net_1**2+p_net_1(rho_1_net_1,rho_2_net_1))*1e-5\
                        -((rho_1_net_2+rho_2_net_2)*v_net_2**2+p_net_2(rho_1_net_2,rho_2_net_2))*1e-5
                if relative_error:
                    value /= ((rho_1_net_1+rho_2_net_1)*v_net_1**2+p_net_1(rho_1_net_1,rho_2_net_1))*1e-5
                prop_values.append(np.abs(value))

        all_values = np.concatenate([prop for prop in prop_values])
        if np.min(all_values) < 0 and np.max(all_values) > 0:
            norm = mcolors.TwoSlopeNorm(vmin=np.min(all_values), vcenter=0, vmax=np.max(all_values))
            cmap = plt.get_cmap('seismic')
        else:
            norm = mcolors.Normalize(vmin=all_values.min(), vmax=all_values.max())
            cmap = cm.viridis


        
        for i,in_id in enumerate(net_1.pipe_in):
            out_id = net_1.pipe_out[i]

            node_in_index = net_1.node_id.index(in_id)
            node_out_index = net_1.node_id.index(out_id)
            x_0 = net_1.node_x_cord[node_in_index]
            x_N_1 = net_1.node_x_cord[node_out_index]
            y_0 = net_1.node_y_cord[node_in_index]           
            y_N_1 = net_1.node_y_cord[node_out_index]

            x_values = np.linspace(x_0,x_N_1,N_x_all_list[i]-1)
            y_values = np.linspace(y_0,y_N_1,N_x_all_list[i]-1)
            

            for j in range(len(x_values)-1):
                x_start = x_values[j]
                x_end = x_values[j+1]
                y_start = y_values[j]
                y_end = y_values[j+1]

                color = cmap(norm(prop_values[i][j]))
                ax.plot([x_start,x_end],[y_start,y_end],color = color,linewidth=5,zorder=1)

        index_source = net_1.node_type == "source"
        index_sink = net_1.node_type == "sink"
        index_in_nodes = net_1.node_type == "in_node"
        ax.scatter(np.array(net_1.node_x_cord)[index_source],
                 np.array(net_1.node_y_cord)[index_source],color="r",
                 label="$\\Gamma_p$",marker="o",
                 s=80,zorder=2)
        ax.scatter(np.array(net_1.node_x_cord)[index_in_nodes],
                 np.array(net_1.node_y_cord)[index_in_nodes],color="b",
                  marker="s", label="$\\Gamma_q$",s=80,zorder=2)
        ax.scatter(np.array(net_1.node_x_cord)[index_sink],
                    np.array(net_1.node_y_cord)[index_sink],
                    marker="s", color="b",s=80,zorder=2)

        #for i,node_id in enumerate(net_1.node_id):
        #    ax.annotate(node_id,(net_1.node_x_cord[i],net_1.node_y_cord[i]+0.5),ha="center")

        ax.legend(loc="lower right")
        ax.set_xticks([])
        ax.set_yticks([])
        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.set_yscale('linear')



        if not relative_error:
            cbar.set_label(f"{prop_name} in {unit}")


        ## Set ticks explicitly
        num_ticks =  8
        ticks = np.linspace(np.min(all_values), np.max(all_values), num_ticks)
        cbar.set_ticks(ticks)                     # set tick positions
        cbar.set_ticklabels([f"{t:.2g}" for t in ticks]) 


        #difference_name = "Relative" if relative_error else "Absolute"

        #ax.set_title(f"{difference_name} difference of {prop_name} in {unit} of \n {net_1.model_name} and {net_2.model_name}")
        if save_name_fig is not None:
            bc_folder = "massflowinflow" if mass_flow_bc else "pressureinflow"
            save_name = f"graphics/networks/pressure_difference_1v/{bc_folder}/{save_name_fig}_{prop}_{net_1.model}_{net_2.model}_{int(algebraic)}_relative_error_{relative_error}"
            save_name_path = Path(save_name) 
            save_name_path.parent.mkdir(parents=True, exist_ok=True)
            

            if save_legend:

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


                fig_legend.savefig(
                    save_name+"_leg.pdf",
                    dpi=300,
                    bbox_inches="tight",
                    transparent=True
                )

                plt.close(fig_legend)

            ax.get_legend().remove()

            fig.savefig(save_name+".pdf",backend="pgf")


    @staticmethod
    def plot_all_diff_network(net_1:Network_1v_time_mass,net_2:Network_1v_time_mass,
                              algebraic:float,mass_flow_bc:bool,save_name_fig:str=None,
                              relative_error:bool=None,save_legend:bool=False):
        """
        plots all quantities 
        Args:
            net_1 (Network_1v_time_mass): network 1 
            net_2 (Network_1v_time_mass): network 2
            algebraic (float): constant in front of non linearity
            mass_flow_bc (bool): whether to use boundary conditions containing the mass inflow at 
            save_name_fig (str, optional): name under which figure is saved.
            The path to the figure is 
            "graphics/networks/models/{save_name_fig}_{prop}_{self.model}_{int(algebraic)}.png"
            If None, then figure is not saved. Defaults to None.
            relative_error(bool, optional): whether to use a relative error
            save_legend (bool, optional): if legend should be saved. Defaults to False.
        """
        
        PlotDiffClass.plot_sol_diff(prop="rho_1",prop_name="$\\rho_1$",unit="$\\frac{kg}{m^3}$",
                                       algebraic=algebraic,net_1=net_1,net_2=net_2,save_name_fig=save_name_fig,relative_error=relative_error,mass_flow_bc=mass_flow_bc,save_legend=save_legend)
        plt.close()
        PlotDiffClass.plot_sol_diff(prop="rho_2",prop_name="$\\rho_2$",unit="$\\frac{kg}{m^3}$",
                                       algebraic=algebraic,net_1=net_1,net_2=net_2,save_name_fig=save_name_fig,relative_error=relative_error,mass_flow_bc=mass_flow_bc,save_legend=save_legend)
        plt.close()
        PlotDiffClass.plot_sol_diff(prop="v",prop_name="$v$",unit="$\\frac{m}{s}$",
                                       algebraic=algebraic,net_1=net_1,net_2=net_2,save_name_fig=save_name_fig,relative_error=relative_error,mass_flow_bc=mass_flow_bc,save_legend=save_legend)
        plt.close()
        PlotDiffClass.plot_sol_diff(prop="pressure",prop_name="$p$",unit="$bar$",
                                       algebraic=algebraic,net_1=net_1,net_2=net_2,save_name_fig=save_name_fig,relative_error=relative_error,mass_flow_bc=mass_flow_bc,save_legend=save_legend)
        plt.close()
        #PlotDiffClass.plot_sol_diff(prop="m_1",prop_name="$m_1$",unit="$\\frac{kg}{m^3s}$",
        #                               algebraic=algebraic,net_1=net_1,net_2=net_2,save_name_fig=save_name_fig,relative_error=relative_error,save_legend=save_legend)
        #plt.close()
        #PlotDiffClass.plot_sol_diff(prop="m_2",prop_name="$m_2$",unit="$\\frac{kg}{m^3s}$",
        #                               algebraic=algebraic,net_1=net_1,net_2=net_2,save_name_fig=save_name_fig,relative_error=relative_error,save_legend=save_legend)
        #plt.close()
        #PlotDiffClass.plot_sol_diff(prop="h",prop_name="$h$",unit="bar",
        #                               algebraic=algebraic,net_1=net_1,net_2=net_2,save_name_fig=save_name_fig,relative_error=relative_error,save_legend=save_legend)

        
# Non class methods    
def plot_pressures_different(file_network:Path, file_data:Path,save_name_fig:str,
                             relative_error:bool,mass_flow_bc:bool):
    """

    Args:
        file_network (Path): network file
        file_data (Path): network file
        mass_flow_bc (bool): whether to use boundary conditions containing the mass inflow at 
        relative_error(bool, optional): whether to use a relative error
    """


    if mass_flow_bc:
        Image_network_1v = Network_1v_time_mass
    else:
        Image_network_1v = Network_1v_time_press


    network_dict = {}

    model_list = ["gerg", "gerg_fit","virial_expansion_mix","virial_expansion","speed_of_sound"]
    model_name_list = ["gerg","ls fit", "virial mix","virial","ideal gas"]
    #model_list = ["virial_expansion","speed_of_sound"]

    stationary_or_instationary = 0
    
    scenario = "gaslib40_stationary" 
    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[stationary_or_instationary]
    dt = int(T/60)
    
    
    
    candidate_dx = 1000



    for i,model in enumerate(model_list):
        network = Image_network_1v(file_network=file_network,file_data=file_data,model=model,
                                   candidate_dx=candidate_dx, dt=dt,T=T,model_name=model_name_list[i])
        network.load_solution_network(algebraic=1.0,scenario=scenario)
        network._precompute_pressure_law()
        network_dict[model] = deepcopy(network)

    for model in model_list:
        network  = network_dict[model]
        network.plot_all_stationary(algebraic=1.0,scenario=scenario)
        
    model_pairs_distinct = [ (model_1,model_2) for i,model_1 in enumerate(model_list) for j,model_2 in enumerate(model_list) if i<j]
    for model_1,model_2 in model_pairs_distinct:
        PlotDiffClass.plot_all_diff_network(network_dict[model_1],network_dict[model_2],algebraic=1,
                                               save_name_fig=save_name_fig,relative_error=relative_error,
                                               mass_flow_bc=mass_flow_bc)
        plt.close()
    




def plot_algebraic_vs_speed_of_sound(file_network:Path, file_data:Path, save_name_fig:str,relative_error:bool,mass_flow_bc:bool):
    """
    plots the difference of algebraic and non algebraic solution

    Args:
        file_network (Path): network file
        file_data (Path): network file
        mass_flow_bc (bool): whether to use boundary conditions containing the mass inflow at 
        relative_error(bool, optional): whether to use a relative error
    """

    if mass_flow_bc:
        Image_network_1v = Network_1v_time_mass
    else:
        Image_network_1v = Network_1v_time_press
    stationary_or_instationary = 0
    
    scenario = "gaslib40_stationary" 
    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[stationary_or_instationary]
    dt = int(T/60)
    
    
    candidate_dx = 1000
     
    network_algebraic = Image_network_1v(file_network=file_network,file_data=file_data,model="speed_of_sound",
                                   candidate_dx=candidate_dx, dt=dt,T=T,model_name="ideal gas")
    network_algebraic.load_solution_network(algebraic=0.0,scenario=scenario)
    network_algebraic._precompute_pressure_law()

    network_ideal = Image_network_1v(file_network=file_network,file_data=file_data,model="speed_of_sound",
                                   candidate_dx=candidate_dx, dt=dt,T=T,model_name="ideal gas")
    network_ideal.load_solution_network(algebraic=1.0,scenario=scenario)

    network_ideal._precompute_pressure_law()
    
    PlotDiffClass.plot_all_diff_network(network_algebraic,network_ideal,algebraic=0,save_name_fig=save_name_fig,
                                        relative_error=relative_error,mass_flow_bc=mass_flow_bc)
    network_algebraic.plot_all_stationary(algebraic=0,scenario=scenario)
    network_ideal.plot_all_stationary(algebraic=1,scenario=scenario)



def plot_algebraic_vs_speed_of_sound_and_pressure_law_difference():
    """
    plots algebraic vs non-algebraic and all pressure law differences
    """
    file_network = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.net")
    file_data = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.lsf")
    save_name_fig = "stationary_gaslib_40_reduced"

    mass_flow_bc = True
    relative_error_list = [True,False]
    for relative_error in relative_error_list:

        ## plots different pressure laws
        plot_pressures_different(file_network, file_data,save_name_fig=save_name_fig,
                                relative_error=relative_error,mass_flow_bc=mass_flow_bc)

        ## plots the difference of algebraic and speed of sound model
        plot_algebraic_vs_speed_of_sound(file_network,file_data,save_name_fig=save_name_fig,
                                        relative_error=relative_error,mass_flow_bc=mass_flow_bc)

def plot_all_algebraic_vs_speed_of_sound():
    """
    plots algebraic vs non-algebraic
    """
    file_network = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.net")
    file_data = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.lsf")
    save_name_fig = "stationary_gaslib_40_reduced"

    mass_flow_bc = True
    relative_error_list = [True,False]
    for relative_error in relative_error_list:

        ## plots the difference of algebraic and speed of sound model
        plot_algebraic_vs_speed_of_sound(file_network,file_data,save_name_fig=save_name_fig,
                                        relative_error=relative_error,mass_flow_bc=mass_flow_bc)

def plot_all_pressure_law_difference():
    """
    plots algebraic vs non-algebraic
    """
    file_network = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.net")
    file_data = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.lsf")
    save_name_fig = "stationary_gaslib_40_reduced"

    mass_flow_bc = True
    relative_error_list = [True,False]
    for relative_error in relative_error_list:

        ## plots different pressure laws
        plot_pressures_different(file_network,file_data,save_name_fig=save_name_fig,
                                        relative_error=relative_error,mass_flow_bc=mass_flow_bc)



if __name__ == "__main__":

    plot_algebraic_vs_speed_of_sound_and_pressure_law_difference()

