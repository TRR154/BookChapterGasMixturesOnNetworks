from network_2v_timedep_dxdtform_massflowinflow import Network_2v_time
from network_1v_timedep_dxdtform_massflowinflow import Network_1v_time
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib
import numpy as np




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


def plot_1v_vs_2v(prop:str,prop_name:str,unit:str,
                  algebraic:float,net_1:Network_1v_time,
                  net_2_list:list[Network_2v_time],
                  f_list:list[float],save_name_fig:None,
                  replace_node_name:list[str]=None,
                  save_legend:bool=False):
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
        replace_node_name (list[str], optional): list of names, which replace the name of nodes.
        This list should be ordered in the same way as the list of net_1.nodes_id 
        and have the same amount of elements.
        Defaults to None.
        save_legend (bool, optional): if legend should be saved. Defaults to False.
    """



    a_list = net_1.pipe_diameter**2*np.pi/4



    # create colormap
    for i,in_id in enumerate(net_1.pipe_in):

        p_net_1 = net_1.p_jax
        N_x_all_list = net_1.N_x_all_list
        x = np.linspace(0,net_1.pipe_length[i],N_x_all_list[i])
        fig, ax = plt.subplots(constrained_layout=True)

        rho_1_net_1 = net_1.u_sol[3*np.sum(N_x_all_list[:i]):3*np.sum(N_x_all_list[:i])+N_x_all_list[i]]
        rho_2_net_1 = net_1.u_sol[3*np.sum(N_x_all_list[:i])+N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]]
        v_net_1 = net_1.u_sol[3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]]

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
        markevery = 3
        for k,net_2 in enumerate(net_2_list):
            rho_1_net_2 = net_2.u_sol[4*np.sum(N_x_all_list[:i]):4*np.sum(N_x_all_list[:i])+N_x_all_list[i]]
            rho_2_net_2 = net_2.u_sol[4*np.sum(N_x_all_list[:i])+N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]]
            v_1_net_2 = net_2.u_sol[4*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]]
            v_2_net_2 = net_2.u_sol[4*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+4*N_x_all_list[i]]
            p1_net_2 = net_2.p1_jax
            p2_net_2 = net_2.p2_jax


            if k < len(marker_list):
                marker = marker_list[k]
            else:
                marker = ""
            markevery += 1

            if prop == "rho_1": 
                ax.plot(x,rho_1_net_2,"--",label=f"2v $f={f_list[k]}$",linewidth=2,marker=marker,markevery=markevery)

            elif prop == "rho_2": 
                ax.plot(x,rho_2_net_2,"--",label=f"2v $f={f_list[k]}$",linewidth=2,marker=marker,markevery=markevery)

            elif prop == "v_1": 
                ax.plot(x,v_1_net_2,"--",label=f"2v $f={f_list[k]}$",linewidth=2,marker=marker,markevery=markevery)

            elif prop == "v_2": 
                ax.plot(x,v_2_net_2,"--",label=f"2v $f={f_list[k]}$",linewidth=2,marker=marker,markevery=markevery)
            
            elif prop == "pressure":
                value = p1_net_2(rho_1_net_2,rho_2_net_2)*1e-5+p2_net_2(rho_1_net_2,rho_2_net_2)*1e-5
                ax.plot(x,value,"--",label=f"2v $f={f_list[k]}$",linewidth=2,marker=marker,markevery=markevery)

            elif prop == "m_1":
                value = a_list[i]*rho_1_net_2*v_1_net_2
                ax.plot(x,value,"--",label=f"2v $f={f_list[k]}$",linewidth=2,marker=marker,markevery=markevery)

            elif prop == "m_2":
                value = a_list[i]*rho_2_net_2*v_2_net_2
                ax.plot(x,value,"--",label=f"2v $f={f_list[k]}$",linewidth=2,marker=marker,markevery=markevery)



        ax.legend(loc="best")
        #ax.set_title(f"Difference of 1v vs 2v model for {prop_name} in {unit}")
        ax.set_xlabel(f"$x$ in m")
        ax.set_ylabel(f"{prop_name} in {unit}")
        if save_name_fig is not None:
            if replace_node_name is None:
                save_name = f"graphics/networks/1v_vs_2v/{net_1.network_data}/massflowinflow_{save_name_fig}_{net_1.pipe_id[i]}_{net_1.pipe_in[i]}_{net_2.pipe_out[i]}_{prop}_{net_1.model}_{net_2.model}_{int(algebraic)}"
            else:
                in_node = replace_node_name[net_1.node_id.index(net_1.pipe_in[i])]
                out_node = replace_node_name[net_1.node_id.index(net_1.pipe_out[i])]
                file_path = Path(f"graphics/networks/1v_vs_2v/{net_1.network_data}/massflowinflow_{save_name_fig}_{in_node}_{out_node}_{prop}_{net_1.model}_{net_2.model}_{int(algebraic)}.pdf")
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if i == 0:
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

                # Save legend
                save_name = Path(f"graphics/networks/1v_vs_2v/{net_1.network_data}/massflowinflow_{save_name_fig}_leg.pdf")
                fig_legend.savefig(
                    save_name,
                    dpi=300,
                    bbox_inches="tight",
                    transparent=True
                )

                plt.close(fig_legend)

            ax.get_legend().remove()

            fig.savefig(file_path,backend="pgf")
        plt.close()


def plot_all_diff_network(net_1:Network_1v_time,net_2_list:list[Network_2v_time],
                          algebraic:float,f_list:list[float],save_name_fig:str=None,
                          replace_node_name:list[str]=None,
                          save_legend:bool=False):
    """
    plots all quantities for different pressures
    Args:
        net_1 (Network_1v_time): network 1 
        net_2_list (list[Network_2v_time]):list of networks 
        algebraic (float): constant in front of non linearity
        f_list (list[float]): list of inner friction parameters
        replace_node_name (list[str], optional): list of names, which replace the name of nodes.
        This list should be ordered in the same way as the list of net_1.nodes_id 
        and have the same amount of elements.
        save_name_fig (str, optional): name under which figure is saved.
        save_legend (bool, optional): if legend should be saved. Defaults to False.
        The path to the figure is 
        graphics/networks/1v_vs_2v/{net_1.network_data}/massflowinflow_{save_name_fig}_\
            {in_node}_{out_node}_{prop}_{net_1.model}_{net_2.model}_{int(algebraic)}.pdf
        If None, then figure is not saved. Defaults to None.
    """

    
    plot_1v_vs_2v(prop="rho_1",prop_name="$\\rho_1$",unit="$\\frac{kg}{m^3}$",
                    algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,
                    save_name_fig=save_name_fig,f_list=f_list,replace_node_name=replace_node_name,save_legend=save_legend)
    plot_1v_vs_2v(prop="rho_2",prop_name="$\\rho_2$",unit="$\\frac{kg}{m^3}$",
                    algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,
                    save_name_fig=save_name_fig,f_list=f_list,replace_node_name=replace_node_name,save_legend=save_legend)
    plot_1v_vs_2v(prop="v_1",prop_name="$v_1$",unit="$\\frac{m}{s}$",
                    algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,
                    save_name_fig=save_name_fig,f_list=f_list,replace_node_name=replace_node_name,save_legend=save_legend)
    plot_1v_vs_2v(prop="v_2",prop_name="$v_2$",unit="$\\frac{m}{s}$",
                    algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,
                    save_name_fig=save_name_fig,f_list=f_list,replace_node_name=replace_node_name,save_legend=save_legend)
    plot_1v_vs_2v(prop="pressure",prop_name="$p$",unit="$bar$",
                    algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,
                    save_name_fig=save_name_fig,f_list=f_list,replace_node_name=replace_node_name,save_legend=save_legend)
    #plot_1v_vs_2v(prop="m_1",prop_name="$m_1$",unit="$\\frac{kg}{m^3s}$",
    #                N=N,algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,save_name_fig=save_name_fig
    # ,f_list=f_list,replace_node_name=replace_node_name)
    #plot_1v_vs_2v(prop="m_2",prop_name="$m_2$",unit="$\\frac{kg}{m^3s}$",
    #                N=N,algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,save_name_fig=save_name_fig,
    # f_list=f_list,replace_node_name=replace_node_name)
    #plot_1v_vs_2v(prop="h",prop_name="$h$",unit="bar",
    #                                N=N,algebraic=algebraic,net_1=net_1,net_2_list=net_2_list,save_name_fig=save_name_fig,
    # f_list=f_list,replace_node_name=replace_node_name)


def plot_l1_error(prop:str,prop_name:str,unit:str,algebraic:float,net_1v:Network_2v_time,
                  net_2_list:list[Network_2v_time],f_list:list[float],save_name_fig:None,
                  pipe_nodes_list:list[str,str]=None,pipe_figure_list:list=None,
                  replace_node_name:list[str]=None):
    """
    plots the l^1 error in space between the stationary 1v solution and the 2v solution

    Args:
        prop (str): property 
        prop_name (str): display name 
        unit (str): unit
        algebraic(float): constant in front of non linear term
        net_1v (Network_2v_time): _description_
        net_2_list (list[Network_2v_time]): _description_
        f_list (list[float]): list of inner friction parameters
        save_name_fig (str, optional): name under which figure is saved.
        The path to the figure is 
        graphics/networks/1v_vs_2v/{net_1v.network_data}/l1_diff_massflowinflow_{save_name_fig}_{prop}.pdf
        If None, then figure is not saved. Defaults to None.
        replace_node_name (list[str], optional): list of names, which replace the name of nodes.
        This list should be ordered in the same way as the list of net_1.nodes_id 
        and have the same amount of elements.
        pipe_nodes_list (list[str,str], optional): list of pipes in the form of [sink1,sink2], for 
        which the l^1 norm should be calculated in a separate figure. Defaults to None.
        pipe_figure_list (list[int], optional): list of figures, where the same figure number indicates the 
        that the pipes given by pipe_node_list are in the same picture.
        This list should have the same length as pipe_node_list
        Defaults to None.

    """



    a_list = net_1v.pipe_diameter**2*np.pi/4


    if pipe_figure_list is not None and pipe_nodes_list is not None:
        n_plots = np.unique(np.array(pipe_figure_list)).shape[0]


        fig_list = []
        ax_list = []
        for i in range(n_plots):
            fig, ax = plt.subplots(constrained_layout=True)
            fig_copy = deepcopy(fig)
            fig_list.append(fig_copy)
            ax_list.append(fig_copy.get_axes()[0])

        linear_fct_drawn_list = [False for i in range(n_plots)]

        

        marker_list = ["x","o","D","^","s"]
        marker_count_list = np.zeros(n_plots)
        for i,in_id in enumerate(net_1v.pipe_in):
            node_in_out = [net_1v.pipe_in[i],net_1v.pipe_out[i]]
        


            if node_in_out  in pipe_nodes_list:



                index_fig = pipe_nodes_list.index(node_in_out)
                ax = ax_list[pipe_figure_list[index_fig]]
                fig = fig_list[pipe_figure_list[index_fig]]

                p_net_1 = net_1v.p_jax
                N_x_all_list = net_1v.N_x_all_list

                rho_1_net_1 = net_1v.u_sol[3*np.sum(N_x_all_list[:i]):3*np.sum(N_x_all_list[:i])+N_x_all_list[i]]
                rho_2_net_1 = net_1v.u_sol[3*np.sum(N_x_all_list[:i])+N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]]
                v_net_1 = net_1v.u_sol[3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]]


                l1_diff_list = []
                for k,net_2 in enumerate(net_2_list):
                    rho_1_net_2 = net_2.u_sol[4*np.sum(N_x_all_list[:i]):4*np.sum(N_x_all_list[:i])+N_x_all_list[i]]
                    rho_2_net_2 = net_2.u_sol[4*np.sum(N_x_all_list[:i])+N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]]
                    v_1_net_2 = net_2.u_sol[4*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]]
                    v_2_net_2 = net_2.u_sol[4*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+4*N_x_all_list[i]]
                    p1_net_2 = net_2.p1_jax
                    p2_net_2 = net_2.p2_jax



                    if prop == "rho_1": 
                        value = rho_1_net_2-rho_1_net_1

                    elif prop == "rho_2": 
                        value = rho_2_net_2-rho_2_net_1

                    elif prop == "v_1": 
                        value = v_1_net_2-v_net_1

                    elif prop == "v_2": 
                        value = v_2_net_2-v_net_1
                    
                    elif prop == "pressure":
                        value = p1_net_2(rho_1_net_2,rho_2_net_2)*1e-5+p2_net_2(rho_1_net_2,rho_2_net_2)*1e-5\
                        -(p_net_1(rho_1_net_1,rho_2_net_1)*1e-5)

                    elif prop == "m_1":
                        value = a_list[i]*rho_1_net_2*v_1_net_2
                        - a_list[i]*rho_1_net_1*v_net_1

                    elif prop == "m_2":
                        value = a_list[i]*rho_2_net_2*v_2_net_2
                        - a_list[i]*rho_2_net_1*v_net_1
                    


                    l1 =  np.sum((np.abs(value[:-1])+np.abs(value[1:]))/2)*net_1v.pipe_length[i]/net_1v.N_x_all_list[i]     

                    l1_diff_list.append(l1)
                
                
                if marker_count_list[pipe_figure_list[index_fig]] < len(marker_list):
                    marker = marker_list[int(marker_count_list[pipe_figure_list[index_fig]])]
                    marker_count_list[pipe_figure_list[index_fig]] += 1
                else:
                    marker = ""
                
                label = f"{net_1v.pipe_in[i]}_{net_1v.pipe_out[i]}"
                if replace_node_name is not None:
                    label = f"({replace_node_name[net_1v.node_id.index(net_1v.pipe_in[i])]},{replace_node_name[net_1v.node_id.index(net_1v.pipe_out[i])]})"
                ax.loglog(f_list,l1_diff_list,"-",label=label,marker=marker,linewidth=2)
                if not linear_fct_drawn_list[pipe_figure_list[index_fig]]:
                    slope,offset = np.polyfit(np.log(f_list),np.log(l1_diff_list),1)
                    ####################################################
                    slope = -1
                    ####################################################
                    ax.loglog(f_list,np.exp(offset*(1-0.05)+slope*np.log(f_list)),"--",label =f"slope {-slope:1.0f}",linewidth=2)
                    linear_fct_drawn_list[pipe_figure_list[index_fig]] = True



        for i,fig in enumerate(fig_list):
            ax = ax_list[i]
            ax.legend(loc="best")
            #ax.set_title(f"Difference of 1v vs 2v model for {prop_name} in {unit}")
            ax.set_xlabel("$f$")
            ax.set_ylabel(f"{prop_name} in {unit}")
            if save_name_fig is not None:
                index_same_plot = np.array(pipe_figure_list)==i
                pipes_list = np.array(pipe_nodes_list)[index_same_plot]

                save_name_str = f"graphics/networks/1v_vs_2v/{net_1v.network_data}/l1_diff_massflowinflow_{save_name_fig}_"
                for pipe in pipes_list:
                    if replace_node_name is not None:
                        save_name_str += f"pipe_{replace_node_name[net_1v.node_id.index(pipe[0])]}_{replace_node_name[net_1v.node_id.index(pipe[1])]}_"
                    else:
                        save_name_str += f"pipe_{pipe[0]}_{pipe[1]}_"
                
                save_name_str += f"{prop}"

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


                # Save legend
                fig_legend.savefig(
                    Path(save_name_str+"_leg.pdf"),
                    dpi=300,
                    bbox_inches="tight",
                    transparent=True
                )

                plt.close(fig_legend)

                ax.get_legend().remove()

                file_path = Path(save_name_str+".pdf")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(file_path,backend="pgf")
                plt.close(fig)
    else:
        # create colormap
        fig, ax = plt.subplots(constrained_layout=True)
        for i,in_id in enumerate(net_1v.pipe_in):

            p_net_1 = net_1v.p_jax
            N_x_all_list = net_1v.N_x_all_list

            rho_1_net_1 = net_1v.u_sol[3*np.sum(N_x_all_list[:i]):3*np.sum(N_x_all_list[:i])+N_x_all_list[i]]
            rho_2_net_1 = net_1v.u_sol[3*np.sum(N_x_all_list[:i])+N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]]
            v_net_1 = net_1v.u_sol[3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]]


            l1_diff_list = []
            for k,net_2 in enumerate(net_2_list):
                rho_1_net_2 = net_2.u_sol[4*np.sum(N_x_all_list[:i]):4*np.sum(N_x_all_list[:i])+N_x_all_list[i]]
                rho_2_net_2 = net_2.u_sol[4*np.sum(N_x_all_list[:i])+N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]]
                v_1_net_2 = net_2.u_sol[4*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]]
                v_2_net_2 = net_2.u_sol[4*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+4*N_x_all_list[i]]
                p1_net_2 = net_2.p1_jax
                p2_net_2 = net_2.p2_jax


                if prop == "rho_1": 
                    value = rho_1_net_2-rho_1_net_1

                elif prop == "rho_2": 
                    value = rho_2_net_2-rho_2_net_1

                elif prop == "v_1": 
                    value = v_1_net_2-v_net_1

                elif prop == "v_2": 
                    value = v_2_net_2-v_net_1
                
                elif prop == "pressure":
                    value = p1_net_2(rho_1_net_2,rho_2_net_2)*1e-5+p2_net_2(rho_1_net_2,rho_2_net_2)*1e-5\
                    -(p_net_1(rho_1_net_1,rho_2_net_1)*1e-5)

                elif prop == "m_1":
                    value = a_list[i]*rho_1_net_2*v_1_net_2
                    - a_list[i]*rho_1_net_1*v_net_1

                elif prop == "m_2":
                    value = a_list[i]*rho_2_net_2*v_2_net_2
                    - a_list[i]*rho_2_net_1*v_net_1
                


                l1 =  np.sum((np.abs(value[:-1])+np.abs(value[1:]))/2)*net_1v.pipe_length[i]/net_1v.N_x_all_list[i]     

                l1_diff_list.append(l1)
            if i == 0:
                offset = 0
                ax.loglog(f_list,np.exp((1-offset)*np.log(l1_diff_list[0])-np.log(f_list)),"--b",label ="slope = 1")
                ax.loglog(f_list,np.exp((1-offset)*np.log(l1_diff_list[0])-0.5*np.log(f_list)),"--g", label ="slope = 0.5")
                ax.loglog(f_list,np.exp((1-offset)*np.log(l1_diff_list[0])-0.25*np.log(f_list)),"--y", label ="slope = 0.25")



        ax.legend(loc="best")
        #ax.set_title(f"Difference of 1v vs 2v model for {prop_name} in {unit}")
        ax.set_xlabel("f")
        ax.set_ylabel(f"{prop_name} in {unit}")
        if save_name_fig is not None:

            file_path = Path(f"graphics/networks/1v_vs_2v/{net_1v.network_data}/l1_diff_massflowinflow_{save_name_fig}_{prop}.pdf")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(file_path,backend="pgf")


def plot_all_l1_error(net_1v:Network_1v_time,net_2_list:list[Network_2v_time],algebraic:float,f_list:list[float],save_name_fig:str=None,
                      pipe_nodes_list:list=None,pipe_figure_list:list=None,replace_node_name:list[str]=None):
    """
    plots the l^1 error in space between the stationary 1v solution and the 2v solution

    Args:
        algebraic(float): constant in front of non linear term
        net_1v (Network_2v_time): _description_
        net_2_list (list[Network_2v_time]): _description_
        f_list (list[float]): list of inner friction parameters
        save_name_fig (str, optional): name under which figure is saved.
        The path to the figure is 
        graphics/networks/1v_vs_2v/{net_1v.network_data}/l1_diff_massflowinflow_{save_name_fig}_{prop}.pdf
        If None, then figure is not saved. Defaults to None.
        replace_node_name (list[str], optional): list of names, which replace the name of nodes.
        This list should be ordered in the same way as the list of net_1.nodes_id 
        and have the same amount of elements.
        pipe_nodes_list (list[str,str], optional): list of pipes in the form of [sink1,sink2], for 
        which the l^1 norm should be calculated in a separate figure. Defaults to None.
        pipe_figure_list (list[int], optional): list of figures, where the same figure number indicates the 
        that the pipes given by pipe_node_list are in the same picture.
        This list should have the same length as pipe_node_list
        Defaults to None.

    """
    
    plot_l1_error(prop="rho_1",prop_name="$\\rho_1$",unit="$\\frac{kg}{m^3}$",
                    algebraic=algebraic,net_1v=net_1v,net_2_list=net_2_list,
                    save_name_fig=save_name_fig,f_list=f_list,pipe_nodes_list=pipe_nodes_list,pipe_figure_list=pipe_figure_list,replace_node_name=replace_node_name)
    plot_l1_error(prop="rho_2",prop_name="$\\rho_2$",unit="$\\frac{kg}{m^3}$",
                    algebraic=algebraic,net_1v=net_1v,net_2_list=net_2_list,
                    save_name_fig=save_name_fig,f_list=f_list,pipe_nodes_list=pipe_nodes_list,pipe_figure_list=pipe_figure_list,replace_node_name=replace_node_name)
    plot_l1_error(prop="v_1",prop_name="$v_1$",unit="$\\frac{m}{s}$",
                    algebraic=algebraic,net_1v=net_1v,net_2_list=net_2_list,
                    save_name_fig=save_name_fig,f_list=f_list,pipe_nodes_list=pipe_nodes_list,pipe_figure_list=pipe_figure_list,replace_node_name=replace_node_name)
    plot_l1_error(prop="v_2",prop_name="$v_2$",unit="$\\frac{m}{s}$",
                    algebraic=algebraic,net_1v=net_1v,net_2_list=net_2_list,
                    save_name_fig=save_name_fig,f_list=f_list,pipe_nodes_list=pipe_nodes_list,pipe_figure_list=pipe_figure_list,replace_node_name=replace_node_name)
    plot_l1_error(prop="pressure",prop_name="$p$",unit="$bar$",
                    algebraic=algebraic,net_1v=net_1v,net_2_list=net_2_list,
                    save_name_fig=save_name_fig,f_list=f_list,pipe_nodes_list=pipe_nodes_list,pipe_figure_list=pipe_figure_list,replace_node_name=replace_node_name)

def plot_all_1v_vs_2v(network_type:str):
    """
    plots l^1 error and difference of the 1v vs 2v model for different inner friction parameter f

    Args:
        network_type (str): "3mix" or "gaslib40"
    """


    if network_type == "3mix":
        file_network = Path("network_data" ,"optimization_data","network_files", "3mixT.net")
        file_data = Path("network_data", "optimization_data","3mix_scenarios", f"3mix_temp_0.lsf")
        candidate_dx = 200
        f_list = [1,5,10,20,50]
        offset_labels = None


    elif network_type == "gaslib40":
        file_network = Path("network_data" ,"optimization_data","network_files", "gaslib40_removed_edit.net")
        file_data = Path("network_data", "optimization_data", "solution_files","gaslib40_removed_edit.lsf")
        candidate_dx = 500
        f_list = [1,5,10,20]#,30]

        label_node_list = []
        offset_labels = None

    # 1v model
    stationary_or_instationary = 0

    
    scenarios = ["gaslib40_stationary", "time_dep"] 
    scenario = scenarios[stationary_or_instationary]
    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[stationary_or_instationary]
    dt = int(T/60)
    
    
    #Parameters which don't depend on stationary/instationary
    algebraic = 1.0 
    
    #Change to fit network
    model ="speed_of_sound"

    network_1v = Network_1v_time(file_network=file_network,file_data=file_data,model=model,
                            dt=dt,T=T,candidate_dx=candidate_dx)
    network_1v._precompute_pressure_law()
    network_1v.load_solution_network(algebraic=algebraic,scenario=scenario)

    if network_type == "gaslib40":
        node_list = network_1v.node_id
        node_name_dict = {"sink_15":"$\\Large{\\textbf{A}}$",
                          "sink_3":"$\\Large{\\textbf{B}}$",
                          "sink_13":"$\\Large{\\textbf{C}}$",
                          #"sink_19":"$\\Large{\\textbf{F}}$",
                          #"innode_3":"$\\Large{\\textbf{G}}$",
                          #"source_3":"$\\Large{\\textbf{H}}$",
                          #"innode_5":"$\\Large{\\textbf{J}}$",
                          "sink_20":"$\\Large{\\textbf{D}}$",
                          "sink_26":"$\\Large{\\textbf{E}}$",
                          "sink_25":"$\\Large{\\textbf{Z}}$"}

        offset_dict = {"sink_15":[-50,170],
                          "sink_3":[250,-100],
                          "sink_13":[-20,150],
                          #"sink_19":[150,150],
                          #"innode_3":[150,150],
                          #"source_3":[150,150],
                          #"innode_5":[150,150],
                          "sink_20":[150,150],
                          "sink_26":[-150,150],
                          "sink_25":[150,150]}
        label_node_list = [node_name_dict[node] if node in node_name_dict.keys() else "" for node in node_list ]
        offset_labels = [offset_dict[node] if node in node_name_dict.keys() else None for node in node_list ]



    network_1v.plot_all_stationary(algebraic=algebraic,plot_pipe=False,scenario=scenario,show_labels=True,
                                   label_node_list=label_node_list,offset_labels=offset_labels)
    

    # 2v sol
    stationary_or_instationary = 0
    
    scenarios = ["gaslib11_stationary_1v", "time_dep"] 
    scenario = scenarios[stationary_or_instationary]
    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[stationary_or_instationary]
    dt = int(T/60)
    
    
    #Fixed computational parameters
    algebraic = 1.0 
    
    #Parameters which can change
    model ="speed_of_sound"
    algebraic = 1.0



    T = 60*60*24*150

    network_2v_list = []
    for f in f_list:
        algebraic = 1.0
        network_2v = Network_2v_time(file_network=file_network,file_data=file_data,model=model,dt=dt,T=T,
                                     candidate_dx=candidate_dx)
        network_2v._precompute_pressure_law()
        network_2v.load_solution_network(algebraic=algebraic,scenario=scenario,f=f)
        network_2v.plot_all_stationary(algebraic=algebraic,f=f,plot_pipe=False,scenario=scenario,show_labels=True,label_node_list=label_node_list,
                                       offset_labels=offset_labels)
        plt.close()
        network_2v_list.append(network_2v)



    if network_type == "gaslib40":
        node_list = network_1v.node_id
        node_name_replace_dict = {
                            "sink_15":"A",
                            "sink_3":"B",
                            "sink_13":"C",
                          #"sink_19":"F",
                          #"innode_3":"G",
                          #"source_3":"H",
                          #"innode_5":"J",
                          "sink_20":"D",
                          "sink_26":"E",
                          "sink_25":"Z"}
        replace_node_name = [node_name_replace_dict[node] if node in node_name_dict.keys() else node for node in node_list ]
    else:
        replace_node_name = None
    

    plot_all_diff_network(net_1=network_1v,net_2_list=network_2v_list,f_list=f_list,
                          algebraic=1.0,save_name_fig="diff_1v_2v_network",replace_node_name= replace_node_name)

    algebraic = 1.0
    net_1v = network_1v

    pipe_nodes_list = [["sink_15","sink_25"] ,["sink_3","sink_25"],["sink_13","sink_25"],
                       ["sink_25","sink_20"] ,["sink_25","sink_26"]]
    pipe_figure_list = [0,0,0,1,1]
    
    
    plot_all_l1_error(net_1v=net_1v,net_2_list=network_2v_list,f_list=f_list,
                          algebraic=1.0,save_name_fig="l1_diff",
                          pipe_nodes_list=pipe_nodes_list,pipe_figure_list=pipe_figure_list,
                          replace_node_name=replace_node_name)

if __name__ == "__main__":
    

    network_type = "gaslib40"
    #network_type = "3mix"
    plot_all_1v_vs_2v(network_type=network_type)

