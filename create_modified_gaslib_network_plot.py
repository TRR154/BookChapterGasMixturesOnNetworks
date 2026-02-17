import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pathlib import Path
from network_1v_timedep_dxdtform_massflowinflow import Network_1v_time


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

def plot_network(net:Network_1v_time,save_name:str,circle_bool:bool=False):
    """
    plots the underlying network of net
    Args:
        net (Network_1v_time): network 
        save_name (str): save name
        circle_bool (bool, optional):add a circle at coordinates (2500,250).
        Defaults to False.
    """

    fig, ax = plt.subplots()
    index_source = net.node_type == "source"
    index_sink = net.node_type == "sink"
    index_in_nodes = net.node_type == "in_node"
    ax.scatter(np.array(net.node_x_cord)[index_source],
                np.array(net.node_y_cord)[index_source],color="r", label="$\\Gamma_p$",marker="o")
    ax.scatter(np.array(net.node_x_cord)[index_in_nodes],
                np.array(net.node_y_cord)[index_in_nodes],color="g", label="$\\Gamma_q$",marker="s" )
    ax.scatter(np.array(net.node_x_cord)[index_sink],
                np.array(net.node_y_cord)[index_sink],color="g",marker="s")

    
    label = "pipe"
    for i,in_id in enumerate(net.pipe_in):
        out_id = net.pipe_out[i]

        node_in_index = net.node_id.index(in_id)
        node_out_index = net.node_id.index(out_id)
        ax.plot([net.node_x_cord[node_in_index],net.node_x_cord[node_out_index]],
        [net.node_y_cord[node_in_index],net.node_y_cord[node_out_index]],'b',label=label)
        if i == 0:
            label = "_nolegend_"

    label = "comp"
    for i,in_id in enumerate(net.comp_in):
        out_id = net.comp_out[i]

        node_in_index = net.node_id.index(in_id)
        node_out_index = net.node_id.index(out_id)
        ax.plot([net.node_x_cord[node_in_index],net.node_x_cord[node_out_index]],
        [net.node_y_cord[node_in_index],net.node_y_cord[node_out_index]],'k--',label=label)
        if i == 0:
            label = "_nolegend_"


    if circle_bool:
        circle = plt.Circle((2500,250), 750, color='orange',fill=False)
        ax.add_patch(circle)
    ax.legend(loc="best")
    ax.set_xticks([])
    ax.set_yticks([])


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
    save_name = f"graphics/{save_name}"

    fig_legend.savefig(
        save_name+"_leg.pdf",
        dpi=300,
        bbox_inches="tight",
        transparent=True
    )

    plt.close(fig_legend)

    ax.get_legend().remove()

    fig.savefig(save_name+".pdf",backend="pgf")



def plot_modified_gaslib40_net():
    """
    plots the original gaslib 40 and the gaslb40-3 file
    and saves them
    """

    file_network = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_edit.net")
    file_data = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_edit.lsf")

    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[0]
    dt = int(T/60)
    
    
    #Change to fit network
    candidate_dx = 200
    model ="speed_of_sound"

    network_gaslib = Network_1v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=candidate_dx,
                            dt=dt,T=T)

    plot_network(net=network_gaslib,save_name="gaslib40",circle_bool=True)


    file_network = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.net")
    file_data = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.lsf")

    network_gaslib = Network_1v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=candidate_dx,
                            dt=dt,T=T)

    plot_network(net=network_gaslib,save_name="gaslib40_modified")


if __name__ == "__main__":

    plot_modified_gaslib40_net()