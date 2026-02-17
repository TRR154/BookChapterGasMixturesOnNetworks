import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import sympy as sym
import jax
import jax.numpy as jnp
from jax import grad

#######################################
#from jax import config
#config.update("jax_enable_x64", True)
# NOTE: standard jax is 32 bit vs 64 sympy
#######################################

from matplotlib import cm
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




def plot_pressure_mixtures(black_white:bool=False):
    """
    plots different pressure models over densities and their differences for T=273.15

    Args:
        black_white(bool, optional): no diverging colormaps are used
    """


    rho_1 = np.linspace(0.1, 65, 2000)
    rho_2 = np.linspace(0.1, 8, 2000)
    T = 273.15

    X, Y = np.meshgrid(rho_1, rho_2)

    # Scaling in bar
    p_spos = pressure_law_mixtures(X, Y, "speed_of_sound", T) * 1e-5
    p_virial = pressure_law_mixtures(X, Y, "virial_expansion", T) * 1e-5
    p_virial_mix = pressure_law_mixtures(X, Y, "virial_expansion_mix", T) * 1e-5
    p_gerg = pressure_law_mixtures(X, Y, "gerg", T) * 1e-5

    # Masking values greater than 80
    p_spos_masked = np.ma.masked_where(p_spos > 80, p_spos)
    p_virial_masked = np.ma.masked_where(p_virial > 80, p_virial)
    p_virial_mix_masked = np.ma.masked_where(p_virial_mix > 80, p_virial_mix)
    p_gerg_masked = np.ma.masked_where(p_gerg > 80, p_gerg)

    rho_2_gerg_80 = rho_2[np.argmin(np.abs(80 - p_gerg), axis=0)]
    rho_2_virial_80 = rho_2[np.argmin(np.abs(80 - p_virial), axis=0)]
    rho_2_virial_mix_80 = rho_2[np.argmin(np.abs(80 - p_virial_mix), axis=0)]
    rho_2_spos_80 = rho_2[np.argmin(np.abs(80 - p_spos), axis=0)]

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))

    # Plot the surfaces with the masked data
    surf = ax[0].contourf(X, Y, p_spos_masked, antialiased=False)
    ax[0].set_title("speed of sound")
    ax[0].set_xlabel("$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$")
    ax[0].set_ylabel("$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$")
    ax[0].plot(rho_1, rho_2_spos_80, "b-", label="$p_I$ = 80")
    ax[0].legend(loc="best")

    surf = ax[1].contourf(X, Y, p_virial_masked, antialiased=False)
    ax[1].set_title("virial expansions")
    ax[1].set_xlabel("$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$")
    ax[1].set_ylabel("$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$")
    ax[1].plot(rho_1, rho_2_virial_80, "b-", label="$p_V$= 80")
    ax[1].legend(loc="best")

    surf = ax[3].contourf(X, Y, p_virial_mix_masked, antialiased=False)
    ax[3].set_title("virial expansions")
    ax[3].set_xlabel("$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$")
    ax[3].set_ylabel("$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$")
    ax[3].plot(rho_1, rho_2_virial_mix_80, "b-", label="$p_M$ = 80")
    ax[3].legend(loc="best")
    fig.colorbar(surf, ax=ax[3], label="$p$ (bar)")

    surf = ax[2].contourf(X, Y, p_gerg_masked, antialiased=False)
    ax[2].set_title("gerg")
    ax[2].set_xlabel("$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$")
    ax[2].set_ylabel("$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$")
    ax[2].plot(rho_1, rho_2_gerg_80, "b-", label="$p_G=80$")
    ax[2].legend(loc="best")




    plt.savefig("graphics/pressure_plots/models.pdf")


    
    num_ticks =  8

    def _plot_pressure_law(value:np.ndarray,
                           rho_2_80:np.ndarray,
                           save_name:str,label_name:str,
                           rho_1:np.ndarray=rho_1,
                           X=X,Y=Y):
        """
        plots pressure laws

        Args:
            value (np.ndarray): plotting value
            rho_1(np.ndarray): rho_1 values
            rho_2_80(np.ndarray): rho_2 values for line
            save_name (str): save name
            label_name (str): label
        """
        vmin = np.min(value)
        vmax = np.max(value)    
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.viridis
        
        fig, ax = plt.subplots(constrained_layout=True)

        # Define the extent of the image in data coordinates
        extent = [
            X.min(), X.max(),   # x-axis limits (rho_1)
            Y.min(), Y.max()    # y-axis limits (rho_2)
        ]

        im = ax.imshow(
            value,
            origin="lower",        # important: match contourf orientation
            extent=extent,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest"
        )

        # Axis labels
        ax.set_xlabel(r"$\rho_1$ $(CH_4)$ in $\frac{kg}{m^3}$")
        ax.set_ylabel(r"$\rho_2$ $(H_2)$ in $\frac{kg}{m^3}$")

        # Overlay line
        ax.plot(rho_1, rho_2_80, "b-", label=label_name)
        ax.legend(loc="best")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, label=r"$p$ (bar)")
        ticks = np.linspace(
            vmin,
            vmax,
            num_ticks
        )
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.1f}" for t in ticks])

        fig.savefig(f"graphics/pressure_plots/model_{save_name}.pdf")
    
    _plot_pressure_law(value=p_spos_masked,rho_2_80=rho_2_spos_80,save_name="ideal",label_name=r"$p_I$")
    _plot_pressure_law(value=p_virial_masked,rho_2_80=rho_2_virial_80,save_name="virial",label_name=r"$p_V$")
    _plot_pressure_law(value=p_virial_mix_masked,rho_2_80=rho_2_virial_mix_80,save_name="virial_mix",label_name=r"$p_M$")
    _plot_pressure_law(value=p_gerg_masked,rho_2_80=rho_2_gerg_80,save_name="gerg",label_name=r"$p_G$")


    plt.close("all")



    ##########################################################
    # difference plots
    M_1 = 16.042460
    M_2 = 2.015880
    x_2 = (Y/M_2)/(X/M_1+Y/M_2)


    relative_error_list = [True,False]
    for relative_error in relative_error_list:
        rho_2_gerg_x2_upper = rho_2[np.argmin(np.abs(0.91 - x_2), axis=0)]
        rho_2_gerg_x2_lower = rho_2[np.argmin(np.abs(0.05 - x_2), axis=0)]

        mask_array = np.logical_or(p_gerg>80,np.logical_or(x_2 > 0.91, x_2<0.05))
        

        p_spos_masked = np.ma.masked_where(mask_array, p_spos)
        p_virial_masked = np.ma.masked_where(mask_array, p_virial)
        p_virial_mix_masked = np.ma.masked_where(mask_array, p_virial_mix)
        p_gerg_masked = np.ma.masked_where(mask_array, p_gerg)


        def _plot_diff(value_1:np.ndarray,value_2:np.ndarray,relative_error:bool,
                       save_name_val_1:str,save_name_val_2):
            fig, ax = plt.subplots(constrained_layout=True) 
            value = value_1-value_2
            if relative_error:
                value = np.abs(value)/value_1

            vmin = np.min(value)
            vmax = np.max(value)
            if black_white:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.get_cmap("Greys")

            elif vmin < 0 and vmax > 0:
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                cmap = plt.get_cmap('seismic')

            elif vmax > 0:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.seismic(np.linspace(0.5, 1, 256))
                cmap = plt.matplotlib.colors.ListedColormap(cmap)

            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.seismic(np.linspace(0,0.5, 256))
                cmap = plt.matplotlib.colors.ListedColormap(cmap)




            # Define the extent of the image in data coordinates
            extent = [
                X.min(), X.max(),   # x-axis limits (rho_1)
                Y.min(), Y.max()    # y-axis limits (rho_2)
            ]

            im = ax.imshow(
                value,
                origin="lower",        # important: match contourf orientation
                extent=extent,
                aspect="auto",
                cmap=cmap,
                norm=norm,
                interpolation="nearest"
            )

            #ax.contourf(X, Y, value,cmap=cmap, norm=norm,levels = 256)


            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            #sm.set_array([])
            if relative_error:
                cbar = fig.colorbar(sm, ax=ax)
            else:
                cbar = fig.colorbar(sm, ax=ax,label="$p$ in bar")
            num_ticks =  8
            ticks = np.linspace(np.min(value), np.max(value), num_ticks)
            round_num = 3 if relative_error else 2
            ticks  = list(set([np.round(tick,round_num) for tick in ticks ]))
            cbar.ax.set_yscale('linear')
            cbar.set_ticks(ticks)                     # set tick positions
            cbar.set_ticklabels([f"{t:.{round_num}f}" for t in ticks]) 
            #ax.set_title("difference gerg and speed of sound in bar")
            ax.set_xlabel("$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$")
            ax.set_ylabel("$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$")
            markevery = int(rho_1.shape[0]/20)
            ax.plot(rho_1,rho_2_gerg_80,"b-^",label="$p_G=80$",markevery=markevery)
            ax.plot(rho_1[rho_2_gerg_x2_upper<np.max(rho_2_gerg_x2_upper)-1e-2],
                    rho_2_gerg_x2_upper[rho_2_gerg_x2_upper<np.max(rho_2_gerg_x2_upper)-1e-2],
                    "g--",label="$x_2=0.91$",markevery=int(markevery/10))
            ax.plot(rho_1,rho_2_gerg_x2_lower,"c-",label="$x_2=0.05$",markevery=markevery)
            ax.legend(loc="best")

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

            save_name = f"graphics/pressure_plots/diff_{save_name_val_1}_{save_name_val_2}_relative_{relative_error}"
            if black_white:
                save_name += "_bw"

            # Save legend
            fig_legend.savefig(
                save_name+"_leg.pdf",
                dpi=300,
                bbox_inches="tight",
                transparent=True
            )

            plt.close(fig_legend)

            ax.get_legend().remove()

            plt.savefig(save_name+".pdf")
            plt.close()
        
        value_1_list = [p_gerg_masked,p_gerg_masked,p_gerg_masked]
        value_2_list = [p_spos_masked,p_virial_mix,p_virial]
        value_1_name_list = ["gerg","gerg","gerg"]
        value_2_name_list = ["sps","virial_mix","viriral"]
        for j,value_1 in enumerate(value_1_list):
            _plot_diff(value_1=value_1,value_2=value_2_list[j],relative_error=relative_error,save_name_val_1=value_1_name_list[j],save_name_val_2=value_2_name_list[j])



def plot_gerg_approximation(relative_error:bool=False,black_white:bool=False):
    """
    plots the gerg approximation and difference
    Args:
        relative_error (bool, optional): if relative errors are needed. Defaults to False.
        black_white(bool, optional): no diverging colormaps are used
    """

    T = 273.15

    poly_deg_list = [1,2,3,3,4,5,10,1,2,3,4,5,10]
    penalty_list = [0,0,50,0,0,0,0,0,0,0,0,0,0,0]
    alloW_non_simple_list = [False,False,False,False,False,False,False,True,True,True,True,True,True]

    for i,poly_deg in enumerate(poly_deg_list):
        penalty = penalty_list[i]

        p1,p2,p_ns = fit_gerg_simple(poly_deg=poly_deg,penalty=penalty,allow_non_simple=alloW_non_simple_list[i],T=T)
        name = f"gerg_simple_approx_k_{poly_deg}_pen_{penalty}_non_simple_{alloW_non_simple_list[i]}"

        pressure_test_convexity(p1,p2,p_ns,name)

        rho_1 = np.linspace(0.1, 65, 2000)
        rho_2 = np.linspace(0.1, 8, 2000)
        rho_1_mesh,rho_2_mesh = np.meshgrid(rho_1,rho_2)
        p_gerg = pressure_law_mixtures(rho_1_mesh,rho_2_mesh,"gerg",T=T)
        p_gerg_fit = p1(rho_1_mesh,rho_2_mesh) + p2(rho_1_mesh,rho_2_mesh)
        if alloW_non_simple_list[i]:
            p_gerg_fit += p_ns(rho_1_mesh,rho_2_mesh)


        rho_2_gerg_80 = rho_2[np.argmin(np.abs(80*1e5-p_gerg),axis=0)]
        

        M_1 = 16.042460
        M_2 = 2.015880
        x_2 = (rho_2_mesh/M_2)/(rho_1_mesh/M_1+rho_2_mesh/M_2)
        rho_2_gerg_x2_upper = rho_2[np.argmin(np.abs(0.91 - x_2), axis=0)]
        rho_2_gerg_x2_lower = rho_2[np.argmin(np.abs(0.05 - x_2), axis=0)]

        p_gerg_fit_masked = np.ma.masked_where(np.logical_or(p_gerg > 80*1e5,
                np.logical_or(x_2<0.05,x_2>0.91)), p_gerg_fit)
        
        p_gerg_masked = np.ma.masked_where(np.logical_or(p_gerg > 80*1e5,
                np.logical_or(x_2<0.05,x_2>0.91)), p_gerg)

        fig, ax = plt.subplots(constrained_layout=True) 
        value = (p_gerg_masked-p_gerg_fit_masked)*1e-5

        if relative_error:
            value = np.abs(value)/(p_gerg_masked*1e-5)

        vmin = np.min(value)
        vmax = np.max(value)
        
        if black_white:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap("Greys")

        elif vmin < 0 and vmax > 0:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            cmap = plt.get_cmap('seismic')

        elif vmax > 0:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.seismic(np.linspace(0.5, 1, 256))
            cmap = plt.matplotlib.colors.ListedColormap(cmap)

        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.seismic(np.linspace(0,0.5, 256))
            cmap = plt.matplotlib.colors.ListedColormap(cmap)


        contour = ax.contourf(rho_1_mesh, rho_2_mesh, value,cmap=cmap, norm=norm,levels=256)


        # Define the extent of the image in data coordinates
        extent = [
            rho_1_mesh.min(), rho_1_mesh.max(),   # x-axis limits (rho_1)
            rho_2_mesh.min(), rho_2_mesh.max()    # y-axis limits (rho_2)
        ]

        im = ax.imshow(
            value,
            origin="lower",        # important: match contourf orientation
            extent=extent,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest"
        )



        rho_2_gerg_80 = rho_2[np.argmin(np.abs(80*1e5-p_gerg),axis=0)]
        #sm.set_array([])
        #ax.set_title(f"difference for {poly_deg}  allowed_non_simple={alloW_non_simple_list[i]}")
        ax.set_xlabel("$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$")
        ax.set_ylabel("$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$")

        markevery = int(rho_1.shape[0]/20)
        ax.plot(rho_1,rho_2_gerg_80,"b-^",label="$p_G=80$",markevery=markevery)
        ax.plot(rho_1[rho_2_gerg_x2_upper<np.max(rho_2_gerg_x2_upper)-1e-2],
                rho_2_gerg_x2_upper[rho_2_gerg_x2_upper<np.max(rho_2_gerg_x2_upper)-1e-2],
                "g--",label="$x_2=0.91$",markevery=int(markevery/10))
        ax.plot(rho_1,rho_2_gerg_x2_lower,"c-",label="$x_2=0.05$",markevery=markevery)
        ax.legend(loc="best")
        if relative_error:
            cbar = fig.colorbar(contour)
        else:
            cbar = fig.colorbar(contour,label="$p$ in bar")
        num_ticks =  8
        ticks = np.linspace(np.min(value), np.max(value), num_ticks)
        cbar.ax.set_yscale('linear')
        cbar.set_ticks(ticks)                     # set tick positions
        cbar.set_ticklabels([f"{t:.2g}" for t in ticks]) 


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
        save_name = f"graphics/pressure_plots/diff_{name}"
        if relative_error:
            save_name += "_rel"
        if black_white:
            save_name += "_bw"
        

        fig_legend.savefig(
            save_name+"_leg.pdf",
            dpi=300,
            bbox_inches="tight",
            transparent=True
        )

        plt.close(fig_legend)

        ax.get_legend().remove()


        plt.savefig(save_name+".pdf")

        
        
        fig, ax = plt.subplots(constrained_layout=True) 
        contour = ax.contourf(rho_1_mesh, rho_2_mesh, p_gerg_fit_masked*1e-5)
        ax.plot(rho_1,rho_2_gerg_80,"b-",label="$p_G=80$")
        ax.plot(rho_1,rho_2_gerg_x2_upper,"g-",label="$x_2=0.91$")
        ax.plot(rho_1,rho_2_gerg_x2_lower,"c-",label="$x_2=0.05$")
        ax.set_title(f"fitted gerg to degree {poly_deg}")
        ax.set_xlabel("$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$")
        ax.set_ylabel("$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$")
        fig.colorbar(contour)
        plt.close("all")


def pressure_law_mixtures(rho_1:np.ndarray,rho_2:np.ndarray,model:str="speed_of_sound",T:float=283,
             test_dict:dict=None,return_partial_pressures:bool=False)->np.ndarray:
    """
    returns different pressure laws for mixtures

    Args:
        rho_1 (np.ndarray): densitiy of CH_4 in kg/m^3
        rho_2 (np.ndarray): densitiy of H_2 in kg/m^3
        model (str, optional): pressure law type. ["speed_of_sound","virial_expansion","virial_expansion_mix","gerg"].
        Defaults to "speed_of_sound".
        T (float, optional): temperature in K. Defaults to 283.
        test_dict (dict, optional):test_dict_values for gerg. Defaults to None.
        return_partial_pressures (bool, optional):except for gerg partial_pressures can be returned.
        Defaults to False.

    Raises:
        ValueError: Wrong model/model_type

    Returns:
        np.nadarray: result
    """

    R = 8.314472
    # Do not rescale before -> otherwhise fitting incorrect
    M_1 = 16.042460
    M_2 = 2.015880


    if model == "virial_expansion":

        if np.abs(T-283)<4:

            # first virial coef
            # CH_4
            B_1 = -47.58
            # H_2
            B_2 =  14.6
            # second virial coef
            # CH_4
            C_1 = 2.440
            # H_2
            C_2 = 0.250
        
        elif np.abs(T-273.15)<4:

            # first virial coef
            # CH_4
            B_1 = -55.0
            # H_2
            B_2 =  (14.0 +14.1 + 14.6 + 13.7 + 13.5 + 13.8 + 13.7 + 13.0 + 13.6)/9
            # second virial coef
            # CH_4
            C_1 = 29.09
            # H_2
            C_2 = (0.305 + 0.423 + 0.923 + 0.415 + 0.389)/5
        
        else: 
            raise ValueError(f"Pressure not implemented for T={T}")
        
        

        # volume is measured in cm**3, and rho_1 given in kg/m**3
        rho_1_scaled = rho_1*1e-3
        rho_2_scaled = rho_2*1e-3


        p1 = 1/M_1*rho_1_scaled+B_1*(1/M_1)**2*rho_1_scaled**2+C_1*(1/M_1)**3*rho_1_scaled**3
        p2 = 1/M_2*rho_2_scaled+B_2*(1/M_2)**2*rho_2_scaled**2+C_2*(1/M_2)**3*rho_2_scaled**3

        
        p1 *= R*T*1e6
        p2 *= R*T*1e6

        p = p1 + p2

        if return_partial_pressures:
            return p1,p2
        else:
            return p
        
    
    elif model == "virial_expansion_mix":
        

        if np.abs(T-283)<4:

            # first/second pure  virial coefs (see above in virial expansions)
            # 
            B_1 = -47.58
            B_2 =  14.6
            #B_1 = 4.4344 *10 - 1.6608 *10**4/T - 3.5430 * 10**6/T**2 + 2.9832 *10**8/T**3 -2.3448 *10**10/T**4
            #B_2 = 1.7472 *10 - 1.2926 *10**2/T - 2.6988 * 10**5/T**2 + 8.0282* 10**6/T**3


            C_1 = 2.440
            C_2 = 0.250
            

            # mixed virial expansions
            B_11 = B_1
            B_12 = 9.2
            B_22 = B_2

            
            #B_12 = -1.0011*10**2 + 7.6037 *10**4/T - 1.2943*10**7/T**2

            C_111 = C_1
            C_112 = 0.6
            C_122 = 1.3
            C_222 = C_2
        
        elif np.abs(T-273.15)<4:

            # first virial coef
            # CH_4
            B_1 = -55.0
            # H_2
            B_2 =  (14.0 +14.1 + 14.6 + 13.7 + 13.5 + 13.8 + 13.7 + 13.0 + 13.6)/9
            # second virial coef
            # CH_4
            C_1 = 29.09
            # H_2
            C_2 = (0.305 + 0.423 + 0.923 + 0.415 + 0.389)/5
        
            # mixed virial expansions
            # TODO: check table and compare
            # TODO: in symbolic also copy
            B_11 = B_1
            B_12 = 3.7
            B_22 = B_2

            

            C_111 = C_1
            C_112 = 0.6
            C_122 = 1.4
            C_222 = C_2
        else: 
            raise ValueError(f"Pressure not implemented for T={T}")

        

        # volume is measured in cm**3, and rho_1 given in kg/m**3
        rho_1_scaled = rho_1*1e-3
        rho_2_scaled = rho_2*1e-3

        #units are cm^3
        rho_mix = 1/M_1*rho_1_scaled+1/M_2*rho_2_scaled
        mu_mass = rho_1_scaled/(rho_1_scaled+rho_2_scaled) 
        x_1 = (mu_mass/M_1)/((mu_mass/M_1)+(1-mu_mass)/M_2)
        x_2 = 1-x_1
        
        B_mix = B_11*x_1**2+ 2*B_12*x_1*x_2+ 2*B_22*x_2**2

        C_mix = C_111*x_1**3+ 3*C_112*x_1**2*x_2 + 3*C_122*x_1*x_2**2+C_222*x_2**3

        p = rho_mix +B_mix*rho_mix**2+C_mix*rho_mix**3



        p *= R*T*1e6

        return p


    elif model == "speed_of_sound":


        p1 = R*T*(1/M_1)*rho_1*1e3 
        p2 = R*T*(1/M_2)*rho_2*1e3 
        p = p1 + p2

        if return_partial_pressures:
            return p1,p2
        else:
            return p

        
    elif model == "gerg":


        if test_dict is not None:
            x_1 = test_dict["x_1"]
            x_2 = 1-x_1
        else:
            # compute molar composition
            rho_1_scaled = rho_1
            rho_2_scaled = rho_2
            mu_mass = rho_1_scaled/(rho_1_scaled+rho_2_scaled) 
            x_1 = (mu_mass/M_1)/((mu_mass/M_1)+(1-mu_mass)/M_2)
            x_2 = 1-x_1


        ## Table A.3.5
        rho_c_1 = 10.139342719
        #critical density of hydrogen 
        rho_c_2 = 14.940000000

        #critical temperature of methane 
        T_c_1 = 190.564000000
        #critical temperature of hydrogen 
        T_c_2 = 33.190000000


        ## Table A.3.8
        # binary paramters for density
        beta_nu_ij = 1
        gamma_nu_ij = 1.018702573

        ## Table A.3.8
        # binary paramters for temperature 
        beta_T_ij = 1
        gamma_T_ij = 1.352643115


        rho_r = 1/((x_1)**2*(1/rho_c_1) + x_2**2*(1/rho_c_2) \
                +2*x_1*x_2*beta_nu_ij*gamma_nu_ij*\
                ((x_1+x_2)/(beta_nu_ij**2*x_1+x_2))*\
                1/8*(1/(rho_c_1**(1/3))+1/(rho_c_2**(1/3)))**3)



        T_r =  x_1**2*T_c_1 + x_2**2*T_c_2 \
                +2*x_1*x_2*beta_T_ij*gamma_T_ij*\
                ((x_1+x_2)/((beta_T_ij**2)*x_1+x_2))*\
                (T_c_1*T_c_2)**(1/2)



        if test_dict is not None:
            rho = test_dict["rho"]
        else:
            rho = rho_1_scaled/M_1 + rho_2_scaled/M_2


        delta = (rho)/(rho_r)
        #print(delta)
        tau = T_r/T

        F_ij = 1 #for binary mixture



        # Note degeree starts at 1
        n_o1k,c_o1k,d_o1k,t_o1k,K_Pol_1,K_Exp_1 = get_data_alpha_r_oi("methane")
        n_o2k,c_o2k,d_o2k,t_o2k,K_Pol_2,K_Exp_2 = get_data_alpha_r_oi("hydrogen")
        n_ijk,d_ijk,t_ijk,K_Pol_ij,K_Exp_ij = get_data_alpha_r_ij()


        
        def alpha_r_o_1(delta_tilde,tau_tilde):
            result = 0
            for k in range(len(n_o1k)):
                if k+1 <= K_Pol_1:
                    result += n_o1k[k]*delta_tilde**(d_o1k[k])*tau_tilde**t_o1k[k]
                elif k+1 > K_Pol_1 and k+1 <= K_Exp_1+K_Pol_1: 
                    result += n_o1k[k]*delta_tilde**(d_o1k[k])*tau_tilde**t_o1k[k]*sym.exp(-delta_tilde**c_o1k[k])
                
            return result


        def alpha_r_o_2(delta_tilde,tau_tilde):
            result = 0
            for k in range(len(n_o2k)):
                if k+1 <= K_Pol_2:
                    result += n_o2k[k]*delta_tilde**(d_o2k[k])*tau_tilde**t_o2k[k]
                elif k+1 > K_Pol_2 and k+1 <= K_Exp_2+K_Pol_2: 
                    result += n_o2k[k]*delta_tilde**(d_o2k[k])*tau_tilde**t_o2k[k]*sym.exp(-delta_tilde**c_o2k[k])
                
            return result
        
        
        def alpha_r_ij(delta_tilde,tau_tilde):
            result = 0
            for k in range(len(n_ijk)):
                if k+1 <= K_Pol_ij:
                    result += n_ijk[k]*delta_tilde**(d_ijk[k])*tau_tilde**t_ijk[k]
                
            return result



        delta_tilde, tau_tilde = sym.symbols('delta_tilde tau_tilde')

        # symbolic differentiatons
        def alpha_r_o_1_deriv_symb(delta_tilde,tau_tilde):
            return sym.diff(alpha_r_o_1(delta_tilde,tau_tilde),delta_tilde)

        def alpha_r_o_2_deriv_symb(delta_tilde,tau_tilde):
            return sym.diff(alpha_r_o_2(delta_tilde,tau_tilde),delta_tilde)

        def alpha_r_ij_deriv_symb(delta_tilde,tau_tilde):
            return sym.diff(alpha_r_ij(delta_tilde,tau_tilde),delta_tilde)

            


        #conversions to numpy functions
        alpha_r_o_1_deriv = sym.lambdify((delta_tilde,tau_tilde),alpha_r_o_1_deriv_symb(delta_tilde,tau_tilde),"numpy")
        alpha_r_o_2_deriv = sym.lambdify((delta_tilde,tau_tilde),alpha_r_o_2_deriv_symb(delta_tilde,tau_tilde),"numpy")
        alpha_r_ij_deriv = sym.lambdify((delta_tilde,tau_tilde),alpha_r_ij_deriv_symb(delta_tilde,tau_tilde),"numpy")


        alpha_r_deriv = x_1 *alpha_r_o_1_deriv(delta ,tau ) \
                        + x_2 *alpha_r_o_2_deriv(delta ,tau ) \
                        +x_1 *x_2 *F_ij*alpha_r_ij_deriv(delta ,tau )



        # from kPc to bar 
        p = rho*R*T*(1+ delta* alpha_r_deriv)*1e3
        
        return p
        


    raise  ValueError(" model admits an invalid value !")


def pressure_law_mixtures_symbolic(model:str="speed_of_sound",T:float=283)->callable:
    """
    symbolic representation, returns a pressure function, which can be used by jax to get a derivaitve

    Args:
        model (str, optional): pressure law type. ["speed_of_sound","virial_expansion","virial_expansion_mix","gerg"].

    Returns:
        p1,p2: except for gerg, returns p
    """

    R = 8.314472
    #T = 288.15
    #T = 283

    # Do not rescale before -> otherwhise fitting incorrect
    M_1 = 16.042460
    M_2 = 2.015880


    if model == "virial_expansion":

        if np.abs(T-283)<4:

            # first virial coef
            # CH_4
            B_1 = -47.58
            # H_2
            B_2 =  14.6
            # second virial coef
            # CH_4
            C_1 = 2.440
            # H_2
            C_2 = 0.250
        
        elif np.abs(T-273.15)<4:

            # first virial coef
            # CH_4
            B_1 = -55.0
            # H_2
            B_2 =  (14.0 +14.1 + 14.6 + 13.7 + 13.5 + 13.8 + 13.7 + 13.0 + 13.6)/9
            # second virial coef
            # CH_4
            C_1 = 29.09
            # H_2
            C_2 = (0.305 + 0.423 + 0.923 + 0.415 + 0.389)/4
        
        else: 
            raise ValueError(f"Pressure not implemented for T={T}")
        
        

        # volume is measured in cm**3, and rho_1 given in kg/m**3

        def p_1(rho_1,rho_2):


            rho_1_scaled = rho_1*1e-3

            #units are cm^3
            p1 = 1/M_1*rho_1_scaled+B_1*(1/M_1)**2*rho_1_scaled**2+C_1*(1/M_1)**3*rho_1_scaled**3
            p1 *= R*T*1e6
            return p1


        def p_2(rho_1,rho_2):


            rho_2_scaled = rho_2*1e-3

            #units are cm^3
            p2 = 1/M_2*rho_2_scaled+B_2*(1/M_2)**2*rho_2_scaled**2+C_2*(1/M_2)**3*rho_2_scaled**3
            p2 *= R*T*1e6

            return p2

        return p_1,p_2




    elif model == "virial_expansion_mix":

        if np.abs(T-283)<4:

            # first/second pure  virial coefs (see above in virial expansions)
            # 
            B_1 = -47.58
            B_2 =  14.6
            #B_1 = 4.4344 *10 - 1.6608 *10**4/T - 3.5430 * 10**6/T**2 + 2.9832 *10**8/T**3 -2.3448 *10**10/T**4
            #B_2 = 1.7472 *10 - 1.2926 *10**2/T - 2.6988 * 10**5/T**2 + 8.0282* 10**6/T**3


            C_1 = 2.440
            C_2 = 0.250
            

            # mixed virial expansions
            B_11 = B_1
            B_12 = 9.2
            B_22 = B_2

            
            #B_12 = -1.0011*10**2 + 7.6037 *10**4/T - 1.2943*10**7/T**2

            C_111 = C_1
            C_112 = 0.6
            C_122 = 1.3
            C_222 = C_2
        
        elif np.abs(T-273.15)<4:

            # first virial coef
            # CH_4
            B_1 = -55.0
            # H_2
            B_2 =  (14.0 +14.1 + 14.6 + 13.7 + 13.5 + 13.8 + 13.7 + 13.0 + 13.6)/9
            # second virial coef
            # CH_4
            C_1 = 29.09
            # H_2
            C_2 = (0.305 + 0.423 + 0.923 + 0.415 + 0.389)/5
        
            # mixed virial expansions
            # TODO: check table and compare
            # TODO: in symbolic also copy
            B_11 = B_1
            B_12 = 3.7
            B_22 = B_2

            

            C_111 = C_1
            C_112 = 0.6
            C_122 = 1.4
            C_222 = C_2
        else: 
            raise ValueError(f"Pressure not implemented for T={T}")

        def p(rho_1,rho_2):
            
            
            rho_1_scaled = rho_1*1e-3
            rho_2_scaled = rho_2*1e-3

            #units are cm^3
            rho_mix = 1/M_1*rho_1_scaled+1/M_2*rho_2_scaled
            mu_mass = rho_1_scaled/(rho_1_scaled+rho_2_scaled) 
            x_1 = (mu_mass/M_1)/((mu_mass/M_1)+(1-mu_mass)/M_2)
            x_2 = 1-x_1
            
            B_mix = B_11*x_1**2+ 2*B_12*x_1*x_2+ 2*B_22*x_2**2

            C_mix = C_111*x_1**3+ 3*C_112*x_1**2*x_2 + 3*C_122*x_1*x_2**2+C_222*x_2**3

            p = rho_mix +B_mix*rho_mix**2+C_mix*rho_mix**3



            p *= R*T*1e6

            return p
        return p


    elif model == "speed_of_sound":

        def p_1(rho_1,rho_2):
        
            p_1 = R*T*(1/M_1*rho_1)*1e3
            return p_1

        def p_2(rho_1,rho_2):
            p_2 = R*T*(1/M_2*rho_2)*1e3
            return p_2

        return p_1,p_2
        
    elif model == "gerg":


        def p_gerg(rho_1,rho_2):

            # no scaling, since g/dm^3 is equivalent to kg/m^3
            rho_1_scaled = rho_1
            rho_2_scaled = rho_2


            # Do not rescale before -> otherwhise fitting incorrect
            M_1 = 16.042460
            M_2 = 2.015880


            ## Table A.3.5
            rho_c_1 = 10.139342719
            #critical density of hydrogen 
            rho_c_2 = 14.940000000

            #critical temperature of methane 
            T_c_1 = 190.564000000
            #critical temperature of hydrogen 
            T_c_2 = 33.190000000


            ## Table A.3.8
            # binary paramters for density
            beta_nu_ij = 1
            gamma_nu_ij = 1.018702573

            ## Table A.3.8
            # binary paramters for temperature 
            beta_T_ij = 1
            gamma_T_ij = 1.352643115



            F_ij = 1 #for binary mixture

            # compute molar composition
            mu_mass = rho_1_scaled/(rho_1_scaled+rho_2_scaled) 
            x_1 = (mu_mass/M_1)/((mu_mass/M_1)+(1-mu_mass)/M_2)
            x_2 = 1-x_1

            


            rho_r = 1/(x_1**2*(1/rho_c_1) + x_2**2*(1/rho_c_2) \
                    +2*x_1*x_2*beta_nu_ij*gamma_nu_ij*\
                    ((x_1+x_2)/(beta_nu_ij**2*x_1+x_2))*\
                    1/8*(1/(rho_c_1**(1/3))+1/(rho_c_2**(1/3)))**3)


            T_r =  x_1**2*T_c_1 + x_2**2*T_c_2 \
                    +2*x_1*x_2*beta_T_ij*gamma_T_ij*\
                    ((x_1+x_2)/(beta_T_ij**2*x_1+x_2))*\
                    (T_c_1*T_c_2)**(1/2)



            rho = rho_1_scaled/M_1 + rho_2_scaled/M_2


            delta = (rho)/(rho_r)
            #print(delta)
            tau = T_r/T

            # Note degeree starts at 1
            n_o1k,c_o1k,d_o1k,t_o1k,K_Pol_1,K_Exp_1 = get_data_alpha_r_oi("methane")
            n_o2k,c_o2k,d_o2k,t_o2k,K_Pol_2,K_Exp_2 = get_data_alpha_r_oi("hydrogen")
            n_ijk,d_ijk,t_ijk,K_Pol_ij,K_Exp_ij = get_data_alpha_r_ij()


            
            def alpha_r_o_1(delta_tilde,tau_tilde):
                result = 0
                for k in range(len(n_o1k)):
                    if k+1 <= K_Pol_1:
                        result += n_o1k[k]*delta_tilde**(d_o1k[k])*tau_tilde**t_o1k[k]
                    elif k+1 > K_Pol_1 and k+1 <= K_Exp_1+K_Pol_1: 
                        result += n_o1k[k]*delta_tilde**(d_o1k[k])*tau_tilde**t_o1k[k]*jnp.exp(-delta_tilde**c_o1k[k])
                    
                return result


            def alpha_r_o_2(delta_tilde,tau_tilde):
                result = 0
                for k in range(len(n_o2k)):
                    if k+1 <= K_Pol_2:
                        result += n_o2k[k]*delta_tilde**(d_o2k[k])*tau_tilde**t_o2k[k]
                    elif k+1 > K_Pol_2 and k+1 <= K_Exp_2+K_Pol_2: 
                        result += n_o2k[k]*delta_tilde**(d_o2k[k])*tau_tilde**t_o2k[k]*jnp.exp(-delta_tilde**c_o2k[k])
                    
                return result
            
            
            def alpha_r_ij(delta_tilde,tau_tilde):
                result = 0
                for k in range(len(n_ijk)):
                    if k+1 <= K_Pol_ij:
                        result += n_ijk[k]*delta_tilde**(d_ijk[k])*tau_tilde**t_ijk[k]
                    
                return result



            alpha_r_o_1_array = jnp.vectorize(grad(alpha_r_o_1, argnums=0))
            alpha_r_o_2_array = jnp.vectorize(grad(alpha_r_o_2,argnums=0))
            alpha_r_ij_array = jnp.vectorize(grad(alpha_r_ij,argnums=0))

            

            rho = rho_1_scaled/M_1 + rho_2_scaled/M_2


            delta = (rho)/(rho_r)
            tau = T_r/T
            
            #New symbolic derivative
            alpha_r_deriv_symb = x_1 *alpha_r_o_1_array(delta,tau)\
                            + x_2 *alpha_r_o_2_array(delta,tau)\
                            +x_1 *x_2 *F_ij*alpha_r_ij_array(delta,tau)

            # from dm to m
            p = rho*R*T*(1+ delta* alpha_r_deriv_symb)*1e3 

            return p
        
        return p_gerg



def fit_gerg_simple(poly_deg:int,penalty:float=0,allow_non_simple:bool=False,T:float= 273.15,
                    return_energy:bool=False):
    """
    least squares fit to gerg as simple pressures

    Args:
        poly_deg (int): degree to fit
        poly_deg (int): penalty ridge regression

        poly_deg (int): _description_
        penalty (float, optional): ridge regression penalty. Defaults to 0.
        allow_non_simple (bool, optional): if non simple term (rho_1+rho_2)**2.
          Defaults to False.
        T (float, optional): temperature. Defaults to 283.15.
        return_energy (bool, optional): returns the energy functional. Defaults to False.

    Returns:
        p1,p2: simple pressures functions
    """

    rho_1 = np.linspace(0.1, 65, 2000)
    rho_2 = np.linspace(0.1, 8, 2000)
    



    rho_1_mesh,rho_2_mesh = np.meshgrid(rho_1,rho_2)




    p_gerg_data_all = pressure_law_mixtures(rho_1_mesh,rho_2_mesh,"gerg",T)

    rho_1_data = rho_1_mesh[p_gerg_data_all<=85*1e5]
    rho_2_data = rho_2_mesh[p_gerg_data_all<=85*1e5]
    p_gerg_data = p_gerg_data_all[p_gerg_data_all<=85*1e5]

    

    #least squares matrix
    rho_1_ex = np.expand_dims(rho_1_data,axis=1)
    rho_2_ex = np.expand_dims(rho_2_data,axis=1)
    ones  = np.ones((rho_1_data.shape[0],1))
    data = [ones]+ [rho_1_ex**i for i in range(1,poly_deg+1)] + [rho_2_ex**i for i in range(1,poly_deg+1)]
    if allow_non_simple: 
        for i in range(poly_deg+1):
            for j in range(1,i+1):
                data += [rho_1_ex**i*rho_2_ex**j]
    A = np.concatenate(data,axis=1)

    b = p_gerg_data

    B = A.T@A
    B += penalty*np.eye(B.shape[0])
    W = np.linalg.solve(B,A.T@b)

    
    def p1(rho_1,rho_2):
        p1 = 1/2*W[0] 
        for i in range(1,poly_deg+1):
            p1 +=  W[i]*rho_1**i 
        return p1

    def p2(rho_1,rho_2):
        p2 = 1/2*W[0] 
        for i in range(poly_deg+1,2*poly_deg+1):
            p2 +=  W[i]*rho_2**(i-poly_deg)
        return p2
    
    if allow_non_simple:
        def p_ns(rho_1,rho_2):
            p_ns = 0
            count = 2*poly_deg+1
            for i in range(poly_deg+1):
                for j in range(1,i+1):
                    p_ns += W[count]*rho_1**i*rho_2**j
                    count += 1
            return  p_ns
    else:
        p_ns = None
    
    if return_energy:
        def e(rho_1,rho_2):
            p = -1/2*W[0] 
            p += W[1]*jnp.log(rho_1)*rho_1
            for i in range(2,poly_deg+1):
                p +=  1/(i-1)*W[i]*rho_1**i 

            p += -1/2*W[0] 
            p += W[poly_deg+1]*jnp.log(rho_2)*rho_2
            for i in range(poly_deg+2,2*poly_deg+1):
                p +=  1/(i-poly_deg-1)*W[i]*rho_2**(i-poly_deg) 
            
            if allow_non_simple:
                p += W[-1]*(rho_1+rho_2)**2
            return p
            
        return p1,p2,p_ns,e
    
    return p1,p2,p_ns




def get_data_alpha_r_oi(model:str="methane"):
    """
    data for gerg

    Args:
        model (str, optional): methane or hydrogen. Defaults to "methane".

    Returns:
        n_o1k,c_o1k,d_o1k,t_o1k,K_Pol_1,K_Exp_1 (list):data 
    """

    
    if model == "methane":

        # Table A3.3
        n_o1k = [
            0.57335704239162,
            -0.16760687523730*10,
            0.23405291834916,
            -0.21947376343441,
            0.16369201404128*10**(-1),
            0.15004406389280*10**(-1),
            0.98990489492918*10**(-1),
            0.58382770929055,
            -0.74786867560390,
            0.30033302857974,
            0.20985543806568,
            -0.18590151133061*10**(-1),
            -0.15782558339049,
            0.12716735220791,
            -0.32019743894346*10**(-1),
            -0.68049729364536*10**(-1),
            0.24291412853736*10**(-1),
            0.51440451639444*10**(-2),
            -0.19084949733532*10**(-1),
            0.55229677241291*10**(-2),
            -0.44197392976085*10**(-2),
            0.40061416708429*10**(-1),
            -0.33752085907575*10**(-1),
            -0.25127658213357*10**(-2)
        ]
        
        c_o1k = [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            6,
            6,
            6,
            6
        ]

        d_o1k = [
            1,
            1,
            2,
            2,
            4,
            4,
            1,
            1,
            1,
            2,
            3,
            6,
            2,
            3,
            3,
            4,
            4,
            2,
            3,
            4,
            5,
            6,
            6,
            7,
        ]

        t_o1k = [
            0.125,
            1.125,
            0.375,
            1.125,
            0.625,
            1.500,
            0.625,
            2.625,
            2.750,
            2.125,
            2.000,
            1.750,
            4.500,
            4.750,
            5.000,
            4.000,
            4.500,
            7.500,
            14.000,
            11.500,
            26.000,
            28.000,
            30.000,
            16.000,
        ]

        K_Pol_1 = 6
        K_Exp_1 = 18 

    elif model == "hydrogen":
        # Table A3.4

        n_o1k = [
            0.53579928451252*10**1,
            -0.62050252530595*10**1,
            0.13830241327086,
            -0.71397954896129*10**(-1),
            0.15474053959733*10**(-1),
            -0.14976806405771,
            -0.26368723988451*10**(-1),
            0.56681303156066*10**(-1),
            -0.60063958030436*10**(-1),
            -0.45043942027132,
            0.42478840244500,
            -0.21997640827139*10**(-1),
            -0.10499521374530*10**(-1),
            -0.28955902866816*10**(-2)
        ]
        
        c_o1k = [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            2,
            2,
            3,
            3,
            5
        ]

        d_o1k = [
            1,
            1,
            2,
            2,
            4,
            1,
            5,
            5,
            5,
            1,
            1,
            2,
            5,
            1
        ]

        t_o1k = [
        0.500,
        0.625,
        0.375,
        0.625,
        1.125,
        2.625,
        0.000,
        0.250,
        1.375,
        4.000,
        4.250,
        5.000,
        8.000,
        8.000
        ]

        K_Pol_1 = 5
        K_Exp_1 = 9

    return n_o1k,c_o1k,d_o1k,t_o1k,K_Pol_1,K_Exp_1



def get_data_alpha_r_ij():
    """
    data for gerg

    Returns:
        n_ijk,d_ijk,t_ijk,K_Pol_ij,K_Exp_ij(list): data gerg 
    """
    
    # Table A3.7
    n_ijk = [
        -0.25157134971934,
        -0.62203841111983*10**(-2),
        0.88850315184396*10**(-1),
        -0.35592212573239*10**(-1)
    ]
    

    d_ijk = [
        1,
        3,
        3,
        4
    ]

    t_ijk = [
        2.000,
        -1.000,
        1.750,
        1.400
    ]

    K_Pol_ij = 4
    K_Exp_ij = 0

    
    return n_ijk,d_ijk,t_ijk,K_Pol_ij,K_Exp_ij

def gerg_test_simple():
    """
    tests whether gerg is a simple pressure law
    """

    p_gerg = pressure_law_mixtures_symbolic("gerg")

    p_d1 = jnp.vectorize(grad(p_gerg,argnums=0))
    p_d2 = jnp.vectorize(grad(p_gerg,argnums=0))
    p_d1_d2 = jnp.vectorize(grad(p_d1,argnums=1))
    p_d1_d1 = jnp.vectorize(grad(p_d1,argnums=0))
    p_d2_d2 = jnp.vectorize(grad(p_d2,argnums=1))

    x = jnp.linspace(2,40,500)
    y = jnp.linspace(1.5,5,500)
    X,Y = jnp.meshgrid(x,y)

    Z = p_d1_d2(X,Y)/(p_d1_d1(X,Y)+p_d2_d2(X,Y))
    p_gerg_masked = np.ma.masked_where(p_gerg(X,Y) > 80*1e5, Z)
    #print(f" gerg is simple: {np.linalg.norm(Z)}")
    fig,ax = plt.subplots()
    surf = ax.contourf(X, Y, p_gerg_masked, antialiased=False)
    ax.set_title("$\\frac{|\\partial_{\\rho_{1} \\rho_{2}} p |}{|\\partial_{\\rho_{1}}^2 p | +|\\partial_{\\rho_{2}}^2 p|}$")
    fig.colorbar(surf,label=f"$p$ bar" )
    fig.savefig("graphics/pressure_plots/gerg_simple.png")
    fig.savefig("graphics/pressure_plots/gerg_simple.pdf")
    plt.show()
    


def pressure_test_convexity(p1:callable,p2:callable,p_non_simple:callable,model_name:str):
    """
    tests whether a pressure law is simple
    NOTE: that p1,p2 are simple parts of partial pressure, but p_non_simple is the non
    simple ENERGY part

    Args:
        p1 (callable):partial simple pressure 
        p2 (callable)::partial simple pressure 
        p_non_simple (callable): non simple part of energy
        model_name (str): save name 
    """


    p1_d1 = jnp.vectorize(grad(p1,argnums=0))
    p2_d2 = jnp.vectorize(grad(p2,argnums=1))
    
    if p_non_simple is not None:
        p_ns_d1 = jnp.vectorize(grad(p_non_simple,argnums=0))
        p_ns_d2 = jnp.vectorize(grad(p_non_simple,argnums=1))
        p_ns_d1_d1 = jnp.vectorize(grad(p_ns_d1,argnums=0))
        p_ns_d1_d2 = jnp.vectorize(grad(p_ns_d1,argnums=1))
        p_ns_d2_d2 = jnp.vectorize(grad(p_ns_d2,argnums=1))

    x = np.linspace(0.1, 65, 500)
    y = np.linspace(0.1, 8, 500)
    X,Y = jnp.meshgrid(x,y)

    Z = (p1_d1(X,Y)/X*p2_d2(X,Y)/Y  )*1e-5
    if p_non_simple is not None:
        Z += (p_ns_d1_d1(X,Y)*p_ns_d2_d2(X,Y)-p_ns_d1_d2(X,Y)**2)*1e-5

    if p_non_simple:
        hess_min = np.min(Z[p1(X,Y)+p2(X,Y)+p_non_simple(X,Y) < 80*1e5])
    else:
        hess_min = np.min(Z[p1(X,Y)+p2(X,Y) < 80*1e5])
    print(f"min of hessian matrix {hess_min}")
    p_masked = np.ma.masked_where(p1(X,Y)+p2(X,Y) > 80*1e5, Z)
    #print(f" gerg is simple: {np.linalg.norm(Z)}")
    fig,ax = plt.subplots()
    surf = ax.contourf(X, Y, p_masked, antialiased=False)
    ax.set_title("$\\mathrm{det(E'')}$ and $\\min(detE'')=$" +f"{hess_min:.0f}")
    fig.colorbar(surf,label=f"$p$ bar" )
    fig.savefig(f"graphics/pressure_plots/{model_name}_convex.pdf")

def gerg_test_values():
    """
    test gerg with different values
    """
    T = 288.15
    test_dict = {
        "x_1":[0.98,0.95,0.8,0.6,0.4,0.2,0.8,0.6,0.4,0.2,0.6,0.6],
        "rho": [2.802,2.776,2.666,2.561,2.490,2.443,1.294,1.269,1.250,1.237,0.209,5.148],
        "p":[60,60,60,60,60,60,30,30,30,30,5,120]
        }
    for i,x1 in enumerate(test_dict["x_1"]):
        param_dict = {
        "x_1":x1,
        "rho": test_dict["rho"][i],
        "p": test_dict["p"][i]
        }
        p = pressure_law_mixtures(None,None,"gerg",T,param_dict)*1e-5
        print(np.abs(p-test_dict["p"][i]))

        T = 300
        param_dict = {
        "x_1":0.8,
        "rho": 2.533,
        "p": 60
        }
        p = pressure_law_mixtures(None,None,"gerg",T,param_dict)*1e-5
        print(np.abs(p-test_dict["p"][i]))
    

def gerg_energy():
    R = 8.314472
    #T = 288.15
    T = 273.15


    def p_gerg_energy(rho_1,rho_2):

        # no scaling, since g/dm^3 is equivalent to kg/m^3
        rho_1_scaled = rho_1
        rho_2_scaled = rho_2


        # Do not rescale before -> otherwhise fitting incorrect
        M_1 = 16.042460
        M_2 = 2.015880


        # account for m**(-3) instead of dm**(-3)
        factor = 10**3        #critical density of methane 
        factor = 1        #critical density of methane 
        rho_c_1 = 10.139342719*factor
        #critical density of hydrogen 
        rho_c_2 = 14.940000000*factor

        #critical temperature of methane 
        T_c_1 = 190.564000000
        #critical temperature of hydrogen 
        T_c_2 = 33.190000000


        ## Table A.3.8
        # binary paramters for density
        beta_nu_ij = 1
        gamma_nu_ij = 1.018702573

        ## Table A.3.8
        # binary paramters for temperature 
        beta_T_ij = 1
        gamma_T_ij = 1.352643115



        F_ij = 1 #for binary mixture

        # compute molar composition
        mu_mass = rho_1_scaled/(rho_1_scaled+rho_2_scaled) 
        x_1 = (mu_mass/M_1)/((mu_mass/M_1)+(1-mu_mass)/M_2)
        x_2 = 1-x_1

        


        rho_r = 1/(x_1**2*(1/rho_c_1) + x_2**2*(1/rho_c_2) \
                +2*x_1*x_2*beta_nu_ij*gamma_nu_ij*\
                ((x_1+x_2)/(beta_nu_ij**2*x_1+x_2))*\
                1/8*(1/(rho_c_1**(1/3))+1/(rho_c_2**(1/3)))**3)


        T_r =  x_1**2*T_c_1 + x_2**2*T_c_2 \
                +2*x_1*x_2*beta_T_ij*gamma_T_ij*\
                ((x_1+x_2)/(beta_T_ij**2*x_1+x_2))*\
                (T_c_1*T_c_2)**(1/2)



        rho = rho_1_scaled/M_1 + rho_2_scaled/M_2


        delta = (rho)/(rho_r)
        #print(delta)
        tau = T_r/T

        # Note degeree starts at 1
        n_o1k,c_o1k,d_o1k,t_o1k,K_Pol_1,K_Exp_1 = get_data_alpha_r_oi("methane")
        n_o2k,c_o2k,d_o2k,t_o2k,K_Pol_2,K_Exp_2 = get_data_alpha_r_oi("hydrogen")
        n_ijk,d_ijk,t_ijk,K_Pol_ij,K_Exp_ij = get_data_alpha_r_ij()


        
        def alpha_r_o_1(delta_tilde,tau_tilde):
            result = 0
            for k in range(len(n_o1k)):
                if k+1 <= K_Pol_1:
                    result += n_o1k[k]*delta_tilde**(d_o1k[k])*tau_tilde**t_o1k[k]
                elif k+1 > K_Pol_1 and k+1 <= K_Exp_1+K_Pol_1: 
                    result += n_o1k[k]*delta_tilde**(d_o1k[k])*tau_tilde**t_o1k[k]*jnp.exp(-delta_tilde**c_o1k[k])
                
            return result


        def alpha_r_o_2(delta_tilde,tau_tilde):
            result = 0
            for k in range(len(n_o2k)):
                if k+1 <= K_Pol_2:
                    result += n_o2k[k]*delta_tilde**(d_o2k[k])*tau_tilde**t_o2k[k]
                elif k+1 > K_Pol_2 and k+1 <= K_Exp_2+K_Pol_2: 
                    result += n_o2k[k]*delta_tilde**(d_o2k[k])*tau_tilde**t_o2k[k]*jnp.exp(-delta_tilde**c_o2k[k])
                
            return result
        
        
        def alpha_r_ij(delta_tilde,tau_tilde):
            result = 0
            for k in range(len(n_ijk)):
                if k+1 <= K_Pol_ij:
                    result += n_ijk[k]*delta_tilde**(d_ijk[k])*tau_tilde**t_ijk[k]
                
            return result




        rho = rho_1_scaled/M_1 + rho_2_scaled/M_2


        delta = (rho)/(rho_r)
        tau = T_r/T
        
        #New symbolic derivative
        alpha_r_deriv_symb = x_1 *alpha_r_o_1(delta,tau)\
                        + x_2 *alpha_r_o_2(delta,tau)\
                        +x_1 *x_2 *F_ij*alpha_r_ij(delta,tau)



        # from dm to m
        #p =  (1/M1*rho_1+1/M2)*alpha_r_deriv_symb*1e3 
        p =  alpha_r_deriv_symb*1e3/rho_r*T_r/(R*T*rho)

        return p
    return p_gerg_energy



def gerg_energy_test():

    R = 8.314472

    #T = 288.15
    T = 273.15

    M1 = 16.042460/1000
    M2 = 2.015880/1000

    rho_c_1 = 10.139342719
    rho_c_2 = 14.940000000
    
    p_gerg_energy = gerg_energy()
    psi_1 = grad(p_gerg_energy,argnums=0)
    psi_2 = grad(p_gerg_energy,argnums=1)
    p = lambda rho_1,rho_2: R*T*(1/M1*rho_1+1/M2*rho_2)+R*T*rho_1*psi_1(rho_1,rho_2)+R*T*rho_2*psi_2(rho_1,rho_2) \
        -R*T*(rho_1+rho_2)*p_gerg_energy(rho_1,rho_2)


    #p = lambda rho_1,rho_2: R*T*(1/M1*rho_1+1/M2*rho_2+ rho_1/M1*psi_1(rho_1,rho_2)+rho_2/M2*psi_2(rho_1,rho_2) \
    #    -(rho_1/M1+rho_2/M2)*p_gerg_energy(rho_1,rho_2))
    #print(p(1.0,2.0))


    #p = lambda rho_1,rho_2: R*T*(1/M1*rho_1+1/M2*rho_2+ rho_1/(M1*rho_c_1)*psi_1(rho_1,rho_2)+rho_2/(M2*rho_c_2)*psi_2(rho_1,rho_2) \
    #    -(rho_1/(M1*rho_c_1)+rho_2/(M2*rho_c_2))*p_gerg_energy(rho_1,rho_2))
    #print(p(1.0,2.0))


    #p = lambda rho_1,rho_2: R*T*(1/M1*rho_1+1/M2)*(psi_1(rho_1,rho_2)+psi_2(rho_1,rho_2))
    x = np.linspace(1,5,10)
    y = np.linspace(1,5,10)
    X,Y = np.meshgrid(x,y)
    p_bothe = p(10.0,1.0)
    p_gerg = pressure_law_mixtures(10.0,1.0,"gerg")
    factor = p_bothe/p_gerg

    rho_1 = 10.0
    rho_2 = 2.0
    p_bothe = p(rho_1,rho_2)
    p_gerg = pressure_law_mixtures(rho_1,rho_2,"gerg")
    print(f"bothe dreyer {p_bothe}")
    print(f"gerg {p_gerg}")


def get_pressure_derivative(T:float=273.15):
    """
    returns the partial derivatives of gerg as numpy functions
    Args:
        T (float, optional): temperature. Defaults to 273.15.15.
    """

    
    def p_gerg(delta_tilde,tau_tilde,x_1,x_2,rho):

        # Note degeree starts at 1
        n_o1k,c_o1k,d_o1k,t_o1k,K_Pol_1,K_Exp_1 = get_data_alpha_r_oi("methane")
        n_o2k,c_o2k,d_o2k,t_o2k,K_Pol_2,K_Exp_2 = get_data_alpha_r_oi("hydrogen")
        n_ijk,d_ijk,t_ijk,K_Pol_ij,K_Exp_ij = get_data_alpha_r_ij()
        
        
        alpha_r_o_1 = 0
        for k in range(len(n_o1k)):
            if k+1 <= K_Pol_1:
                alpha_r_o_1 += n_o1k[k]*delta_tilde**(d_o1k[k])*tau_tilde**t_o1k[k]
            elif k+1 > K_Pol_1 and k+1 <= K_Exp_1+K_Pol_1: 
                alpha_r_o_1 += n_o1k[k]*delta_tilde**(d_o1k[k])*tau_tilde**t_o1k[k]*sym.exp(-delta_tilde**c_o1k[k])
            


        alpha_r_o_2 = 0
        for k in range(len(n_o2k)):
            if k+1 <= K_Pol_2:
                alpha_r_o_2 += n_o2k[k]*delta_tilde**(d_o2k[k])*tau_tilde**t_o2k[k]
            elif k+1 > K_Pol_2 and k+1 <= K_Exp_2+K_Pol_2: 
                alpha_r_o_2 += n_o2k[k]*delta_tilde**(d_o2k[k])*tau_tilde**t_o2k[k]*sym.exp(-delta_tilde**c_o2k[k])
            
    
    
        alpha_r_ij = 0
        for k in range(len(n_ijk)):
            if k+1 <= K_Pol_ij:
                alpha_r_ij += n_ijk[k]*delta_tilde**(d_ijk[k])*tau_tilde**t_ijk[k]
            

        alpha_r_o_1_deriv_symb = sym.diff(alpha_r_o_1,delta_tilde)
        alpha_r_o_2_deriv_symb = sym.diff(alpha_r_o_2,delta_tilde)
        alpha_r_ij_deriv_symb = sym.diff(alpha_r_ij,delta_tilde)

        F_ij = 1 #for binary mixture
        alpha_r_deriv = x_1 *alpha_r_o_1_deriv_symb \
                        + x_2 *alpha_r_o_2_deriv_symb \
                        +x_1 *x_2 *F_ij*alpha_r_ij_deriv_symb



        # from kPc to bar 
        R = 8.314472
        p = rho*R*T*(1+ delta_tilde* alpha_r_deriv)*1e3
        return p


    # Do not rescale before -> otherwhise fitting incorrect
    M_1 = 16.042460
    M_2 = 2.015880


    rho_1, rho_2, delta_tilde, tau_tilde, rho = sym.symbols('rho_1 rho_2 delta_tilde tau_tilde rho')
    delta = sym.Function("delta")(rho_1,rho_2)
    tau = sym.Function("tau")(rho_1,rho_2)
    rho = sym.Function("rho")(rho_1,rho_2)

    # compute molar composition
    rho_1_scaled = rho_1
    rho_2_scaled = rho_2
    mu_mass = rho_1_scaled/(rho_1_scaled+rho_2_scaled) 
    x_1 = (mu_mass/M_1)/((mu_mass/M_1)+(1-mu_mass)/M_2)
    x_2 = 1-x_1


    ## Table A.3.5
    rho_c_1 = 10.139342719
    #critical density of hydrogen 
    rho_c_2 = 14.940000000

    #critical temperature of methane 
    T_c_1 = 190.564000000
    #critical temperature of hydrogen 
    T_c_2 = 33.190000000


    ## Table A.3.8
    # binary paramters for density
    beta_nu_ij = 1
    gamma_nu_ij = 1.018702573

    ## Table A.3.8
    # binary paramters for temperature 
    beta_T_ij = 1
    gamma_T_ij = 1.352643115


    rho_r = 1/((x_1)**2*(1/rho_c_1) + x_2**2*(1/rho_c_2) \
            +2*x_1*x_2*beta_nu_ij*gamma_nu_ij*\
            ((x_1+x_2)/(beta_nu_ij**2*x_1+x_2))*\
            1/8*(1/(rho_c_1**(1/3))+1/(rho_c_2**(1/3)))**3)



    T_r =  x_1**2*T_c_1 + x_2**2*T_c_2 \
            +2*x_1*x_2*beta_T_ij*gamma_T_ij*\
            ((x_1+x_2)/((beta_T_ij**2)*x_1+x_2))*\
            (T_c_1*T_c_2)**(1/2)



    rho = rho_1_scaled/M_1 + rho_2_scaled/M_2


    delta = (rho)/(rho_r)
    #print(delta)
    tau = T_r/T





    p_gerg_sym =  p_gerg(delta_tilde,tau_tilde,x_1,x_2,rho)
    p_gerg_sym_sub = p_gerg_sym.subs(delta_tilde,delta)
    p_gerg_sym_sub = p_gerg_sym_sub.subs(tau_tilde,tau)
    #test_sub = alpha_r_o_1_deriv_symb.subs(delta_tilde,delta)
    #test_sub = test_sub.subs(tau_tilde,tau)
    rho_1_deriv = sym.diff(p_gerg_sym_sub,rho_1)
    #rho_1_deriv_simplified = sym.simplify(rho_1_deriv)
    rho_1_deriv_np = sym.lambdify((rho_1, rho_2), rho_1_deriv, modules='numpy')


    rho_2_deriv = sym.diff(p_gerg_sym_sub,rho_2)
    #rho_1_deriv_simplified = sym.simplify(rho_1_deriv)
    rho_2_deriv_np = sym.lambdify((rho_1, rho_2), rho_2_deriv, modules='numpy')

    #print("Comp_time")
    #tmp = rho_1_deriv_np(2*np.ones(10),np.ones(10))


    #print(tmp)

    #p = pressure_law_mixtures_symbolic(model="gerg",T=283.15)

    #p_deriv = jax.jacfwd(p)
    #test = p_deriv(2*jnp.ones(10),jnp.ones(10))
    #print(test)
    return rho_1_deriv_np,rho_2_deriv_np 
    

        

        



if __name__ == "__main__":


    plot_pressure_mixtures(black_white=True)

    #plot_gerg_approximation(relative_error=False,black_white=True)
    
    #gerg_test_simple()

    #gerg_energy_test()

    #gerg_test_values()

    #different results for 32 vs 64 bit
    #print(pressure_law_mixtures(0.0,1.0,"gerg"))
    #print(pressure_law_mixtures_symbolic("gerg")(0.0,1.0))

    #get_pressure_derivative()

