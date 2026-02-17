import numpy as np
import matplotlib.pyplot as plt
import jax

from scipy.integrate import solve_ivp
from scipy.optimize import newton

from jax import numpy as jnp
from jax import grad

from pressure_laws_comparison import pressure_law_mixtures,pressure_law_mixtures_symbolic


def newton_solver_bc_data_1_v(m1:float,m2:float,p:float,p_fct:callable,x0:np.ndarray,a:float):
    """
    newton solver to find bc data

    Args:
        m1 (float):mass flow in kg/m^3 
        m2 (float):mass flow in kg/m^3 
        p (float): pressure in Pa 
        p_fct (callable): pressure function
        x0 (np.ndarray): bc value
        a (float): cross section area
    """


    def F(x):
        result = p-p_fct(m1/(a*x),m2/(a*x))
        return result
    
    bc_data,inf_dict = newton(F,x0=x0,maxiter=1000,tol=1e-8,full_output=True)
    print(f"bc data Newton Solver {inf_dict['converged']}")
    if not inf_dict['converged']:
        raise Exception("Newton solver for bc data did not converge !")

    return bc_data

    
def get_bc_data_1v(m1:float,m2:float,p:float,a:float,model="speed_of_sound",T:float=273.15,p_precomp:callable=None)->np.ndarray:
    """
    returns the bc data related to m1,m2,p

    Args:
        m1 (float):mass flow in kg/m^3 
        m2 (float):mass flow in kg/m^3 
        p (float): pressure in Pa 
        p_fct (callable): pressure function
        a (float): cross section area
        model (str, optional): pressure law type. ["speed_of_sound","virial_expansion","gerg"].
        Defaults to "speed_of_sound".
        p_precomp (callable): precomputed pressure law, which is used instead of the model.
        Defaults to None.

    Raises:
        Exception: if no interval for the bc velocity is found
    Returns:
        np.ndarray: bc data [rho_1,rho_2,v]
    """
    # compute bc guess for Newton solver
    c1,c2 = pressure_law_mixtures(1,1,"speed_of_sound",return_partial_pressures=True,T=T)

    rho_2_guess = p*m2/(c1*(m1+c2*m2/c1))
    rho_1_guess = (p-c2*rho_2_guess)/c1
    v_guess = (m1+m2)/(a*(rho_1_guess+rho_2_guess))

    if model == "speed_of_sound":
        return np.array([rho_1_guess,rho_2_guess,v_guess]) 

    if p_precomp is None:
        p_law = lambda rho_1,rho_2: pressure_law_mixtures(rho_1=rho_1,rho_2=rho_2,model=model,T=T)
    else:
        p_law = p_precomp


    # brute force bc value
    #find interval, which contains the correct value
    factor = 1.5 
    # number of interval enlargements
    n_inv_max = 15
    # max number of points
    n_points_max = 10000
    # step size between points
    tau = 1e-2
    middle =  p_law(m1/(a*v_guess),m2/(a*v_guess))-p
    lhs = middle
    rhs = middle

    i = 0
    while middle*lhs>0 and middle*rhs>0 and i<=n_inv_max:
        middle =  p_law(m1/(a*v_guess),m2/(a*v_guess))-p
        lhs = p_law(m1/(a*factor*v_guess),m2/(a*factor*v_guess))-p
        rhs = p_law(m1/(a/factor*v_guess),m2/(a/factor*v_guess))-p
        factor *= 2
        i+=1

    if middle*lhs<0:
        n_points = min(int(np.round(np.abs(factor*v_guess-v_guess)/tau)),n_points_max)
        x = np.sign(v_guess)*np.linspace(np.abs(v_guess),factor*np.abs(v_guess),n_points)
    elif middle*rhs<0:
        n_points = min(int(np.round(np.abs(v_guess-v_guess/factor)/tau)),n_points_max)
        x = np.sign(v_guess)*np.linspace(np.abs(v_guess)/factor,np.abs(v_guess),n_points)
    else: 
        raise Exception(f"Failed finding a root for the bc data with n_inv_max={n_inv_max}")
    
    
    i = np.argmin(np.abs(p_law(m1/(a*x),m2/(a*x))-p))
    
    v_0 = newton_solver_bc_data_1_v(m1=m1,m2=m2,p=p,p_fct=p_law,x0=x[i],a=a)

    rho_1_0 = m1/(a*v_0)
    rho_2_0 = m2/(a*v_0)

    bc_data = np.array([rho_1_0,rho_2_0,v_0])
    
    return bc_data



def newton_solver_bc_data_2_v(m1:float,m2:float,p1:float,p2:float,
                                    p_1_fct:callable,p_2_fct:callable,x0:np.ndarray,a:float):
    """
    newton solver for bc data for the 2 velocity ode

    Args:
        m1 (float):mass flow in kg/m^3 
        m2 (float):mass flow in kg/m^3 
        p1 (float): partial pressure in Pa 
        p2 (float): partial pressure in Pa 
        a (float): cross section area
        p_1_fct (callable): partial pressure function  p1
        p_2_fct (callable): partial pressure function  p2
        x0 (np.ndarray): bc value
        a (float): cross section area
    """


    def F(x):
        m_1_solv,m_2_solv,v_1_solv,v_2_solv = x
        rho_1 = (m_1_solv)/(a*v_1_solv)
        rho_2 = (m_2_solv)/(a*v_2_solv)
        result = np.array([m1-m_1_solv,m2-m_2_solv,p1-p_1_fct(rho_1,rho_2),p2-p_2_fct(rho_1,rho_2)])
        print(result)
        return result

    bc_data = newton(F,x0=x0,maxiter=1000,tol=1e-8)

    return bc_data

    
def get_bc_data_2v(m1:float,m2:float,p1:float,p2:float,a:float,model="speed_of_sound",T:float=283):
    """
    returns the bc data for the 2 velocity problem

    Args:
        m1 (float): mass flow in kg/m^3 
        m2 (float): mass flow in kg/m^3 
        p1 (float): partial pressure in Pa 
        p2 (float): partial pressure in Pa 
        a (float): cross section area
        model (str, optional): pressure law type. ["speed_of_sound","virial_expansion","gerg"].

    Returns:
        np.ndarray: bc data [rho_1,rho_2,v_1,v_2]
    """

    # compute bc guess for Newton solver
    c1,c2 = pressure_law_mixtures(1,1,"speed_of_sound",return_partial_pressures=True,T=T)

    rho_1_guess = (p1)/c1
    rho_2_guess = (p2)/c2
    v_1_guess = m1/(a*rho_1_guess)
    v_2_guess = m2/(a*rho_2_guess)


    if model == "speed_of_sound":
        return np.array([rho_1_guess,rho_2_guess,v_1_guess,v_2_guess]) 

    bc_guess = np.array([m1,m2,v_1_guess,v_2_guess])
    p_1_fct,p_2_fct = pressure_law_mixtures_symbolic(model=model,T=T)
    m_1_0, m_2_0, v_1,v_2 = newton_solver_bc_data_2_v(m1=m1,m2=m2,p1=p1,p2=p2,p_1_fct=p_1_fct
                                                           ,p_2_fct=p_2_fct,a=a,x0=bc_guess)

    rho_1_0 = m_1_0/(a*v_1)
    rho_2_0 = m_2_0/(a*v_2)

    bc_data = np.array([rho_1_0,rho_2_0,v_1,v_2])
    
    return bc_data

    








def solve_ode_1v(rho_1_0:np.ndarray,rho_2_0:np.ndarray,v_0:np.ndarray, lmb:float,
                 p_1_d:callable,p_2_d:callable,pipe_length:float,pipe_diameter:float,
                 algebraic:bool=False):
    """
    solves the ode
    Args:
        rho_1_0 (np.ndarray): bc rho_1 in kg/m^3
        rho_2_0 (np.ndarray):  bc rho_1 kg/m^3
        v_0 (np.ndarray): bc v in m/s
        lmb (float): friction parameter
        p_1_d (callable): derivative w.r.t rho_1 
        p_2_d (callable): derivative w.r.t rho_1 
        pipe_length (float): length pipe in m
        pipe_diameter (float): diameter pipe in m

    Returns:
        result_dict: solution 
    """
    eps = 0 if algebraic else 1

    # rescalling non_linaer term
    # Note that \partial_x m = 0 in partial_x( (rho)*v**2) = rho*v \partial_x v
    def F(t,y):
        rho_1,rho_2,v = y
        lhs_mult = np.array(
            [ [v , 0,rho_1],
            [0 , v,rho_2],
            [p_1_d(rho_1,rho_2) ,p_2_d(rho_1,rho_2),eps*(rho_1+rho_2)*v]]
        )
        eq_rhs = -lmb/(2*pipe_diameter)*(rho_1+rho_2)*np.abs(v)*v 
        rhs = np.array([0,0,eq_rhs])
        return np.linalg.solve(lhs_mult,rhs)

    y0 = np.array([rho_1_0,rho_2_0,v_0])
    t_eval = np.linspace(0,pipe_length,100)
    result = solve_ivp(fun=F,t_span=(0,pipe_length),y0=y0,t_eval=t_eval)

    rho_1,rho_2,v = result["y"][:,-1]
    p_out = p_1_d(rho_1,rho_1)*rho_1+p_2_d(rho_1,rho_2)*rho_2
    print(f"pressure out {p_out}")

    return result

    
    
def solve_ode_2v(rho_1_0:np.ndarray,rho_2_0:np.ndarray,v_1_0:np.ndarray,v_2_0:np.ndarray,
                 lmb:float, p_1_d_1:callable,p_1_d_2:callable, p_2_d_1:callable,p_2_d_2:callable,
                 pipe_diameter:float,pipe_length:float,f:float):
    """


    Args:
        rho_1_0 (np.ndarray): bc rho_1 in kg/m^3
        rho_2_0 (np.ndarray):  bc rho_1 kg/m^3
        v_1_0 (np.ndarray): bc v_1 in m/s
        v_2_0 (np.ndarray): bc v_2 in m/s
        lmb (float): friction parameter
        p_1_d_1 (callable): derivative of p1 w.r.t rho_1
        p_1_d_2 (callable): derivative of p1 w.r.t rho_2
        p_2_d_1 (callable): derivative of p2 w.r.t rho_1
        p_2_d_2 (callable): derivative of p2 w.r.t rho_2
        pipe_length (float): length pipe in m
        pipe_diameter (float): diameter pipe in m
        f (float): inner friction paramter
    """

    def F(t,y):
        rho_1,rho_2,v_1,v_2 = y
        lhs_mult = np.array(
            [ [v_1 , 0,rho_1,0],
              [0 , v_2,0,rho_2],
              [p_1_d_1(rho_1,rho_2) , p_1_d_2(rho_1,rho_2),rho_1*v_1,0],
              [p_2_d_1(rho_1,rho_2) , p_2_d_2(rho_1,rho_2),0,rho_2*v_2]
              ]
        )
        # note conservative formulation
        eq_rhs_1 = -f*rho_1*rho_2*(v_1-v_2)  -lmb/(2*pipe_diameter)*(rho_1)*np.abs(v_1)*v_1 
        eq_rhs_2 = -f*rho_1*rho_2*(v_2-v_1)  -lmb/(2*pipe_diameter)*(rho_2)*np.abs(v_2)*v_2 
        rhs = np.array([0,0,eq_rhs_1,eq_rhs_2])
        return np.linalg.solve(lhs_mult,rhs)

    y0 = np.array([rho_1_0,rho_2_0,v_1_0,v_2_0])
    t_eval = np.linspace(0,pipe_length,100)
    result = solve_ivp(fun=F,t_span=(0,pipe_length),y0=y0,method="BDF",t_eval=t_eval)#,min_step=1e-10)
    return result
    
    
def plot_1v_different_pressures_vs_algebraic():
    """
    plots 1v  models with different pressures
    """

    mass_flow = [[25,25],[30,20], [40,10],[10,40],[45,5],[49,1]]
    #mass_flow = [[48,2]]
    # km
    pipe_length = 20000
    # m 
    pipe_diameter = 0.5 
    A = pipe_diameter**2*np.pi/4
    # pressure 
    p = 70*1e5
    # friction parameter
    #lmb = 0.00192
    lmb = 0.008
    result_list = []

    for i,(m_1,m_2) in enumerate(mass_flow):


        model_list = [
        "algebraic",
        "speed_of_sound"
        ]
        #"virial_expansion",
        #"virial_expansion_mix",
        #"gerg" ]
        for model in model_list: 


            model_tmp = model if model != "algebraic" else "speed_of_sound"
            [rho_1_0,rho_2_0,v_0] = get_bc_data_1v(m1=m_1,m2=m_2,p=p,a=A,model=model_tmp)
            print(f"bc data {rho_1_0} {rho_2_0} {v_0}")

            
            if model == "speed_of_sound" or model =="algebraic":
                c1,c2 = pressure_law_mixtures(1,1,"speed_of_sound",return_partial_pressures=True)
                p_fct = lambda rho_1,rho_2 : c1*rho_1+c2*rho_2
                p_d1 = lambda rho_1,rho_2: c1*np.ones(rho_1.shape)
                p_d2 = lambda rho_1,rho_2: c2*np.ones(rho_2.shape)


            param_dict = {
                "rho_1_0":rho_1_0,
                "rho_2_0":rho_2_0,
                "v_0":v_0,
                "lmb":lmb,
                "p_1_d": p_d1,
                "p_2_d": p_d2,
                "pipe_length":pipe_length,
                "pipe_diameter":pipe_diameter,
                "algebraic":False
            }
            
            if model == "algebraic":
                param_dict["algebraic"] = True
            result_dict = solve_ode_1v(**param_dict)
            print(result_dict)

            p_fct_vec = np.vectorize(p_fct)
            rho_1 = result_dict["y"][0,:]
            rho_2 = result_dict["y"][1,:]
            result_dict["y"] =  np.vstack((result_dict["y"],np.expand_dims(p_fct_vec(rho_1,rho_2)*1e-5,axis=0)))
            
            result_list.append(result_dict)



        label_list = ["$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$","$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$",
                        "$v$ in $\\frac{m}{s}$", "$p$ in bar"]
        label_list_save = ["density_1","density_2","velocity", "pressure_sum"]
        for j,label in enumerate(label_list):
            plt.figure(i)
            plt.plot(result_list[0]["t"],np.abs(result_list[0]["y"][j,:]-result_list[1]["y"][j,:]),label=f"{label}")
            plt.xlabel("x")
            plt.ylabel(label)
        plt.title("Diffrence of algebraic and non algebraic model")
        plt.legend(loc="upper left")
        plt.savefig(f"graphics/one_velocity_model_vs_algebraic/difference_m1_{m_1}_m2_{m_2}_p_{int(p*1e-5)}.png")

    plt.show()

    


def plot_1v_different_pressures():
    """
    plots 1v  models with different pressures
    """

    mass_flow = [[25,25],[30,20], [40,10],[10,40],[45,5],[49,1]]
    #mass_flow = [[48,2]]
    # km
    pipe_length = 20000
    # m 
    pipe_diameter = 0.5 
    A = pipe_diameter**2*np.pi/4
    # pressure 
    p = 70*1e5
    # friction parameter
    #lmb = 0.00192
    lmb = 0.008

    for i,(m_1,m_2) in enumerate(mass_flow):


        model_list = [
        "speed_of_sound",
        "virial_expansion",
        "virial_expansion_mix",
        "gerg" ]
        for model in model_list: 

            [rho_1_0,rho_2_0,v_0] = get_bc_data_1v(m1=m_1,m2=m_2,p=p,a=A,model=model)
            print(f"bc data {rho_1_0} {rho_2_0} {v_0}")

            
            if model == "speed_of_sound":
                c1,c2 = pressure_law_mixtures(1,1,"speed_of_sound",return_partial_pressures=True)
                p_fct = lambda rho_1,rho_2 : c1*rho_1+c2*rho_2
                p_d1 = lambda rho_1,rho_2: c1*np.ones(rho_1.shape)
                p_d2 = lambda rho_1,rho_2: c2*np.ones(rho_2.shape)

            
            elif model == "virial_expansion":
                p_1,p_2 = pressure_law_mixtures_symbolic("virial_expansion")
                p_fct = lambda rho_1,rho_2 : p_1(rho_1,rho_2)+p_2(rho_1,rho_2)
                p_d1 = jnp.vectorize(grad(p_1,argnums=0))
                p_d2 = jnp.vectorize(grad(p_2,argnums=1))
            
            elif model == "virial_expansion_mix":
                p_fct = pressure_law_mixtures_symbolic("virial_expansion_mix")
                p_d1 = jnp.vectorize(grad(p_fct,argnums=0))
                p_d2 = jnp.vectorize(grad(p_fct,argnums=1))
                
            elif model == "gerg":
                p_fct = pressure_law_mixtures_symbolic("gerg")
                p_d1 = jnp.vectorize(grad(p_fct,argnums=0))
                p_d2 = jnp.vectorize(grad(p_fct,argnums=1))


            param_dict = {
                "rho_1_0":rho_1_0,
                "rho_2_0":rho_2_0,
                "v_0":v_0,
                "lmb":lmb,
                "p_1_d": p_d1,
                "p_2_d": p_d2,
                "pipe_length":pipe_length,
                "pipe_diameter":pipe_diameter
            }
            result_dict = solve_ode_1v(**param_dict)
            print(result_dict)

            p_fct_vec = np.vectorize(p_fct)
            rho_1 = result_dict["y"][0,:]
            rho_2 = result_dict["y"][1,:]
            result_dict["y"] =  np.vstack((result_dict["y"],np.expand_dims(p_fct_vec(rho_1,rho_2)*1e-5,axis=0)))

            label_list = ["$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$","$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$",
                          "$v$ in $\\frac{m}{s}$", "$p$ in bar"]
            label_list_save = ["density_1","density_2","velocity", "pressure_sum"]
            for j,label in enumerate(label_list):
                plt.figure(i*4+j)
                plt.plot(result_dict["t"],result_dict["y"][j,:],label=f"{model}")
                plt.xlabel("x")
                plt.ylabel(label)
                plt.title("$ m_{CH_{4}}$" +f" = {m_1} and " + "$m_{H_{2}} = $" +f"{m_2} p={p*1e-5} in bar")
                plt.legend(loc="upper right")
                plt.savefig(f"graphics/one_velocity_model/{label_list_save[j]}_m1_{m_1}_m2_{m_2}_p_{int(p*1e-5)}.png")

    plt.show()

    
    
    

def plot_1v_relative_error_over_mass_fractions():
    """
    plots relative pressure errors over mass fractions 
    """

    m_list = [50]
    fig2,ax2 = plt.subplots()
    fig1,ax1 = plt.subplots()
    for m in m_list:

    
    
        m_1_list = np.linspace(5,49,30)
        m_2_list = m-m_1_list

        m_1_list = m_1_list[m_2_list>0]
        m_2_list = m_2_list[m_2_list>0]

        # km
        pipe_length = 20000
        # m 
        pipe_diameter = 0.5 
        A = pipe_diameter**2*np.pi/4
        # pressure 
        p = 70*1e5
        # friction parameter
        #lmb = 0.00192
        lmb = 0.008


        relative_error = []
        for i,m_1 in enumerate(m_1_list):
            m_2 = m_2_list[i]


            model_list = [
            "gerg" ,
            "speed_of_sound",
            "virial_expansion",
            "virial_expansion_mix"
            ]

            result_dict = {}
            for model in model_list: 

                [rho_1_0,rho_2_0,v_0] = get_bc_data_1v(m1=m_1,m2=m_2,p=p,a=A,model=model)
                print(f"bc data {rho_1_0} {rho_2_0} {v_0}")

                
                if model == "speed_of_sound":
                    c1,c2 = pressure_law_mixtures(1,1,"speed_of_sound",return_partial_pressures=True)
                    p_fct = lambda rho_1,rho_2 : c1*rho_1+c2*rho_2
                    p_d1 = lambda rho_1,rho_2: c1*np.ones(rho_1.shape)
                    p_d2 = lambda rho_1,rho_2: c2*np.ones(rho_2.shape)

                
                elif model == "virial_expansion":
                    p_1,p_2 = pressure_law_mixtures_symbolic("virial_expansion")
                    p_fct = lambda rho_1,rho_2 : p_1(rho_1,rho_2)+p_2(rho_1,rho_2)
                    p_d1 = jnp.vectorize(grad(p_1,argnums=0))
                    p_d2 = jnp.vectorize(grad(p_2,argnums=1))

                    
                elif model == "virial_expansion_mix":
                    p_fct = pressure_law_mixtures_symbolic("virial_expansion_mix")
                    p_d1 = jnp.vectorize(grad(p_fct,argnums=0))
                    p_d2 = jnp.vectorize(grad(p_fct,argnums=1))
                    
                    
                elif model == "gerg":
                    p_fct = pressure_law_mixtures_symbolic("gerg")
                    p_d1 = jnp.vectorize(grad(p_fct,argnums=0))
                    p_d2 = jnp.vectorize(grad(p_fct,argnums=1))


                param_dict = {
                    "rho_1_0":rho_1_0,
                    "rho_2_0":rho_2_0,
                    "v_0":v_0,
                    "lmb":lmb,
                    "p_1_d": p_d1,
                    "p_2_d": p_d2,
                    "pipe_length":pipe_length,
                    "pipe_diameter":pipe_diameter
                }
                result_dict[model] = solve_ode_1v(**param_dict)
            rho_1_err = {}
            rho_2_err = {}
            v_err = {}
            for model in model_list[1:]:
                rho_1_err [model] = np.max(np.abs((result_dict["gerg"]["y"][0,:]-result_dict[model]["y"][0,:])/result_dict["gerg"]["y"][0,:]))
                rho_2_err [model] = np.max(np.abs((result_dict["gerg"]["y"][1,:]-result_dict[model]["y"][1,:])/result_dict["gerg"]["y"][1,:]))
                v_err [model] = np.max(np.abs((result_dict["gerg"]["y"][2,:]-result_dict[model]["y"][2,:])/result_dict["gerg"]["y"][2,:]))
            relative_error.append([rho_1_err,rho_2_err,v_err])

        print(f" m1 = {m_1} ")


        label_list = ["$\\rho_1$","$\\rho_2$","$v$"]
        short_model = {
            "virial_expansion":"vi",
            "virial_expansion_mix":"vi mix",
            "speed_of_sound":"sp"
        }
        for model in model_list[1:]:
            for j,label in enumerate(label_list):
                rel_err = [ err[j][model] for err in relative_error ]
                label_str = f"{short_model[model]} {label}"
                if len(m_list) >1:
                    label_str +=  f"m={m}"
                ax1.plot(m_1_list,rel_err,"-x",label=label_str)

        for model in model_list[1:]:
            rel_err = [ err[j][model] for err in relative_error ]
            label_str = f"{model}"
            if len(m_list) >1:
                label_str +=  f"m={m}"
            ax2.plot(m_1_list,rel_err,"-x",label=label_str)
    
    ax1.set_title(f"relative errors")
    ax1.set_ylabel("relative error")
    ax1.set_xlabel("$m_1$ $(CH_4)$")

    title = "$\\frac{|\\rho_{1,\\mathrm{gerg}}-\\rho_1|}{|\\rho_{1,\\mathrm{gerg}}|}$"
    if len(m_list)==1:
        title += f" for m = {m}"
    ax2.set_title(title)
    ax2.set_ylabel("relative error")
    ax2.set_xlabel("$m_1$ $(CH_4)$")

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    fig1.savefig("graphics/mass_flux_1v/all_relative_errors.png")
    fig2.savefig("graphics/mass_flux_1v/rho_1_relative_errors.png")
    plt.show()


def plot_2v_different_pressures():
    """
    plots 2v with different pressures
    """

    mass_flow = [[25,25],[30,20], [40,10],[10,40]]
    #mass_flow = [[10,40]]
    # km
    pipe_length = 20000
    # m 
    pipe_diameter = 0.5 
    A = pipe_diameter**2*np.pi/4
    # pressure 
    p1 = 35*1e5
    p2 = 35*1e5
    # friction parameter
    
    
    ## divided by 2, since we add up both equations to get 1v
    lmb = 0.008

    f = 1


    

    #mass_flow = [[25,25],[30,25]]#, [40,10]]

    ##### problem case see partial_pressures of one velocity
    #mass_flow = [ [40,10]] # and p1=p2=35

    model_list = [
    "speed_of_sound",
    "virial_expansion"]

    for i,(m1,m2) in enumerate(mass_flow):
        for model in model_list:


            rho_1_0,rho_2_0,v_1_0,v_2_0 = get_bc_data_2v(m1=m1,m2=m2,p1=p1,p2=p2,a=A,model=model)

            p_1_fct,p_2_fct = pressure_law_mixtures_symbolic(model)
            p_1_d_1 = jnp.vectorize(grad(p_1_fct,argnums=0))
            p_1_d_2 = jnp.vectorize(grad(p_1_fct,argnums=1))
            p_2_d_1 = jnp.vectorize(grad(p_2_fct,argnums=0))
            p_2_d_2 = jnp.vectorize(grad(p_2_fct,argnums=1))

            result_dict = solve_ode_2v(rho_1_0=rho_1_0,rho_2_0=rho_2_0,v_1_0=v_1_0,v_2_0=v_2_0,
                                       lmb=lmb,p_1_d_1=p_1_d_1,p_1_d_2=p_1_d_2,p_2_d_1=p_2_d_1,p_2_d_2=p_2_d_2,
                                       pipe_length=pipe_length,pipe_diameter=pipe_diameter,
                                       f=f)
            print(result_dict)
            
            rho_1 = result_dict["y"][0,:]
            rho_2 = result_dict["y"][1,:]
            result_dict["y"] =  np.vstack((result_dict["y"],np.expand_dims(p_1_fct(rho_1,rho_2)*1e-5,axis=0)))
            result_dict["y"] =  np.vstack((result_dict["y"],np.expand_dims(p_2_fct(rho_1,rho_2)*1e-5,axis=0)))
            result_dict["y"] =  np.vstack((result_dict["y"],np.expand_dims(p_1_fct(rho_1,rho_2)*1e-5+p_2_fct(rho_1,rho_2)*1e-5,axis=0)))

            label_list = ["$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$","$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$",
                          "$v_1$ $(CH_4)$ in $\\frac{m}{s}$", "$v_2$ $(H_2)$ in $\\frac{m}{s}$", "$p_1$ $(CH_4)$ in bar",
                          "$p_2$ $(H_2)$ in bar","$p$ in bar"]
            label_list_save = ["density_1","density_2","velocity_1", "velocity_2", "pressure_1","pressure_2","pressure_sum"]
            for j,label in enumerate(label_list):
                plt.figure(i*7+j)
                plt.plot(result_dict["t"],result_dict["y"][j,:],label=f"{model}")
                plt.xlabel("x")
                plt.ylabel(label)
                plt.title("$ m_{CH_{4}}$" +f" = {m1} and " + "$m_{H_{2}} = $" +f"{m2} p1={int(p1*1e-5)} p2={int(1e-5*p2)} in bar")
                plt.legend(loc="upper right")
                plt.savefig(f"graphics/two_velocity_model/{label_list_save[j]}_m1_{m1}_m2_{m2}_p1_{int(p2*1e-5)}_p2_{int(1e-5*p2)}_f_{f}.png")

    plt.show()



def plot_1v_2v_inner_friction():
    """
    plots 1v vs 2v  models with different pressures
    """

    #mass_flow = [[25,25]]#,[30,20]]#, [40,10],[10,40]]
    m_1 = 48
    m_2 = 2
    # km
    pipe_length = 20000
    # m 
    pipe_diameter = 0.5 
    A = pipe_diameter**2*np.pi/4
    # pressure 
    p = 70*1e5
    # friction parameter
    lmb = 0.008
    f_list =  [0.01,0.1,1,10,100]

    f_list_err_rho_1 = []
    f_list_err_rho_2 = []

    for i,f in enumerate(f_list):

        model= "speed_of_sound"
        [rho_1_0_1v,rho_2_0_1v,v_0_1v] = get_bc_data_1v(m1=m_1,m2=m_2,p=p,a=A,model=model)
        print(f"bc data {rho_1_0_1v} {rho_2_0_1v} {v_0_1v}")

        
        c1,c2 = pressure_law_mixtures(1,1,"speed_of_sound",return_partial_pressures=True)
        p_fct = lambda rho_1,rho_2 : c1*rho_1+c2*rho_2
        p_d1 = lambda rho_1,rho_2: c1*np.ones(rho_1.shape)
        p_d2 = lambda rho_1,rho_2: c2*np.ones(rho_2.shape)


        param_dict = {
            "rho_1_0":rho_1_0_1v,
            "rho_2_0":rho_2_0_1v,
            "v_0":v_0_1v,
            "lmb":lmb,
            "p_1_d": p_d1,
            "p_2_d": p_d2,
            "pipe_length":pipe_length,
            "pipe_diameter":pipe_diameter
        }

        if i == 0:
            result_dict_1v = solve_ode_1v(**param_dict)
            print(result_dict_1v)

            p_fct_vec = np.vectorize(p_fct)
            rho_1 = result_dict_1v["y"][0,:]
            rho_2 = result_dict_1v["y"][1,:]
            result_dict_1v["y"] =  np.vstack((result_dict_1v["y"],np.expand_dims(p_fct_vec(rho_1,rho_2)*1e-5,axis=0)))

            label_list = ["$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$","$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$",
                            "$v$ in $\\frac{m}{s}$", "$p$ in bar"]
            label_list_save = ["density_1","density_2","velocity", "pressure_sum"]
            for j,label in enumerate(label_list):
                plt.figure(j)
                plt.plot(result_dict_1v["t"],result_dict_1v["y"][j,:],label=f" 1v ")
                plt.xlabel("x")
                plt.ylabel(label)
                plt.title("$ m_{CH_{4}}$" +f" = {m_1} and " + "$m_{H_{2}} = $" +f"{m_2} p={p*1e-5} in bar")
                plt.legend(loc="upper right")


        ########################################################################## 
        # get 1 v data
        p1 = c1*rho_1_0_1v
        p2 = c2*rho_2_0_1v
        

        rho_1_0,rho_2_0,v_1_0,v_2_0 = get_bc_data_2v(m1=m_1,m2=m_2,p1=p1,p2=p2,a=A,model=model)


        p_1_d_1 = lambda rho_1,rho_2: c1*np.ones(rho_1.shape)
        p_1_d_2 = lambda rho_1,rho_2: np.zeros(rho_1.shape)
        p_2_d_1 = lambda rho_1,rho_2: np.zeros(rho_2.shape)
        p_2_d_2 = lambda rho_1,rho_2: c2*np.ones(rho_2.shape)

        result_dict = solve_ode_2v(rho_1_0=rho_1_0,rho_2_0=rho_2_0,v_1_0=v_1_0,v_2_0=v_2_0,
                                    lmb=lmb,p_1_d_1=p_1_d_1,p_1_d_2=p_1_d_2,p_2_d_1=p_2_d_1,p_2_d_2=p_2_d_2,
                                    pipe_length=pipe_length,pipe_diameter=pipe_diameter,
                                    f=f)

        print(result_dict)

        p_fct_vec = lambda rho_1,rho_2: c1*rho_1+c2*rho_2
        rho_1 = result_dict["y"][0,:]
        rho_2 = result_dict["y"][1,:]
        result_dict["y"] =  np.vstack((result_dict["y"],np.expand_dims(p_fct_vec(rho_1,rho_2)*1e-5,axis=0)))
        result_dict["y"] =  np.vstack((result_dict["y"],np.expand_dims(c1*rho_1*1e-5,axis=0)))
        result_dict["y"] =  np.vstack((result_dict["y"],np.expand_dims(c2*rho_2*1e-5,axis=0)))

        x_label_list = ["$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$","$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$",
                        "$v_i$ in $\\frac{m}{s}$", "$v_i$ in $\\frac{m}{s}$", "$p$ in bar", "$p1$","$p2$"]
        label_list = [f" 2v {f:1.2f}" for i in range(len(x_label_list))]
        label_list[2] = [f"$v_1$ {f:1.2f}"]
        label_list[3] = [f" $v_2$ {f:1.2f}"]


        label_list_save = ["density_1","density_2","velocity","velocity", "pressure_sum","pressure_1","pressure_2"]

        fig_list = [0,1,2,2,3,4,5]
        for j,x_label in enumerate(x_label_list):
            plt.figure(fig_list[j])
            plt.plot(result_dict["t"],result_dict["y"][j,:],"--",label=label_list[j])
            plt.xlabel("x")
            plt.ylabel(x_label)
            plt.title("$ m_{CH_{4}}$" +f" = {m_1} and " + "$m_{H_{2}} = $" +f"{m_2} p={p*1e-5} in bar")
            plt.legend(loc="upper right")
            plt.savefig(f"graphics/friction_comparison/{label_list_save[j]}_m1_{m_1}_m2_{m_2}_p_{p*1e-5}.png")

        rho_1_v = result_dict_1v["y"][0,:]
        rho_1_err = np.max(np.abs((result_dict["y"][0,:]- rho_1_v)/rho_1_v))
        rho_2_v = result_dict_1v["y"][1,:]
        rho_2_err = np.max(np.abs((result_dict["y"][1,:]- rho_2_v)/rho_2_v))
        f_list_err_rho_1.append(rho_1_err)
        f_list_err_rho_2.append(rho_2_err)

    fig,ax = plt.subplots()
    ax.set_title(f"relative error for m1={m_1} m2={m_2} p={p*1e-5:.0f}")
    poly_rho_1 = np.polyfit(np.log(f_list),np.log(f_list_err_rho_1),1)
    poly_rho_2 = np.polyfit(np.log(f_list),np.log(f_list_err_rho_2),1)

    ax.loglog(np.array(f_list),f_list_err_rho_1,"-x",label="$\\rho_1$ " +f"slope={poly_rho_1[0]:1.2f}")
    ax.loglog(np.array(f_list),f_list_err_rho_2,"-x",label="$\\rho_2$ " +f"slope={poly_rho_2[0]:1.2f}")
    plt.legend(loc="upper right")
    fig.savefig(f"graphics/friction_comparison/f_relative_erros_m1_{m_1}_m2_{m_2}_p_{p*1e-5}.png")
    print(f"relative error rho_1 {f_list_err_rho_1}")
    print(f"relative error rho_2 {f_list_err_rho_2}")


    plt.show()


def plot_1v_2v_partial_pressures():
    """
    plots 1v vs 2v  models with different pressures
    """

    #mass_flow = [[25,25]]#,[30,20]]#, [40,10],[10,40]]
    m_1 = 30
    m_2 = 20
    # km
    pipe_length = 20000
    # m 
    pipe_diameter = 0.5 
    A = pipe_diameter**2*np.pi/4
    # pressure 
    p = 70*1e5
    # friction parameter
    lmb = 0.008
    f =  0.1

    p1_list = np.array([1,2,5,-1,0,-1,-2,-5])*1e5



    model= "speed_of_sound"
    [rho_1_0_1v,rho_2_0_1v,v_0_1v] = get_bc_data_1v(m1=m_1,m2=m_2,p=p,a=A,model=model)
    print(f"bc data {rho_1_0_1v} {rho_2_0_1v} {v_0_1v}")

    
    c1,c2 = pressure_law_mixtures(1,1,"speed_of_sound",return_partial_pressures=True)
    p_fct = lambda rho_1,rho_2 : c1*rho_1+c2*rho_2
    p_d1 = lambda rho_1,rho_2: c1*np.ones(rho_1.shape)
    p_d2 = lambda rho_1,rho_2: c2*np.ones(rho_2.shape)
    p1 = c1*rho_1_0_1v


    param_dict = {
        "rho_1_0":rho_1_0_1v,
        "rho_2_0":rho_2_0_1v,
        "v_0":v_0_1v,
        "lmb":lmb,
        "p_1_d": p_d1,
        "p_2_d": p_d2,
        "pipe_length":pipe_length,
        "pipe_diameter":pipe_diameter
    }

    result_dict = solve_ode_1v(**param_dict)
    print(result_dict)


    p_fct_vec = np.vectorize(p_fct)
    rho_1 = result_dict["y"][0,:]
    rho_2 = result_dict["y"][1,:]
    result_dict["y"] =  np.vstack((result_dict["y"],result_dict["y"][2,:]))
    result_dict["y"] =  np.vstack((result_dict["y"],np.expand_dims(p_fct_vec(rho_1,rho_2)*1e-5,axis=0)))

    label_list = ["$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$","$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$",
                    "$v$ in $\\frac{m}{s}$", "$v$ in $\\frac{m}{s}$", "$p$ in bar"]
    label_list_save = ["density_1","density_2","velocity", "pressure_sum"]
    for j,label in enumerate(label_list):
        plt.figure(j)
        plt.plot(result_dict["t"],result_dict["y"][j,:],label=f" 1v p1={p1*1e-5:1.1f} ")
        plt.xlabel("x")
        plt.ylabel(label)
        plt.title("$ m_{CH_{4}}$" +f" = {m_1} and " + "$m_{H_{2}} = $" +f"{m_2} p={p*1e-5} in bar")
        plt.legend(loc="upper right")
        
        
        
    for i,p1_sub in enumerate(p1_list):



        ########################################################################## 
        # get 1 v data
        p1 = c1*rho_1_0_1v-p1_sub
        p2 = p-p1
        

        rho_1_0,rho_2_0,v_1_0,v_2_0 = get_bc_data_2v(m1=m_1,m2=m_2,p1=p1,p2=p2,a=A,model=model)


        p_1_d_1 = lambda rho_1,rho_2: c1*np.ones(rho_1.shape)
        p_1_d_2 = lambda rho_1,rho_2: np.zeros(rho_1.shape)
        p_2_d_1 = lambda rho_1,rho_2: np.zeros(rho_2.shape)
        p_2_d_2 = lambda rho_1,rho_2: c2*np.ones(rho_2.shape)

        result_dict = solve_ode_2v(rho_1_0=rho_1_0,rho_2_0=rho_2_0,v_1_0=v_1_0,v_2_0=v_2_0,
                                    lmb=lmb,p_1_d_1=p_1_d_1,p_1_d_2=p_1_d_2,p_2_d_1=p_2_d_1,p_2_d_2=p_2_d_2,
                                    pipe_length=pipe_length,pipe_diameter=pipe_diameter,
                                    f=f)

        print(result_dict)

        p_fct_vec = lambda rho_1,rho_2: c1*rho_1+c2*rho_2
        rho_1 = result_dict["y"][0,:]
        rho_2 = result_dict["y"][1,:]
        result_dict["y"] =  np.vstack((result_dict["y"],np.expand_dims(p_fct_vec(rho_1,rho_2)*1e-5,axis=0)))
        result_dict["y"] =  np.vstack((result_dict["y"],np.expand_dims(c1*rho_1*1e-5,axis=0)))
        result_dict["y"] =  np.vstack((result_dict["y"],np.expand_dims(c2*rho_2*1e-5,axis=0)))

        x_label_list = ["$\\rho_1$ $(CH_4)$ in $\\frac{kg}{m^3}$","$\\rho_2$ $(H_2)$ in $\\frac{kg}{m^3}$",
                        "$v_i$ in $\\frac{m}{s}$", "$v_i$ in $\\frac{m}{s}$", "$p$ in bar", "$p1$","$p2$"]
        label_list = [f" 2v p1={p1*1e-5:1.1f}" for i in range(len(x_label_list))]
        label_list[2] = [f"$v_1$ p1={p1*1e-5:1.1f}"]
        label_list[3] = [f" $v_2$ p1={p1*1e-5:1.1f}"]


        label_list_save = ["density_1","density_2","velocity","velocity", "pressure_sum","pressure_1","pressure_2"]

        for j,x_label in enumerate(x_label_list):
            plt.figure(j)
            plt.plot(result_dict["t"],result_dict["y"][j,:],"--",label=label_list[j])
            plt.xlabel("x")
            plt.ylabel(x_label)
            plt.title("$ m_{CH_{4}}$" +f" = {m_1} and " + "$m_{H_{2}} = $" +f"{m_2} p={p*1e-5} in bar")
            plt.savefig(f"graphics/partial_pressures/{label_list_save[j]}_m1_{m_1}_m2_{m_2}_p_{p*1e-5}.png")
            plt.legend(loc="upper right")


    plt.show()


if __name__ == "__main__":

    #plot_1v_relative_error_over_mass_fractions()
    #plot_1v_different_pressures()
    # plot_1v_different_pressures_vs_algebraic()
    #plot_2v_different_pressures()
    plot_1v_2v_inner_friction()
    #plot_1v_2v_partial_pressures()

