"""
This Python program contains code for computing numerical solutions to the one-velocity model on a general network.

NOTE: This code considers the case where we set MASS FLOW on the inflow nodes. 
"""
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt


import xmltodict
import jax
from jax import numpy as jnp
import pickle

from pathlib import Path
from one_pipe_setinflow import get_bc_data_1v
from pressure_laws_comparison import pressure_law_mixtures,pressure_law_mixtures_symbolic,\
                                    fit_gerg_simple
from copy import deepcopy

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



# Enable 64 Bit
from jax import config
config.update("jax_enable_x64", True)

class Network_1v_time:
    #network and initial file names
    network_name:str
    network_init_and_bdry_data:str
    file_network:str
    file_data:str

    ## nodes 
    # concatinated id's
    node_id:list
    # coordinates
    node_x_cord:list
    node_y_cord:list

    # source, inner, sink nodes
    node_type: list 

    temperature: float

    # data on nodes
    p_list: list
    m_node_flow: list
    mu_node:list

    ## pipes 
    pipe_id:list
    # in nodes id
    pipe_in:list
    # outer_nodes_id
    pipe_out :list
    
    #data_on pipes
    m_list : list
    mu_list : list

    pipe_length: list
    pipe_diameter: list
    # calculate friction formula from Nikuraze
    pipe_roughness: list
    pipe_friction: list

    # compressors
    comp_in: list
    comp_out: list

    #solution (after computation)
    u_sol:np.ndarray
    u_store:np.ndarray
    # if solution converged at each time step
    conv:bool

    # pressure laws
    model:str
    # name of pressure law for display
    model_name:str

    # after precomputation of model (sympy faster, but jax needed for derivative)
    # NOTE: standard accuracy of jax is 32 bit -> different results possible
    p_sympy:callable
    p_jax:callable
    
    #computational parameters
    candidate_dx:float
    N_x_all_list:np.ndarray
    max_dx:float
    dt:float
    T:float
    
    



    def __init__(self,file_network:Path,file_data:Path,model:str,
                 candidate_dx:float,dt:float,T:float,model_name:str=None):
        """
        This function initializes the network by 2 given xml files, 
        where the network structure is given in file_network
        and the data in file_data. The file "file_network" should have the same
        keys and structure as "network_data/optimization_data/testm_new.net".
        The file "file_data" should have the same structure as 
        "network_data/optimization_data/testm-testm_70_b.lsf".

        Args:
            file_network (Path): path to network structure 
            file_data (Path): path to data
            model (str): pressure law type. 
            ["speed_of_sound","virial_expansion","virial_expansion_mix","gerg_fit"],
            see pressure_law.py for more information

        
        Then the following parameters are available:
        ## node structure
        node_id (list): id of each node
        node_x_cord (list)  x coordinate of each node
        node_y_cord (list) y coordinate of each node
        node_type (list) type of ["source","in_node","sink"]


        # node data
        p_list (list): list of pressure at node in Pa
        m_node_flow (list): list of exit/entry mass flows at each boundary node, 
        with + for inflow and - for outflow
        mu_node (list): list of percentage mass of CH_4 of the OUTFLOW
        at each node.
        temperature (float): temperature in Kelvin

        ## pipes 
        pipe_id (list): list of id's
        pipe_in (list): list of inflow nodes of each pipe
        pipe_out  (list): list of outflow nodes of each pipe
        
        #data_on pipes
        m_list  (list): mass flows on each pipe in kg/s^2, + for along pipe direction
        and - against
        mu_list  (list): list of percentage mass of CH_4 


        pipe_length (list): length of pipe
        pipe_diameter (list): pipe diameters
        pipe_roughness (list): pipe roughness

        # The pipe friction is via the formula of Nikuraze via calling self._calculate_friction_parameter()! 
        Available only after the call of self._calculate_friction_parameter().
        pipe_friction (list): pipe friction 


        # pressure_laws 
        Only after call of self._precompute_pressure()!
        p_jax(callable): pressure law "self.model" via automatic differentiation
        p_sympy(callable): pressure law "self.model" via symbolic differentiation
        
        # boundary data  
        Only after call of self.convert_boundary_conditions()!

        rho_1_0 (list[list]): list of pairs for inflow/outflow boundary value of rho_1 for each pipe
        rho_2_0 (list[list]): list of pairs for inflow/outflow boundary value of rho_2 for each pipe
        v_0 (list[list]): list of pairs for inflow/outflow boundary value of rho_2 for each pipe

        # solution 
        Only after call of self.solve()!
        u_sol(np.ndarray): last step of computed solution
        conv(bool): if computed solution converged
        """

        self.node_id = []
        self.node_x_cord = []
        self.node_y_cord = []
        self.node_type  = []
        self.temperature  = []
        self.m_list  = []
        self.mu_list  = []
        self.p_list = []
        self.pipe_id = []
        self.pipe_in = []
        self.pipe_out  = []
        self.pipe_length = []
        self.pipe_diameter = []
        self.pipe_roughness = []
        self.comp_in = []
        self.comp_out = []
        self.u_sol = None
        self.u_store = None
        self.conv = False
        self.network_name = Path(file_network).name[:-4]
        self.network_data = Path(file_data).name[:-4]
        self.file_network = file_network
        self.file_data = file_data
        self._read_network_architecture(file_network)
        self._read_network_data(file_data)
        self.model = model

        ####################################################################
        ####################################################################
        ## MARKER START
        """ 
        Swap which nodes are inflow and outflow so we can set mass flow on the inflow nodes.
        """

        #swap node types (bc's )
        ####################################################################
        node_type_swap = {"source":"sink","sink":"source","in_node":"in_node"}
        self.node_type = np.array([node_type_swap[node] for node in self.node_type])
        ####################################################################
        

        ## MARKER END 
        ####################################################################
        ####################################################################


        if model_name is None:
            self.model_name = model
        else:
            self.model_name = model_name
        self.p_sympy = None
        self.p_jax = None

        self.candidate_dx = candidate_dx
        self.dt = dt
        self.T = T
        
        nrofpieces = np.round(self.pipe_length/candidate_dx)
        self.dx_list = self.pipe_length/nrofpieces
        self.N_x_all_list = np.array([int(n_x) + 1 for n_x in nrofpieces])
        self.max_dx = np.max(self.dx_list)
        self.N_time = int(np.round(T/dt)) + 1
    
    
    @staticmethod
    def get_key_nodes():

        key = "nodes"
        
        sub_key = {
        "source":"source",
        "in_node":"innode",
        "sink":"sink",
        }

        sub_sub_key = {
            "id": "@id",
            "x":"@x",
            "y":"@y"
        }
        

        return key, sub_key, sub_sub_key



    @staticmethod
    def get_key_nodes_data():

        key = "nodes"
        
        sub_key = {
        "source":"source",
        "in_node":"innode",
        "sink":"sink",
        }

        sub_sub_key = {
            "p":"pressure",
            "m":"flow",
            "mu":"mixingratio"
                       }
        

        return key, sub_key, sub_sub_key


    @staticmethod
    def get_key_pipes():

        key = "connections"
        
        sub_key = {
            "pipe":"pipe",
            "comp":"compressorStation"
        }

        sub_sub_key = {
            "id":"@id",
            "pipe_in":"@from",
            "pipe_out":"@to",
            "pipe_diameter":"diameter",
            "pipe_roughness":"roughness",
            "pipe_length":"length",
            "comp_in":"@from",
            "comp_out":"@to"
        }

        return key, sub_key, sub_sub_key


    @staticmethod
    def get_key_pipes_data():

        key = "connections"
        
        sub_key = "pipe"

        sub_sub_key = {
            "m":"flow",
            "mu":"mixingratioMass"
        }

        return key, sub_key, sub_sub_key

    @staticmethod
    def get_key_information():

        key = "information"

        return key



    def _read_network_architecture(self,file:Path):
        """
        Reads in network architecture

        Args:
            file (Path): path to file
        """

        with open(file, "r") as f:
            xml_data = f.read()

        # Convert XML to a dictionary
        xml_dict = xmltodict.parse(xml_data)

        framework = "framework"
        data = xml_dict["network"]

        # read information
        key_info = Network_1v_time.get_key_information()
        information = data[f"{framework}:{key_info}"]
        # print(information)

        # read nodes
        key_node,key_node_category,key_node_prop = Network_1v_time.get_key_nodes()

        nodes = data[f"{framework}:{key_node}"]

        for name_node_type,node_type in key_node_category.items():

            node_list = nodes.get(node_type)
            if node_list is not None:
                node_list = [node_list] if not isinstance(node_list,list) else node_list

                self.node_id += [node_list[i][key_node_prop["id"]] for i in range(len(node_list))] 
                self.node_x_cord += [float(node_list[i][key_node_prop["x"]]) for i in range(len(node_list))] 
                self.node_y_cord += [float(node_list[i][key_node_prop["y"]]) for i in range(len(node_list))] 

                self.node_type += [name_node_type for i in range(len(node_list))]

                if name_node_type == "source":
                    conv_dict_temperature = {"Celsius":273.15}
                    temperature_dict = node_list[0]["gasTemperature"]
                    self.temperature = float(temperature_dict["@value"]) + conv_dict_temperature[temperature_dict["@unit"]]
        self.node_type = np.array(self.node_type)


        # read pipes
        key_pipe,key_pipe_category,key_pipe_properties = Network_1v_time.get_key_pipes()
        pipe_list = data[f"{framework}:{key_pipe}"][key_pipe_category["pipe"]]

        pipe_list = pipe_list if isinstance(pipe_list,list) else [pipe_list] 

        # pipe id's 
        self.pipe_id = [pipe_list[i][key_pipe_properties["id"]] 
                        for i in range(len(pipe_list))]
        # pipe in/out
        self.pipe_in = [pipe_list[i][key_pipe_properties["pipe_in"]] 
                        for i in range(len(pipe_list))]
        self.pipe_out = [pipe_list[i][key_pipe_properties["pipe_out"]] 
                         for i in range(len(pipe_list))]

        pipe_length_units = [pipe_list[i][key_pipe_properties["pipe_length"]] 
                             for i in range(len(pipe_list))]
        pipe_diameter_units = [pipe_list[i][key_pipe_properties["pipe_diameter"]] 
                               for i in range(len(pipe_list))]
        pipe_roughness_units = [pipe_list[i][key_pipe_properties["pipe_roughness"]] 
                                for i in range(len(pipe_list))]

        conversion_dict = {
            "km":1000,
            "m":1,
            "dm":0.1,
            "cm":0.01,
            "mm":0.001,
        }


        # convert to meter
        self.pipe_length = np.array([ conversion_dict[p_dict["@unit"]]*float(p_dict["@value"]) 
                            for p_dict in pipe_length_units])
        self.pipe_diameter = np.array([ conversion_dict[p_dict["@unit"]]*float(p_dict["@value"] )
                              for p_dict in pipe_diameter_units])
        self.pipe_roughness = np.array([ conversion_dict[p_dict["@unit"]]*float(p_dict["@value"])
                               for p_dict in pipe_roughness_units])



        pipe_list = data[f"{framework}:{key_pipe}"].get(key_pipe_category["comp"],[])
        self.comp_in = [pipe_list[i][key_pipe_properties["comp_in"]] 
                        for i in range(len(pipe_list))]
        self.comp_out = [pipe_list[i][key_pipe_properties["comp_out"]] 
                         for i in range(len(pipe_list))]

        


    def _read_network_data(self,file):
        """
        Reads in network data

        Args:
            file (Path): path to data
        """



        with open(file, "r") as f:
            xml_data = f.read()

        # Convert XML to a dictionary
        xml_dict = xmltodict.parse(xml_data)

        data = xml_dict["solution"]
        key_node,key_node_category,key_node_prop = Network_1v_time.get_key_nodes_data()

        nodes = data[f"{key_node}"]
        p_dict_list = []
        m_node_flow = []
        mu_node = []
        for name_node_type,node_type in key_node_category.items():

            node_list = nodes.get(f"{node_type}s")
            if node_list is not None:
                node_list = node_list[node_type]
                node_list = [node_list] if not isinstance(node_list,list) else node_list

                p_dict_list += [node_list[i][key_node_prop["p"]] for i in range(len(node_list))] 
                # flows in/out of node

                conv_dict_m = {
                    "kg_per_s":1,
                }

                if name_node_type == "source":
                    m_node_flow += [conv_dict_m[node_list[i][key_node_prop["m"]]["@unit"]]*\
                        float(node_list[i][key_node_prop["m"]]["@value"]) for i in range(len(node_list))] 
                    mu_node += [ node_list[i][key_node_prop["mu"]] for i in range(len(node_list))] 

                elif name_node_type == "sink":
                    # negative sign for out going gas
                    m_node_flow += [-conv_dict_m[node_list[i][key_node_prop["m"]]["@unit"]]*\
                        float(node_list[i][key_node_prop["m"]]["@value"]) for i in range(len(node_list))] 

                    mu_node += [ node_list[i][key_node_prop["mu"]] for i in range(len(node_list))] 

                elif name_node_type == "in_node":
                    m_node_flow += [0 for i in range(len(node_list))] 
                    mu_node += [{"@unit":"mass-percent","@value":"0"} for i in range(len(node_list))] 

        conv_dict_pres = {
            "pa":1,
            "bar":1e5,
        }

        self.m_node_flow = np.array(m_node_flow)

        self.p_list  = np.array([ conv_dict_pres[p_dict["@unit"]]*float(p_dict["@value"]) 
                            for p_dict in p_dict_list])


        # mass percent 
        self.mu_node  = np.array([ float(mu_dict["@value"]) 
                            for mu_dict in mu_node])

                            
                            
        # read pipes
        key_pipe,key_pipe_category,key_pipe_properties = Network_1v_time.get_key_pipes_data()
        pipe_list = data[f"{key_pipe}"][f"{key_pipe_category}s"][key_pipe_category]
        pipe_list = pipe_list if isinstance(pipe_list,list) else [pipe_list] 

        # pipe id's 
        m_dict_list = [pipe_list[i][key_pipe_properties["m"]] 
                        for i in range(len(pipe_list))]



        self.m = np.array([ conv_dict_m[m_dict["@unit"]]*float(m_dict["@value"]) 
                             for m_dict in m_dict_list])


        self.lamb =   np.array([float(pipe_list[i]["LAMBDA"]["@value"])
                        for i in range(len(pipe_list))])

        self.mu = np.array([float(pipe_list[i][key_pipe_properties["mu"]]["@value"])
                        for i in range(len(pipe_list))])


    def _precompute_pressure_law(self):
        """
        Computes and stores the pressure function. 

        """

        if self.p_sympy is None:

            if self.model == "speed_of_sound":
                self.p_sympy = lambda x,y: pressure_law_mixtures(x,y,model=self.model,T=self.temperature)
                p1,p2 = pressure_law_mixtures_symbolic(model=self.model,T=self.temperature)
                self.p_jax = lambda rho1,rho2: p1(rho1,rho2)+p2(rho1,rho2)

            elif self.model == "virial_expansion_mix" or self.model == "gerg":
                self.p_sympy = lambda x,y: pressure_law_mixtures(x,y,model=self.model,T=self.temperature)
                self.p_jax = pressure_law_mixtures_symbolic(model=self.model,T=self.temperature)

            elif self.model == "virial_expansion":
                self.p_sympy = lambda x,y: pressure_law_mixtures(x,y,model=self.model,T=self.temperature,return_partial_pressures=False)

                p1,p2 = pressure_law_mixtures_symbolic(model=self.model,T=self.temperature)
                self.p_jax = lambda rho1,rho2: p1(rho1,rho2)+p2(rho1,rho2)
            
            elif self.model == "gerg_fit":
                p1,p2,p_ns = fit_gerg_simple(poly_deg=4,penalty=0,allow_non_simple=True,T=self.temperature)
                self.p_sympy = lambda rho1,rho2: p1(rho1,rho2)+p2(rho1,rho2)+p_ns(rho1,rho2)
                self.p_jax = lambda rho1,rho2: p1(rho1,rho2)+p2(rho1,rho2)+p_ns(rho1,rho2)
            else: 
                raise Exception(f"This pressure law is not implemented: {self.model}")




    def convert_boundary_conditions(self):
        """
        This method converts the boundary conditions m,mu,p 
        to rho_1,rho_2,v for EACH pipe.
        Note, this yields values for rho_1,rho_2,v
        on both ends of EACH pipe.
        Then the results are saved in self.rho_1_0, self.rho_2_0, 
        self.v_0.
        """

        # m constant along pipe
        m1_list = self.m*self.mu
        m2_list = self.m*(1-self.mu)
        a_list = self.pipe_diameter**2*np.pi/4


        self.rho_1_0 = []
        self.rho_2_0 = []
        self.v_0 = []
        for i,m1 in enumerate(m1_list):
            m2 = m2_list[i]
            a = a_list[i]

            index_in = self.node_id.index(self.pipe_in[i])
            index_out = self.node_id.index(self.pipe_out[i])

            p_left = self.p_list[index_in]
            p_right = self.p_list[index_out]

            self._precompute_pressure_law()
            rho_1_0_l,rho_2_0_l,v_0_l = get_bc_data_1v(m1,m2,p_left,a,self.model,T=self.temperature,p_precomp=self.p_sympy)
            rho_1_0_r,rho_2_0_r,v_0_r = get_bc_data_1v(m1,m2,p_right,a,self.model,T=self.temperature,p_precomp=self.p_sympy)
            self.rho_1_0.append([rho_1_0_l,rho_1_0_r])
            self.rho_2_0.append([rho_2_0_l,rho_2_0_r])
            self.v_0.append([v_0_l,v_0_r])

            



    def _calculate_friction_parameter(self):
        """
        Calculates and stores the pipe friction from the pipe roughness. Uses a scaled version of the Nikuraze formula
        (as discussed in Section 1.5).
        """

        self.pipe_friction = 0.65*(2*np.log10(self.pipe_diameter /self.pipe_roughness)+1.138)**(-2)






    def discrete_system_jax(self,algebraic:bool):

        """
        Implements the box scheme discretisation of the one-velocity model as a function F.

        Args:
            algebraic (bool): constant in front of nonlinear term

        Returns:
            F (function): corresponds to the scheme. Has arguments for solutions and boundary conditions.
        """
    
        self._precompute_pressure_law()
        p = self.p_jax

        def F(y,y_old,bc_m,bc_mu,bc_p):
            """

            Stores the box scheme discretisation as a function which contains arguments for the new 
            solution y and the old solution y_old, as well as the boundary values. 
            
            For each pipe (including the endpoints at the nodes), y and y_old contain values of rho_1, rho_2, and v at the 
            corresponding set of discrete points. 
                
            F contains
                1. mass conservation equations for each constituent on each pipe
                2. the momentum conservation equation for the mixture on each pipe
                3. coupling conditions on internal nodes:
                    - total pressure continuity 
                    - total mass conservation 
                    - perfect mixing, i.e. the mixture ratio is the same for all outflows
                4. boundary data:
                    - m = rho*v on the INFLOW nodes 
                    - mu (mixing ratio) on the INFLOW nodes
                    - p(rho_1,rho_2) on the OUTFLOW nodes
                    (this is done by SWITCHING which nodes are treated as sinks and sources, see the init function)

            NOTE: 
                - Here we need that the pipe directions correspond to the flow directions and do not change.

            Args:
                y (jnp.ndarray): solution at new time step
                y_old (jnp.ndarray): soltuion at old time step
                bc_m1 (jnp.ndarray): array of mass flow boundary values for each node (0 if not bc node)
                bc_p1 (jnp.ndarray): array of pressure boundary values for each node (0 if not bc node)
                bc_m2 (jnp.ndarray): array of mass flow boundary values for each node (0 if not bc node)
                bc_p2 (jnp.ndarray): array of pressure for each node (0 if not bc node)

            Returns:
                result (jnp.ndarray): concatenation of all equations in F
            """
            # order of equations
            block_rho_1_list = []
            block_rho_2_list = []
            block_v_list = []
            mass_cons_list = []
            mass_1_cons_inner_list = []
            pressure_equality_list = []
            mu_exit_equality_list = []


            N_x_all_list = self.N_x_all_list
            dx_list = self.dx_list
            dt = self.dt

            # pipe discretizations
            for i,id in enumerate(self.pipe_id):
                
                previous_steps=int(np.sum(N_x_all_list[:i]))


                # get corresponding values
                rho_1_h = y[3*previous_steps:3*previous_steps+N_x_all_list[i]]
                rho_2_h = y[3*previous_steps+N_x_all_list[i]:3*previous_steps+2*N_x_all_list[i]]
                rho_p_h = rho_1_h+rho_2_h
                v_h = y[3*previous_steps+2*N_x_all_list[i]:3*previous_steps+3*N_x_all_list[i]]


                rho_1_h_old = y_old[3*previous_steps:3*previous_steps+N_x_all_list[i]]
                rho_2_h_old = y_old[3*previous_steps+N_x_all_list[i]:3*previous_steps+2*N_x_all_list[i]]
                rho_p_h_old = rho_1_h+rho_2_h
                v_h_old = y_old[3*previous_steps+2*N_x_all_list[i]:3*previous_steps+3*N_x_all_list[i]]



                # mass conservation CH4
                block_rho_1 = ((rho_1_h[1:]+rho_1_h[:-1])/2-(rho_1_h_old[1:]+rho_1_h_old[:-1])/2)\
                    +dt*jnp.diff(rho_1_h*v_h)/dx_list[i]

                # mass conservation H2
                block_rho_2 = ((rho_2_h[1:]+rho_2_h[:-1])/2-(rho_2_h_old[1:]+rho_2_h_old[:-1])/2) \
                    + dt*jnp.diff(rho_2_h*v_h)/dx_list[i]

                # momentum conservation
                #NOTE: *1e-5 for stability (calculate in bar instead of Pa)
                block_v = (((rho_p_h[1:]*v_h[1:]+rho_p_h[:-1]*v_h[:-1])/2 - (rho_p_h_old[1:]*v_h_old[1:]+rho_p_h_old[:-1]*v_h_old[:-1])/2)\
                    + dt*jnp.diff(algebraic*rho_p_h*v_h**2+p(rho_1_h,rho_2_h))/dx_list[i]\
                +dt*1/2*((self.pipe_friction[i]/(2*self.pipe_diameter[i])*(rho_1_h+rho_2_h)*jnp.abs(v_h)*v_h)[1:]\
                +(self.pipe_friction[i]/(2*self.pipe_diameter[i])*(rho_1_h+rho_2_h)*jnp.abs(v_h)*v_h)[:-1]))*1e-5
                    

                block_rho_1_list.append(block_rho_1)
                block_rho_2_list.append(block_rho_2)
                block_v_list.append(block_v)


            # coupling conditions on nodes
            for i,id in enumerate(self.node_id):

                a_list = self.pipe_diameter**2*jnp.pi/4

                # pipes where out going node == current node
                out_pipes_index_list = np.where(np.array(self.pipe_out) == id)[0] 
                # pipes where in node == current node
                in_pipes_index_list = np.where(np.array(self.pipe_in) == id)[0] 
                n_in_pipes = len(in_pipes_index_list)
                n_out_pipes = len(out_pipes_index_list)

                #mass conservation at inner nodes
                mass_cons = 0
                mass_1_cons_inner = 0
                for in_pipe_index in in_pipes_index_list:
                    

                    rho_1_0_h = y[3*np.sum(N_x_all_list[:in_pipe_index])]
                    rho_2_0_h = y[3*np.sum(N_x_all_list[:in_pipe_index])+N_x_all_list[in_pipe_index]]
                    v_0_h = y[3*np.sum(N_x_all_list[:in_pipe_index])+2*N_x_all_list[in_pipe_index]]

                    mass_cons -= a_list[in_pipe_index]*(rho_1_0_h+rho_2_0_h)*v_0_h
                    mass_1_cons_inner -= a_list[in_pipe_index]*rho_1_0_h*v_0_h


                for out_pipe_index in out_pipes_index_list:
                    rho_1_N1_h = y[3*np.sum(N_x_all_list[:out_pipe_index])+N_x_all_list[out_pipe_index]-1]
                    rho_2_N1_h = y[3*np.sum(N_x_all_list[:out_pipe_index])+2*N_x_all_list[out_pipe_index]-1]
                    v_N1_h = y[3*np.sum(N_x_all_list[:out_pipe_index])+3*N_x_all_list[out_pipe_index]-1]
                    mass_cons += a_list[out_pipe_index]*(rho_1_N1_h+rho_2_N1_h)*v_N1_h
                    mass_1_cons_inner += a_list[out_pipe_index]*rho_1_N1_h*v_N1_h



                ####################################################################
                ####################################################################
                ## MARKER START

                if  n_in_pipes>=1 and n_out_pipes >=1: 
                    # take any value of outgoing pipe (same value by perfect mixing)
                    out_pipe_index = self.pipe_in.index(id)
                    rho_1_0_h = y[3*np.sum(N_x_all_list[:out_pipe_index])]
                    rho_2_0_h = y[3*np.sum(N_x_all_list[:out_pipe_index])+N_x_all_list[out_pipe_index]]
                    mu_out = rho_1_0_h/(rho_1_0_h+rho_2_0_h)

                    q_out = mass_cons
                    mass_1_cons_inner -= mu_out*q_out
                    mass_1_cons_inner_list.append(jnp.array([mass_1_cons_inner]))

                if n_in_pipes>=1 and n_out_pipes >=1 and self.node_type[i]=="in_node":
                    mass_cons_list.append(jnp.array([mass_cons]))

                elif self.node_type[i] == "sink":

                    mass_cons += bc_m[i]
                    mass_cons_list.append(jnp.array([mass_cons]))

                ## MARKER END 
                ####################################################################
                ####################################################################
            
            

                # pressure equalities only where at 2 pipes or source node
                if (n_in_pipes + n_out_pipes ) >=2 or self.node_type[i] == "source":


                    p_value_first = 0 
                    # boundary condition
                    if self.node_type[i] == "source":
                        p_value_first = bc_p[i]


                    elif n_in_pipes >= 1:
                        index_first = in_pipes_index_list[0]
                        rho_1_0_l = y[3*np.sum(N_x_all_list[:index_first])]
                        rho_2_0_l = y[3*np.sum(N_x_all_list[:index_first]) + N_x_all_list[index_first]]
                        
                        p_value_first =  p(rho_1_0_l,rho_2_0_l)
                        # remove first pipe entry
                        in_pipes_index_list = in_pipes_index_list[1:]
                    else:
                        index_first = out_pipes_index_list[0]
                        rho_1_0_r = y[3*np.sum(N_x_all_list[:index_first]) + N_x_all_list[index_first]-1]
                        rho_2_0_r = y[3*np.sum(N_x_all_list[:index_first]) + 2*N_x_all_list[index_first]-1]

                        p_value_first = p(rho_1_0_r,rho_2_0_r)

                        # remove first pipe entry 
                        out_pipes_index_list = out_pipes_index_list[1:]
                            
                    for pipe_in_index in in_pipes_index_list:

                        rho_1_0_l = y[3*np.sum(N_x_all_list[:pipe_in_index])]
                        rho_2_0_l = y[3*np.sum(N_x_all_list[:pipe_in_index])+N_x_all_list[pipe_in_index]]
                        p_value = p(rho_1_0_l,rho_2_0_l)



                        h_cond = (p_value_first-p_value)*1e-5
                        pressure_equality_list.append(jnp.array([h_cond]))

                    for pipe_out_index in out_pipes_index_list:


                        rho_1_0_r = y[3*np.sum(N_x_all_list[:pipe_out_index])+N_x_all_list[pipe_out_index]-1]
                        rho_2_0_r = y[3*np.sum(N_x_all_list[:pipe_out_index])+2*N_x_all_list[pipe_out_index]-1]

                        p_value = p(rho_1_0_r,rho_2_0_r)
                        h_cond = (p_value_first-p_value)*1e-5
                        pressure_equality_list.append(jnp.array([h_cond]))


                # pipes where in node == current node
                in_pipes_index_list = np.where(np.array(self.pipe_in) == id)[0] 


                ####################################################################
                ####################################################################
                ## MARKER START

                # mu equalities only on out going
                if n_in_pipes >= 1:

                    if self.node_type[i] ==  "sink":
                        mu_value_first = bc_mu[i]
                    else:
                        index_first = in_pipes_index_list[0]
                        rho_1_0_l = y[3*np.sum(N_x_all_list[:index_first])]
                        rho_2_0_l = y[3*np.sum(N_x_all_list[:index_first]) + N_x_all_list[index_first]]
                        
                        mu_value_first = rho_1_0_l/(rho_1_0_l + rho_2_0_l)
                        # remove first pipe entry
                        in_pipes_index_list = in_pipes_index_list[1:]

                        
                    for pipe_in_index in in_pipes_index_list:

                        rho_1_0_l = y[3*np.sum(N_x_all_list[:pipe_in_index])]
                        rho_2_0_l = y[3*np.sum(N_x_all_list[:pipe_in_index])+N_x_all_list[pipe_in_index]]
                        mu_value = rho_1_0_l/(rho_1_0_l+rho_2_0_l)



                        mu_cond = (mu_value_first-mu_value)
                        mu_exit_equality_list.append(jnp.array([mu_cond]))
                ## MARKER END 
                ####################################################################
                ####################################################################




            
            result = jnp.concatenate(
                [ item for item in block_rho_1_list ]
               + [ item for item in block_rho_2_list ]
               + [ item for item in block_v_list ]
               + [ item for item in mass_cons_list ]
               + [ item for item in mass_1_cons_inner_list ]
               + [ item for item in  pressure_equality_list]
               +[item for item in mu_exit_equality_list])
            return result
            
        return F



    def solve(self,algebraic:float,tol:float,scenario:str):
        """
        This routine implements Newton's method to compute numerical solutions of the one-velocity model
        discretised using the box scheme. 

        Args:
            algebraic (float): algebraic value in front of nonlinearity
            tol (float): Newton tolerance
            scenario (str):choice of initial and boundary condition (passed to self.get_initial_and_bc_data)
        """

        self._calculate_friction_parameter()
        self._precompute_pressure_law()

        # boundary data of algebraic solution
        self.convert_boundary_conditions()


        initial_vec, bc_m, bc_mu, bc_p = self.get_initial_and_bc_data(scenario = scenario,algebraic=algebraic,tol=tol)

        
        F = self.discrete_system_jax(algebraic=algebraic)

        #newton iteration steps
        n_iter = 100

        self.conv = []
        self.u_sol = initial_vec
        u_new = deepcopy(initial_vec)
        self.u_store = np.zeros(((self.N_time),initial_vec.shape[0]))
        self.u_store[0] = initial_vec
        if scenario == "time_dep":
            long_time_conv = True
        else: 
            long_time_conv = False
        
        for i in range(self.N_time-1):
            
            print("".center(80,"-"))
            print(f"Time Step {i+1}")

            #Newton solver implementation at each time step
            u_time_t = deepcopy(u_new)
            F_step = lambda x: F(x,u_time_t,bc_m[i,:],bc_mu[i,:],bc_p[i,:])
            F_deriv = jax.jacfwd(F_step)
            
            for j in range(n_iter):

                rhs = F_step(u_new)

                rhs_norm = jnp.linalg.norm(rhs)
                if rhs_norm < tol:
                    break
                elif not np.isfinite(rhs_norm):
                    raise Exception(f"Newton solver not converging - norm is NaN at {j} Newton steps")

                dk = jnp.linalg.solve(F_deriv(u_new),-rhs)
                u_new = u_new + dk
                
                if j == n_iter-1:
                    raise Exception(f"Newton solver not converged after {n_iter} steps")
                     
            self.u_sol = u_new
            
            self.u_store[i+1] = u_new

            if j==0 and scenario != "time_dep":
                long_time_conv = True
                break


            if rhs_norm < tol:
                self.conv.append(True)
            else:
                self.conv.append(False)
        if all(self.conv):
            self.conv = True
        
        if not scenario =="time_dep" and not long_time_conv: 
            raise Exception("The solution did not converge to a long time limit!")





    def get_initial_and_bc_data(self,scenario:str, algebraic:float=1.0, tol:float=1e-7):
        """
        Computes and stores initial and boundary data.

        Args:
            scenario (str): Choice of initial and boundary data scenario. Choices:
                - gaslib11_stationary (for stationary solutions initialised using constant values)
                - gaslib40_stationary (for stationary solutions initialised using algebraic solution)
                - time_dep (for instationary aka "time dependent" solutions)
            algebraic (float): coefficient in front of nonlinearity (needed for instationary solutions)
            tol (float): Newton tolerance (needed for instationary solutions)

        Raises:
            Exception: If scenario not implemented.

        Returns:
            (initial_vec,bc_m,bc_mu,bc_p) (tuple): initial vector and arrays of boundary conditions 
        """
        



        N_x_all_list = self.N_x_all_list
        N_time = self.N_time
        T = self.T

        bc_m = jnp.zeros((N_time,len(self.node_id)))
        bc_mu = jnp.zeros((N_time,len(self.node_id)))
        bc_p = jnp.zeros((N_time,len(self.node_id)))


        # constant values everywhere
        if scenario == "one_pipe":

            a_list = self.pipe_diameter**2*np.pi/4
            m1 = 41.16666666*2/3
            m2 = 41.16666666/3
            p = 70*1e5

            rho_1_source, rho_2_source,v_source = get_bc_data_1v(m1=m1,m2=m2,p=p,a=a_list[0])
            initial_vec = jnp.zeros(3*jnp.sum(N_x_all_list))

            for i,id in enumerate(self.pipe_id):
                rho_1_h =  rho_1_source*jnp.ones(N_x_all_list[i])
                rho_2_h =  rho_2_source*jnp.ones(N_x_all_list[i])
            
                v_h = v_source*jnp.ones((N_x_all_list[i]))
            
                initial_vec = initial_vec.at[3*np.sum(N_x_all_list[:i]):3*np.sum(N_x_all_list[:i]) + N_x_all_list[i]].set(rho_1_h )
                initial_vec = initial_vec.at[3*np.sum(N_x_all_list[:i])+N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]].set(rho_2_h)
                initial_vec = initial_vec.at[3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]].set(v_h)
            
            

            
            # boundary conditons in m_form
            for i,id in enumerate(self.node_id):
                if self.node_type[i] in ["sink"]:
                    bc_m = bc_m.at[:,i].set(-a_list[0]*(rho_1_source+rho_2_source)*v_source*np.ones(N_time))

                    mu = rho_1_source/(rho_1_source+rho_2_source)
                    bc_mu = bc_mu.at[:,i].set(mu*np.ones(N_time))
                    
                    

                if self.node_type[i] in ["source"]:


                    p = self.p_jax(rho_1_source,rho_2_source)
                    bc_p = bc_p.at[:,i].set(p*np.ones(N_time))




        elif scenario == "gaslib11_stationary":
            #This scenario initialises the values as constants everywhere in the network. 
            #The boundary conditions are determined as follows: identify source values 
            #(i.e. boundary values compatible with the initial state) and goal values 
            #(i.e. boundary values compatible with a steady state of ISO-4). Then we 
            #interpolate from the source value to the goal value for the FIRST HALF
            #of the time period, after which we keep the boundary conditions fixed.

            a_list = self.pipe_diameter**2*np.pi/4
            m1 = 41.16666666*2/3
            m2 = 41.16666666/3
            p = 70*1e5

            rho_1_source, rho_2_source,v_source = get_bc_data_1v(m1=m1,m2=m2,p=p,a=a_list[0])


            initial_vec = jnp.zeros(3*jnp.sum(N_x_all_list))
            for i,id in enumerate(self.pipe_id):
                rho_1_h =  rho_1_source*jnp.ones(N_x_all_list[i])
                rho_2_h =  rho_2_source*jnp.ones(N_x_all_list[i])
        
                v_h = v_source*jnp.ones(N_x_all_list[i])
            
                initial_vec = initial_vec.at[3*np.sum(N_x_all_list[:i]):3*np.sum(N_x_all_list[:i])+N_x_all_list[i]].set(rho_1_h )
                initial_vec = initial_vec.at[3*np.sum(N_x_all_list[:i])+N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]].set(rho_2_h)
                initial_vec = initial_vec.at[3*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]:3*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]].set(v_h)

            bc_m = jnp.zeros((N_time,len(self.node_id)))
            bc_mu = jnp.zeros((N_time,len(self.node_id)))
            bc_p = jnp.zeros((N_time,len(self.node_id)))
            # boundary conditons in m_form
            for i,id in enumerate(self.node_id):
                
                N_time_half = round((N_time)/2)
                if self.node_type[i] in ["sink"]:
                    m_goal = self.m_node_flow[i]

                    # pipes where out going node == current node
                    out_pipes_index_list = np.where(np.array(self.pipe_out) == id)[0] 
                    # pipes where in node == current node
                    in_pipes_index_list = np.where(np.array(self.pipe_in) == id)[0] 
                    n_in_pipes = len(in_pipes_index_list)
                    n_out_pipes = len(out_pipes_index_list)

                    m_initial = -a_list[0]*(rho_1_source+rho_2_source)*v_source*(n_out_pipes-n_in_pipes)
                    bc_m_int = np.linspace(m_initial,m_goal,N_time_half)
                    bc_const = m_goal*np.ones(N_time-N_time_half)

                    bc_m = bc_m.at[:,i].set(np.hstack((bc_m_int,bc_const)))
 

                    mu_goal= self.mu_node[i]
                    mu_initial = np.linspace(rho_1_source/(rho_1_source+rho_2_source),mu_goal,N_time_half)
                    mu_const = mu_goal*np.ones(N_time-N_time_half)
                    

                    bc_mu = bc_mu.at[:,i].set(np.hstack((mu_initial,mu_const)))


                if self.node_type[i] in ["source"]:


                    p_initial = self.p_jax(rho_1_source,rho_2_source)
                    
                    p_goal = p_list[i]


                    p_int = np.linspace(p_initial,p_goal,N_time_half)
                    p_const = p_goal*np.ones(N_time-N_time_half)

                    bc_p = bc_p.at[:,i].set(np.hstack((p_int,p_const)))
            
        


        elif scenario == "gaslib40_stationary":
            #This is the other major scenario for arriving at stationary solutions.
            #Here we prepare initial data by making use of all the "goal" values
            #at each node as discussed in gaslib11_stationary, and then using the
            #Weymouth equation to build the rest of the solution in the network. 
            #In contrast to gaslib11_stationary, this works also on gaslib40/networks 
            #larger than the small ones. On the other hand, it is not clear if it
            #will work for the 2v case. It is also not clear if, for networks where 
            #both gaslib11_stationary and gaslib40_stationary are viable, whether 
            #we arrive at the same solutions. 

            initial_vec = []
            rho_1_initial = []
            rho_2_initial = []
            v_initial = []
            m_initial = []
            mu_initial = []
            p_initial = []
            a_list = self.pipe_diameter**2*np.pi/4
            p_dict_exact_sol = {}
            for i,pipe_id in enumerate(self.pipe_id):

                m1_const = a_list[i]*self.rho_1_0[i][0]*self.v_0[i][0]
                m2_const = a_list[i]*self.rho_2_0[i][0]*self.v_0[i][0]
                mu_const = self.rho_1_0[i][0]/(self.rho_1_0[i][0]+self.rho_2_0[i][0])
                p_in_const = self.p_jax(self.rho_1_0[i][0],self.rho_2_0[i][0])
                
                x_points = np.linspace(0,self.pipe_length[i],N_x_all_list[i])
                p_list = [p_in_const]
                p_in = p_in_const


                def compute_weymouth(mu,m,p_in,pipe_index,x):
                    
                    R = 8.314472
                    M_1 = 16.042460/1000
                    M_2 = 2.015880/1000

                    c_mu = jnp.sqrt(R*self.temperature*(mu/M_1 +(1-mu)/M_2))

                    A = (self.pipe_diameter[pipe_index])**2*jnp.pi/4
                    rhs = (x)*(self.pipe_friction[pipe_index])*c_mu**2/(A**2*self.pipe_diameter[pipe_index]) *jnp.abs(m)*m*1e-10

                    rhs -= (p_in*1e-5)**2
                    p_out = np.sqrt(-rhs)
                    return p_out*1e5
            


                for k in range(N_x_all_list[i]-1):
                    p_out = compute_weymouth(mu=mu_const,m=(m1_const+m2_const),p_in=p_in,pipe_index=i,x=x_points[k+1])
                    p_list.append(p_out)
                    
                
                rho_1_pipe = []
                rho_2_pipe = []
                v_pipe = []
                for p in p_list:
                    rho_1,rho_2,v = get_bc_data_1v(m1=m1_const,m2=m2_const,p=p,a=a_list[i],
                                                   model=self.model,T=self.temperature,p_precomp=self.p_sympy)
                    rho_1_pipe.append(rho_1)
                    rho_2_pipe.append(rho_2)
                    v_pipe.append(v)
                
                p_dict_exact_sol[pipe_id] = p_list

                rho_1_initial += rho_1_pipe
                rho_2_initial += rho_2_pipe
                v_initial += v_pipe


                initial_vec += rho_1_pipe+rho_2_pipe+v_pipe
            initial_vec = jnp.array(initial_vec)

            for i,id in enumerate(self.node_id):
            
                if self.node_type[i] in ["sink"]:
                    m_goal = self.m_node_flow[i]
                    bc_const = m_goal*np.ones(N_time)

                    m_goal = self.m_node_flow[i]


                    #no need for different m !
                    bc_const = m_goal*np.ones(N_time)

                    bc_m = bc_m.at[:,i].set(bc_const)


                    ####################################################################
                    ####################################################################
                    ## MARKER START
                    # mu bc
                    mu_goal= self.mu_node[i]
                    mu_const = mu_goal*np.ones(N_time)

                    bc_mu = bc_mu.at[:,i].set(mu_const)

                    ## MARKER END 
                    ####################################################################
                    ####################################################################

                if self.node_type[i] in ["source"]:

                    
                    p_goal = self.p_list[i]

                    p_const = p_goal*np.ones(N_time)
                    bc_p = bc_p.at[:,i].set(p_const)
                    
        
        elif scenario == "time_dep":

            timesteplength = round(T/(N_time-1))

            
            timerange = [timesteplength*j for j in range(N_time)]

            
            
            #First we try and load the initial data. If it doesn't already exist, we generate it.
            # self.load_solution_network(N=N_space,algebraic=algebraic,scenario="gaslib40_stationary")
            try:
                self.load_solution_network(algebraic=algebraic, scenario="gaslib40_stationary")
                print("Stationary one-velocity solution loaded")
                
            except:
                print("Need to first compute stationary one-velocity solution")
                network_stationary = Network_1v_time(self.file_network,self.file_data,self.model,candidate_dx=self.candidate_dx,dt=60*24*150*15,T=60*60*24*150*15)
                network_stationary.solve(algebraic=algebraic,tol=tol,scenario="gaslib40_stationary")
                network_stationary.save_solution_network(algebraic=algebraic,scenario="gaslib40_stationary")
                self.load_solution_network(algebraic=algebraic,scenario="gaslib40_stationary")
                print("Stationary one-velocity solution computed")


            initial_vec = self.u_sol
            
            t_mid = T/2
            t_fourth = T/4
            bc_m = jnp.zeros((N_time,len(self.node_id)))
            bc_mu = jnp.zeros((N_time,len(self.node_id)))
            bc_p = jnp.zeros((N_time,len(self.node_id)))
            # boundary conditons in m_form
            for i,id in enumerate(self.node_id):
                
                
                if self.node_type[i] in ["sink"]:

                    m_goal = self.m_node_flow[i]

                    
                    def bc_m_values(t):
                        if t<t_fourth:
                            m_value = (1.02-0.08*np.abs(t-t_fourth)/T)*m_goal
                        elif t_fourth<=t<t_mid:
                            m_value = 1.02*m_goal
                        elif t_mid<=t<3*t_fourth:
                            m_value = (1.02-0.08*np.abs(t-t_mid)/T)*m_goal
                        else:
                            m_value = m_goal
                        return m_value

                    bc_m_vectorised = np.vectorize(bc_m_values)
                    bc_m = bc_m.at[:,i].set(bc_m_vectorised(timerange))



                    mu_goal= self.mu_node[i]

                    def bc_mu_values(t):
                        if t<t_fourth:
                            mu_value = mu_goal
                        elif t_fourth<=t<3*t_fourth:
                            mu_value = (1.02-0.08*np.abs(t-t_mid)/T)*mu_goal 

                        else:
                            mu_value = mu_goal
                        return mu_value
                            
                    bc_mu_vectorised = np.vectorize(bc_mu_values)
                    bc_mu = bc_mu.at[:,i].set(bc_mu_vectorised(timerange))

                
                if self.node_type[i] in ["source"]:

                    
                    p_goal = self.p_list[i]

                    def bc_p_values(t):
                        if t <= t_mid:
                            p_value = (1.02-0.08*np.abs(t-t_fourth)/T)*p_goal
                        else: 
                            p_value = p_goal
                        return p_value

                    bc_p_vectorised = np.vectorize(bc_p_values)
                    bc_p = bc_p.at[:,i].set(bc_p_vectorised(timerange))
         


        else:
            raise Exception("Initial Data NOT implemented!")

        return initial_vec,bc_m,bc_mu,bc_p



        

            
    def save_solution_network(self,algebraic:float,scenario:str):
        """
        Saves the solutions inside the folder "save_solution/solutions/network_1v_timedep" for later use/reuse.
        
        Args:
            algebraic (float): algebraic value in front of nonlinearity
            scenario (str): choice of initial and boundary condition (passed to self.get_initial_and_bc_data)
        """
        
        if scenario == "time_dep":
            file_path = Path(f"save_solution/solutions/network_1v_timedep/massflowinflow/1v_{self.network_data}_dx_{self.candidate_dx}_dt_{self.dt}_T_{self.T}_SCENARIO_{scenario}_pressure_{self.model}_algebraic_{algebraic}.pkl")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as file:
                pickle.dump(self.u_store, file)
        else:
            file_path = Path(f"save_solution/solutions/network_1v_timedep/massflowinflow/1v_{self.network_data}_dx_{self.candidate_dx}_SCENARIO_{scenario}_pressure_{self.model}_algebraic_{algebraic}.pkl")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as file:
                pickle.dump(self.u_sol, file)
            
        
    def load_solution_network(self,algebraic:float,scenario:str):
        """
        Loads a solution for further computations/plots. 

        Args: as in save_solution_network
        """
        
        if scenario == "time_dep":
            with open(f"save_solution/solutions/network_1v_timedep/massflowinflow/1v_{self.network_data}_dx_{self.candidate_dx}_dt_{self.dt}_T_{self.T}_SCENARIO_{scenario}_pressure_{self.model}_algebraic_{algebraic}.pkl", "rb") as file:
                self.u_store = pickle.load(file)
        else:
            with open(f"save_solution/solutions/network_1v_timedep/massflowinflow/1v_{self.network_data}_dx_{self.candidate_dx}_SCENARIO_{scenario}_pressure_{self.model}_algebraic_{algebraic}.pkl", "rb") as file:
                self.u_sol = pickle.load(file)

    def plot_all_stationary(self,algebraic:float,plot_pipe=False,scenario:str=None,show_labels:bool=False,
                            label_node_list:str=None,offset_labels:float=None,
                            save_legend:bool=False):
        """
        Calls the plot function plot_sol_stationary for various quantities. 

        Args:
            algebraic (float): value in front of nonlinearity (used for name when saving plots)
            plot_pipe (bool): whether or not to plot results for each individual pipe.  
            scenario (str): choice of initial and boundary data (used for name when saving plots)
            show_labels (bool): whether node labels are shown
            label_node_list (bool): whether or not to add names of every node to plot. 
            offset_labels (float): amount by which to offset label from node

        """


        self.plot_sol_stationary(prop="rho_1",prop_name="$\\rho_1$",unit="$\\frac{kg}{m^3}$",
                                 algebraic=algebraic,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="rho_2",prop_name="$\\rho_2$",unit="$\\frac{kg}{m^3}$",
                                 algebraic=algebraic,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="v",prop_name="$v$",unit="$\\frac{m}{s}$",
                                 algebraic=algebraic,arrows=True,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="pressure",prop_name="$p$",unit="$bar$",
                                 algebraic=algebraic,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="m_1",prop_name="$m_1$",unit="$\\frac{kg}{m^3s}$",
                                 algebraic=algebraic,arrows=True,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="m_2",prop_name="$m_2$",unit="$\\frac{kg}{m^3s}$",
                                 algebraic=algebraic,arrows=True,plot_pipe=plot_pipe, scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="mu",prop_name="$\\mu$",unit="none",
                                 algebraic=algebraic,arrows=True,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        # self.plot_sol_stationary(prop="h",prop_name="$h$",unit="bar",N=N,algebraic=algebraic,save_name_fig=save_name_fig,plot_pipe=plot_pipe,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)

    
    def plot_sol_stationary(self,prop:str,prop_name:str,unit:str,algebraic:float,arrows:bool=False
                            ,plot_pipe:bool=False,scenario:str=None,show_labels:bool=False,
                            label_node_list:list[str]=None,offset_labels:float=None,save_legend:bool=False):
        """
        Plots a specific quantity (either solution quantities i.e. rho_1, rho_2, v or computed quantities e.g. m = rho*v).

        Args:
            prop (str): choice of quantity/property 
            prop_name (str): name of property to be displayed in plot. SHOULD CORRESPOND TO prop. 
            unit (str): unit. SHOULD CORRESPOND TO prop. 
            algebraic (float): value in front of nonlinearity (used for name when saving plots)
            arrows (bool): whether or not to add arrows to indicate direction. Used for vector quantities e.g. velocity.
            plot_pipe (bool): whether or not to plot results for each individual pipe.
            scenario (str): choice of initial and boundary data (used for name when saving plots)
            show_labels (bool): whether node labels are shown
            label_node_list (bool): whether or not to add names of every node to plot. 
            offset_labels (float): amount by which to offset label from node
        """
    
            



        self._precompute_pressure_law()
        p = self.p_jax
        a_list = self.pipe_diameter**2*np.pi/4


        # create colormap
        prop_values = []

        for i,in_id in enumerate(self.pipe_in):
            rho1 = self.u_sol[3*np.sum(self.N_x_all_list[:i]):3*np.sum(self.N_x_all_list[:i])+self.N_x_all_list[i]]
            rho2 = self.u_sol[3*np.sum(self.N_x_all_list[:i])+self.N_x_all_list[i]:3*np.sum(self.N_x_all_list[:i])+2*self.N_x_all_list[i]]
            v = self.u_sol[3*np.sum(self.N_x_all_list[:i])+2*self.N_x_all_list[i]:3*np.sum(self.N_x_all_list[:i])+3*self.N_x_all_list[i]]
            
            if plot_pipe or self.network_name=="one_pipe":
                fig_pipe,ax_pipe = plt.subplots(constrained_layout=True)

            if prop == "rho_1": 
                prop_values.append(rho1)

            elif prop == "rho_2": 
                prop_values.append(rho2)

            elif prop == "v": 
                prop_values.append(v)
            
            elif prop == "pressure":
                prop_values.append(p(rho1,rho2)*1e-5)

            elif prop == "m_1":
                prop_values.append(a_list[i]*rho1*v)

            elif prop == "m_2":
                prop_values.append(a_list[i]*rho2*v)
            elif prop == "h":
                prop_values.append(((rho1+rho2)*v**2+p(rho1,rho2))*1e-5)
            elif prop == "mu":
                prop_values.append(rho1/(rho1 + rho2))

            if self.network_name == "one_pipe" or plot_pipe == True:

                x_values = np.linspace(0,self.pipe_length[i],len(prop_values[i]))
                
                
                if self.network_name == "one_pipe":
                    ax_pipe.plot(x_values,prop_values[i])
                    ax_pipe.set_xlabel("x in m")
                    ax_pipe.set_ylabel(f"{prop_name}")
                    file_path = Path(f"graphics/networks/network_1v_timedep/massflowinflow/{self.network_data}/1v_dx_{self.candidate_dx}_dt_{self.dt}_T_{self.T}_SCENARIO_{scenario}_pressure_{self.model}_algebraic_{algebraic}_PROP_{prop}")
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    fig_pipe.savefig(file_path, backend="pgf")
                    plt.close()
                else:
                    fig_pipe, ax_pipe = plt.subplots(constrained_layout=True)
                    ax_pipe.plot(x_values,prop_values[i])
                    ax_pipe.set_xlabel("x in m")
                    ax_pipe.set_ylabel(f"{prop_name}")
                    file_path = Path(f"graphics/networks/network_1v_timedep/massflowinflow/{self.network_data}/1v_dx_{self.candidate_dx}_dt_{self.dt}_T_{self.T}_SCENARIO_{self.network_data}_pressure_{self.model}_algebraic_{algebraic}_PROP_{prop}_pipe_{i}.pdf")
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    fig_pipe.savefig(file_path, backend="pgf")
                    plt.close()

        if self.network_name != "one_pipe":
            fig, ax = plt.subplots(constrained_layout=True)
            all_values = np.concatenate([prop for prop in prop_values])
            if np.min(all_values) < 0 and np.max(all_values) > 0:
                norm = mcolors.TwoSlopeNorm(vmin=np.min(all_values), vcenter=0, vmax=np.max(all_values))
                cmap = plt.get_cmap('seismic')
            else:
                norm = mcolors.Normalize(vmin=all_values.min(), vmax=all_values.max())
                cmap = cm.viridis
    
    
            
            for i,in_id in enumerate(self.pipe_in):
                out_id = self.pipe_out[i]
    
                node_in_index = self.node_id.index(in_id)
                node_out_index = self.node_id.index(out_id)
                x_0 = self.node_x_cord[node_in_index]
                x_N_1 = self.node_x_cord[node_out_index]
                y_0 = self.node_y_cord[node_in_index]           
                y_N_1 = self.node_y_cord[node_out_index]
        
    
                x_values = np.linspace(x_0,x_N_1,self.N_x_all_list[i])
                y_values = np.linspace(y_0,y_N_1,self.N_x_all_list[i])
    
                if arrows:
                    n_arrows = 10
                    dist = int(np.ceil(len(x_values)/n_arrows))
                
    
                for j in range(len(x_values)-1):
                    x_start = x_values[j]
                    x_end = x_values[j+1]
                    y_start = y_values[j]
                    y_end = y_values[j+1]
    
                    color = cmap(norm(prop_values[i][j]))
                    ax.plot([x_start,x_end],[y_start,y_end],color = color,linewidth=5,zorder=1)
    
    
                    if arrows and j%dist == 0:
                        xm = (x_start+ x_end) / 2
                        ym = (y_start+ y_end) / 2
                        # Calculate the direction of the arrow
                        dx = x_end - x_start
                        dy = y_end - y_start
                        if prop_values[i][j] <0:
                            dx*= -1
                            dy*= -1
                        ax.annotate('', xy=(xm + dx*0.1, ym + dy*0.1), xytext=(xm, ym),
                                    arrowprops=dict(arrowstyle='->', color='lightgrey'),fontsize=8,zorder=2)
    
    
    
            index_source = self.node_type == "source"
            index_sink = self.node_type == "sink"
            index_in_nodes = self.node_type == "in_node"
            ax.scatter(np.array(self.node_x_cord)[index_source],
                     np.array(self.node_y_cord)[index_source],color="r",
                      marker="o", label="$\\Gamma_p$",s=80,zorder=3)
            ax.scatter(np.array(self.node_x_cord)[index_in_nodes],
                     np.array(self.node_y_cord)[index_in_nodes],color="b",
                      marker="s", label="$\\Gamma_q$",s=80,zorder=3)
            ax.scatter(np.array(self.node_x_cord)[index_sink],
                        np.array(self.node_y_cord)[index_sink],
                        marker="s",color="b",s=80,zorder=3)
    
            if show_labels:
                for i,node_id in enumerate(self.node_id):
                    if label_node_list is not None:
                        node_id = label_node_list[i]
                    offset_x = 0.5
                    offset_y = 0
                    if offset_labels is not None:
                        if offset_labels[i] is not None:
                            offset_x = offset_labels[i][0]
                            offset_y = offset_labels[i][1]
                    ax.annotate(node_id,(self.node_x_cord[i]+offset_x,self.node_y_cord[i]+offset_y),ha="center")
    
    
            ax.legend(loc="lower right")
    
    
    
            ax.set_xticks([])
            ax.set_yticks([])
            # Add colorbar
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.ax.set_yscale('linear')
    

            cbar.set_label(f"{prop_name} in {unit}")
    
    
    
            save_name = f"graphics/networks/network_1v_timedep/massflowinflow/{self.network_data}/1v_dx_{self.candidate_dx}_dt_{self.dt}_T_{self.T}_SCENARIO_{scenario}_pressure_{self.model}_algebraic_{algebraic}_PROP_{prop}_colourmap"
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
            plt.close()

    
############################################  
#END OF DEFINITION OF Network_1v_time class
############################################  


def one_pipe_computations():

    """
    Computations on a single pipe. Set up to run stationary solutions.
    """

    file_network = Path("network_data" ,"optimization_data", "network_files", "one_pipe.net")
    file_data = Path("network_data", "optimization_data", "solution_files","one_pipe.lsf")

    stationary_or_instationary = 0
    
    scenarios = ["gaslib40_stationary", "time_dep"] 
    scenario = scenarios[stationary_or_instationary]
    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[stationary_or_instationary]
    dt = int(T/60)
    
    
    #Parameters which don't depend on stationary/instationary
    algebraic = 1.0 
    tol = 1e-7
    
    #Change to fit network
    candidate_dx = 200

    model ="speed_of_sound"
    algebraic = 1.0
    
    
    network = Network_1v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=candidate_dx,
                              dt=dt,T=T)
    try:
        network.load_solution_network(algebraic=algebraic, scenario=scenario)
    except FileNotFoundError: 
        network.solve(algebraic=algebraic,tol=tol,scenario=scenario)
        network.save_solution_network(algebraic=algebraic, scenario=scenario)
    network.plot_all_stationary(algebraic=algebraic,scenario=scenario)


def compute_3mix_scenarios():
    """
    Computations on a 3-star/Y-shape network. Set up to run stationary solutions.
    """

    
    #We can also adapt this to do stationary_or_instationary
    stationary_or_instationary = 0
    
    scenarios = ["gaslib40_stationary", "time_dep"] 
    scenario = scenarios[stationary_or_instationary]
    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[stationary_or_instationary]
    dt = int(T/60)
    
    
    #Parameters which don't depend on stationary/instationary
    algebraic = 1.0 
    tol = 1e-7
    multiplicative_factor = 1/50
    
    #Change to fit network
    candidate_dx = 200

    model ="speed_of_sound"


    tol = 1e-7
    algebraic = 1.0

    file_network = Path("network_data" ,"optimization_data","network_files","3mixT.net")
    file_data = Path("network_data", "optimization_data","3mix_scenarios", "3mix_temp_0.lsf")
    network = Network_1v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=candidate_dx,
                            dt=dt,T=T)

    try:
        network.load_solution_network(algebraic=algebraic,scenario=scenario)
    except FileNotFoundError: 
        network.solve(candidate_dx=candidate_dx, dt=dt,T=T,algebraic=algebraic,tol=tol,scenario=scenario, multiplicative_factor=multiplicative_factor)
        network.save_solution_network(algebraic=algebraic, scenario=scenario)
    network.plot_all_stationary(algebraic=algebraic,scenario=scenario)

def compute_gaslib40_modified_smaller_disc(model:str,algebraic:float):
    """
    Computations on GasLib40-3. Set up to run stationary solutions.
    """


    file_network = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.net")
    file_data = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.lsf")
    
    # Parameters needed for network.solve

    stationary_or_instationary = 0
    
    scenarios = ["gaslib40_stationary", "time_dep"] 
    scenario = scenarios[stationary_or_instationary]
    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[stationary_or_instationary]
    dt = int(T/60)
    
    
    #Parameters which don't depend on stationary/instationary
    tol = 1e-7
    
    #Change to fit network
    candidate_dx = 500

    
    
    network = Network_1v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=candidate_dx,
                              dt=dt,T=T)
    try:
        network.load_solution_network(algebraic=algebraic,scenario=scenario)
    except FileNotFoundError: 
        network.solve(algebraic=algebraic,tol=tol,scenario=scenario)
        network.save_solution_network(algebraic=algebraic, scenario=scenario)
    network.plot_all_stationary(algebraic=algebraic,scenario=scenario)

def compute_gaslib40_modified(model:str,algebraic:float):
    """
    Computations on GasLib40-3. Set up to run stationary solutions.
    """


    file_network = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.net")
    file_data = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.lsf")
    
    # Parameters needed for network.solve

    stationary_or_instationary = 0
    
    scenarios = ["gaslib40_stationary", "time_dep"] 
    scenario = scenarios[stationary_or_instationary]
    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[stationary_or_instationary]
    dt = int(T/60)
    
    
    #Parameters which don't depend on stationary/instationary
    tol = 1e-7
    
    #Change to fit network
    candidate_dx = 1000

    
    
    network = Network_1v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=candidate_dx,
                              dt=dt,T=T)
    try:
        network.load_solution_network(algebraic=algebraic,scenario=scenario)
    except FileNotFoundError: 
        network.solve(algebraic=algebraic,tol=tol,scenario=scenario)
        network.save_solution_network(algebraic=algebraic, scenario=scenario)
    network.plot_all_stationary(algebraic=algebraic,scenario=scenario)

# if __name__ == "__main__":


    # #Network and data specification
    # index = 2
    # network_files = ["3mixT","gaslib40_edit","gaslib40_removed_edit","one_pipe","testm_new"]
    # network_file = network_files[index]
    # file_network = Path("network_data" ,"optimization_data", "network_files", str(network_file)+".net")
   
    # init_and_bdry_data = ['3mix_scen_1', 'gaslib40_edit','gaslib40_removed_edit','one_pipe','testm-testm_new']
    # init_and_bdry_choice = init_and_bdry_data[index]
    # file_data = Path("network_data" ,"optimization_data", "solution_files", str(init_and_bdry_choice)+".lsf")
   
    # #Pressure laws 
    # models = ["speed_of_sound", "virial_expansion", "virial_expansion_mix","gerg", "gerg_fit"] 
    # model = models[0]
   
    # #Declare network
   
    # #Choose type of solution to be computed, affecting the time horizon and the dt. 
    # stationary_or_instationary = 1
   
    # scenarios = ["gaslib40_stationary", "time_dep"] 
    # scenario = scenarios[stationary_or_instationary]
   
    # timehorizons = [60*60*24*150*15, 60*60*4]
    # T = timehorizons[stationary_or_instationary]
   
    # #Parameters which don't depend on stationary/instationary
    # algebraic = 1.0 
    # tol = 1e-7
    # multiplicative_factor = 1/50


    # for i in range(2):
    #     actualexponent = -i+7
    #     scale = 1.5**(-(actualexponent))
    #     candidate_dx = 1500*scale
    #     candidate_dt = T*3/60*scale
    #     dt = T/int(np.round(T/candidate_dt))
       
        
    #     network = Network_1v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=candidate_dx,
    #                               dt=dt,T=T)
                
    #     print("")
    #     print(f"scaling exponent = {actualexponent}")
    #     print(f"candidate dx = {candidate_dx}")
    #     print(f"mean dx = {np.mean(network.dx_list)}")
    #     print(f"mean dx ratio = {np.mean(network.dx_list/candidate_dx)}")
    #     print(f"std dev dx ratio = {np.std(network.dx_list/candidate_dx)}")
    #     print(f"candidate dt = {candidate_dt}")
    #     print(f"real dt = {dt}")
    #     print(f"dt ratio = {dt/candidate_dt}")
    #     print(f"Actual time = {(network.N_time-1)*network.dt}")
    #     network.solve(algebraic=algebraic, tol=tol, scenario=scenario)
    #     network.save_solution_network(algebraic=algebraic, scenario=scenario)
    
    # print("Finished computation!")
  
    
