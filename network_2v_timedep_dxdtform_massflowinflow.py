"""
This Python program contains code for computing numerical solutions to the two-velocity model on a general network.

NOTE: This code considers the case where we set MASS FLOW on the inflow nodes. 
"""
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt


import xmltodict
import jax
import pickle

from pathlib import Path
from one_pipe_setinflow import get_bc_data_1v
from pressure_laws_comparison import pressure_law_mixtures_symbolic
from network_1v_timedep_dxdtform_massflowinflow import Network_1v_time
from jax import numpy as jnp
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

class Network_2v_time:
    ##network files
    network_file:str
    network_data:str
    

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
    
    #1v solutions (for boundary conditions)
    u_1v_sol:np.ndarray
    u_1v_store:np.ndarray
    
    #friction
    f:float
    
    #tolerance
    tol:float
    # pressure laws
    model:str
    # name of pressure law for display
    model_name:str

    # after precomputaion of model (sympy faster, but jax needed for derivative)
    # NOTE: standard accuracy of jax is 32 bit -> different results possible
    p1_jax:callable
    p2_jax:callable
    
    #computational parameters
    candidate_dx:float
    N_x_all_list:np.ndarray
    max_dx:float
    dt:float
    T:float
    N_time:int



    def __init__(self,file_network:Path,file_data:Path,model:str,
                 candidate_dx:float,dt:float,T:float,model_name:str=None):
        """
        This function initializes the network by 2 given .xml files, 
        where the network structure is given in file_network
        and the data in file_data. The file "file_network" should have the same
        keys and structure as "network_data/optimization_data/gaslib40/gaslib40_removed_edit.net".
        The file "file_data" should have the same structure as 
        "network_data/optimization_data/gaslib40/gaslib40_removed_edit.".

        Args:
            file_network (Path): path to network structure 
            file_data (Path): path to data
            model (str): pressure law type. 
            ["speed_of_sound","virial_expansion","virial_expansion_mix","gerg_fit"],
            see pressure_law.py for more information
        
            
        Then the following parameters are available:
        ## Parameters
        # node structure
        node_id (list): id of each node\n
        node_x_cord (list)  x coordinate of each node\n
        node_y_cord (list) y coordinate of each node\n
        node_type (list) type of ["source","in_node","sink"]\n


        # node data
        p_list (list): list of pressure at node in Pa\n
        m_node_flow (list): list of exit/entry mass flows at each boundary node, \n
        with + for inflow and - for outflow\n
        mu_node (list): list of percentage mass of CH_4 of the OUTFLOW\n
        at each node.\n
        temperature (float): temperature in Kelvin\n

        # pipes 
        pipe_id (list): list of id's\n
        pipe_in (list): list of inflow nodes of each pipe\n
        pipe_out (list): list of outflow nodes of each pipe\n
        
        # data_on pipes
        m_list (list): mass flows on each pipe in kg/s^2, + for along pipe direction\n
        and - against\n
        mu_list (list): list of percentage mass of CH_4 \n


        pipe_length (list): length of pipe\n
        pipe_diameter (list): pipe diameters\n
        pipe_roughness (list): pipe roughness\n

        pipe_friction (list): pipe friction 
        The pipe friction is via the formula of Nikuraze via calling self._calculate_friction_parameter()! 
        Available only after the call of self._calculate_friction_parameter().\n

        # pressure_laws 
        Only after call of self._precompute_pressure()!\n
        p1_jax (callable): partial pressure "self.model" via automatic differentiation\n
        p2_jax (callable): partial pressure "self.model" via automatic differentiation\n
        
        # boundary data  
        Only after call of self.convert_boundary_conditions()!

        rho_1_0 (list[list]): list of pairs for inflow/outflow boundary value of rho_1 for each pipe\n
        rho_2_0 (list[list]): list of pairs for inflow/outflow boundary value of rho_2 for each pipe\n
        v_0 (list[list]): list of pairs for inflow/outflow boundary value of rho_2 for each pipe\n

        # solution 
        Only after call of self.solve()!\n
        u_sol (np.ndarray): last step of computed solution\n
        conv (bool): if computed solution converged\n

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
        self.u_1v_store = None
        self.u_1v_sol = None
        self.conv = False
        self.f = None
        self.tol = None
        self._read_network_architecture(file_network)
        self.file_network = file_network
        self._read_network_data(file_data)
        self.file_data = file_data


        ####################################################################
        ####################################################################
        ## MARKER START

        """ 
        Swap which nodes are inflow and outflow so we can set mass flow on the inflow nodes.
        """
        
        ####################################################################
        node_type_swap = {"source":"sink","sink":"source","in_node":"in_node"}
        self.node_type = np.array([node_type_swap[node] for node in self.node_type])
        ####################################################################

        ## MARKER END 
        ####################################################################
        ####################################################################
        self.network_name = Path(file_network).name[:-4]
        self.network_data = Path(file_data).name[:-4]
        self.model = model
        if model_name is None:
            self.model_name = model
        else:
            self.model_name = model_name
        self.p1_jax = None
        self.p2_jax = None

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
        Computes and stores the partial pressure functions. 

        """

        if self.p1_jax is None:

            if self.model == "speed_of_sound":
                p1,p2 = pressure_law_mixtures_symbolic(model=self.model,T=self.temperature)
                self.p1_jax = p1
                self.p2_jax = p2

            elif self.model == "virial_expansion":
                p1,p2 = pressure_law_mixtures_symbolic(model=self.model,T=self.temperature)
                self.p1_jax = p1
                self.p2_jax = p2
            
            else: 
                raise Exception(f"This pressure law is not implemented: {self.model}")






    def _calculate_friction_parameter(self):
        """
        Calculates and stores the pipe friction from the pipe roughness. Uses a scaled version of the Nikuraze formula
        (as discussed in Section 1.5).
        """

        self.pipe_friction = 0.65*(2*np.log10(self.pipe_diameter /self.pipe_roughness)+1.138)**(-2)



    def discrete_system_jax(self,algebraic:bool,f:float):        
        """
        Implements the box scheme discretisation of the two-velocity model as a function F.

        Args:
            algebraic (bool): constant in front of nonlinear term
            f (float): inner friction parameter

        Returns:
            F (function): corresponds to the scheme. Has arguments for solutions and boundary conditions.
        """


        self._precompute_pressure_law()
        p1 = self.p1_jax
        p2 = self.p2_jax

        def F(y,y_old,bc_m1,bc_m2,bc_p1,bc_p2):
            """

            Stores the box scheme discretisation as a function which contains arguments for the new 
            solution y and the old solution y_old, as well as the boundary values. 
            
            For each pipe (including the endpoints at the nodes), y and y_old contain values of rho_1, rho_2, v_1, and v_2 at the 
            corresponding set of discrete points. 
                
            F contains
                1. mass conservation equations for each constituent on each pipe
                2. momentum conservation equations for each constituent on each pipe
                3. coupling conditions on internal nodes:
                    - PARTIAL pressure continuity
                    - mass conservation for EACH constituent
                4. boundary data:
                    - m_i=rho_i*v_i on the INFLOW nodes 
                    - p_i(rho_1,rho_2) on the OUTFLOW nodes
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
            block_rho_1_list = []
            block_rho_2_list = []
            block_v_1_list = []
            block_v_2_list = []
            mass_cons_1_list = []
            mass_cons_2_list = []
            pressure_equality_1_list = []
            pressure_equality_2_list = []

            N_x_all_list = self.N_x_all_list
            dx_list = self.dx_list
            dt = self.dt


            for i,id in enumerate(self.pipe_id):
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
                    +dt*1/2*((self.pipe_friction[i]/(2*self.pipe_diameter[i])*rho_1_h*jnp.abs(v_1_h)*v_1_h)[1:]\
                    +f*rho_1_h[1:]*rho_2_h[1:]*(v_1_h[1:]-v_2_h[1:])\
                    +(self.pipe_friction[i]/(2*self.pipe_diameter[i])*rho_1_h*jnp.abs(v_1_h)*v_1_h)[:-1]\
                    +f*rho_1_h[:-1]*rho_2_h[:-1]*(v_1_h[:-1]-v_2_h[:-1])))*1e-5
                    
                #momentum conservation H2
                block_v_2 = (((rho_2_h[1:]*v_2_h[1:]+rho_2_h[:-1]*v_2_h[:-1])/2\
                              -(rho_2_h_old[1:]*v_2_h_old[1:]+rho_2_h_old[:-1]*v_2_h_old[:-1])/2)\
                    +dt*jnp.diff(algebraic*rho_2_h*v_2_h**2+p2(rho_1_h,rho_2_h))/dx_list[i]\
                    +dt*1/2*((self.pipe_friction[i]/(2*self.pipe_diameter[i])*rho_2_h*jnp.abs(v_2_h)*v_2_h)[1:]\
                    +f*rho_1_h[1:]*rho_2_h[1:]*(v_2_h[1:]-v_1_h[1:])\
                    +(self.pipe_friction[i]/(2*self.pipe_diameter[i])*rho_2_h*jnp.abs(v_2_h)*v_2_h)[:-1]\
                    +f*rho_1_h[:-1]*rho_2_h[:-1]*(v_2_h[:-1]-v_1_h[:-1])))*1e-5 

                block_rho_1_list.append(block_rho_1)
                block_rho_2_list.append(block_rho_2)
                block_v_1_list.append(block_v_1)
                block_v_2_list.append(block_v_2)
                
            
            #Coupling and boundary conditions on nodes
            for i,id in enumerate(self.node_id):

                a_list = self.pipe_diameter**2*jnp.pi/4

                # pipes where out going node == current node
                out_pipes_index_list = np.where(np.array(self.pipe_out) == id)[0] 
                # pipes where in node == current node
                in_pipes_index_list = np.where(np.array(self.pipe_in) == id)[0] 

                

                # mass conservation on each inner junctions or outflow
                mass_cons_1 = 0
                mass_cons_2 = 0

                #NOTE: not allowed to have pipes going into source node
                if not self.node_type[i] == "source":
                    #either sink or internal
    
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
                

                n_in_pipes = len(in_pipes_index_list)
                n_out_pipes = len(out_pipes_index_list)
                if not self.node_type[i] == "sink" or n_in_pipes+n_out_pipes>=2:
                    if self.node_type[i] == "source":
                        p_1_value_first = bc_p1[i] 
                        p_2_value_first = bc_p2[i]

                    elif n_in_pipes >= 1:
                        index_first = in_pipes_index_list[0]
                        rho_1_0_l = y[4*np.sum(N_x_all_list[:index_first])] 
                        rho_2_0_l = y[4*np.sum(N_x_all_list[:index_first])+N_x_all_list[index_first]]
                        
                        p_1_value_first = p1(rho_1_0_l,rho_2_0_l)
                        p_2_value_first = p2(rho_1_0_l,rho_2_0_l)
                        # remove first pipe entry
                        in_pipes_index_list = in_pipes_index_list[1:]
                    
                    else:
                        index_first = out_pipes_index_list[0]
                        rho_1_0_r = y[4*np.sum(N_x_all_list[:index_first])+N_x_all_list[index_first]-1]
                        rho_2_0_r = y[4*np.sum(N_x_all_list[:index_first])+2*N_x_all_list[index_first]-1]

                        p_1_value_first = p1(rho_1_0_r,rho_2_0_r)
                        p_2_value_first = p2(rho_1_0_r,rho_2_0_r)
                        # remove first pipe entry 
                        out_pipes_index_list = out_pipes_index_list[1:]
                                
                    for pipe_in_index in in_pipes_index_list:

                        rho_1_0_l = y[4*np.sum(N_x_all_list[:pipe_in_index])]
                        rho_2_0_l = y[4*np.sum(N_x_all_list[:pipe_in_index])+N_x_all_list[pipe_in_index]]

                        p_1_value = p1(rho_1_0_l,rho_2_0_l)

                        p_2_value = p2(rho_1_0_l,rho_2_0_l)


                        p_1_cond = (p_1_value_first-p_1_value)*1e-5
                        p_2_cond = (p_2_value_first-p_2_value)*1e-5
                        pressure_equality_1_list.append(jnp.array([p_1_cond]))
                        pressure_equality_2_list.append(jnp.array([p_2_cond]))

                    for pipe_out_index in out_pipes_index_list:


                        rho_1_0_r = y[4*np.sum(N_x_all_list[:pipe_out_index])+N_x_all_list[pipe_out_index]-1]
                        rho_2_0_r = y[4*np.sum(N_x_all_list[:pipe_out_index])+2*N_x_all_list[pipe_out_index]-1]


                        p_1_value = p1(rho_1_0_r,rho_2_0_r)

                        p_2_value = p2(rho_1_0_r,rho_2_0_r)
                        p_1_cond = (p_1_value_first-p_1_value)*1e-5
                        p_2_cond = (p_2_value_first-p_2_value)*1e-5
                        pressure_equality_1_list.append(jnp.array([p_1_cond]))
                        pressure_equality_2_list.append(jnp.array([p_2_cond]))
            
            result = jnp.concatenate(
                [ item for item in block_rho_1_list ]
               + [ item for item in block_rho_2_list ]
               + [ item for item in block_v_1_list ]
               + [ item for item in block_v_2_list ]
               + [ item for item in mass_cons_1_list ]
               + [ item for item in mass_cons_2_list ]
               + [ item for item in  pressure_equality_1_list]
               + [ item for item in  pressure_equality_2_list])
 
            return result
        
        
        
        return F



    def solve(self,algebraic:float,tol:float,scenario:str,f:float):
        """
        This routine implements Newton's method to compute numerical solutions of the two-velocity model
        discretised using the box scheme. 

        Args:
            algebraic (float): algebraic value in front of nonlinearity
            tol (float): Newton tolerance
            scenario (str): choice of initial and boundary condition (passed to self.get_initial_and_bc_data)
            f (float): inner friction parameter
        """

        self._calculate_friction_parameter()
        self._precompute_pressure_law()
        
        
        initial_vec, bc_m1,bc_m2, bc_p1,bc_p2 = self.get_initial_and_bc_data(scenario = scenario)

        F = self.discrete_system_jax(algebraic=algebraic,f=f)

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
            F_step = lambda x: F(x,u_time_t,bc_m1[i,:],bc_m2[i,:],bc_p1[i,:],bc_p2[i,:])
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

            if (scenario != "time_dep" and j==0 and i>=self.N_time/2):
                long_time_conv = True
                break

            if rhs_norm < tol:
                self.conv.append(True)
            else:
                self.conv.append(False)
            
        if all(self.conv):
            self.conv = True
        else: 
            self.conv = False

        if scenario =="gaslib11_stationary_1v" and not long_time_conv: 
            raise Exception("The solution did not converge to a long time limit!")


    def get_initial_and_bc_data(self,scenario:str):
        """
        Computes and stores initial and boundary data.

        Args:
            scenario (str): Choice of initial and boundary data scenario. Choices:
                - gaslib11_stationary_1v (for stationary solutions)
                - time_dep (for instationary aka "time dependent" solutions)

        Raises:
            Exception: If scenario not implemented.

        Returns:
            (initial_vec,bc_m,bc_mu,bc_p1,bc_p1) (tuple): initial vector and arrays of boundary conditions 
        """
        
        N_x_all_list = self.N_x_all_list
        N_time = self.N_time
        T = self.T


        bc_m1 = jnp.zeros((N_time,len(self.node_id)))
        bc_p1 = jnp.zeros((N_time,len(self.node_id)))
        bc_m2 = jnp.zeros((N_time,len(self.node_id)))
        bc_p2 = jnp.zeros((N_time,len(self.node_id)))


        if scenario == "gaslib11_stationary_1v":
            ### load 1v model for partial pressures
            initial_vec = jnp.zeros(4*jnp.sum(N_x_all_list))

            a_list = self.pipe_diameter**2*np.pi/4
            m1 = 41.16666666*2/3
            m2 = 41.16666666/3
            p = 70*1e5
            rho_1_source, rho_2_source,v_source = get_bc_data_1v(m1=m1,m2=m2,p=p,a=a_list[0],T=self.temperature)


            #const values
            for i,id in enumerate(self.pipe_id):
                

                rho_1_h =  rho_1_source*jnp.ones(N_x_all_list[i])
                rho_2_h =  rho_2_source*jnp.ones(N_x_all_list[i])
       
                v_1_h = v_source*jnp.ones((N_x_all_list[i]))
                v_2_h = v_source*jnp.ones((N_x_all_list[i]))

            
                initial_vec = initial_vec.at[4*np.sum(N_x_all_list[:i]):4*np.sum(N_x_all_list[:i])+N_x_all_list[i]].set(rho_1_h )
                initial_vec = initial_vec.at[4*np.sum(N_x_all_list[:i])+N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]].set(rho_2_h)
                initial_vec = initial_vec.at[4*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]].set(v_1_h)
                initial_vec = initial_vec.at[4*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+4*N_x_all_list[i]].set(v_2_h)

            # boundary conditons in m_form
            for i,id in enumerate(self.node_id):
                
                N_time_half = int((N_time)/2) 
                if self.node_type[i] == "sink":
                    
                    m1_goal = self.m_node_flow[i]*self.mu_node[i] 
                    m2_goal = self.m_node_flow[i]*(1-self.mu_node[i]) 
                    

                    # pipes where out going node == current node
                    out_pipes_index_list = np.where(np.array(self.pipe_out) == id)[0] 
                    # pipes where in node == current node
                    in_pipes_index_list = np.where(np.array(self.pipe_in) == id)[0] 
                    n_in_pipes = len(in_pipes_index_list)
                    n_out_pipes = len(out_pipes_index_list)

                    m1_initial = -a_list[0]*(rho_1_source)*v_source*(n_out_pipes-n_in_pipes)
                    m2_initial = -a_list[0]*(rho_2_source)*v_source*(n_out_pipes-n_in_pipes)
    
                    bc_m1_int = np.linspace(m1_initial,m1_goal,N_time_half)
                    bc_m2_int = np.linspace(m2_initial,m2_goal,N_time_half)
                    bc_m1_const = m1_goal*np.ones(N_time-N_time_half)
                    bc_m2_const = m2_goal*np.ones(N_time-N_time_half)

                    bc_m1 = bc_m1.at[:,i].set(np.hstack((bc_m1_int,bc_m1_const)))
                    bc_m2 = bc_m2.at[:,i].set(np.hstack((bc_m2_int,bc_m2_const)))


                if self.node_type[i] == "in_node":

                    m1_goal = 0
                    m2_goal = 0
                    

                    # pipes where out going node == current node
                    out_pipes_index_list = np.where(np.array(self.pipe_out) == id)[0] 
                    # pipes where in node == current node
                    in_pipes_index_list = np.where(np.array(self.pipe_in) == id)[0] 
                    n_in_pipes = len(in_pipes_index_list)
                    n_out_pipes = len(out_pipes_index_list)

                    m1_initial = -a_list[0]*(rho_1_source)*v_source*(n_out_pipes-n_in_pipes)
                    m2_initial = -a_list[0]*(rho_2_source)*v_source*(n_out_pipes-n_in_pipes)


                    bc_m1_int = np.linspace(m1_initial,m1_goal,N_time_half)
                    bc_m2_int = np.linspace(m2_initial,m2_goal,N_time_half)

                    bc_m1_const = m1_goal*np.ones(N_time-N_time_half)
                    bc_m2_const = m2_goal*np.ones(N_time-N_time_half)

                    bc_m1 = bc_m1.at[:,i].set(np.hstack((bc_m1_int,bc_m1_const)))
                    bc_m2 = bc_m2.at[:,i].set(np.hstack((bc_m2_int,bc_m2_const)))


                if self.node_type[i] in ["source"]:

                    p1_initial = self.p1_jax(rho_1_source,rho_2_source)
                    p2_initial = self.p2_jax(rho_1_source,rho_2_source)
                    



                    ####################################################################
                    ####################################################################
                    ## MARKER START

                    mu_node_out = self.mu_node[i]
                    p_node = self.p_list[i]
                    # only temporary mass flow (rho_1_bc and rho_2_bc independent of mass flow)
                    m_tmp = 10
                    m1_tmp = mu_node_out*m_tmp
                    m2_tmp = (1-mu_node_out)*m_tmp
                    rho_1_bc,rho_2_bc,_ = get_bc_data_1v(m1=m1_tmp,m2=m2_tmp,p=p_node,a=a_list[0],T=self.temperature)
                    p1_goal = self.p1_jax(rho_1_bc,rho_2_bc)
                    p2_goal = self.p2_jax(rho_1_bc,rho_2_bc)


                    ## MARKER END 
                    ####################################################################
                    ####################################################################

                    bc_p1_int = np.linspace(p1_initial,p1_goal,N_time_half)
                    bc_p2_int = np.linspace(p2_initial,p2_goal,N_time_half)


                    bc_p1_const = p1_goal*np.ones(N_time-N_time_half)
                    bc_p2_const = p2_goal*np.ones(N_time-N_time_half)

                    bc_p1 = bc_p1.at[:,i].set(np.hstack((bc_p1_int,bc_p1_const)))
                    bc_p2 = bc_p2.at[:,i].set(np.hstack((bc_p2_int,bc_p2_const)))

        elif scenario == "time_dep":
            a_list = self.pipe_diameter**2*np.pi/4
            t_mid = T/2
            t_fourth = T/4
            timesteplength = round(T/(N_time-1))       
            timerange = [timesteplength*j for j in range(N_time)]
            initial_vec = jnp.zeros(4*jnp.sum(N_x_all_list))

            a_list = self.pipe_diameter**2*np.pi/4
            m1 = 41.16666666*2/3
            m2 = 41.16666666/3
            p = 70*1e5
            rho_1_source = 10
            rho_2_source = 5
            v_source = 0

            for i,id in enumerate(self.pipe_id):

                rho_1_h =  rho_1_source*jnp.ones(N_x_all_list[i])
                rho_2_h =  rho_2_source*jnp.ones(N_x_all_list[i])
       
                v_1_h = v_source*jnp.ones((N_x_all_list[i]))
                v_2_h = v_source*jnp.ones((N_x_all_list[i]))


                initial_vec = initial_vec.at[4*np.sum(N_x_all_list[:i])+0*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+1*N_x_all_list[i]].set(rho_1_h )
                initial_vec = initial_vec.at[4*np.sum(N_x_all_list[:i])+1*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]].set(rho_2_h)
                initial_vec = initial_vec.at[4*np.sum(N_x_all_list[:i])+2*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]].set(v_1_h)
                initial_vec = initial_vec.at[4*np.sum(N_x_all_list[:i])+3*N_x_all_list[i]:4*np.sum(N_x_all_list[:i])+4*N_x_all_list[i]].set(v_2_h)


            # boundary conditons in m_form
            for i,id in enumerate(self.node_id):      
                if not self.node_type[i] == "source":
                    m1_goal = 0 
                    m2_goal = 0
                    
                    def bc_m1_values(t):

                        if t <= t_fourth:
                            m1_value = (4*t/T)*10
                        elif t_fourth < t <= t_mid:
                            m1_value = 1*10 
                        elif t_mid < t <= 3*t_fourth:
                            m1_value = ((-4*t/T) + 3)*10 
                        else:
                            m1_value = 0

                        return m1_value

                    bc_m1_vectorised = np.vectorize(bc_m1_values)
                    bc_m1 = bc_m1.at[:,i].set(bc_m1_vectorised(timerange))

                    
                    def bc_m2_values(t):
                        
                        if t <= t_fourth:
                            m2_value = m2_goal
                        elif t_fourth < t <= t_mid:
                            m2_value = (4*t/T -1)*5 
                        elif t_mid < t <= 3*t_fourth:
                            m2_value = ((-4*t/T) + 3)*5 
                        else:
                            m2_value = 0

                        return m2_value

                    bc_m2_vectorised = np.vectorize(bc_m2_values)
                    bc_m2 = bc_m2.at[:,i].set(bc_m2_vectorised(timerange))
                    

                else:
                    
                    p1_goal = self.p1_jax(rho_1_source,rho_2_source) 
                    p2_goal = self.p2_jax(rho_1_source,rho_2_source)

                    def bc_p1_values(t):
                        
                        if t <= t_mid:
                            p1_value = (1.02-0.08*np.abs(t-t_fourth)/T)*p1_goal
                        else: 
                            p1_value = p1_goal
                        return p1_value

                    bc_p1_vectorised = np.vectorize(bc_p1_values)
                    bc_p1 = bc_p1.at[:,i].set(bc_p1_vectorised(timerange))
                    def bc_p2_values(t):
                        
                        if t <= t_mid:
                            p2_value = (1.04-0.16*np.abs(t-t_fourth)/T)*p2_goal
                        else: 
                            p2_value = p2_goal
                        
                
                        p2_value = p2_goal
                        return p2_value

                    bc_p2_vectorised = np.vectorize(bc_p2_values)
                    bc_p2 = bc_p2.at[:,i].set(bc_p2_vectorised(timerange))
                    
        else:
            raise Exception("Initial Data NOT implemented!")


        return initial_vec,bc_m1,bc_m2,bc_p1,bc_p2



        
        
    def save_solution_network(self,algebraic:float,scenario:str,f:float):
        """
        Saves the solutions inside the folder "save_solution/network/network_2v_timedep" for later use/reuse.       

        Args:
            algebraic (float): algebraic value in front of nonlinearity
            scenario (str): choice of initial and boundary condition (passed to self.get_initial_and_bc_data)
            f (float): inner friction parameter
        """
        if scenario == "time_dep":
            file_path = Path(f"save_solution/solutions/network_2v_f_{f}_timedep/massflowinflow/2v_{self.network_data}_dx_{self.candidate_dx}_dt_{self.dt}_T_{self.T}_SCENARIO_{scenario}_pressure_{self.model}_algebraic_{algebraic}.pkl")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as file:
                pickle.dump(self.u_store, file)
        else:
            file_path = Path(f"save_solution/solutions/network_2v_f_{f}_timedep/massflowinflow/2v_{self.network_data}_dx_{self.candidate_dx}_SCENARIO_{scenario}_pressure_{self.model}_algebraic_{algebraic}.pkl")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as file:
                pickle.dump(self.u_sol, file)
        
    
        
    def load_solution_network(self,algebraic:float,scenario:str,f:float):
        """
        Loads a solution for further computations/plots. 

        Args: as in save_solution_network
        """
        
        if scenario == "time_dep":
            file_path = Path(f"save_solution/solutions/network_2v_f_{f}_timedep/massflowinflow/2v_{self.network_data}_dx_{self.candidate_dx}_dt_{self.dt}_T_{self.T}_SCENARIO_{scenario}_pressure_{self.model}_algebraic_{algebraic}.pkl")
            with open(file_path, "rb") as file:
                self.u_store = pickle.load(file)
        else:
            file_path = Path(f"save_solution/solutions/network_2v_f_{f}_timedep/massflowinflow/2v_{self.network_data}_dx_{self.candidate_dx}_SCENARIO_{scenario}_pressure_{self.model}_algebraic_{algebraic}.pkl")
            with open(file_path, "rb") as file:
                self.u_sol = pickle.load(file)

                
    def save_solution_1v_network(self,algebraic:float,scenario:str):
        """
        If computing a 1v solution, saves it in the correct location

        Args:
            scenario (str): choice of initial and boundary condition (passed to self.get_initial_and_bc_data)
            algebraic (float): value in front non linear term
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
                

    def plot_all_stationary(self,algebraic:float,f:float,plot_pipe:bool=False,scenario:str=None,show_labels:bool=False,
                            label_node_list:list[str]=None,offset_labels:float=None,save_legend:bool=False):
        """
        Plots all required solution quantities

        Args:
            algebraic (float): value in front of nonlinearity (used for name when saving plots)
            f (float): inner friction parameter (used for name when saving plots)
            plot_pipe (bool): whether or not to plot results for each individual pipe.  
            scenario (str): choice of initial and boundary data (used for name when saving plots)
            show_labels (bool): whether node labels are shown
            label_node_list (bool): whether or not to add names of every node to plot. 
            offset_labels (float): amount by which to offset label from node

        """


        self.plot_sol_stationary(prop="rho_1",prop_name="$\\rho_1$",unit="$\\frac{kg}{m^3}$",algebraic=algebraic,
                                 f=f,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="rho_2",prop_name="$\\rho_2$",unit="$\\frac{kg}{m^3}$",algebraic=algebraic,
                                 f=f,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="v_1",prop_name="$v_1$",unit="$\\frac{m}{s}$",algebraic=algebraic,
                                 arrows=True,f=f,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="v_2",prop_name="$v_2$",unit="$\\frac{m}{s}$",algebraic=algebraic,
                                 arrows=True,f=f,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="pressure",prop_name="$p$",unit="$bar$",algebraic=algebraic,
                                 f=f,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="p_1",prop_name="$p_1$",unit="$bar$",algebraic=algebraic,
                                 f=f,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="p_2",prop_name="$p_2$",unit="$bar$",algebraic=algebraic,
                                 f=f,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="m_1",prop_name="$m_1$",unit="$\\frac{kg}{m^3s}$",algebraic=algebraic,
                                 arrows=True,f=f,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,offset_labels=offset_labels,save_legend=save_legend)
        self.plot_sol_stationary(prop="m_2",prop_name="$m_2$",unit="$\\frac{kg}{m^3s}$",algebraic=algebraic,
                                 arrows=True,f=f,plot_pipe=plot_pipe,scenario=scenario,show_labels=show_labels,label_node_list=label_node_list,save_legend=save_legend)
        # self.plot_sol_stationary(prop="h_1",prop_name="$h_1$",unit="bar",algebraic=algebraic,save_name_fig=save_name_fig,f=f,plot_pipe=plot_pipe,scenario=scenario)
        # self.plot_sol_stationary(prop="h_2",prop_name="$h_2$",unit="bar",algebraic=algebraic,save_name_fig=save_name_fig,f=f,plot_pipe=plot_pipe,scenario=scenario)
        # self.plot_sol_stationary(prop="h",prop_name="$h$",unit="bar",algebraic=algebraic,save_name_fig=save_name_fig,f=f,plot_pipe=plot_pipe,scenario=scenario)

    
    def plot_sol_stationary(self,prop:str,prop_name:str,unit:str,algebraic:float,f:str,arrows:bool=False,plot_pipe:bool=False,scenario:str=None,
                            show_labels=False,label_node_list:list[str]=None,offset_labels:float=None,save_legend:bool=False):
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
    
            
        fig, ax = plt.subplots(constrained_layout=True)



        self._precompute_pressure_law()

        p1 = self.p1_jax
        p2 = self.p2_jax

        a_list = self.pipe_diameter**2*np.pi/4

        # create colormap
        prop_values = []
        for i,in_id in enumerate(self.pipe_in):
            
            rho_1 = self.u_sol[4*np.sum(self.N_x_all_list[:i]):4*np.sum(self.N_x_all_list[:i])+self.N_x_all_list[i]]
            rho_2 = self.u_sol[4*np.sum(self.N_x_all_list[:i])+self.N_x_all_list[i]:4*np.sum(self.N_x_all_list[:i])+2*self.N_x_all_list[i]]
            v_1 = self.u_sol[4*np.sum(self.N_x_all_list[:i])+2*self.N_x_all_list[i]:4*np.sum(self.N_x_all_list[:i])+3*self.N_x_all_list[i]]
            v_2 = self.u_sol[4*np.sum(self.N_x_all_list[:i])+3*self.N_x_all_list[i]:4*np.sum(self.N_x_all_list[:i])+4*self.N_x_all_list[i]]
            

            if plot_pipe or self.network_name =="one_pipe":
                fig_pipe,ax_pipe = plt.subplots(constrained_layout=True)
            if prop == "rho_1": 
                prop_values.append(rho_1)

            elif prop == "rho_2": 
                prop_values.append(rho_2)

            elif prop == "v_1": 
                prop_values.append(v_1)
            
            elif prop == "v_2": 
                prop_values.append(v_2)
            
            elif prop == "pressure":
                prop_values.append((p1(rho_1,rho_2)+p2(rho_1,rho_2))*1e-5)

            elif prop == "p_1":
                prop_values.append(p1(rho_1,rho_2)*1e-5)

            elif prop == "p_2":

                prop_values.append(p2(rho_1,rho_2)*1e-5)

            elif prop == "m_1":
                prop_values.append(a_list[i]*rho_1*v_1)

            elif prop == "m_2":
                prop_values.append(a_list[i]*rho_2*v_2)


            elif prop == "h_1":
                prop_values.append((algebraic*rho_1*v_1+p1(rho_1,rho_2))*1e-5)
            

            elif prop == "h_2":
                prop_values.append((algebraic*rho_2*v_2**2+p2(rho_1,rho_2))*1e-5)

            
            elif prop == "h":
                prop_values.append((algebraic*rho_1*v_1**2+p1(rho_1,rho_2)+\
                                    algebraic*rho_2*v_2**2+p2(rho_1,rho_2))*1e-5)


            if self.network_name != "one_pipe" and plot_pipe == True:

                x_values = np.linspace(0,self.pipe_length[i],len(prop_values[i]))
            
                fig_pipe, ax_pipe = plt.subplots(constrained_layout=True)
                ax_pipe.plot(x_values,prop_values[i])
                ax_pipe.set_xlabel("Distance [m]")
                ax_pipe.set_ylabel(f"{prop_name}")

                file_path = Path(f"graphics/networks/network_2v_timedep/massflowinflow/{self.network_data}/2v_f_{f}_dx_{self.candidate_dx}_dt_{self.dt}_T_{self.T}_SCENARIO_{scenario}_pressure_{self.model}_algebraic_{algebraic}_PROP_{prop}_pipe_{i}.png")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                fig_pipe.savefig(file_path)
                plt.close()
                
        if self.network_name == "one_pipe":
            x_values = np.linspace(0,self.pipe_length[0],len(prop_values[0]))
            ax_pipe.plot(x_values,prop_values[i])
            ax_pipe.set_xlabel("Distance [m]")
            ax_pipe.set_ylabel(f"{prop_name}")
            file_path = Path(f"graphics/networks/network_2v_timedep/massflowinflow/{self.network_data}/_2v_f_{f}_dx_{self.candidate_dx}_dt_{self.dt}_T_{self.T}_SCENARIO_{scenario}_pressure_{self.model}_algebraic_{algebraic}_PROP_{prop}.png")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            fig_pipe.savefig(file_path)
            plt.close()
        else:
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
                        marker="s", color="b",s=80,zorder=3)
    
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
    
    
    

            save_name = f"graphics/networks/network_2v_timedep/massflowinflow/{self.network_data}/2v_f_{f}_dx_{self.candidate_dx}_dt_{self.dt}_T_{self.T}_SCENARIO_{scenario}_pressure_{self.model}_algebraic_{algebraic}_PROP_{prop}"
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
#END OF DEFINITION OF Network_2v_time class
############################################  

def compute_one_pipe():
    """
    Computes 2v stationary solutions on a pipe.
    """

    file_network = Path("network_data" ,"optimization_data", "network_files","one_pipe.net")
    file_data = Path("network_data", "optimization_data", "solution_files","one_pipe.lsf")


    model ="speed_of_sound"
    algebraic = 1.0

    stationary_or_instationary = 0
    scenarios = ["gaslib11_stationary_1v", "time_dep"] 
    scenario = scenarios[stationary_or_instationary]
    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[stationary_or_instationary]
    dt = int(T/60)
    
    
    #Fixed computational parameters
    algebraic = 1.0 
    tol = 1e-7

    
    #Parameters which can change
    candidate_dx = 200
    tol = 1e-7
    f_list = [0.01,0.1,1,10,100]
    for f in f_list:
        print(f"inner friction = {f}")
        algebraic = 1.0
        network = Network_2v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=candidate_dx,dt=dt,T=T)
        try:
            network.load_solution_network(algebraic=algebraic, scenario=scenario, f=f)
        except FileNotFoundError: 
            network.solve(algebraic=algebraic, tol=tol, scenario=scenario, f=f)
            network.save_solution_network(algebraic=algebraic, scenario=scenario, f=f)


def compute_3mix_scenarions():
    """
    Computes 2v stationary solutions on a 3-star/Y-shape network. 

    """
    model = "speed_of_sound"

    stationary_or_instationary = 0
    scenarios = ["gaslib11_stationary_1v", "time_dep"] 
    scenario = scenarios[stationary_or_instationary]
    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[stationary_or_instationary]
    dt = int(T/60)
    
    
    #Fixed computational parameters
    algebraic = 1.0 
    tol = 1e-7

    
    #Parameters which can change
    candidate_dx = 200
    f_list = [1,5,10,20,50]

    for f in f_list:


        file_network = Path("network_data" ,"optimization_data","network_files", "3mixT.net")
        file_data = Path("network_data", "optimization_data","3mix_scenarios", f"3mix_temp_0.lsf")


        network = Network_2v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=candidate_dx,dt=dt,T=T)
        try: 
            network.load_solution_network(algebraic=algebraic,scenario=scenario,f=f)
        except (FileNotFoundError, IOError): 
            network.solve(algebraic=algebraic, tol=tol, scenario=scenario, f=f)
            network.save_solution_network(algebraic=algebraic, scenario=scenario, f=f)
            network.plot_all_stationary(algebraic,f,plot_pipe=False,scenario=scenario)

def compute_gaslib40_scenarions():
    """
    Computes 2v stationary solutions for GasLib40-3 and various friction parameters f 
    """

    model = "speed_of_sound"

    stationary_or_instationary = 0
    
    scenarios = ["gaslib11_stationary_1v", "time_dep"] 
    scenario = scenarios[stationary_or_instationary]
    
    timehorizons = [60*60*24*150*15, 60*60*12]
    T = timehorizons[stationary_or_instationary]
    dt = int(T/60)
    
    
    #Fixed computational parameters
    algebraic = 1.0 
    tol = 1e-7

    
    #Parameters which can change
    candidate_dx = 500
    f_list = [1,5,10,20,30]

    for f in f_list:


        file_network = Path("network_data" ,"optimization_data","network_files", "gaslib40_removed_edit.net")
        file_data = Path("network_data", "optimization_data", "solution_files","gaslib40_removed_edit.lsf")


        network = Network_2v_time(file_network=file_network,file_data=file_data,model=model,candidate_dx=candidate_dx,dt=dt,T=T)
        try: 
            network.load_solution_network(algebraic=algebraic,scenario=scenario,f=f)
        except (FileNotFoundError, IOError): 
            network.solve(algebraic=algebraic, tol=tol, scenario=scenario, f=f)
            network.save_solution_network(algebraic=algebraic, scenario=scenario, f=f)
            network.plot_all_stationary(algebraic,f,plot_pipe=False,scenario=scenario)



