from pathlib import Path
import xmltodict
import numpy as np


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
        "mu":"mixingratioMass",
        "posFlow" : "posFlowBinvar",
        "negFlow" : "negFlowBinvar"
    }

    return key, sub_key, sub_sub_key


def read_and_write(file_net:Path, file_data:Path,
                   save_file_data:Path, save_file_network:Path):
    
    node_type_list = []


    with open(file_net, "r") as f:
        xml_network = f.read()

    # Convert XML to a dictionary
    xml_network_dict = xmltodict.parse(xml_network)

    framework = "framework"
    network_data = xml_network_dict["network"]


    # read nodes
    key_node,key_node_category,key_node_prop = get_key_nodes()

    nodes = network_data[f"{framework}:{key_node}"]

    for name_node_type,node_type in key_node_category.items():

        node_list = nodes.get(node_type)
        if node_list is not None:
            node_list = [node_list] if not isinstance(node_list,list) else node_list

            node_type_list += [name_node_type for i in range(len(node_list))]

    node_type_list = np.array(node_type_list)


    # read pipes
    key_pipe,key_pipe_category,key_pipe_properties = get_key_pipes()
    pipe_list = network_data[f"{framework}:{key_pipe}"][key_pipe_category["pipe"]]

    pipe_list = pipe_list if isinstance(pipe_list,list) else [pipe_list] 



        

    with open(file_data, "r") as f:
        xml_data = f.read()

    # Convert xml_data to a dictionary
    xml_data_dict = xmltodict.parse(xml_data)

    data = xml_data_dict["solution"]

    key_data_node,key_data_node_category,key_data_node_prop = get_key_nodes_data()
 
                         
     # read pipes
    key_data_pipe,key_data_pipe_category,key_data_pipe_properties = get_key_pipes_data()
    pipe_list_data = data[f"{key_data_pipe}"][f"{key_data_pipe_category}s"][key_data_pipe_category]
    pipe_list_data = pipe_list_data if isinstance(pipe_list_data,list) else [pipe_list_data] 

    ### change direction of flow 
    m_dict_list = [pipe_list_data[i][key_data_pipe_properties["m"]] 
                    for i in range(len(pipe_list_data))]
    for i,m in enumerate(m_dict_list):
        node_data_in = pipe_list_data[i][key_pipe_properties["pipe_in"]] 
        node_data_out = pipe_list_data[i][key_pipe_properties["pipe_out"]]

        node_in = pipe_list[i][key_pipe_properties["pipe_in"]] 
        node_out = pipe_list[i][key_pipe_properties["pipe_out"]]

        if int(pipe_list_data[i][key_data_pipe_properties["posFlow"]]["@value"]) == 0:
            pipe_list_data[i][key_data_pipe_properties["m"]]["@value"] = str(-float(m["@value"]))
            pipe_list_data[i][key_data_pipe_properties["posFlow"]]["@value"] = "1"
            pipe_list_data[i][key_data_pipe_properties["negFlow"]]["@value"] = "0"

            # change node_datas
            pipe_list_data[i][key_pipe_properties["pipe_in"]]  = node_data_out
            pipe_list_data[i][key_pipe_properties["pipe_out"]]  = node_data_in

            # change in network file
            pipe_list[i][key_pipe_properties["pipe_in"]]  = node_out
            pipe_list[i][key_pipe_properties["pipe_out"]]  = node_in

            ## change  id 
            #pipe_list_data[i][key_pipe_properties["id"]]  = f"pipe_{i}_" +node_data_out +"_"+ node_data_in
            ## change  id in network
            #pipe_list[i][key_pipe_properties["id"]]  = f"pipe_{i}_" +node_out +"_"+ node_in

        #else:
        #    # change  id 
        #    pipe_list_data[i][key_pipe_properties["id"]]  = f"pipe_{i}_" +node_data_in +"_"+ node_data_out
        #    # change  id in network
        #    pipe_list[i][key_pipe_properties["id"]]  = f"pipe_{i}_" +node_in +"_"+ node_out

    # update data
    data[f"{key_data_pipe}"][f"{key_data_pipe_category}s"][key_data_pipe_category] = pipe_list_data
    xml_data_dict["solution"] = data

    network_data[f"{framework}:{key_pipe}"][key_pipe_category["pipe"]] = pipe_list 
    xml_network_dict["network"] = network_data


    ### remove compressors

    # read pipes
    key_pipe,key_pipe_category,key_pipe_properties = get_key_pipes()
    pipe_list = network_data[f"{framework}:{key_pipe}"][key_pipe_category["pipe"]]

    pipe_list = pipe_list if isinstance(pipe_list,list) else [pipe_list] 
    comp_list = network_data[f"{framework}:{key_pipe}"].get(key_pipe_category["comp"],[])


    key_data_pipe,key_data_pipe_category,key_data_pipe_properties = get_key_pipes_data()
    pipe_list_data = data[f"{key_data_pipe}"][f"{key_data_pipe_category}s"][key_data_pipe_category]
    pipe_list_data = pipe_list_data if isinstance(pipe_list_data,list) else [pipe_list_data] 


    comp_in_list = [comp_list[i][key_pipe_properties["comp_in"]] 
                    for i in range(len(comp_list))]
    comp_out_list = [comp_list[i][key_pipe_properties["comp_out"]] 
                        for i in range(len(comp_list))]


    nodes = data[f"{key_data_node}"]




    # read nodes
    key_node,key_node_category,key_node_prop = get_key_nodes()

    nodes = network_data[f"{framework}:{key_node}"]
    node_type_list = []
    node_id = []

    for name_node_type,node_type in key_node_category.items():

        node_list = nodes.get(node_type)
        if node_list is not None:
            node_list = [node_list] if not isinstance(node_list,list) else node_list

            node_id += [node_list[i][key_node_prop["id"]] for i in range(len(node_list))] 
            node_type_list += [name_node_type for i in range(len(node_list))]

    node_type_list = np.array(node_type_list)

    removed_node_id_list = []
    pipe_list_data = data[f"{key_data_pipe}"][f"{key_data_pipe_category}s"][key_data_pipe_category]
    for i,comp_in in enumerate(comp_in_list):
        comp_out = comp_out_list[i]

        if node_type_list[node_id.index(comp_in)] in ["source","sink"]:
            keep_node_id = comp_in
            remove_node_id = comp_out

        else: 
            keep_node_id = comp_out
            remove_node_id = comp_in
        
        
        for j,pipe in enumerate(pipe_list):
            remove_node_flag = False
            pipe_data = pipe_list_data[j]
            if pipe[key_pipe_properties["pipe_in"]] == remove_node_id:
                pipe[key_pipe_properties["pipe_in"]] = keep_node_id
                pipe_data[key_pipe_properties["pipe_in"]] = keep_node_id
                remove_node_flag= True


            if pipe[key_pipe_properties["pipe_out"]] == remove_node_id:
                pipe[key_pipe_properties["pipe_out"]] = keep_node_id
                pipe_data[key_pipe_properties["pipe_out"]] = keep_node_id
                remove_node_flag= True

            # update pipe id
            if remove_node_flag:
                removed_node_id_list.append(remove_node_id)
                pipe_list_data[j] = pipe_data

                # change in network file
                pipe_list[j]  = pipe

                ## change  id 
                #pipe_list_data[j][key_pipe_properties["id"]]  = f"pipe_{j}_" +node_data_out +"_"+ node_data_in
                ## change  id in network
                #pipe_list[j][key_pipe_properties["id"]]  = f"pipe_{j}_" +node_out +"_"+ node_in

    # remove compressors
    network_data[f"{framework}:{key_pipe}"].pop(key_pipe_category["comp"])
    data[f"{key_data_pipe}"].pop(f"{key_pipe_category["comp"]}s")


    # update data
    data[f"{key_data_pipe}"][f"{key_data_pipe_category}s"][key_data_pipe_category] = pipe_list_data
    xml_data_dict["solution"] = data

    network_data[f"{framework}:{key_pipe}"][key_pipe_category["pipe"]] = pipe_list 
    xml_network_dict["network"] = network_data



    # remove nodes
    nodes = network_data[f"{framework}:{key_node}"]
    data = xml_data_dict["solution"]
    nodes_solution = data[key_node]

    node_type_list = []

    for name_node_type,node_type in key_node_category.items():

        node_list = nodes.get(node_type)
        node_list_solution = nodes_solution.get(f"{node_type}s").get(node_type)

        if node_list is not None:

            node_list_update = []
            node_list_solution_update = []
            for s,node in enumerate(node_list):
                if not (node_list[s]["@id"] in removed_node_id_list):
                    node_list_update.append(node)
                    node_list_solution_update.append(node_list_solution[s])

            nodes[node_type] = node_list_update 
            nodes_solution[f"{node_type}s"][node_type] = node_list_solution_update 


    network_data[f"{framework}:{key_node}"] = nodes
    xml_network_dict["network"] = network_data

    data[key_node] = nodes_solution
    xml_data_dict["solution"] = data


    ### save data
    xml_data_transform = xmltodict.unparse(xml_data_dict,pretty=True)
    with open(save_file_network,"w+") as f:
        f.write(xml_data_transform)


    # data on network
    xml_data_transform = xmltodict.unparse(xml_network_dict,pretty=True)
    with open(save_file_data,"w+") as f:
        f.write(xml_data_transform)





if __name__ ==  "__main__":
    
    #file_network = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40.net")
    #file_data = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40.lsf")

    #save_file_data = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_edit.net")
    #save_file_network = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_edit.lsf")



    #file_network = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40.net")
    #file_data = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_high_mix.lsf")

    #save_file_data = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_high_mix_edit.net")
    #save_file_network = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_high_mix_edit.lsf")

    file_network = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed.net")
    file_data = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed.lsf")

    save_file_network = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.lsf")
    save_file_data = Path("network_data" ,"optimization_data", "gaslib40", "gaslib40_removed_edit.net")

    read_and_write(file_net=file_network,file_data=file_data,
                   save_file_data=save_file_data,save_file_network=save_file_network)