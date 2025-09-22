# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 16:00:01 2024

@author: rayno
"""

import networkx as nx
from collections import Counter
from itertools import combinations
#from openpyxl import load_workbook 
from write2excel import excel_write_line

# ——————————————————————————————————————

def cal_net_statistics( net_id, d_time, net):

    """
    lty创建

    计算网络统计特征，结果输出为
    ['Net_ID', 'D_Times', 'Net_size', 'ave_degree', 'ave_diameter', 'ave_shortest_path', 'ave_clustering' ]

    """ 
    
    net_stat = list()
    
    net_size = nx.number_of_nodes(net)
    ave_degree = round(nx.number_of_edges(net) * 2 / net_size, 4)
    
    # 网络连通时才计算直径和平均最短路径长度
    if nx.is_connected(net):
        ave_diameter = round(nx.diameter(net), 4)
        ave_shortest_path = round(nx.average_shortest_path_length(net), 4)
    else:
        ave_diameter = -1
        ave_shortest_path = -1
    
    ave_clustering = round(nx.average_clustering(net), 4)
    
    # 计算度分布
    degree_sequence = list(dict(net.degree()).values())
    degree_counts = Counter(degree_sequence)
    degree_stat = list()
    i = 0
    # 前述度分布中度可能不连续，缺少的度，补充对应节点数为0
    for degree, count in sorted(degree_counts.items()):
        while i<degree:
            degree_stat.append(0)
            i += 1
        degree_stat.append(count)
        i += 1
        #print(str(degree)+'\t'+str(count)+'\n') 
    
    net_stat = [net_id, d_time, net_size, ave_degree, ave_diameter, ave_shortest_path, ave_clustering] + degree_stat
    
    return net_stat

# ——————————————————————————————————————
# 按照netConfigs中各子网的配置顺序计算各子网拓扑特征，最后计算整体网的拓扑特征
# 

def cal_HE_nets_statistics( netConfigs, nets, G_inter, net_loop, r, pro_id, D_time ):

    """
    lty创建

    计算异质网络统计特征，结果输出为
    ['Net_ID', 'r', 'pro_ID', 'D_Times', 'Net_type', 'Net_Seq', 'Net_size', 'ave_degree', 'ave_diameter', 'ave_shortest_path', 'ave_clustering' ]

    """ 
    
    net_stat = list()
    
    # 计算各子网和整体网统计特征
    for jj in range(len(nets)+1):
        
        if jj<len(nets):
            net = nets[jj]
            net_type = netConfigs[jj][0]
        else:
            net = HE_nets_combine( nets, G_inter )
            net_type = 'HE'
    
        net_size = nx.number_of_nodes(net)
        ave_degree = round(nx.number_of_edges(net) * 2 / net_size, 4)
        
        # 网络连通时才计算直径和平均最短路径长度
        if nx.is_connected(net):
            ave_diameter = round(nx.diameter(net), 4)
            ave_shortest_path = round(nx.average_shortest_path_length(net), 4)
        else:
            ave_diameter = -1
            ave_shortest_path = -1
        
        ave_clustering = round(nx.average_clustering(net), 4)
        
        # 计算度分布
        degree_sequence = list(dict(net.degree()).values())
        degree_counts = Counter(degree_sequence)
        degree_stat = list()
        i = 0
        # 前述度分布中度可能不连续，缺少的度，补充对应节点数为0
        for degree, count in sorted(degree_counts.items()):
            while i<degree:
                degree_stat.append(0)
                i += 1
            degree_stat.append(count)
            i += 1
        
        # ['Net_ID', 'r', 'Pro_ID', 'D_Times', 'Net_type', 'Net_Seq', 'Net_size', 'ave_degree', 'ave_diameter', 'ave_shortest_path', 'ave_clustering' ]
        net_stat.append([net_loop, r, pro_id, D_time, net_type, jj, net_size, ave_degree, ave_diameter, ave_shortest_path, ave_clustering] + degree_stat)
    
    return net_stat

def excel_column_number(column_title):  
    """将Excel列的字母表示转换为列号"""  
    number = 0  
    for c in column_title:  
        if 'A' <= c <= 'Z':  
            number = number * 26 + (ord(c) - ord('A') + 1)  
        else:  
            raise ValueError("Invalid column title")  
    return number-1  # excel中列数从0开始

# ——————————————————————————————————————
# 将各个子网合并为一个整网

def HE_nets_combine( nets, G_inter ):
    
    if len(nets)==1:
        return nets[0]
    
    whole_net = nx.union(nets[0], nets[1])
    
    for i in range(len(nets)-2):
        ##### Bug检查，子网间存在节点交叉
        if len( list( set(list(whole_net.nodes())).intersection(set(list(nets[i+2].nodes()))) ) )>0:
            print('错误，不同子网间{0}、{1}存在节点交叉{2}！！！'.format(list(whole_net.nodes()), \
                                                       list(nets[i+2].nodes()), list( set(list(whole_net.nodes())).intersection(set(list(nets[i+2].nodes()))) )  ) )
        
        whole_net = nx.union(whole_net, nets[i+2])
    
    for edge in G_inter.edges():
        whole_net.add_edge(edge[0], edge[1])
        
    #nodeList = list(whole_net.nodes())
    #edgeList = list(whole_net.edges())
    
    return whole_net

# ——————————————————————————————————————
# 获取网络中指定规模的连通子网
# 

def get_connected_subgraphs_of_size(G, size):
    """
    获取图 G 中指定规模的连通子图
    :param G: 输入的图
    :param size: 指定的子图规模（节点数量）
    :return: 符合规模要求的连通子图列表
    """
    
    all_subgraphs = []
    nodes = list(G.nodes())
    # 生成所有指定规模的节点组合
    for node_combination in combinations(nodes, size):
        subgraph = G.subgraph(node_combination)
        
        # 检查子图是否连通
        if nx.is_connected(subgraph):
            all_subgraphs.append(subgraph)
            
    return all_subgraphs
            

"""
        # 计算度分布
        degree_sequence = list(dict(net.degree()).values())
        degree_counts = Counter(degree_sequence)
        for degree, count in sorted(degree_counts.items()):          
            f.write(str(degree)+'\t'+str(count)+'\n') 


"""