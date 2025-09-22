
"""

This is a Python script for input, output, draw network of PGG

Last revised on March 2025

@author: Mrh, Dan, Lty, Grj

添加了读取真实网络的方法

"""

import networkx as nx
import os
import sys

#import xlrd
#import xlwt
#import numpy as np
#from xlutils.copy import copy
import logging
from collections import Counter
import datetime
import math
import matplotlib.pyplot as plt

# ——————————————————————————————————————
# Grj 创立
# 由网络数据文件读取网络拓扑信息，目前仅支持Female、Male、Nyangatoms三个真实网络
# ——————————————————————————————————————

''' 
Female和Male的.net文件的节点包含三种属性：节点ID、性别、影响度，边包含一种属性：节点之间的关系；
Nyangatom的.net文件的节点包含两种属性：节点ID、影响度，边包含一种属性：节点之间的关系

（1）Nyangatom社会网络揭示了友谊关系在群体协作中的关键作用，尤其是在突袭方面的影响。网络中的年龄层次反映了
社交结构的代际特征，而互惠关系则展现了社会联系的稳定性。该网络表明，友谊不仅影响资源的流动和分配，还在合作模式
和群体凝聚方面发挥着重要作用。

（2）Female网络和Male网络在性别差异方面揭示Hadza社会中男性和女性在社交网络中的不同角色和互动模式；在合作行为
方面分析性别对合作行为传播的影响，了解各自网络中的合作倾向；在社会动态方面帮助理解性别在食物分享和育儿等任务中的
分工；在进化视角方面反映性别在合作和社会结构适应中的重要性，为人类社会演化提供新视角。

References
    ----------
    [1]Luke G ,Alexander I ,W R W , et al.Formation of raiding parties for intergroup violence is 
    mediated by social network structure.[J].Proceedings of the National Academy of Sciences of the 
    United States of America,2016,113(43):12114-12119.
    
    [2]L C A ,W F M ,H J F , et al.Social networks and cooperation in hunter-gatherers.[J].Nature,2012,481(7382):497-501.
'''

def load_realnet( network_name="network" ):  

                
    if getattr(sys, 'frozen', False):
        # 如果是打包后的可执行文件
        base_path = os.path.dirname(sys.executable)
    else:
        # 如果是作为脚本运行
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    # 定义子目录名称和网络文件名称
    sub_dir = 'realnets'
    
    # 根据网络名称选择文件路径
    if network_name.lower() in ["female", "male", "nyangatom"]:
        net_file = network_name + '.net'
    else:
        raise ValueError("不支持的网络名称，请选择 'Female', 'Male' 或 'Nyangatom'")
        
    # 构建配置文件的完整路径
    net_file_path = os.path.join(base_path, sub_dir, net_file)
    logging.info('Network file is located at {0}.'.format(net_file_path))
       
    # 根据网络名称选择文件路径
    #if network_name.lower() in ["female", "male", "nyangatom"]:
    #    file_path = network_path + "/" + network_name + ".net"
    #else:
    #    raise ValueError("不支持的网络名称，请选择 'Female', 'Male' 或 'Nyangatom'")
    
    try:
        with open(net_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：未找到配置文件 {net_file_path}")
        logging.error(f"Cannot locate the network file {net_file_path}!")
    except Exception as e:
        print(f"发生未知错误：{e}")
        logging.error(f"Unexpected error in loadding the network file {net_file_path}!")

    vertices = {}   #节点
    edges = []      #边
    edge_relations = {}  # 用于存储节点的关系
    reading_vertices = True  # 用于指示是否正在读取节点信息

    for line in lines:
        line = line.strip()  # 去掉首尾空白字符
        if line.startswith("*Vertices"):
            continue  # 跳过这一行
        elif line.startswith("*Edges"):
            reading_vertices = False  # 开始读取边的信息
            continue  # 跳过这一行
        elif reading_vertices:  # 处理节点信息
        
            parts = line.split()
            if len(parts) == 2:  # 两列：节点ID和影响力，针对Nyangatom网络
                vertex_id = int(parts[0])  # 节点 ID
                influence = parts[1]  # 影响力
                vertices[vertex_id] = {'influence': influence}
                
            elif len(parts) == 3:  # 三列：节点ID、性别和影响力，针对Female和male网络
                vertex_id = int(parts[0])  # 节点 ID
                gender = parts[1]  # 性别
                influence = parts[2]  # 影响力
                vertices[vertex_id] = {'gender': gender, 'influence': influence}

        else:  # 处理边的信息
            parts = line.split()
            if len(parts) >= 2:
                source = int(parts[0])  # 边的起始节点
                target = int(parts[1])  # 边的目标节点
                relation = parts[2] if len(parts) == 3 else None  # 节点之间的关系（第三列）
                # 确保边的顺序是无向的
                edge = (min(source, target), max(source, target))  # 创建无向边元组
                # 存储边的关系（如果存在）
                if relation:
                    edge_relations[edge] = relation
                # 检查边是否已经存在
                if edge in edges:
                    continue  # 如果边已存在，跳过
                else:
                    edges.append(edge)  # 添加边

    # 创建图
    G = nx.Graph()

    # 添加节点和属性
    for vertex_id, attrs in vertices.items():
        G.add_node(vertex_id, **attrs)

    # 添加边
    G.add_edges_from(edges)

    # 计算网络统计信息
    #num_nodes = G.number_of_nodes()
    #num_edges = G.number_of_edges()
    #ave_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

    # 打印边的数量
    '''
    print(f"实际添加的边数量: {len(edges)}")  # 输出边的数量
    print(f"图中的边数量: {num_edges}")  # 输出图中的边数量
    '''
        
    #nodeList = list(G.nodes())
   
    return G

# Dan 创立，lty完善
# 计算节点邻居中的合作者数量
# ——————————————————————————————————————

def C_nei_count(net, node):
    
    neighbors = nx.all_neighbors(net, node)
    count = 0
    for node in neighbors:
        if net.nodes[node]['select'] == 1:
            count += 1
    
    return count


# ——————————————————————————————————————
# Dan 创立，lty完善
# 输出演化稳定的网络状态
# ——————————————————————————————————————

def save_game_stable_net( netDirectory, net, netType, D_time, filename ):

    if not os.path.exists(netDirectory):
        os.mkdir(netDirectory)
    f = open(os.path.join(netDirectory, filename), 'w')

    nodes = list(nx.nodes(net))
    f.write('*Vertices ' +str(len(nodes))+'\n')
    
    for nodelabel in nodes:
        f.write(str(nodelabel+1) +' Str is '+ str(net.nodes[nodelabel]['select'])+ \
                ', survive '+ str(D_time) +' stages and ' + str(net.nodes[nodelabel]['survival_time']) +' times.\n')

    f.write('*Edges\n')
    edges = list(nx.edges(net))
    if netType == 'WS':
        for i in range(len(edges)):
            if 'ori' in net[edges[i][0]][edges[i][1]].keys():
                f.write(str(edges[i][0] + 1) + ' ' + str(edges[i][1] + 1) + '\n')
            else:
                f.write(str(edges[i][0] + 1) + ' ' + str(edges[i][1] + 1) + '\n')
    else:
        for i in range(len(edges)):
            f.write(str(edges[i][0] + 1) + ' ' + str(edges[i][1] + 1) + '\n')

    f.close()
    
    return


# ——————————————————————————————————————
# Dan 创立，lty完善
# 保存生成的网络
# 不同类型的网络输出不同的节点属性
# tree 节点树层级、family 所属家庭、Grid 所属切片、BA 节点度数
# ——————————————————————————————————————

def saveNet( netDirectory, net, netType, filename ):

    if not os.path.exists(netDirectory):
        os.mkdir(netDirectory)

    f = open(os.path.join(netDirectory, filename), 'w')
    f.write('*Vertices '+str( len(net.nodes()) )+'\n')
    
    # 写入节点编号及属性信息
    for nodeLabel in net.nodes():
        
        nodeAttr = ''
        if netType=='REG':
            nodeAttr = '_'+str(net.nodes[nodeLabel]['grid'])
        elif 'FAM' in netType:
            nodeAttr = '_'+str(net.nodes[nodeLabel]['fam'])
        elif netType=='TREE':
            nodeAttr = '_'+str(net.nodes[nodeLabel]['tree_level'])
        elif netType=='BA':
            nodeAttr = '_'+str(net.degree(nodeLabel))
        elif netType=='GP' or netType=='ML':
            nodeAttr = '_'+str(net.nodes[nodeLabel]['subnet_type'])    
        
        f.write(str(nodeLabel+1) + nodeAttr + '\n')
    
    f.write('*Edges\n')    
    edges = list(nx.edges(net))
    if netType == 'WS':
        for i in range(len(edges)):
            if 'ori' in net[edges[i][0]][edges[i][1]].keys():
                f.write(str(edges[i][0]+1)+' '+str(edges[i][1]+1)+' '+str([j+1 for j in net[edges[i][0]][edges[i][1]]['ori']])+'\n')
            else:
                f.write(str(edges[i][0]+1)+' '+str(edges[i][1]+1)+'\n')
    else:
        for i in range(len(edges)):
            #print(edges[i])
            f.write(str(edges[i][0]+1)+' '+str(edges[i][1]+1)+'\n')
            
    f.close()
    
    return
    
    
# 读取.net文件中的网络   
def readNet(filepath):
    f = open(filepath, 'r')
    data = f.readlines()
    net = nx.Graph() 
    for i in range(len(data)):
        dataline = data[i][:-1]
        if dataline.startswith('*Vertices') or dataline.startswith('*Edges'):
            continue
        node = dataline.split(' ')
        if len(node)==1:
            net.add_node(int(node[0])-1) 
        elif len(node) > 1:
            net.add_edge(int(node[0])-1,int(node[1])-1)
        if len(node)>2:
            net[int(node[0])-1][int(node[1])-1]['ori']=(int(node[2][1:-1])-1, int(node[3][0:-1])-1)
    return net

# ——————————————————————————————————————
# lty创立
# 输出网络的各种统计信息，拓扑特征统计，最后统计平均值及方差
# ——————————————————————————————————————

def saveNetStatistics(fileName, net, loop, netType):
    
    f = open(fileName, "a")
    
    # 网络的拓扑统计
    f.write('net_'+str(loop)+'_'+str(len(net.nodes))+'\n')        
    f.write('平均度为：'+'\t'+str(round(nx.number_of_edges(net) * 2 / nx.number_of_nodes(net), 4))+'\n')
    
    # 网络连通时才计算直径和平均最短路径长度
    if nx.is_connected(net):
        f.write('直径为：'+'\t'+str(round(nx.diameter(net), 4))+'\n')
        f.write('平均最短路径长度为：'+'\t'+str(round(nx.average_shortest_path_length(net), 4))+'\n')
    else:
        print(str(netType)+'网络出现不连通！')
        f.write('网络不连通！不再计算直径、平均最短路径。')
        logging.error('{0}网络出现不连通！'.format(str(netType)))
    
    f.write('平均聚集系数为：'+'\t'+str(round(nx.average_clustering(net), 4))+'\n')
            
    # 计算度分布
    f.write('度分布为：' + '\n')
    degree_sequence = list(dict(net.degree()).values())
    degree_counts = Counter(degree_sequence)
    for degree, count in sorted(degree_counts.items()):          
        f.write(str(degree)+'\t'+str(count)+'\n') 
    f.write('____________________________\n\n')
    
    f.close()
    
    return

# ——————————————————————————————————————
""" 
lty 创建

创建网络结果保存的根目录

 end_N =-1表示为静态网络

"""

def creat_net_output_root( netType, start_N, end_N, k, numCreate, ws_P):

    if netType == "WS":
        rootDirectory = netType + '+P' + str(ws_P) + '_N' + str(start_N) + 'to' + str(end_N) \
                + '+K'+str(k)+ '_' + str(numCreate) +'_'+str(datetime.date.today()) 
    
    elif netType == "REG":
        
        # 根据度值和网络规模测算格子网构型及其边长L    
        if k==4:
            L = round( math.sqrt(start_N) )
        elif k==5:
            L = round( math.sqrt(start_N/2) )
        elif k==6:
            L = round( math.pow(start_N, 1/3) )
        rootDirectory = netType + '_L' + str(L) + '+K' + str(k) +'_'+str(numCreate) \
                +'_'+str(datetime.date.today())
    else:
        rootDirectory = netType + '_N' + str(start_N) + 'to' + str(end_N) + '+K' + str(k) \
                +'_'+str(numCreate)+'_'+str(datetime.date.today())  #+'x'+str(numBoyi)
    
    if end_N==-1:
        rootDirectory = 'Sig_S_' + rootDirectory
    else:
        rootDirectory = 'Sig_D_' + rootDirectory
    
    if not os.path.exists(rootDirectory):
        os.makedirs(rootDirectory)  
        
    return rootDirectory

# ——————————————————————————————————————
# 快速调试入口

if __name__ == '__main__':
    
    # 获取当前脚本文件的绝对路径
    current_script_path = os.path.abspath(__file__)
    # 获取当前脚本所在的目录
    rootDirectory = os.path.dirname(current_script_path)
    rootDirectory += '/realnets'
    
    # 加载数据    
    net1 = load_net_data(rootDirectory, "Female")
    #net1 = load_net_data("Male")
    #net1 = load_net_data("Nyangatom")
    
    plt.figure(figsize=(10, 8))  # 设置图形大小
    pos = nx.spring_layout(net1)  # 计算节点位置
    nx.draw(net1, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_color='black', edge_color='gray')
    
    plt.show()
