"""

This is a Python script for creating social networks for gaming (PGG)

including BA, WS, ER, REG, TREE(202407), FAM(202407)

Last revised on Aug 2024

@author: Mrh, Lty

"""
import multiprocessing
import datetime
import networkx as nx   # 导入建网络模型包，命名
#from networkx import datasets

import numpy as np
import random as rd
import math
import statistics
from collections import Counter
import logging
import time
import configparser
import os

# ERGM函数包
import pandas as pd
#import rpy2.robjects as ro
#from rpy2.robjects.packages import importr
#from rpy2.robjects import r, globalenv, Environment
#from rpy2.robjects import conversion, pandas2ri
#from rpy2.robjects.vectors import IntVector, ListVector, StrVector

from write2excel import save_nets_statistics, excel_creat_with_sheetS, excel_write_line 
from net_analysis import cal_net_statistics
from net_save_load import creat_net_output_root, load_realnet
#from net_visualization import draw_small_net

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
file = time.strftime('%y%m%d',time.localtime(time.time()))
logging.basicConfig(filename='net_creat_stat_'+file+'.log', level=logging.DEBUG, format=LOG_FORMAT)


#参数获取，暂时没有用
paraPath = "bestnetplay.conf"
def readPara(paraPath):
    paraDic = {}                                
    cf = configparser.ConfigParser() 
    cf.read(paraPath)
   
    
    paraDic['numNet'] = cf.getint("netP", "numNet")
    paraDic['numV'] = cf.getint("netP", "numV")
    paraDic['numK'] = cf.getint("netP", "numK")#相同参数组合下重复模拟次数
    paraDic['numLoop'] = cf.getint("netP", "numLoop")#一个博弈的最大循环次数
    paraDic['imaxLabel'] = cf.getint("netP", "imaxLabel")#
    paraDic['k'] = cf.getint("netP", "k")
    paraDic['radius'] = cf.getfloat("netP", "radius")
    paraDic['dijpct'] = cf.getfloat("netP", "dijpct")
    paraDic['subpct'] = cf.getfloat("netP", "subpct")
    paraDic['conWay'] = cf.get("netP", "conWay")
    
    paraDic['bmin'] = cf.getfloat("play", "bmin")#
    paraDic['binter'] = cf.getfloat("play", "binter")#
    paraDic['bmax'] = cf.getfloat("play", "bmax")#
    paraDic['c'] =cf.getfloat("play", "c")#
    paraDic['s'] = cf.getfloat("play", "s")#
    paraDic['q'] = cf.getfloat("play", "q")#
    
    paraDic['randomSelect'] = cf.getfloat("dynamic", "randomSelect")#

    return paraDic

#_____________________________________________
# 创建用于检验博弈收益计算结果的小型样例网络 

def creat_profit_test_Net( ):
    
    # 构建k=4的近邻耦合网络，ws_P=0.1
    ws_net = nx.Graph()
    for i in range(10):
        j = i+1
        if i+1>=10:
            j -= 10
        if not ws_net.has_edge(i, j):
            ws_net.add_edge(i, j)
        j = i+2
        if i+2>=10:
            j -= 10
        if not ws_net.has_edge(i, j):
            ws_net.add_edge(i, j)
    
    # 2条远连接
    ws_net.remove_edge(0, 1)
    ws_net.add_edge(0, 5)
    ws_net.remove_edge(3, 4)
    ws_net.add_edge(4, 8)
    
    # 奇数节点为合作者，偶数节点为背叛者
    for node in ws_net.nodes():
        if int(node)%2==1:
            ws_net.nodes[node]['select'] = 1
        else:
            ws_net.nodes[node]['select'] = 0
        ws_net.nodes[node]['preference'] = int(node)/10.0
    nx.set_node_attributes(ws_net, 0, 'profit')         #初始收益均为0
    nx.set_edge_attributes(ws_net, 0, 'study_p')    
    
    #draw_small_net(ws_net, 'WS', 'WS test network')
    
    # 构建k=4的BA网络
    ba_net = nx.Graph()
    ba_net.add_edge(10, 15)
    ba_net.add_edge(10, 13)
    ba_net.add_edge(10, 11)
    ba_net.add_edge(10, 19)
    ba_net.add_edge(10, 16)
    ba_net.add_edge(11, 14)
    ba_net.add_edge(11, 17)
    ba_net.add_edge(11, 12)
    ba_net.add_edge(11, 15)
    ba_net.add_edge(12, 18)
    
    ba_net.add_edge(12, 15)
    ba_net.add_edge(12, 13)
    ba_net.add_edge(13, 17)
    ba_net.add_edge(13, 19)
    ba_net.add_edge(14, 16)
    ba_net.add_edge(14, 17)
    ba_net.add_edge(15, 18)
    ba_net.add_edge(15, 16)
    ba_net.add_edge(16, 18)
    ba_net.add_edge(18, 19)
    
    # 奇数节点为合作者，偶数节点为背叛者
    for node in ba_net.nodes():
        if int(node)%2==1:
            ba_net.nodes[node]['select'] = 1
        else:
            ba_net.nodes[node]['select'] = 0
        ba_net.nodes[node]['preference'] = int(node)/10.0-1.0
    
    nx.set_node_attributes(ba_net, 0, 'profit')         #初始收益均为0
    nx.set_edge_attributes(ba_net, 0, 'study_p')
    
    #draw_small_net(ba_net, 'BA', 'BA test network')
    
    # 构建子网间连边，连边密度比0.1，连边
    inter_net = nx.Graph()
    inter_net.add_edge(0, 10)
    inter_net.add_edge(2, 15)
    inter_net.add_edge(7, 13)
    inter_net.add_edge(8, 18)
    
    for node in inter_net.nodes():
        if int(node)%2==1:
            inter_net.nodes[node]['select'] = 1
        else:
            inter_net.nodes[node]['select'] = 0    
    nx.set_node_attributes(inter_net, 0, 'profit')         #初始收益均为0
    nx.set_edge_attributes(inter_net, 0, 'study_p')
    
    print('网间联系节点属性状态检查')
    for node in ws_net.nodes():
        print(node, ws_net.nodes[node]['select'], ws_net.nodes[node]['profit'])
    print('\n')
    for node in ba_net.nodes():
        print(node, ba_net.nodes[node]['select'], ba_net.nodes[node]['profit'])
    print('\n')
    for node in inter_net.nodes():
        print(node, inter_net.nodes[node]['select'], inter_net.nodes[node]['profit'])
        
    
    return ws_net, ba_net, inter_net

# ——————————————————————————————————————
# lty 引用networkx包中的生成程序，修改、完善奇数度情况
# 修改输入参数为预期平均度，当度为奇数是，新节点随机增加k//2或k//2+1条边
# ——————————————————————————————————————

def improved_barabasi_albert_graph(n, k, seed=None, initial_graph=None):
    """Returns a random graph using Barabási–Albert preferential attachment

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    k : int
        Expected average degree of the network
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    initial_graph : Graph or None (default)
        Initial network for Barabási–Albert algorithm.
        It should be a connected graph for most use cases.
        A copy of `initial_graph` is used.
        If None, starts from a star graph on (m+1) nodes.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n``, or
        the initial graph number of nodes m0 does not satisfy ``m <= m0 <= n``.

    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """
    
    # m : int Number of edges to attach from a new node to existing nodes
    m1 = k//2+1
    if k%2==1:
        m2 = k//2
    if m1 < 1 or m1 >= n:
        raise nx.NetworkXError(
            f"Barabási–Albert network must have m >= 1 and m < n, m = {k//2+1}, n = {n}"
        )

    if initial_graph is None:
        # Default initial graph : star graph on (m + 1) nodes
        G = nx.star_graph(m1)
    else:
        if len(initial_graph) < m1 or len(initial_graph) > n:
            raise nx.NetworkXError(
                f"Barabási–Albert initial graph needs between m={k//2+1} and n={n} nodes"
            )
        G = initial_graph.copy()

    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
    # Start adding the other n - m0 nodes.
    source = len(G)
    m = k//2
    while source < n:
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        if k%2==1:
            m = rd.randint(m2, m1)            
        targets = rd.sample(repeated_nodes, m)
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)

        source += 1
    
    return G

# ——————————————————————————————————————
# mrh 引用networkx包中的生成程序，有改动，记录了每条长程边的原始边
# lty 修改、完善奇数度情况
# 以偶数度近邻耦合网络为基础，随机增补左或右邻接边
# ——————————————————————————————————————
 
def improved_watts_strogatz_graph(n, k, p):
    """Returns a Watts–Strogatz small-world graph.

    Attempts to generate a connected graph by repeated generation of
    Watts–Strogatz small-world graphs.  An exception is raised if the maximum
    number of tries is exceeded.

    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of rewiring each edge
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    newman_watts_strogatz_graph
    connected_watts_strogatz_graph

    Notes
    -----
    First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined
    to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).
    Then shortcuts are created by replacing some edges as follows: for each
    edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"
    with probability $p$ replace it with a new edge $(u, w)$ with uniformly
    random choice of existing node $w$.

    In contrast with :func:`newman_watts_strogatz_graph`, the random rewiring
    does not increase the number of edges. The rewired graph is not guaranteed
    to be connected as in :func:`connected_watts_strogatz_graph`.

    References
    ----------
    .. [1] Duncan J. Watts and Steven H. Strogatz,
       Collective dynamics of small-world networks,
       Nature, 393, pp. 440--442, 1998.
    """
    
    if k > n:
        raise nx.NetworkXError("k>n, choose smaller k or larger n")

    # If k == n, the graph is complete not Watts-Strogatz
    if k == n:
        return nx.complete_graph(n)

    G = nx.Graph()
    nodes = list(range(n))  # nodes are labeled 0 to n-1
    # connect each node to k//2 neighbors
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        G.add_edges_from(zip(nodes, targets))
    
    # 处理奇数度情况，以0.5的概率增加距离k//2+1的邻居
    if k%2 == 1:
        targets = nodes[k//2+1:] + nodes[0:k//2+1]  # first k//2+1 nodes are now last in list
        for i in range(n):
            if rd.random() < 0.5:
                G.add_edge(nodes[i], targets[i])
                # print(nodes[i], targets[i])
        
    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for j in range(1, round(k / 2) + 1):  # outer loop is neighbors
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        # inner loop in node order
        for u, v in zip(nodes, targets): 
            # 跳过奇数度情况下不存在的边
            if not G.has_edge(u, v):
                break  
                
            if rd.random() < p:
                w = rd.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = rd.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
                    G[u][w]['ori']=(u,v) #mrh增加
    
    #draw_small_net(G, 'WS', 'WS net visualization')
    
    return G


# ——————————————————————————————————————
# mrh创建引用networkx中的方法，保证WS网络连通
# ——————————————————————————————————————
def connected_watts_strogatz_graph(n, k, p, tries=100):
    
    """Returns a connected Watts–Strogatz small-world graph.

    Attempts to generate a connected graph by repeated generation of
    Watts–Strogatz small-world graphs.  An exception is raised if the maximum
    number of tries is exceeded.

    """
    
    for i in range(tries):
        G = improved_watts_strogatz_graph(n, k, p)
        if nx.is_connected(G):
            return G
        
    raise nx.NetworkXError("Maximum number of tries exceeded")

# 从first_label开始，递次给节点设置新的编号
# 引用networkx中的方法 ，有改动
def convert_node_labels_to_integers(G, first_label=0, ordering="default", edge_attribute=None,
                                    label_attribute=None ):
    """Returns a copy of the graph G with the nodes relabeled using
    consecutive integers.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    first_label : int, optional (default=0)
       An integer specifying the starting offset in numbering nodes.
       The new integer labels are numbered first_label, ..., n-1+first_label.

    ordering : string
       "default" : inherit node ordering from G.nodes()
       "sorted"  : inherit node ordering from sorted(G.nodes())
       "increasing degree" : nodes are sorted by increasing degree
       "decreasing degree" : nodes are sorted by decreasing degree

    label_attribute : string, optional (default=None)
       Name of node attribute to store old label.  If None no attribute
       is created.

    Notes
    -----
    Node and edge attribute data are copied to the new (relabeled) graph.

    There is no guarantee that the relabeling of nodes to integers will
    give the same two integers for two (even identical graphs).
    Use the `ordering` argument to try to preserve the order.

    See Also
    --------
    relabel_nodes
    """
    N = G.number_of_nodes() + first_label
    if ordering == "default":
        mapping = dict(zip(G.nodes(), range(first_label, N)))
    elif ordering == "sorted":
        nlist = sorted(G.nodes())
        mapping = dict(zip(nlist, range(first_label, N)))
    elif ordering == "increasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    elif ordering == "decreasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        dv_pairs.reverse()
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    else:
        raise nx.NetworkXError('Unknown node ordering: %s' % ordering)
        
    H = nx.relabel_nodes(G, mapping)
    # create node attribute with the old label
    
    if label_attribute is not None:
        nx.set_node_attributes(H, {v: ki for ki, v in mapping.items()},
                               label_attribute)
        
    if edge_attribute is not None:
        edges = list(nx.edges(H))
        for i in range(len(edges)):
            if edge_attribute in H[edges[i][0]][edges[i][1]].keys():        
                H[edges[i][0]][edges[i][1]][edge_attribute]=(mapping[H[edges[i][0]][edges[i][1]][edge_attribute][0]],mapping[H[edges[i][0]][edges[i][1]][edge_attribute][1]])
    
    return H

# ——————————————————————————————————————
# 202507 lty修改、完善
# 根据预期规模和度，换算出最接近的边长为L、层数为layers的分层格子网
# numV 目标网络规模
# k    网络度数
# ——————————————————————————————————————

def grid_netsize_to_edgelen( numV, k ):
    
    # 网络规模检查，k为5，规模不小于50，k为6，规模不小于64，否则开始调降生成grid的k值
    if k in (5, 6) and numV<4*4*2:
        k = 4
        print('由于网络规模小于32！生成Grid网络平均度调整为：'+str(k))
    elif k==6 and numV<64:
        k = 5
        print('由于网络规模过小64！生成Grid网络平均度调整为：'+str(k))

    # 根据度值和网络规模测算格子网构型及其边长L    
    if k==4:
        L = round( math.sqrt(numV) )
        layers = 1
    elif k==5:
        L = round( math.sqrt(numV/2) )
        layers = 2
    elif k==6:
        L = round( math.pow(numV, 1/3) )
        layers = L 
    
    return L, layers

# ——————————————————————————————————————
# Dan 创立，202407 lty修改、完善
# 根据预期规模和度创建最接近的边长为L的分层格子网
# numV 目标网络规模
# k    网络度数
# ——————————————————————————————————————

def creat_grid_based_graph(numV, k):
    
    # 根据预期规模和度，换算出最接近的边长为L、层数为layers的分层格子网
    L, layers = grid_netsize_to_edgelen( numV, k )
    
    grid_num = L * L    # 一层格子网的节点数
    
    # 测算构建几个立方体，现有k值只支持1个Cube
    grids = []
    for cube_i in range(1)  :
        
        # 构建layers层格子网
        for grid_j in range( layers )  :
            
            #print(cube_i, grid_j)
            
            # 生成有L * L个节点的无边网络
            net = nx.random_graphs.erdos_renyi_graph(grid_num, 0)
            net = convert_node_labels_to_integers(net, first_label=grid_num*grid_j, ordering="default")
            #nodesList = list(net.nodes())
            grids.append(net)
            
            # 为节点添加Grid中的位置属性
            # (a, b, c, d) Ca 表示第a个立方体，Lb表示第b层，(c, d)表示c行d列
            # k=4时，记录(C0-L0-c-d)
            # k=5时，记录(C0-Lb-c-d)，仅有2层L0、L1，2层间对应位置节点相互连接
            # k=6时，记录(C0-Lb-c-d)，有L层，上下层间对应位置节点相互连接
            for node in net.nodes():
                x = (node%grid_num)//L
                y = (node%grid_num)%L
                net.nodes[node]['grid'] = 'C'+str(cube_i)+'-L'+str(grid_j)+'-'+str(x)+'-'+str(y)
                #print(net.nodes[node]['grid'])
    
            for node in net.nodes():
                if((node+L<grid_num)):
                    net.add_edge(node, node + L)
                if((node+1)%L==0):
                    continue
                else:
                    net.add_edge(node, node+1)
        
        # 建立不同层网络节点的连接关系
        net = grids[0]
        for i in range(1, len(grids), 1):
            net = nx.union( net, grids[i] )
            for j in range(grid_num):
                net.add_edge(j+grid_num*(i-1), j+grid_num*(i))
                #print(j+grid_num*(i-1), j+grid_num*i)
        
        #edgeList = list(net.edges())
            
    return net

# ——————————————————————————————————————
# 2025.04 lty 创立
# 为指定节点集合生成全互联网络

def generate_fully_connected_network(nodes):
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            G.add_edge(nodes[i], nodes[j])
            
    return G

# ——————————————————————————————————————
# 2024.07 lty 创立
# 生成以家庭+独立个体为基础构成的社会网络
# 小型家庭由2-5个个体构成，为全互联网络，同一家庭的fam属性为其家庭编号，Fx
# 部分独立个体，Fam属性为其个体编号，Px
# 家庭与独立个体之间按照所在群体的网络构型famType相互连接
# 网络需要连通，并达到指定平均度k
# ——————————————————————————————————————

def creat_family_based_graph(famType, numV, k, ws_P=0.1):

    """Returns a network founded on small families and independent adults



    References
    ----------
    [1] Robin Dunbar. Human Evolution. P81.
    [2] Hamilton, M. J., etc. The complex structure of hunter-gatherer social networks. 2007
    [3] Hill, R. A., etc. Network scaling reveals consistent fractal pattern in hierarchical mammalian societies. 2008
    [4] Zhou, W. X., etc. Discrete hierarchical organization of social group size. 2004
    [5] Lehmann, J., etc. Unravelling the evolutionary function of communities. 2014
    """
    
    # 生成一组均值为3.5的数，使得80%节点有所属家庭，并呈正态分布，作为不同家庭的规模
    fam_ave_size = 3.5
    fam_num = round(numV*0.8/3.5) # 
    fam_sizes = np.random.normal( fam_ave_size, 1.5, fam_num )   # 均值、标准差、家庭数
    
    # 生成不同家庭的全互联子网，如果随机数小于1，则作为独立个体后续处理
    families = []
    label = 0
    sumV = 0
    for i in range(len(fam_sizes)):
        
        size = round( fam_sizes[i] )
        if size>1 and sumV+size<=numV: # 确保家庭成员数量不超过预期网络规模
            
            sumV += size
            
            # 添加当前家庭的全互联网络
            fam_i = nx.complete_graph( size )
            
            # 设置节点的家庭编号
            for j in range(size):
                fam_i.nodes[j]['fam'] = 'F'+str(i)    
                
            # 将各家庭子网的节点编号递增重编
            fam_i = convert_node_labels_to_integers(fam_i, first_label=label, ordering="default")
            label += size
            #print( list(fam_i.nodes()) )                      
            
            # 记录各家庭构成的子网
            families.append(fam_i)
            
    # 补足个体节点，直至达到指定规模
    for i in range( numV-sumV ):        
        fam_i = nx.Graph()
        fam_i.add_node(label)
        fam_i.nodes[label]['fam'] = 'P'+str(i)
        label += 1
        families.append(fam_i)  
        
    # 合并家庭网络
    net = families[0]    
    fam_num = len(families)
    i = 1
    while i<fam_num:
        net = nx.union( net, families[i] )
        i += 1 
    
    # 网络合并情况检查点
    # nodelist = list(net.nodes())
    # edgelist = list(net.edges())
           
    # 测算家庭及独立个体网络的平均度
    fam_edge_num = numV*k/2 - len(net.edges())
    k_fam = round( fam_edge_num*2/fam_num ) 
    if k_fam<3 and famType in ('BA', 'WS', 'ER'):
        k_fam = 3
    elif k_fam<4 and famType in ('TREE'):
        k_fam = 4
    
    # 以不同连通分支为节点，构建小型网络
    print('以'+str(fam_num)+'个家庭和个体构成小型网络，平均度为：'+str(k_fam))
    logging.info('以{0}个家庭和个体构成小型网络，平均度为：{1}'.format(str(fam_num), str(k_fam)))
    famNet = creat_Net(famType, fam_num, k_fam, ws_P)
    
    # fam_degree_dis = degree_distributions(famNet)
    # print( '部落网络度分布为：')
    # print( fam_degree_dis )
    famEdgeList = list(famNet.edges())
    #allEdgeList = list(net.edges())
    
    # 随机排列families中的家庭或独立个体，安置在famNet的某个节点上
    rd.shuffle(families)
    
    # 获取每个家庭或独立个体对应节点在famNet的连边，并建立对应的连接关系      
    for j in range( len(famEdgeList) ):
        curEdge = famEdgeList[j]

        u = rd.choice( list(families[curEdge[1]].nodes()) )
        v = rd.choice( list(families[curEdge[0]].nodes()) )

        if not net.has_edge(u, v):
            net.add_edge(u, v)
        else:
            print('两个独立家庭间出现连边！报错！！')
            logging.error('两个独立家庭间出现连边！报错！！')
                
    # 观察网络平均度情况，家庭网络统计情况
    #allEdgeList = list(net.edges())
    print('家庭网络平均度为：'+'\t'+str(round(nx.number_of_edges(net) * 2 / nx.number_of_nodes(net), 4))+'\n')
    logging.info('家庭网络平均度为：{0}'.format(str(round(nx.number_of_edges(net) * 2 / nx.number_of_nodes(net), 4))))
    #fam_sizes_R, fam_size_stat = fam_Statistics(net)
    
    return net

# ——————————————————————————————————————
# 2024.07 lty 创立
# 网络度分布统计
# 返回 degree_dis   list  每个位置记录度值，和对应节点数
# ——————————————————————————————————————

def degree_distributions(net):
    
    degree_dis = []
    
    degree_sequence = list(dict(net.degree()).values())
    degree_counts = Counter(degree_sequence)
    for degree, count in sorted(degree_counts.items()):   
        degree_dis.append([degree, count])
        
    return degree_dis


# ——————————————————————————————————————
# 2024.07 lty 创立
# 家庭网络统计，使用各节点的家庭属性信息'fam'
# 返回 fam_list   list  每个位置记录家庭编号，和对应规模
# 返回 size_stat  list  每个位置记录规模，和具有相应规模的家庭数量
# ——————————————————————————————————————

def fam_Statistics(fam_net):
    
    nodesList = list(fam_net.nodes())
    fam_list = []    
    fam_labels = []  # 记录全部家庭编号
    
    # 获取家庭网络中出现的家庭列表，独立个体统一记为P，计算家庭规模
    for i in range( len(nodesList) ):
        fam_label = fam_net.nodes[i]['fam']
        if 'P' in fam_label:
            fam_label = 'P'   
            
        if fam_label not in fam_labels: 
            fam_labels.append(fam_label)
            fam_list.append([fam_label, 1])
        else:
            
            for j in range( len(fam_list) ):
                if fam_list[j][0]==fam_label:
                    fam_list[j][1] += 1       
    #print(fam_list)
    
    # 统计不同规模家庭的数量
    size_stat = []
    maxSize = 0      # 家庭最大规模
    for i in range( len(fam_list) ):
        if fam_list[i][1]>maxSize and not fam_list[i][0]=='P':
            maxSize = fam_list[i][1]
    sizes = [0 for _ in range(maxSize+1)]
    for i in range( len(fam_list) ):
        if not fam_list[i][0]=='P':
            sizes[fam_list[i][1]] += 1
        else:
            sizes[1] += fam_list[i][1]
    for i in range(1, len(sizes), 1):
        size_stat.append([i, sizes[i]])
    #print(size_stat)
    
    return fam_list, size_stat

# ——————————————————————————————————————
# 2024.07 lty 创立
# 向不连通网络net的不连通分支间补充最小数量连边
#
# net 不连通网络
# limit 补充连边数量上限为平均度误差，暂定为5%
#
# 返回补充连边的数量，-1表示网络连通
# ——————————————————————————————————————

def component_connecting(net):
    
    # 计算不连通的分支
    connected_components = list(nx.connected_components(net))    
    # print("不连通的分支:", connected_components)
    compNum = len(connected_components)
    
    if compNum==1 :
        return 0
    
    # 优先选取孤立点与大的分支随机建立连边，并逐渐扩大连通分支范围
        
    # 将连通分支按照从小到大顺序排序
    connected_components = sorted(connected_components, key=len)
        
    maxComp = list(connected_components[compNum-1])
    for i in range(compNum-1):
        u = rd.choice(list(connected_components[i]))
        v = rd.choice( maxComp )
        maxComp += list(connected_components[i])
        if not net.has_edge(u, v):
            net.add_edge(u, v)
                # edgeList = list(net.edges())
        else:
            print('此时两个连通分支不应有连边！报错！！')
        
    return compNum-1
    

# ——————————————————————————————————————
# 2024.07 lty 创立
# 获取指定度数的节点集合
# degree  指定度数
# ——————————————————————————————————————

def find_nodes_with_given_degree(net, degree):
    
    degreeList = list(net.degree())
    resultList = []
    
    j = 0
    num = len(net.nodes)
    for i in range(num):
        if degreeList[i][1]==degree:
            resultList.append(degreeList[i])
            j += 1
    # print(str(resultList))
    
    return resultList

# ——————————————————————————————————————
# 2024.07 lty 创立
# 在完全平衡树基础上，通过调整叶子节点数量
# 在叶子节点间增加随机连边获取指定节点数（numV）和平均度的类树形网络
# 2024.10 增加补边方式comp_way，'ALL'在全体节点中补边，'LEAF'在叶节点中补边，'LEVEL'倾向在高层节点中补边
# 
# ——————————————————————————————————————

def creat_Tree(numV, k, comp_way='ALL'):   # 'LEVEL'

    """Returns a balanced-tree-based network with embedded hierarch

    科层制（Bureaucracy）是一种高度结构化、等级化的组织管理模式，起源于马克斯·韦伯的理论。
    一般认为，人类由原始的相对平等的社会结构，逐渐演化到由具有相应权力层级的社会结构。
    与科层制相对应的社会网络最直观的就是树形结构。
    但是，人与人之间的复杂互动又使得这种树形结构必然伴随着随机联系。
    
    为此，以networkX中完全平衡树为基础，构建了隐含层级结构复杂社会网络。
    
    根据，邓巴等人的研究[1-5]原始人类社会关系的拓展系数为3（不超过4）。n-ary balanced tree
    即由5人核心团体，以3为系数，逐步5-15-50-150-500扩大到1500人。
    
    因此，首先构建3叉完全平衡树，而后删除超过预期规模的多余叶子结点。
    由于平衡树的平均度仅为2，随机在叶子结点间补足所缺连边，从而构建节点数和平均度均满足预期的网络。
    网络中的层级结构隐藏在社会网络中。

    References
    ----------
    [1] Robin Dunbar. Human Evolution. P81.
    [2] Hamilton, M. J., etc. The complex structure of hunter-gatherer social networks. 2007
    [3] Hill, R. A., etc. Network scaling reveals consistent fractal pattern in hierarchical mammalian societies. 2008
    [4] Zhou, W. X., etc. Discrete hierarchical organization of social group size. 2004
    [5] Lehmann, J., etc. Unravelling the evolutionary function of communities. 2014
    """
    
    # 测算节点数超过numV的最小深度3或4叉树
    n_ary = 3     # 或为4
    nodeNum = 1
    depth = 0     # 只有根节点时深度为0
    comp_way = str.upper(comp_way)
    
    while nodeNum<numV:
        nodeNum = int( (1-math.pow(n_ary, depth+1))/(1-n_ary) )
        depth = depth + 1
    print('最接近指定规模的{0}叉树深度为：{1}。'.format(n_ary, str(depth)))
    
    # 生成略微超过numV的3叉树
    net = nx.balanced_tree(n_ary, depth-1)
    
    # 根据节点编号，为节点添加层级信息
    treeCount = 1
    i = 0
    for nodeLabel in net.nodes():
        while nodeLabel>=treeCount:
            i += 1
            treeCount += int(math.pow(n_ary, i))
        net.nodes[nodeLabel]['tree_level'] = i
        #print(nodeLabel, net.nodes[nodeLabel]['tree_level'])
    
    # 删除多出numV的叶节点
    # 如果当前叶节点不足删除，则只删除当前叶节点
    # 否则进行下一次循环，删除新出现的叶节点，直至达到预期节点数
    while (len(net.nodes)>numV):
        delNum = len(net.nodes)-numV
        leafNodes = find_nodes_with_given_degree(net, 1)
        
        if len(leafNodes)>delNum:
            delNodes = rd.sample(sorted(leafNodes), delNum)
        else:
            delNodes = sorted(leafNodes)
            
        for i in range(len(delNodes)):        
            net.remove_node(delNodes[i][0])
    
    # 检查平均度是否满足条件，由于叶节点的存在平均度一般偏低（k=2）
    # 由于tree结构平均度不超过5，剩余的叶节点间可满足补边要求
    
    if comp_way=='LEAF':
        nodes = find_nodes_with_given_degree(net, 1)    # 在叶节点间补边的策略
    elif comp_way=='ALL':
        nodes = list(net.nodes())     # 在全部节点中进行补边
    elif comp_way=='LEVEL':           # 倾向于对高层节点补边进行补边
        nodes = []     
        for nodeLabel in net.nodes():
            nodes = nodes + [nodeLabel]*(depth-net.nodes[nodeLabel]['tree_level'])     ###### 略微倾向于给高层节点增边
    
    # leafNum = len(leafNodes)
    # print('补边前树形网络节点数为：'+str(len(net.nodes))+'，边数为：'+str(nx.number_of_edges(net)))
    # degreeList = list(net.degree())
    
    # 标记现有边均为原始边，非补边
    for u, v in net.edges():
        net[u][v]['tree']= 'tree'
    
    if nx.number_of_edges(net)*2<k*len(net.nodes):
        edgeNum = int(k*len(net.nodes)/2-nx.number_of_edges(net))
        
        # 在k<degreeLimit的节点间补边，最初为叶节点，如果仍然不够补边则逐步放大补边
        #dLimit = 1
        for i in range(edgeNum):
            u = rd.choice( nodes )  
            e = rd.choice( nodes )  # int((rd.sample(sorted(nodes),1)[0])[0])
            loopcount = 1
            
            # 节点度变化检查
            # print(str(rd.sample(sorted(leafNodes),1)[0]))
            # print(str(u)+', '+str(e))
            
            #degree_u = net.degree(u)
            #degree_e = net.degree(e)            
            # print(str(degree_u)+', '+str(net.degree(e)))
            
            while (u==e) or (net.has_edge(u,e)): # or degree_u>dLimit or degree_e>dLimit  or degree_u>=4 or degree_e>=4
                loopcount += 1
                u = rd.choice( nodes )
                e = rd.choice( nodes )
                # print(str(u)+', '+str(e))
                
                #degree_u = net.degree(u)
                #degree_e = net.degree(e)
                # print(str(degree_u)+', '+str(net.degree(e)))
        
                #当只剩少量节点时，无以下判断会出现死循环
                #if loopcount>=leafNum*(leafNum-1)//2:
                #    dLimit += 1
                    #leafNodes = find_nodes_with_given_degree(net, dLimit)
                    #leafNum = len(leafNodes)
                    # print('反复尝试'+str(loopcount)+'，无法获取叶节点间的有效补边! 扩大节点筛取范围至度为：'+str(dLimit))
                #    break

            #if loopcount>=leafNum*(leafNum-1)//2 and dLimit>5:
            #    print('反复尝试，无法获取节点间的有效补边!')
            #    break 
            
            net.add_edge(u, e)
            net[u][e]['tree']= 'non-tree'  # 针对Tree网络，标记这条边不是树的自有边，而是补边
            
            # 补边确认，存在补边后增加新节点的情况，需要核查
            if len(net.nodes)>numV:  
                # nodeList = list(net.nodes())
                # edgeList = list(net.edges())
                # print(str(u)+', '+str(e))
                print('补边后树形网络规模突破上限，当前节点数为：'+str(len(net.nodes))+'，边数为：'+str(nx.number_of_edges(net)))
    
    # 由于删边操作可能导致生成的树形网络节点不连号，重置节点编号连续
    net = convert_node_labels_to_integers(net, first_label=0, ordering="default")
    
    # 补边调整结果核查
    # nodesList = list(net.nodes())
    # edgeList = list(net.edges())
    # degreeList = list(net.degree())
    # print('调整后树形网络节点数为：'+str(len(net.nodes))+'，边数为：'+str(nx.number_of_edges(net)))

    #print(degreeList)
    
    return net


# ——————————————————————————————————————
# 创建网络，初始化时须确保网络的连通性
# netType：网络构型
# numV：节点个数
# k：网络平均度
# ——————————————————————————————————————

def creat_Net(netType, numV, k, ws_P=0.1):
    
    try_times = 1
    
    while try_times<100 :        
        #print('生成'+netType+'连通网络，当前尝试次数：', try_times)  
        #logging.info('生成{0}连通网络，当前尝试次数：{1}!!'.format(netType, try_times))
        
        # 生成BA网络，实验发现，BA网小规模时平均度偏差较大，且无法扭转
        if netType == 'BA':              
            net = improved_barabasi_albert_graph(numV, k)
            aveDegree = round(nx.number_of_edges(net) *2 / nx.number_of_nodes(net), 4)
            if ( aveDegree<k*0.90 and numV<=50) or (aveDegree<k*0.95 and numV>50):
                
                # 由于NetworkX包中的BA网络生成机制问题，小规模时存在误差，按照度优先的规则补边
                # print('生成{0}网络平均度较低，误差超5%，进行补边。'.format(netType))
                # logging.info('生成{0}网络平均度较低，误差超5%，进行补边。'.format(netType))
                
                #节点度越大，在samplesArray中出现的次数就越多,被连边概率就越大
                node_degrees = nx.degree(net)
                samplesArray = []
                for node in node_degrees:                  
                    j = node[1]
                    samplesArray = samplesArray+[node[0]]*j
                rd.shuffle(samplesArray) #每5次打乱一下抽样数组，为使得抽样更随机
                
                edge_num = round((k-aveDegree)*numV/2)
                nodelist = list(net.nodes())
                for ii in range(edge_num):
                    if ii%5==0:
                        rd.shuffle(samplesArray) #每5次打乱一下抽样数组，为使得抽样更随机
                        
                    s_node = rd.choice(nodelist)
                    t_node = rd.choice(samplesArray)
                    while s_node==t_node or net.has_edge(s_node, t_node):
                        t_node = rd.choice(samplesArray)                    
                    net.add_edge(s_node, t_node)
                    samplesArray += [s_node, t_node]
                
                break
            
            elif ( aveDegree>k*1.1 and numV<=50) or (aveDegree>k*1.05 and numV>50):
                #print('生成{0}网络平均度较高，误差超5%，重新生成。'.format(netType))
                try_times = try_times+1
                continue                
            else:
                break
                
        # 实验发现，ER网平均度偏差较大如果ER网的平均度误差大于10%，则重新生成        
        elif netType == 'ER':
            net = nx.random_graphs.erdos_renyi_graph(numV, k / (numV - 1))
            aveDegree = round(nx.number_of_edges(net) *2 / nx.number_of_nodes(net), 4)
            if (aveDegree>k*1.05 or aveDegree<k*0.95):
                #print('生成{0}网络平均度误差超5%，重新生成。'.format(netType))
                try_times = try_times+1
                continue
            else:
                break
        
        # 生成WS网络
        elif netType == 'WS':
            net = connected_watts_strogatz_graph(numV, k, ws_P)  
            break
        
        # 生成树形网络
        elif( netType == "REG" ):
            net = creat_grid_based_graph(numV, k)
            break
            
        # 生成树形网络    
        elif netType  == 'TREE':
            net = creat_Tree(numV, k)
            break
            
        # 家庭组织网络
        elif 'FAM' in netType:
            famType = netType[4: ]
            net = creat_family_based_graph(famType, numV, k)
            break
        
        # 读取真实网络
        elif 'REAL' in netType:          
               
            # 加载数据    
            net = load_realnet( netType[5:])
            
            break
        else:
            logging.warning('netType is not valid! BA default')
            net = nx.random_graphs.barabasi_albert_graph(numV, int(k / 2)) 
            break                                     
        
        try_times = try_times+1 
        
    if not nx.is_connected(net):
        #print('生成满足平均度误差的{0}网络，最终尝试次数：{1}!!'.format(netType, try_times))
        #logging.info('生成满足平均度误差的{0}网络，最终尝试次数：{1}!!'.format(netType, try_times))
        
    # 网络不连通，尝试在不连通分支间补少量连边，使其连通
    #else: 
        component_connecting(net)
    
    return net

# ____________________________________________________________
# 
# 创建者dan, 修改lty
# 
# nets为需建立连边的子网列表
# 子网内外连边密度比例dijpct要求，随机生成所有子网之间的连边，inter_con_way为连边方式，R随机、P度优先
# 返回子网与子网的连边和子网间节点的连边

def connect_Subnet( nets, inter_con_way='R', dijpct=0.1 ):
    
    if len(nets)<2:
        print('子网数量小于2，无法建立子网间连边！')
        return
    if dijpct<=0:
        print('子网间连边概率不大于0，无法建立子网间连边！')
        return
    if inter_con_way not in ['R', 'P']:
        print('子网间连边方式不是可支持选项，无法建立子网间连边！')
        return

    G_inter = nx.Graph()
    for i in range(len(nets)-1):
        net_i = nets[i]
        Mi = nx.number_of_edges(net_i)
        Ni = nx.number_of_nodes(net_i)

        for j in range(i+1, len(nets), 1):
            net_j = nets[j]
            Mj = nx.number_of_edges(net_j)
            Nj = nx.number_of_nodes(net_j)
            dij = dijpct * min(Mi*2/(Ni*(Ni-1)), Mj*2/(Nj*(Nj-1)))  # 两子网间的边密度
            
            # 子网间连边总数
            Mij = int(dij * Ni * Nj)
            
            # 随机连边
            if inter_con_way=='R':  
                for l in range(Mij):
                    u = rd.sample(sorted(net_i.nodes()), 1)[0]
                    e = rd.sample(sorted(net_j.nodes()), 1)[0]
                    while G_inter.has_edge(u, e):
                        u = rd.sample(sorted(net_i.nodes()), 1)[0]
                        e = rd.sample(sorted(net_j.nodes()), 1)[0]
                    G_inter.add_edge(u, e)
            
            # 度优先连边
            else:
                edge_list = find_interconnected_HiDgree(net_i, net_j, G_inter, Mij)
                for edge in edge_list:
                    G_inter.add_edge(edge[0], edge[1])
    
    # 子网间连边情况检查点
    #edgeList = list(G_inter.edges())
    #nodeList = list(G_inter.nodes())
    
    return G_inter

"""
# ——————————————————————————————————————
# 按照度优先原则在两个网间选择高度节点建立连接，返回边集合
# 边的首节点为source网络、尾节点为target网络
"""

def find_interconnected_HiDgree(net_source, net_target, G_inter, edge_count):
    
    newEdges = list()
    
    # 节点度越大，在samplesArray中出现的次数就越多,被连边概率就越大
    samples_source = []
    for node_s in net_source:
        samples_source += [node_s]*net_source.degree(node_s)    
    samples_target = []
    for node_t in net_target:
        samples_target += [node_t]*net_target.degree(node_t)

    if len(samples_target)==0 or len(samples_source)==0:
        return None

    for j in range(edge_count):
        
        node_s = samples_source[rd.randint(0, len(samples_source)-1)]
        node_t = samples_target[rd.randint(0, len(samples_target)-1)]     
        iter_times = 0
        while ([node_s, node_t] in newEdges or G_inter.has_edge(node_s, node_t)) and iter_times<len(samples_source)*len(samples_target)*5:
            node_s = samples_source[rd.randint(0, len(samples_source)-1)]
            node_t = samples_target[rd.randint(0, len(samples_target)-1)] 
            iter_times += 1
        
        if iter_times<len(samples_source)*len(samples_target)*5:
            newEdges.append([node_s, node_t])
        else:
            print('Fail to add high degree inter-edges between nets with size [{0}, {1}].'.format(nx.number_of_nodes(net_source), nx.number_of_edges(net_target)))
            logging.error('Fail to add high degree inter-edges between nets with size [{0}, {1}].'.format(nx.number_of_nodes(net_source), nx.number_of_edges(net_target)))

    return newEdges 

"""——————————————————————————————————————

# lty创建
# 测试网络生成时的拓扑特征

——————————————————————————————————————"""

def main_static_net_analysis( netType, sizes, k, numCreate=10, ws_P=0.1 ):
    
    rootDirectory = creat_net_output_root( netType, sizes[0], -1, k, numCreate, ws_P )
    
    # 创建网络统计结果汇总excel
    net_excel_name = rootDirectory+'/'+netType+'_'+str(k)+'_net_statistics_'+str(datetime.date.today())+'.xls'
    excel_creat_with_sheetS(net_excel_name, ['stat_detail', 'stat_sum', 'stat_change'] )
    collumns_title = ['Net_ID', 'D_Times', 'Net_size', 'ave_degree', 'ave_diameter', 'ave_shortest_path', 'ave_clustering', 'degree_distributions' ]
    excel_write_line(net_excel_name, 0, 0, collumns_title)   # 写入表头从第1行、第1列开始
    
    all_net_detail = list()
    d_times = 0
    
    for numV in sizes:
        
        for i in range(numCreate):
        
            #if i%100==0:
                #print( '生成第{0}次网络。'.format( i ) )
        
            net = creat_Net(netType, numV, k, ws_P=0.1)
        
            all_net_detail.append( cal_net_statistics( i, d_times, net) )
        
        d_times += 1
        
    # 输出全部网络的统计特征
    print("\n输出全部网络的统计特征!!")
    save_nets_statistics( net_excel_name, netType, k, sizes, all_net_detail)
    
    return

# ——————————————————————————————————————
# lty 创立
# 网络增减幅度平滑，即每次网络规模变化幅度不超过一定限度，否则自动测算中间过渡规模
# sizes_arr            list   用户输入的预期规模变化  
# 返回smoothed_sizes   list   平滑后的网络演变规模
# ——————————————————————————————————————

def smoothing_net_size_changes( sizes_arr, maxScale=0.5 ):
    
    if len(sizes_arr)<2:
        print("未输入2个及以上的网络动态规模!!") 
        return sizes_arr
    
    smoothed_sizes = list()
    smoothed_sizes.append(sizes_arr[0])
    for i in range(len(sizes_arr)-1):
        
        # 当下一阶段的网络规模在现有规模的变化幅度内
        if sizes_arr[i]*(1-maxScale) <= sizes_arr[i+1] <= sizes_arr[i]*(1+maxScale):
            smoothed_sizes.append(sizes_arr[i+1])
        
        # 超过变化幅度需测算过渡步数及步幅，采取非线性插值方法
        else:
            if sizes_arr[i]<sizes_arr[i+1]:
                steps = int(math.log(sizes_arr[i+1]/sizes_arr[i], (1+maxScale)))+1
                j = 1
                while j<steps :
                   smoothed_sizes.append( int(sizes_arr[i]*math.pow(1+maxScale, j)) )
                   j += 1
                smoothed_sizes.append( sizes_arr[i+1] )
            else:
                steps = int(math.log(sizes_arr[i+1]/sizes_arr[i], (1-maxScale)))+1
                j = 1
                while j<steps :
                   smoothed_sizes.append( int(sizes_arr[i]*math.pow(1-maxScale, j)) )
                   j += 1
                smoothed_sizes.append( sizes_arr[i+1] )

    print("The smoothed dynamic netsizes are: "+str(smoothed_sizes))
    
    return smoothed_sizes
'''
# ——————————————————————————————————————
# lty 创立
# 输出networkx中支持的网络类型
# ——————————————————————————————————————

def list_nxEmbeded_nets():
    # 列出所有可用的内置网络函数
    social_networks = [
        func for func in dir(nx) 
        if func.endswith('_graph') or func in ['dolphins', 'les_miserables']
    ]
    
    print("NetworkX 内置社会网络：")
    for net in social_networks:
        print(f"- {net}")    
    
    # 创建一个示例网络（使用networkx）
    G = nx.karate_club_graph()  # 经典的空手道俱乐部网络

# 手动注册pandas Series到R向量的转换规则
def convert_series(series):
    return IntVector(series.tolist())

def pandas_df_to_r_df(pd_df):
    """将pandas DataFrame转换为R数据框"""
    # 转换每一列
    r_columns = {}
    for col_name in pd_df.columns:
        # 转换为R向量
        if pd_df[col_name].dtype.kind in 'iu':  # 整数类型
            r_vec = IntVector(pd_df[col_name].tolist())
        else:  # 其他类型
            r_vec = StrVector(pd_df[col_name].astype(str).tolist())
        r_columns[col_name] = r_vec
    
    # 创建R列表
    r_list = ListVector(r_columns)
    
    # 转换为R数据框
    r_df = r['data.frame'](r_list)
    
    # 修正：使用R的方式设置列名
    # 这里使用R的colnames<-函数进行赋值
    r_df = r('`colnames<-`')(r_df, StrVector(pd_df.columns.tolist()))
    
    return r_df   


# ——————————————————————————————————————
# 快速调试入口

if __name__ == '__main__':
 
    # 读取真实网络，分析其ERGM网络特征
    # 导入R的相关包
    utils = importr('utils')
    
    # 安装必要的R包（首次运行时需要）
    #utils.install_packages('statnet', dependencies=True)
    #utils.install_packages('ergm')
    
    ergm = importr('ergm', on_conflict="warn")
    network = importr('network', on_conflict="warn")

    # netType in []: # ['REG', 'BA', 'ER', 'WS', 'REAL-Nyangatom', 'REAL-Female', 'REAL-Male']
    netType = 'REAL-Nyangatom'
    G = creat_Net( netType, numV=50, k=4, ws_P=0.1 )
    
    edges = list(G.edges())
    edge_df = pd.DataFrame(edges, columns=['from', 'to'])
    edge_df['from'] += 1  # 转换为R的节点编号（从1开始）
    edge_df['to'] += 1
       
    # 执行转换
    r_edge_df = pandas_df_to_r_df(edge_df)
        
    net = network.network(r_edge_df, directed=False)

    # 关键步骤：多种方式确保net在R环境中可用
    # 1. 放入全局环境
    globalenv['net'] = net
    
    # 2. 显式创建R环境并绑定变量
    env = Environment()
    env['net'] = net
    
    # 3. 直接使用R命令创建net变量（最可靠的方式）
    ro.r.assign("net", net)   
    
    # 解决方法：在gwdegree中增加cutoff参数（根据实际最大度设置）
    # ERGM 中，gwdegree()（广义加权度，Generalized Weighted Degree）术语中的截断值（cutoff）
    # 是一个控制参数，用于限制模型对高度数节点（度值极大的节点） 的权重计算，
    # 避免这类节点对模型估计产生过度影响
    # 首先检查网络中节点的最大度
    degrees = [G.degree(node) for node in G.nodes()]
    max_degree = max(degrees) if degrees else 0
    print(f"网络中最大节点度: {max_degree}")
    
    # 设置cutoff为最大度 + 10（留有余地）
    cutoff_value = max_degree + 10
    print(f"将gwdegree的cutoff设置为: {cutoff_value}")
    
    # 关键修复：将Python的cutoff_value变量传递到R环境
    globalenv['cutoff_value_r'] = cutoff_value  # 在R中创建名为cutoff_value_r的变量
    
    # 验证net是否存在于R环境中
    print("R环境中的变量列表:", ro.r("ls()"))  # 应包含'net'
    print("net的类型:", ro.r("class(net)"))    # 应显示"network"
    
    # 拟合包含度分布和路径特征的ERGM
    # 模型项说明：
    # - edges：控制网络密度
    # - gwdegree(0.3)：几何加权度数，控制度分布
    # - triangles：促进聚类，缩短路径
    # - esp(2)：控制长度为2的路径，间接影响直径
    # - geo.mean.age：几何平均路径长度，间接反映网络的 "紧凑性"（路径越短，直径可能越小）
    # - connected：约束网络为连通图（直径为有限值），避免网络分裂为多个孤立组件。
    # + gwdegree(0.3, cutoff={cutoff_value_r}) + esp(2)
    model = ergm.ergm(
        ro.Formula('net ~ edges + triangles ') # 拟合 ERGM 模型（估计参数），而非直接生成仿真网络
    )
    
    # 2. 调整MCMC控制参数
    # 避免采样过程没有混合（mixing），即采样链未能充分探索参数空间，导致无法继续优化
    # 增加燃烧期(burnin)和样本量，调整步骤大小
    #control = ergm.control_ergm(
    #    MCMC_burnin=10000,      # 增加燃烧期（丢弃的初始样本），至少设置为网络节点数的 100 倍（如 50 节点→5000 步）
    #    MCMC_samplesize=10,   # 增加样本量
    #    MCMC_interval=10,       # 采样间隔，减少自相关
    #    MCMC_adjust=1,         # 自动调整提议分布
    #    seed=42                 # 固定种子，便于复现
    #)
        
    # 查看模型结果
    print("模型参数估计：")
    print(ergm.summary_ergm(model))   
    
    # 模拟符合模型的网络
    sim_nets = ergm.simulate_ergm(model, nsim=3)

    # 分析模拟网络的度分布和直径
    for i, sim_net in enumerate(sim_nets):
        edges_sim = list(zip(
            sim_net.rx2('mel')[0],  # 边的起点
            sim_net.rx2('mel')[1]   # 边的终点
        ))
        G_sim = nx.Graph(edges_sim)
        
        # 计算生成网络的拓扑信息
        # [net_id, d_time, net_size, ave_degree, ave_diameter, ave_shortest_path, ave_clustering] + degree_stat
        net_stat = cal_net_statistics( i, 0, G_sim)
        
        print("\n模拟网络特征："+net_stat)

# ——————————————————————————————————————
# 快速调试入口

#if __name__ == '__main__':
    
    multiprocessing.freeze_support()
    m = multiprocessing.Manager()
    lock = m.Lock()
    
    #draw_net(30, 4)
    
    #[30, 45, 67, 101, 151, 227, 341, 512, 768, 1000, 500, 250, 125, 62, 31, 30]
    
    date_time_1 = datetime.datetime.now()
    print("Gaming Started at " + str(date_time_1) + "!!\n")
    
    #creat_profit_test_Net()

    # 反复生成网络，评价网络的拓扑状态
    # main_static_net_analysis( netType, numV, k, numCreate, ws_P )
    #main_static_net_analysis('BA', 40, 5, 1, 0.1)
    
    sizes = [15]  #
    sizes = smoothing_net_size_changes( sizes )

    numCreate = 20
    pool = multiprocessing.Pool(processes=min(2, numCreate))
    
    for netType in ['WS']: #'BA', 'REG','FAM', , 'WS', 'TREE' 
    
        k = 3
        if netType=='REG':
            k = 4
            
        while k>=3 and k<=6:  
            #if netType in ['BA', 'ER', 'FAM', 'REG', 'TREE', 'WS'] : #or (netType==)and k%4==0
                
            print('Dynamic net analysis begins for {0} nets with k={1} and sizes {2} and iterates {3} times.'.format(netType, k, sizes, numCreate))
            #logging.info('Dynamic net analysis begins for {0} nets with k={1} and sizes {2} and iterates {3} times.'.format(netType, k, sizes, numCreate))
                
            main_static_net_analysis ( netType, sizes, k, numCreate, 0.1 )
            #pool.apply_async(main_static_net_analysis, (netType, sizes, k, numCreate, 0.1))
            
            k += 1
    
    pool.close()
    pool.join()  

    date_time_2 = datetime.datetime.now()
    print("Gaming Ended at " + str(date_time_2) + "!!\n")
    


# 加载库
library(ergm)
library(network)

# 1. 准备数据（空手道俱乐部网络）
data(karate)

# 2. 拟合ERGM模型
model <- ergm(
  karate ~ edges + triangles + gwdegree(0.3),
  control = control.ergm(
    MCMC.burnin = 10000,    # 预热次数
    MCMC.samplesize = 500,  # 样本量
    MCMC.interval = 100     # 抽样间隔
  )
)

# 3. 绘制迹线图（诊断MCMC收敛性）
mcmc.diagnostics(model, type = "trace")  # type="trace"指定绘制迹线图

# 4. 解读迹线图的关键指标
# - 平稳性：迹线是否无趋势地随机波动
# - 混合性：是否覆盖较广的取值范围（无长时间停滞）
# - 预热效果：预热期后是否进入平稳状态
    '''
