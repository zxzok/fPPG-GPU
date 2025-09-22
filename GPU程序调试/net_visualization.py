"""

This is a Python script for visualizing small networks

Last revised on Aug 2024

@author: Lty

"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx   # 导入建网络模型包，命名
import net_creat 

#______________________________________

max_G_size = 100    # 可视化网络的最大规模


# 尝试绘制网络图
def draw_small_net(G, netType, pic_title):


    # 绘制网络图
    plt.figure(figsize=(10, 8))
    
    # 定义节点布局
    if netType=='WS' :
        pos = nx.circular_layout( G )  # 环形布局
    else:
        pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes

    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=15, font_weight='bold')
    plt.title(pic_title)
    plt.show()

    return

def draw_GEO_gaming_net(G, netType, pic_title):

    # 网络规模过大则不显示，直接退出
    if nx.number_of_nodes(G)>max_G_size:
        print(f"网络规模 {nx.number_of_nodes(G)} 超出上限 {max_G_size}！！！")
        return

    fig = plt.figure(figsize=(10, 8))
    if netType=='WS' :
        pos = nx.circular_layout( G )  # 环形布局
    elif netType=='REG': 
        
        """ Grid经尝试确定以下布局
        # 4x4 grid, iterations=100, seed=1, node_size=150 
        # 5x5 grid, iterations=100, seed=1, node_size=120 
        # 6x6 grid, iterations=100, seed=4, node_size=80 
        # 7x7 grid, iterations=100, seed=13, node_size=60  
        # 8x8 grid, iterations=100, seed=100, node_size=50
        # 9x9 grid, iterations=100, seed=11100, node_size=45
        # 10x10 grid, iterations=100, seed=111009, node_size=35
        pos = nx.spring_layout(G, iterations=100, seed=100)
        
        BA网经尝试，spring_layout最优，seed取值无明显影响
        
        ER网经尝试，kamada_kawai_layout较好，节点相对平铺，适合ER网节点均匀的情况
        FAM网经尝试，kamada_kawai_layout较好，FAM节点聚集，不同家庭间连边清晰
        """
        
        pos = nx.spring_layout(G, iterations=100, seed=39775)
    else:
        pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
    
    return

# 绘制小型WS网络可视化图，节点布局采用环形布局
# 研究人机交互

"""circular_layout：
将节点均匀放置在圆周上。
适用于需要强调网络环形结构的场景。"""

def draw_small_gaming_net(G, netType, pic_title):
    
    # 网络规模过大则不显示，直接退出
    if nx.number_of_nodes(G)>max_G_size:
        print(f"网络规模 {nx.number_of_nodes(G)} 超出上限 {max_G_size}！！！")
        return
    
    fig = plt.figure(figsize=(10, 8))
    if netType=='WS' :
        pos = nx.circular_layout( G )  # 环形布局
    elif netType=='REG': 
        
        """ Grid经尝试确定以下布局
        # 4x4 grid, iterations=100, seed=1, node_size=150 
        # 5x5 grid, iterations=100, seed=1, node_size=120 
        # 6x6 grid, iterations=100, seed=4, node_size=80 
        # 7x7 grid, iterations=100, seed=13, node_size=60  
        # 8x8 grid, iterations=100, seed=100, node_size=50
        # 9x9 grid, iterations=100, seed=11100, node_size=45
        # 10x10 grid, iterations=100, seed=111009, node_size=35
        pos = nx.spring_layout(G, iterations=100, seed=100)
        
        BA网经尝试，spring_layout最优，seed取值无明显影响
        
        ER网经尝试，kamada_kawai_layout较好，节点相对平铺，适合ER网节点均匀的情况
        FAM网经尝试，kamada_kawai_layout较好，FAM节点聚集，不同家庭间连边清晰
        """
        
        pos = nx.spring_layout(G, iterations=100, seed=39775)
    else:
        pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
    
    # 绘制节点
    C_List = list()
    B_List = list()
    N_List = list()
    C_Pre_List = list()
    B_Pre_List = list()
    for nodeLabel in G.nodes:
        
        # fPGG参与情况绘制
        #if G.nodes[nodeLabel]['participation'] == 2:    # participation
        #    B_List.append(nodeLabel)
        #elif G.nodes[nodeLabel]['participation'] == 1:    # participation:
        #    C_List.append(nodeLabel)
        #else:
        #    N_List.append(nodeLabel)
        
        # 节点策略情况绘制
        if G.nodes[nodeLabel]['select'] == 0:    # participation
            B_List.append(nodeLabel)
            B_Pre_List.append(nodeLabel) 
        elif G.nodes[nodeLabel]['select'] == 1:    # participation:
            C_List.append(nodeLabel)
            C_Pre_List.append(nodeLabel)
        else:
            N_List.append(nodeLabel)
    
    options = {"edgecolors": "tab:gray", "node_size": 600, "alpha": 0.9}     # 结果图片大小为A4半页时600，1500
    nx.draw_networkx_nodes(G, pos, nodelist=B_List, node_color="tab:red", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=C_List, node_color="tab:blue", **options)
    
    '''
    # 计算每个节点的个体倾向性（作为颜色映射的依据）
    #node_preference = [G.nodes[node]['preference'] for node in G.nodes()]  
    # 定义颜色规则：度数>5的节点为红色，否则为蓝色
    #node_colors = ['red' if d > 5 else 'blue' for d in node_degrees]
    #nx.draw_networkx_nodes(G, pos, nodelist=B_List, node_shape='o',  # 圆形
    #                   node_color=B_Pre_List, cmap=plt.cm.bwr,      # 选择颜色映射（viridis是常用的渐变色）
                       **options)
    nx.draw_networkx_nodes(G, pos, nodelist=C_List, node_shape='s',  # 正方形
                       node_color=C_Pre_List, cmap=plt.cm.bwr,      # 选择颜色映射（viridis是常用的渐变色）
                       **options)  '''
    
    nx.draw_networkx_nodes(G, pos, nodelist=N_List, node_color="tab:gray", **options)
    
    # 绘制节点属性（当前收益）位置，使用position参数调整标签位置
    #offset = 0.1  # 偏移量大小，可以根据你的图和标签长度进行调整  
    #label_pos = {node: (pos[node][0] + offset * (1 if node % 2 == 0 else -1),  # x方向偏移  
    #                    pos[node][1] + offset * (1 if node % 2 == 0 else -1))  # y方向偏移  
    #             for node in G.nodes()}  
    #node_labels = {node: G.nodes[node]['profit'] for node in G.nodes()}
    #nx.draw_networkx_labels(G, label_pos, labels=node_labels, font_size=16)
    
    # 绘制边  
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # 绘制边属性
    edges = G.edges(data=True)  
    #widths = [G[u][v]['weight'] for u, v, _ in edges] 
    #edge_labels = {(u, v): f"{d['study_p']}" for u, v, d in edges}  
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=18) 
    
    CC_edges = list()
    CB_edges = list()
    BB_edges = list()
    for edge in G.edges:
        if edge[0] in C_List and edge[1] in C_List:
            CC_edges.append(edge)
        elif edge[0] in B_List and edge[1] in B_List:
            BB_edges.append(edge)
        else:
            CB_edges.append(edge)
            
    #nx.draw_networkx_edges( G, pos, edgelist=CB_edges, width=4, alpha=0.5, edge_color="tab:orange" )
    #nx.draw_networkx_edges( G, pos, edgelist=CC_edges, width=4, alpha=0.5, edge_color="tab:blue" )
    #nx.draw_networkx_edges( G, pos, edgelist=BB_edges, width=4, alpha=0.5, edge_color="tab:gray" )       
     
    # 绘制参与公共物品活动的发起关系
    #PGG_edges = list()
    #for edge in G.edges:
    #    if G.edges[edge[0], edge[1]]['participation']==1:
    #        PGG_edges.append(edge)
    #nx.draw_networkx_edges( G, pos, edgelist=PGG_edges, width=3, alpha=0.5, edge_color="tab:orange" ) # 4
    
    # 绘制节点编号
    nx.draw_networkx_labels(G, pos, font_size=14, font_color="whitesmoke")  # 18
    
    #nx.draw(G, pos, with_labels=True, node_size=700, font_size=10, font_weight='bold') #, picker=True, ax=ax
    #plt.title(pic_title)
    plt.tight_layout()
    plt.axis("off")   
    plt.show()
    
    return
    
    """
    # 绘制网络  
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)  
      
    # 事件处理函数  
    def on_pick(event):  
        node = event.artist  
        node_info = G.nodes[node.get_label()]['info']  
        print(f"Clicked on node {node.get_label()} with info: {node_info}")  
      
    # 为每个节点添加拾取事件  
    for node, attrs in G.nodes(data=True):  
        nx.draw_networkx_nodes(G, pos, nodelist=[node], ax=ax, node_color='lightblue', picker=True)  
        nx.draw_networkx_labels(G, pos, labels={node: node}, ax=ax)  
      
    # 连接事件处理函数  
    fig.canvas.mpl_connect('pick_event', on_pick)  
      
    # 显示图形  
    plt.show()  """
    
    
    """
    # Create a 2x2 subplot
    fig, all_axes = plt.subplots(2, 2)
    ax = all_axes.flat    
    
    #G = net_creat.creat_grid_based_graph(32, 5)  # 4x4 grid
    G = net_creat.creat_Net('TREE', 50, 4)
    # nodesList = G.nodes()
    # edgeList = G.edges()
    
    G_tree = G.copy()    # 只保留图中树相关的边，用于获取最佳的可视化布局
    edgeList = list()
    for u, v in G_tree.edges():
        if G_tree[u][v]['tree']=='non-tree':
            edgeList.append((u, v))
            G_tree.remove_edge(u, v)
            
    pos = nx.spring_layout(G_tree, iterations=100, seed=3, scale=100)      # 获取树形结构布局
    nx.draw(G_tree, pos, ax=ax[0], node_size=0, edgelist=edgeList, edge_color="tab:gray", with_labels=False)
    nx.draw(G_tree, pos, ax=ax[0], node_size=40, with_labels=False, font_size=8)   """
    

# ——————————————————————————————————————
# 快速调试入口

if __name__ == '__main__':
    
    print("网络可视化的方法调试开始 !\n") 
    
    # Create a 2x2 subplot
    #fig, all_axes = plt.subplots(2, 2)
    #ax = all_axes.flat    
    
    #G = net_creat.creat_grid_based_graph(32, 5)  # 4x4 grid
    G = net_creat.creat_Net('TREE', 50, 4)
    # nodesList = G.nodes()
    # edgeList = G.edges()
    
           
    pos = nx.kamada_kawai_layout(G)      # 获取树形结构布局
    nx.draw(G, pos, node_size=100, with_labels=False)  
     
    #nx.draw(G, pos, ax=ax[1], node_size=0, edge_color="tab:gray",  with_labels=False, font_size=8)

    #nx.draw(G, pos, ax=ax[2], node_size=60, edgelist=edgeList, edge_color="tab:gray", with_labels=False)
    #nx.draw(G_tree, pos, ax=ax[3], node_size=60, with_labels=False, font_size=8)
    
    # 4x4 grid, iterations=100, seed=1, node_size=150 
    # 5x5 grid, iterations=100, seed=1, node_size=120 
    # 6x6 grid, iterations=100, seed=4, node_size=80 
    # 7x7 grid, iterations=100, seed=13, node_size=60  
    # 8x8 grid, iterations=100, seed=100, node_size=50
    # 9x9 grid, iterations=100, seed=11100, node_size=45
    # 10x10 grid, iterations=100, seed=111009, node_size=35
    
    #G = net_creat.creat_grid_based_graph(25, 4)  # 4x4 grid
    #G = net_creat.creat_Net('ER', 50, 4)

    # Set margins for the axes so that nodes aren't clipped
    #for a in ax:
    #    a.margins(0.10)
    #fig.tight_layout()
    plt.show()

"""    
    G = nx.cubical_graph()
    
    # 为seed参数提供特定的值，只要图G保持不变，生成的布局也将保持不变
    pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes 
    # 为seed参数提供特定的值，只要图G保持不变，生成的布局也将保持不变

    # nodes
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    nx.draw_networkx_nodes(G, pos, nodelist=[0, 1, 2, 3], node_color="tab:red", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=[4, 5, 6, 7], node_color="tab:blue", **options)

    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_edges(
                            G,
                            pos,
                            edgelist=[(0, 1), (1, 2), (2, 3), (3, 0)],
                            width=8,
                            alpha=0.5,
                            edge_color="tab:red",
                            )
    nx.draw_networkx_edges(
                            G,
                            pos,
                            edgelist=[(4, 5), (5, 6), (6, 7), (7, 4)],
                            width=8,
                            alpha=0.5,
                            edge_color="tab:blue",
                            )
    
    # some math labels
    labels = {}
    labels[0] = r"$a$"
    labels[1] = r"$b$"
    labels[2] = r"$c$"
    labels[3] = r"$d$"
    labels[4] = r"$\alpha$"
    labels[5] = r"$\beta$"
    labels[6] = r"$\gamma$"
    labels[7] = r"$\delta$"
    nx.draw_networkx_labels(G, pos, labels, font_size=22, font_color="whitesmoke")

    plt.tight_layout()
    plt.axis("off")
    plt.show()   """

""" 
NetworkX中的draw函数是用于绘制图形的一个便捷函数，它基于Matplotlib库来实现图形的可视化。draw函数提供了多个参数来定制绘制图形的样式和布局。以下是一些常用的draw函数参数及其含义：

基本参数
G：要绘制的图形对象，必须是NetworkX中的一个图对象。
pos（可选，默认值为None）：节点的位置布局。可以是一个以节点为键、以位置（通常是(x, y)坐标元组）为值的字典，也可以是NetworkX提供的布局算法（如spring_layout, circular_layout等）返回的位置字典。如果不指定，则默认使用spring_layout算法计算节点位置。
样式参数
node_size（可选，默认值为300）：节点的大小。可以是一个标量值，表示所有节点的大小相同；也可以是一个数组，其长度应与节点列表相同，表示每个节点的大小不同。
node_color（可选，默认值为单一颜色字符串或颜色代码）：节点的颜色。可以是一个颜色字符串（如'r'表示红色），也可以是一个颜色数组，对每个节点进行着色。
node_shape（可选，默认值为'o'）：节点的形状。Matplotlib支持的标记类型均可使用，如's'（正方形）、'^'（三角形）等。
alpha（可选，默认值为1.0）：节点和边的透明度。0表示完全透明，1表示完全不透明。
width（可选，默认值为1.0）：边的宽度。可以是一个标量值，表示所有边的宽度相同；也可以是一个与边列表长度相同的数组，表示每条边的宽度不同。
edge_color（可选，默认值为'k'（黑色））：边的颜色。可以是一个颜色字符串，也可以是一个颜色数组，对每条边进行着色。
style（可选，默认值为'solid'）：边的线条样式。可选值包括'solid'（实线）、'dashed'（虚线）、'dotted'（点线）和'dashdot'（点划线）。
标签和文本参数
with_labels（可选，默认值为True）：是否在节点上绘制标签。
font_size（可选，默认值为12）：节点标签的字体大小。
font_color（可选，默认值为'k'（黑色））：节点标签的字体颜色。
其他参数
ax（可选）：Matplotlib坐标轴对象。如果指定了这个参数，图形将在指定的坐标轴中绘制。
nodelist（可选，默认值为G.nodes()）：只绘制指定的节点。
edgelist（可选，默认值为G.edges()）：只绘制指定的边。
需要注意的是，draw函数虽然方便，但它在某些情况下可能不够灵活。NetworkX还提供了draw_networkx、draw_networkx_nodes、draw_networkx_edges等更细粒度的绘图函数，允许用户更精确地控制绘图过程。

draw_networkx_nodes 是 NetworkX 库中用于绘制图形节点的一个函数。这个函数允许用户自定义节点的绘制样式，包括节点的大小、颜色、形状等。以下是对 draw_networkx_nodes 函数参数的详细解释：

基本参数
G：要绘制的图对象，必须是 NetworkX 库中的一个图实例。
pos：节点位置字典，其中键是节点，值是对应的二维坐标（如(x, y)）。这个参数定义了每个节点在画布上的位置。
样式参数
nodelist（可选）：一个节点列表，指定要绘制的节点。如果未指定，则默认绘制图中的所有节点。
node_size（可选）：节点的大小。可以是一个标量值，表示所有节点的大小相同；也可以是一个列表或数组，其长度应与 nodelist 的长度相同，表示每个节点的大小不同。默认大小可能因 NetworkX 版本而异，但通常是一个较大的值，如 300。
node_color（可选）：节点的颜色。可以是一个颜色字符串（如 'r' 表示红色）、颜色代码（如 '#FF0000' 表示红色）或一个颜色列表/数组，用于指定每个节点的颜色。默认颜色可能是红色或其他颜色，具体取决于 NetworkX 的版本和配置。
node_shape（可选）：节点的形状。Matplotlib 支持的标记类型均可使用，如 'o'（圆形）、's'（正方形）、'^'（三角形）等。默认形状是圆形。
alpha（可选）：节点的透明度。值范围从 0（完全透明）到 1（完全不透明）。默认值通常是 1.0。
cmap（可选）：Matplotlib 的颜色映射（colormap），用于将节点属性映射到颜色上。如果指定了此参数，则 node_color 应该是一个与节点属性相对应的数值列表/数组。
vmin, vmax（可选）：与 cmap 参数一起使用，分别指定颜色映射的最小值和最大值。
linewidths（可选）：节点边界的线宽。如果未指定，则默认线宽可能取决于 Matplotlib 的配置。
label（可选）：节点标签（注意：这个参数在标准的 draw_networkx_nodes 函数中可能不存在，这里可能是对类似功能的通用描述）。通常，节点标签是通过 draw_networkx_labels 函数单独绘制的。
其他参数
ax（可选）：Matplotlib 的坐标轴对象。如果指定了这个参数，节点将在指定的坐标轴上绘制。这对于在复杂布局中绘制多个图形特别有用。
****kwds**：其他关键字参数，这些参数将直接传递给 Matplotlib 的绘图函数，以允许进一步的自定义。

nx.draw_networkx_edges 是 NetworkX 库中用于绘制图形边的函数。这个函数允许用户自定义边的绘制样式，包括边的颜色、宽度、线条样式等。以下是对 draw_networkx_edges 函数参数的详细解释：

基本参数
G：要绘制的图对象，必须是 NetworkX 库中的一个图实例。
pos：节点位置字典，其中键是节点，值是对应的二维坐标（如 (x, y)）。这个参数定义了每个节点在画布上的位置。
样式参数
edgelist（可选）：一个边的列表，指定要绘制的边。如果未指定，则默认绘制图中的所有边。
width（可选）：边的宽度。可以是一个标量值，表示所有边的宽度相同；也可以是一个列表或数组，其长度应与 edgelist 的长度相同，表示每条边的宽度不同。
edge_color（可选）：边的颜色。可以是一个颜色字符串（如 'r' 表示红色）、颜色代码（如 '#FF0000' 表示红色）或一个颜色列表/数组，用于指定每条边的颜色。
style（可选）：边的线条样式。可选值包括 'solid'（实线）、'dashed'（虚线）、'dotted'（点线）和 'dashdot'（点划线）。
alpha（可选）：边的透明度。值范围从 0（完全透明）到 1（完全不透明）。
edge_cmap（可选）：Matplotlib 的颜色映射（colormap），用于将边属性映射到颜色上。如果指定了此参数，则 edge_color 应该是一个与边属性相对应的数值列表/数组。
edge_vmin, edge_vmax（可选）：与 edge_cmap 参数一起使用，分别指定颜色映射的最小值和最大值。
arrows（可选）：布尔值或字典，指定是否在边上绘制箭头。如果为 True，则在所有边上绘制箭头；如果为 False，则不绘制箭头；如果为字典，则可以指定某些边上的箭头样式。
arrowstyle（可选）：箭头的样式，如 '-|>' 表示带有直线和大于符号的箭头。仅当 arrows 为 True 或字典时才有效。
arrowsize（可选）：箭头的大小。仅当 arrows 为 True 或字典时才有效。
connectionstyle（可选）：边的连接样式，如 'arc3,rad=0.5' 表示使用半径为 0.5 的圆弧连接节点。
其他参数
ax（可选）：Matplotlib 的坐标轴对象。如果指定了这个参数，边将在指定的坐标轴上绘制。
****kwds**：其他关键字参数，这些参数将直接传递给 Matplotlib 的绘图函数，以允许进一步的自定义。

一些常见的networkx布局及其简要描述：

spring_layout：
以弹簧模型为基础，模拟节点间的斥力和引力，尝试达到平衡状态。
适用于较小规模的网络，可以生成较为美观的布局。
G（图）：这是spring_layout函数的主要参数，表示要进行布局的图。图应该是NetworkX库中的一个图对象。
k（可选，默认值为None）：这个参数控制弹簧系统的“劲度系数”，影响节点间的排斥力。如果设置为None，则根据图的节点数自动计算一个合适的值。较大的k值会使节点分布得更加分散。
pos（可选，默认值为None）：这个参数允许用户提供一个初始位置字典，字典的键是节点，值是该节点的(x, y)坐标。如果提供了初始位置，算法将尝试从这些位置开始布局。
fixed（可选，默认值为None）：这个参数接受一个节点集合或字典，指定哪些节点在布局过程中应该保持固定位置不变。如果是一个集合，则集合中的节点位置将保持不变；如果是一个字典，则字典的键是节点，值是该节点的固定(x, y)坐标。
iterations（可选，默认值为50）：这个参数控制布局算法迭代的次数。增加迭代次数可能会使布局更加稳定，但也会增加计算时间。
threshold（可选，默认值为0.0001）：这个参数是收敛阈值，用于判断布局算法是否收敛。如果节点位置的变化小于这个阈值，则认为算法已经收敛，停止迭代。
weight（可选，默认值为'weight'）：这个参数指定图中边的权重属性的名称。算法将根据边的权重来调整节点间的吸引力。如果图中没有权重属性，或者不想考虑权重，可以将其设置为None。
scale（可选，默认值为1.0）：这个参数用于缩放整个布局。乘以一个小于1的值会使布局更加紧凑，乘以一个大于1的值会使布局更加分散。
center（可选，默认值为(0.0, 0.0)）：这个参数指定布局的中心点。算法将尝试将布局的中心移动到这个指定的位置。
dim（可选，默认值为2）：这个参数指定布局的维度。虽然spring_layout主要用于二维布局，但也可以设置为其他值（尽管NetworkX可能不支持非二维布局）。
seed（可选，默认值为None）：这个参数设置随机数生成器的种子。使用相同的种子值可以确保每次运行布局算法时得到相同的结果。

circular_layout：
将节点均匀放置在圆周上。
适用于需要强调网络环形结构的场景。
G（图）：
这是circular_layout函数的主要参数，表示要进行布局的图。图应该是NetworkX库中的一个图对象。
scale（可选，默认值为1.0）：
这个参数用于缩放整个布局。乘以一个小于1的值会使布局更加紧凑，乘以一个大于1的值会使布局更加分散。这有助于根据图形的显示需求调整节点之间的间距。
center（可选，默认值为(0.0, 0.0)）：
这个参数指定布局的中心点。算法将尝试将布局的中心移动到这个指定的位置。这有助于将图形放置在画布上的特定位置，以便与其他图形或元素对齐。
dim（可选，默认值为2）：
这个参数指定布局的维度。虽然circular_layout主要用于二维布局，但也可以设置为其他值（尽管NetworkX可能不支持非二维布局）。在大多数情况下，这个参数可以保持默认值。

random_layout：
随机分配节点的位置。
适用于对网络布局无特殊要求的情况。
shell_layout：
将节点放置在同心圆上，通常用于展示具有层次结构的网络。
spectral_layout：
基于图的拉普拉斯特征向量的布局算法。
适用于需要揭示网络谱性质的场景。
planar_layout：
尝试将图以平面形式绘制，减少边的交叉。
适用于需要清晰展示网络平面结构的场景。
kamada_kawai_layout：
通过优化节点间的距离来模拟弹簧模型，达到全局能量最小化。
适用于需要较为均匀布局的网络。
fruchterman_reingold_layout：
另一种基于弹簧模型的布局算法，尝试通过迭代达到力的平衡。
适用于大多数网络，特别是节点间相互作用较强的情况。
reingold_tilford_layout：
一种树形布局算法，适用于具有明显层次或树形结构的网络。
graphviz_layout（包括dot、neato、fdp、sfdp、twopi、circo）：
利用Graphviz软件包中的布局算法。
适用于需要高级布局优化的场景。

一些常用的颜色名称和它们在 networkx 中的使用示例：

基本颜色名称：
'red'
'blue'
'green'
'yellow'
'cyan'
'magenta'
'black'
'white'
更多颜色名称：
'orange'
'purple'
'brown'
'gray' 或 'grey'
'pink'
'lime'
十六进制颜色代码：
'#FF0000'（红色）
'#00FF00'（绿色）
'#0000FF'（蓝色）
RGB元组：
(255, 0, 0)（红色）
(0, 255, 0)（绿色）
(0, 0, 255)（蓝色）

"""

"""
import networkx as nx  
import matplotlib.pyplot as plt  
  
# 创建一个图并添加一些节点和边  
G = nx.Graph()  
G.add_node('A', info='Node A')  
G.add_node('B', info='Node B')  
G.add_edge('A', 'B')  
  
# 布局  
pos = nx.spring_layout(G)  
  
# 绘制网络  
fig, ax = plt.subplots()  
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)  
  
# 事件处理函数  
def on_pick(event):  
    node = event.artist  
    node_info = G.nodes[node.get_label()]['info']  
    print(f"Clicked on node {node.get_label()} with info: {node_info}")  
  
# 为每个节点添加拾取事件  
for node, attrs in G.nodes(data=True):  
    nx.draw_networkx_nodes(G, pos, nodelist=[node], ax=ax, node_color='lightblue', picker=True)  
    nx.draw_networkx_labels(G, pos, labels={node: node}, ax=ax)  
  
# 连接事件处理函数  
fig.canvas.mpl_connect('pick_event', on_pick)  
  
# 显示图形  
plt.show()

人机交互，获取节点属性信息
"""

"""
import networkx as nx  
import matplotlib.pyplot as plt  
from matplotlib.backend_bases import MouseButton  
  
# 创建一个图并添加一些节点和边  
G = nx.Graph()  
G.add_node('A')  
G.add_node('B')  
G.add_edge('A', 'B')  
  
# 初始布局  
pos = nx.spring_layout(G)  
  
# 绘制网络  
fig, ax = plt.subplots()  
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax, node_size=700)  
  
# 用于存储拖拽状态的变量  
dragging = {}  
drag_start = None  
  
# 事件处理函数  
def on_button_press(event):  
    if event.button is MouseButton.LEFT:  
        # 检查是否点击了节点  
        for node, attrs in G.nodes(data=True):  
            node_pos = pos[node]  
            if event.xdata is not None and event.ydata is not None:  
                distance = ((event.xdata - node_pos[0]) ** 2 + (event.ydata - node_pos[1]) ** 2) ** 0.5  
                if distance < 0.1:  # 0.1是节点的点击半径  
                    dragging[node] = (event.xdata, event.ydata)  
                    drag_start = node_pos  
                    break  
  
def on_button_release(event):  
    if event.button is MouseButton.LEFT:  
        dragging.clear()  
        drag_start = None  
  
def on_mouse_move(event):  
    if dragging:  
        node, drag_pos = dragging.popitem()  
        if drag_start:  
            delta_x = event.xdata - drag_pos[0]  
            delta_y = event.ydata - drag_pos[1]  
            pos[node] = (pos[node][0] + delta_x, pos[node][1] + delta_y)  
            nx.draw_networkx_nodes(G, pos, nodelist=[node], ax=ax, node_color='lightblue', node_size=700)  
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=G.edges(), edge_color='gray')  
            fig.canvas.draw_idle()  
            dragging[node] = (event.xdata, event.ydata)  
  
# 连接事件处理函数  
fig.canvas.mpl_connect('button_press_event', on_button_press)  
fig.canvas.mpl_connect('button_release_event', on_button_release)  
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)  
  
# 显示图形  
plt.show()

网络可视化及度分布等信息
https://networkx.org/documentation/latest/auto_examples/drawing/plot_degree.html

Grid可视化
import matplotlib.pyplot as plt
import networkx as nx

G = nx.grid_2d_graph(4, 4)  # 4x4 grid

pos = nx.spring_layout(G, iterations=100, seed=39775)

# Create a 2x2 subplot
fig, all_axes = plt.subplots(2, 2)
ax = all_axes.flat

nx.draw(G, pos, ax=ax[0], font_size=8)
nx.draw(G, pos, ax=ax[1], node_size=0, with_labels=False)
nx.draw(
    G,
    pos,
    ax=ax[2],
    node_color="tab:green",
    edgecolors="tab:gray",  # Node surface color
    edge_color="tab:gray",  # Color of graph edges
    node_size=250,
    with_labels=False,
    width=6,
)
H = G.to_directed()
nx.draw(
    H,
    pos,
    ax=ax[3],
    node_color="tab:orange",
    node_size=20,
    with_labels=False,
    arrowsize=10,
    width=2,
)

# Set margins for the axes so that nodes aren't clipped
for a in ax:
    a.margins(0.10)
fig.tight_layout()
plt.show()

"""
