"""

This is a Python script (PGG_A) for evolution gaming simulation of PGG

based on single network including static BA, WS, ER, REG, TREE(202407), FAM(202407)

and dynamic BA, WS, ER

Last revised on Aug 2024

@author: Mrh, Dan, Lty

"""

import datetime
import math
import networkx as nx   # 导入建网络模型包，命名
import os
import numpy as np
import random as rd
import multiprocessing
from multiprocessing import Pool
import logging
import time
from collections import Counter

from gaming_models import calc_C_num, calc_profit_PGG, game_stra_learn, game_stra_learn_withPrefer

from net_creat import  creat_Net

from write2excel import excel_creat_with_sheetS, excel_write_line, save_nets_statistics, save_gaming_result_statistics
    
from net_save_load import saveNet

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
file=time.strftime('%y%m%d',time.localtime(time.time()))
logging.basicConfig(filename='PGG'+file+'-static.log', level=logging.DEBUG, format=LOG_FORMAT)

# ——————————————————————————————————————
# 以下参数由main_single_net统一设置，全局统一
dynamic_flag = 'S'  # 网络是否动态的标志，S代表静态、D代表动态
del_way = 'R'       # 网络衰减节点选取规则（R随机、S适者生存）
inc_stra = 'R'      # 新增节点策略确定方式（R随机、L学习邻居）

netType = ''       # 'BA'无标度, 'WS'小世界, 'ER'随机, 'REG'网格, 'TREE'树形, 'FAM'家庭
k = 0 
ws_P = 0.1

KP = 0.5  
numCreate = 10 
numBoyi = 10 

rootDirectory = ''
# ——————————————————————————————————————

pC = 1    # 合作者向PGG游戏投入的资源份数
mag = 100

#单次博弈迭代次数
numLoop = 2000  # 20
group_iter_num = 100 # 5

#学习的子网比例或要删除的节点比例
subpct = 0.1

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

# ——————————————————————————————————————
# Dan创立 Lty修改
# 初始化网络中节点的策略、收益和生存回合数
# 在博弈的最初阶段和更新一个子网的构型时，才会用到此方法
# 返回随机选择策略后的合作者的数量
# ——————————————————————————————————————

def init_game_strategy( net ):
    
    nx.set_node_attributes(net, 0, 'profit')         # 初始收益均为0
    nx.set_node_attributes(net, 0, 'survival_time')  # 初始生存回合数均为0
    nx.set_node_attributes(net, 0, 'preference')     # 个体决策的倾向性
    
    iniC = 0      #初始合作者为0
    for node in net.nodes():
        status = rd.sample(range(2),1)[0]     #0表示背叛，1表示合作
        net.nodes[node]['select']=int(status) #记录每个节点的策略
        if status == 1:
            iniC +=1         #记录合作者个数
    
    """
    生成指定总数量且按等箱分布的随机数序列
    
    """
    # 计算每个箱子应分配的样本量（处理不能整除的情况）
    num_bins = 10
    total_count = nx.number_of_nodes( net )
    base_per_bin = total_count // num_bins
    remainder = total_count % num_bins  # 剩余样本分配给前remainder个箱子
    
    # 计算每个箱子的区间
    bin_edges = np.linspace(0, 1.0, num_bins + 1)  # 箱子边界min_val=0, max_val=1.0
    
    # 生成随机数
    random_numbers = []
    for i in range(num_bins):
        # 确定当前箱子的样本量
        count = base_per_bin + (1 if i < remainder else 0)
        
        # 箱子的起止范围
        start = bin_edges[i]
        end = bin_edges[i + 1]
        
        # 在当前箱子内生成均匀随机数
        bin_samples = np.random.uniform(low=start, high=end, size=count)
        random_numbers.extend(bin_samples)
    
    np.random.shuffle(random_numbers)
    
    # 随机给节点指定倾向性
    ii = 0
    for node in net.nodes():
        net.nodes[node]['preference'] = random_numbers[ii]
        ii += 1

    return iniC

# 返回groupNum次博弈结果的平均值，状态1；
# 如果某次博弈合作者比例为0或1，则直接返回，状态0。
    
def group_iterations(net, groupNum, r, alpha=0):
    
    numV = nx.number_of_nodes(net)
    pctC_ = 0
    
    for j in range(groupNum):
        calc_profit_PGG(net, r)

        if alpha==0:
            game_stra_learn(net, r)
        else:
            game_stra_learn_withPrefer(net, r, alpha, 1-abs(alpha))
        
        countC = calc_C_num(net)
        pctC_ += countC/numV
        
        # 如果当前回合合作者比例为0，则不需要再次演化
        if countC==0 or countC==numV:
            return [0, np.round(countC/numV, 4), j+1]

    return [1, np.round(pctC_/groupNum, 4), j+1]


# ——————————————————————————————————————
# Dan创建，lty修订
# 节点之间互相博弈直到整体合作者比例稳定
# 前2000次为连续5次合作者比例稳定终止博弈，超过2000次则按照100次博弈合作者比例均值判断稳定性
# 最大博弈次数设为10000
# 202408 增加了合作者比例一旦为0或为1，直接结束博弈的判断
# 思考博弈稳定的判定条件，即阈值0.001或允许至少一个节点改变策略
# 
# pctC, i   返回博弈稳定时合作者比例、及博弈次数
# ——————————————————————————————————————

def gaming_iterations(net, r, alpha=0):
    stableNum = 0
    numVTotal = nx.number_of_nodes(net)
    pctC = round(calc_C_num(net)/numVTotal, 4)  # 合作者比例
    stab_p = max(1.001/numVTotal, 0.001)            # 针对小型网络，仅1个节点的策略变化，视为整体稳定
  
    # 迭代numLoop次，或者节点状态稳定后停止博弈过程
    for i in range(numLoop):
        #if i>0 and i%10==0:
        #    draw_small_WS_net(net, netType+' net+N'+str(numVTotal)+' iterated'+str(i)+' times visualization')

        # 一次完整的学习，学习过程中策略会改变，学完要重新计算收益
        calc_profit_PGG(net, r)
        
        if alpha==0:
            game_stra_learn(net, r)
        else:
            game_stra_learn_withPrefer(net, r, alpha, 1-abs(alpha))
      
        # 计算当期轮次学习完成后的合作者比例
        curPctC = round(calc_C_num(net) / numVTotal, 4)
        
        # 如果当前回合合作者比例为0或全为合作者，不需要再次演化
        if curPctC==0 or curPctC==1.0:
            pctC = curPctC     
            break
           
        # 博弈稳定条件判定
        if abs(curPctC - pctC) <= stab_p:
            pctC = curPctC # 将当期比例作为下一次比较基础，允许合作者比例的微小调整
            stableNum += 1
            
            # 连续四次博弈，合作者比例变化度小于0.001，即稳定
            if stableNum>=4:
                break
        else:            
            pctC = curPctC
            stableNum = 0

    # 达到最大迭代次数前收敛，返回单次稳定合作者比例
    # 达到最大迭代次数依然没有收敛，采用新的稳定状态判定策略，每group_iter_num次迭代的平均值连续两次稳定，即稳定
    stab_p = max(1.001/numVTotal, 0.005)            # 针对小型网络，仅1个节点的策略变化，视为群组稳定
    if i>=numLoop-1:
        groupPctC_ = 0        
        groupStableNum = 0
        
        # 总迭代次数不超过8000次
        for j in range(79):  
            group_result = group_iterations(net, group_iter_num, r, alpha)
            groupPctCTemp = group_result[1]
            i += group_result[2]
            if group_result[0]==0:
                break           
            
            if abs(groupPctCTemp - groupPctC_)<stab_p:
                groupPctC_ = groupPctCTemp
                groupStableNum += 1
                if groupStableNum>=2:  # 稳定
                    break
            else:
                groupPctC_ = groupPctCTemp
                groupStableNum = 0
        
        pctC = groupPctCTemp
        
    # 更新各节点生存回合数
    for nodeLabel in net.nodes():
        net.nodes[nodeLabel]['survival_time'] += i+1
        
    return pctC, i+1


# ——————————————————————————————————————
# Dan 创立 2024.07 lty 修改
# 单一静态网络的博弈演化程序
# 
# 返回['r', 'Net_ID', 'D_time', 'Net_Size', 'Pro_ID', 'Iter_Times', 'Initial_C', 'S_C_Ratio']
# ——————————————————————————————————————
"""    # 计算初始化策略后网络中合作者的整体拓扑情况
    C_topology_stat = ave_clustering_coefficient_of_cooperators(net)
    
    C_list = list()
    for nodeLabel in net.nodes:
        if net.nodes[nodeLabel]['select'] == 1:
            C_list.append(nodeLabel)
    rd.shuffle(C_list)    
    bfs_shortest_path(net, C_list[0], C_list[1])   """
    
def gaming_on_static_net(net, r, net_loop, process_id, excel_name, alpha): #, lock
    
    # 初始化每个节点的策略 合作者数量
    numV = len(net.nodes())
    
    # 尝试不同的合作者初始化策略、个体倾向性
    iniC = init_game_strategy(net)
    #iniC = init_strategy_with_degree_preference(net, 0.1)
    
    #draw_small_WS_net(net, netType+' net'+ str(net_loop)+'+N'+str(numV)+' initial state visualization')
    #draw_small_gaming_net(net, netType, netType+' network-initialization')
     
    # 计算节点间学习概率平均分布
    calc_profit_PGG(net, r)
    
    pctC, iterTimes = gaming_iterations(net, r, alpha)
    
    gaming_stat = [r, alpha, 'N'+str(net_loop), '0', numV, process_id, iterTimes, round(iniC/numV, 4), pctC] #+ C_topology_stat
    #with lock:
    #    excel_write_line(excel_name, process_id, 0, gaming_stat)      # process_id从1计数，跳过了首行

    return gaming_stat



# ——————————————————————————————————————
# 初始化生成numNet个网络，每个网络节点个数为numV,平均度为k，同时保存这些网络，并随机生成每个网络的坐标
# loop为批次，主要用于确定保存网络的文件名称
# 返回网络的类型列表，网络，网络的坐标
# numNet暂时为1
# ——————————————————————————————————————

def initial_single_net( netType, k, numV, directory, loop ):
    
    # 创建网络
    net = creat_Net( netType, numV, k, ws_P )
    
    # 真实网络规模不遵从预设值
    if 'REAL' in netType:
        numV = nx.number_of_nodes(net)
        k = round(nx.number_of_edges(net)/nx.number_of_nodes(net), 2)
    
    # 转换网络中节点编号，避免分别生成网络时节点编号重复
    # net = convert_node_labels_to_integers(net, first_label=start, edge_attribute='ori')
    
    ws_Node_Seq = list()
    if netType=="WS":
        net_file = 'Sin_' + dynamic_flag +'0_' + str(netType) + '_P'+str(ws_P)+ '_N' + str(numV) + '+K' + str(k) + '_' + str(loop) + '.net'  
        for i in net.nodes():
            ws_Node_Seq.append(i)
    else:
        net_file = 'Sin_' + dynamic_flag +'0_' + str(netType)+'_N'+str(len(net.nodes()))+'+K'+str(k)+'_'+str(loop)+'.net'
    
    if not 'REAL' in netType:
        saveNet( directory, net, netType, net_file )

    return net, ws_Node_Seq



# ——————————————————————————————————————
# lty 创立
# 根据输入，设置博弈和网络演化全局参数，避免函数传参过多
# ——————————————————————————————————————

def set_global_para(netType_i, DynamicNum_arr, del_way_i, inc_stra_i, KP_i, \
                    k_i, ws_P_i):
    
    global netType
    netType = netType_i
    
    global dynamic_flag
    if len(DynamicNum_arr)==1 or 'REAL' in netType_i:
        dynamic_flag = 'S'
    else:
        dynamic_flag = 'D'   
        
    global del_way
    del_way = del_way_i
    global inc_stra
    inc_stra = inc_stra_i
    
    global k
    k = k_i 
    global ws_P
    ws_P = ws_P_i
    
    global KP
    KP = KP_i  
    
    return


# ——————————————————————————————————————
# lty 创立
# 创建保存各类结果的根目录
# ——————————————————————————————————————

def creat_output_root_folder( numV, start, end, interval ):

    if netType == "WS":
        rootDirectory = netType + '+P' + str(ws_P) + '_N' + str(numV) + '+K'+str(k) \
                +'_PGG+r'+str(start)+'to'+str(end)+'by'+str(interval)+'_'+str(datetime.date.today())  #+'_'+str(numCreate)+'x'+str(numBoyi)
    
    elif netType == "REG":
        if k==4:
            L = round( math.sqrt(numV) )
        elif k==5:
            L = round( math.sqrt(numV/2) )
        elif k==6:
            L = round( math.pow(numV, 1/3) )            
        rootDirectory = netType + '_L' + str(L) + '+K' + str(k) + '_PGG+r'+str(start) \
                +'to'+str(end)+'by'+str(interval)+'_'+str(datetime.date.today()) #+'_'+str(numCreate)+'x'+str(numBoyi)
    else:
        rootDirectory = netType + '_N' + str(numV) + '+K' + str(k) + '_PGG+r'+str(start) \
                +'to'+str(end)+'by'+str(interval)+'_'+str(datetime.date.today())  #+'_'+str(numCreate)+'x'+str(numBoyi)
    
    if dynamic_flag=='S':
        rootDirectory = 'Sig_' + dynamic_flag + '_' + rootDirectory
    else:
        rootDirectory = 'Sig_' + dynamic_flag + '+' + del_way + '+' + inc_stra \
            + '_' + rootDirectory
    
    return rootDirectory

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

"""
# ——————————————————————————————————————
# 单一网络生成和博弈演化的主程序
# 
# netType 'BA'无标度, 'WS'小世界, 'ER'随机, 'REG'网格, 'TREE'树形, 'FAM'家庭
# 
"""

def main_single_net(netType_i, DynamicNum_arr, del_way_i, inc_stra_i, KP_i, k_i, ws_P_i, \
                    r_range, numthread, numCreate=10, numBoyi=10, alpha_range=[0, 0, 0]):   #lock, 
    
    # 全局变量赋值更新
    set_global_para(netType_i, DynamicNum_arr, del_way_i, inc_stra_i, KP_i, k_i, ws_P_i)
    
    numV = DynamicNum_arr[0]  
    smoothed_sizes = smoothing_net_size_changes( DynamicNum_arr )  # 静态网络时smoothed_sizes只记录初始规模
    
    # 创建保存实验结果的根目录、网络目录
    global rootDirectory
    rootDirectory = creat_output_root_folder( numV, r_range[0], r_range[1], r_range[2] )
    if not os.path.exists(rootDirectory):
        os.makedirs(rootDirectory)
    netDirectory = rootDirectory+'/networks'
    if not os.path.exists(netDirectory):
        os.makedirs(netDirectory) 

    # 创建博弈结果汇总excel    
    excel_name = rootDirectory+'/'+netType+'_'+str(k)+'_result_Sum_'+str(datetime.date.today())+'.xls'
    excel_creat_with_sheetS(excel_name, ['Game_Result_Detail', 'stat_sum', 'stat_change', 'mesh_grid', 'learnP_Dis'] )
    collumns_title = ['r', 'alpha_G', 'Net_ID', 'D_time', 'Net_Size', 'Pro_ID', 'Iter_Times', 'Initial_C', dynamic_flag+'_C_Ratio', \
                      'CC_of_Cooperators', '合作者聚集在邻居整体的情况', '合作者间聚集的情况']
    excel_write_line(excel_name, 0, 0, collumns_title)   # 写入表头从第1行、第1列开始
    result_list = list()                # 收取网络博弈演化结果
    
    # 创建网络统计结果汇总excel
    net_excel_name = rootDirectory+'/'+netType+'_'+str(k)+'_net_statistics_'+str(datetime.date.today())+'.xls'
    excel_creat_with_sheetS(net_excel_name, ['stat_detail', 'stat_sum', 'stat_change'] )
    collumns_title = ['Net_ID', 'D_Times', 'Net_size', 'ave_degree', 'ave_diameter', 'ave_shortest_path', 'ave_clustering', 'degree_distributions' ]
    excel_write_line(net_excel_name, 0, 0, collumns_title)   # 写入表头从第1行、第1列开始
    
    logging.info('\n________________________________\n')
    logging.info('netType:{0}'.format(netType))    
    logging.info('Net evolution in {0} way'.format(dynamic_flag))
    logging.info('Number of nodes:{0}'.format(numV))
    logging.info('Average degree k:{0}'.format(k))
    logging.info('The start, end and interval value of r are: {0}, {1}, {2}'.format(r_range[0], r_range[1], r_range[2]))    
    if netType=="WS":
        logging.info('The probability of the small-world network reconnecting edges is:{0}'.format(ws_P))
    
    m = multiprocessing.Manager()
    lock = m.Lock()
    
    # ————————————————————————————————————————————————————————————————————————
    # 网络生成及博弈演化
    all_net_detail = list()
    dynamic_Net_List = list()
    
    pool = Pool(processes=numthread)
    process_id = 0
    learnP_dis = list()
    learnP_dis_withPre = list()
    
    for net_loop in range(numCreate):
        
        # 初始化网络
        net, ws_Node_Seq = initial_single_net(netType, k, numV, netDirectory, net_loop)
        
        if 'REAL' in netType:
            smoothed_sizes = [nx.number_of_nodes(net)]
            numV = nx.number_of_nodes(net)  
        maxLabel = numV-1  # 记录累计出现过的节点数量-1，以确定新加节点标记
        
        dynamic_Net_List = [ net ]
        
        # 统计并输出全部网络的特征
        for d_time in range(len(dynamic_Net_List)):
            net_stat = cal_net_statistics( net_loop, d_time, dynamic_Net_List[d_time])
            excel_write_line(net_excel_name, net_loop+1, 0, net_stat)
            all_net_detail.append( net_stat )
 
        for r in np.arange(r_range[0]*mag, r_range[1]*mag+1, r_range[2]*mag):            
            for alpha in np.arange(alpha_range[0]*mag, alpha_range[1]*mag+1, alpha_range[2]*mag): 
                for game_it in range(numBoyi):
                    process_id += 1
                    print('\nProcess {0} starts with r={1}.'.format(process_id, r/mag))
                    logging.info( '\nProcess {0} starts with r={1}.'.format(process_id, r/mag) )
    
                    # 进行博弈演化
                    if dynamic_flag == 'S':
                        game_net = dynamic_Net_List[0].copy()
                        
                        # 在alpha启用时，计算节点间学习概率平均分布
                        #if alpha_range[1]>0:
                            #init_game_strategy(net)
                            #draw_small_gaming_net(net, netType, netType+' network-initialization')
                            #calc_profit_PGG(net, r/mag)
                        
                            # [[概率分布],..., [概率分布]]，分别记录c-c、d-c、c-d、d-d节点对间的学习概率分布
                            #learnP_dis.append( [r/mag] + calc_learnP_distribution(net, r/mag, 0, 1.0) )
                            #learnP_dis_withPre.append( [r/mag] + calc_learnP_distribution(net, r/mag, 0.5, 0.5) )
                        
                        #result_list.append(gaming_on_static_net(game_net, r/mag, net_loop, process_id, excel_name, alpha/mag))   #, lock
                        result_list.append(pool.apply_async(gaming_on_static_net, \
                                                               (game_net, r/mag, net_loop, process_id, excel_name, alpha/mag)) ) #, lock
                       
    pool.close()
    pool.join()

    # 输出全部网络的特征
    print("\n输出全部网络的统计特征!!")
    if dynamic_flag=='D' and (del_way=='S' or inc_stra=='L') :  # 网络动态演化生成的结果输出 
        for i in range(len(result_list)):
            cur_result = result_list[i].get()      #   
            for j in range(len(cur_result)):
                net_stat = cur_result[j][1]
                if len(net_stat)>0:
                    all_net_detail.append(net_stat)
    # 在网络变化次数太多时，不输出明细结果，标志位设为0
    save_nets_statistics( net_excel_name, netType, k, smoothed_sizes, all_net_detail, 0)      
    
    # 全部演化结果汇总输出
    print("\n输出全部网络博弈演化结果的统计特征!!")
    r_list = list()
    for r in np.arange(r_range[0]*mag, r_range[1]*mag+1, r_range[2]*mag):
        r_list += [r/mag]
    alpha_list = list()
    for alpha in np.arange(alpha_range[0]*mag, alpha_range[1]*mag+1, alpha_range[2]*mag): 
        alpha_list += [alpha/mag]
    
    result_convert = list()
    if dynamic_flag=='D' and (del_way=='S'or inc_stra=='L') :  # 网络动态演化生成的结果输出
        for i in range(len(result_list)):
            cur_result = result_list[i].get()     #   
            for j in range(len(cur_result)):
                result_convert.append( cur_result[j][0] )
    else:     
        # 其他情况下，需要从多进程结果中提取为可输出样式
        for j in range(len(result_list)):     # 多进程结果采用get获取.get()result in result_list
            cur_result = result_list[j].get()        #
            if dynamic_flag=='D':
                for i in range(len(cur_result)):
                    result_convert.append( cur_result[i] )
            else:
                result_convert.append( cur_result )
    result_stat = save_gaming_result_statistics( excel_name, netType, k, r_list, alpha_list, smoothed_sizes, result_convert)  
    
    # 在alpha启用时，输出alpha_G和r关系的热力图
    if alpha_range[1]>0:
        
        excel_write_line( excel_name, 0, 0, ['alpha_G/r'] + r_list, 3)       # 输出r取值列表，第一格为行列信息说明
        excel_x = 1  
        
        for alpha in alpha_list:
            curline = [alpha]
            
            # 从第2行开始写每个alpha取值对应的平均合作率，其中第1列为对应alpha值
            for r in r_list:
                found = 0
                for cur_result in result_stat:
                    # [ r, alpha, D_time, net_size, mean(ave_C), variance(ave_C), mean(ave_iter), variance(ave_iter) ]
                    if round(cur_result[0], 1)==round(r, 1) and round(cur_result[1], 4)==round(alpha, 4):
                        curline += [ str(round(cur_result[4], 4)) ]
                        found = 1
                        break
                    
                if found==0:     # 解决有些情况未能获取对应合作率的情况
                    curline += [ '--' ]
                            
            excel_write_line(excel_name, excel_x, 0, curline, 3)
            excel_x += 1
    
    return 

# ——————————————————————————————————————
# 快速调试入口

if __name__ == '__main__':
         
    multiprocessing.freeze_support()
         
    date_time_1 = datetime.datetime.now()
    print("Gaming Started at " + str(date_time_1) + "!!\n")
   
    sizes = [100]    # 30, 60, 30
    r_range = [1, 5, 0.1]
    alpha_range = [0, 0.5, 0.01]
    numthread = min(4, int(numCreate*numBoyi*(r_range[1]-r_range[0])/r_range[2]+1)) #5  
    
    # 真实网络：'Female', 'Male', 'Nyangatom'
    for netType in ['ER']: #   'REAL-Female', 'REAL-Male', 'BA''REAL-Nyangatom''BA', , 'WS', 'TREE', 'FAM-ER','REG' 
        
        k = 4
        numBoyi = 20
        numCreate = 10
        
        if netType=='REG':     #  and dynamic_flag=='S'格子网无随机性，静态情况下只生成一个网络即可
            numBoyi = numBoyi*10  
            numCreate = 1
            if k<4:
                k = 4
                
        elif 'REAL' in netType:
            numBoyi = numBoyi*10  
            numCreate = 1
            
        elif netType=='WS':
            k = 4
        
        while k>=4 and k<=6:
            #if netType=='ER' or netType=='BA' or (netType=='WS' and k%4==0):
                #for size in sizes:
            
            # del_way R随机、S适者生存，inc_stra R随机、L学习邻居
            main_single_net(netType, sizes, 'R', 'R', 0.5, k, 0.1, r_range, numthread, numCreate, numBoyi, alpha_range) 
            
            k += 1

    date_time_2 = datetime.datetime.now()
    print("Gaming Ended at " + str(date_time_2) + "!!\n")   

"""    # _________________________________________________________________
    # 博弈收益测试
    ws_net, ba_net, inter_net = creat_profit_test_Net() 
    calc_profit_PGG(ws_net, 3)   # 人工演算：0收益6.7，7收益2.3
    
    #set_Rand_Stra_2_NewNode(ws_net, 3, 20, add_nodes=[2, 3, 4], add_nodes_life=[1])
    learn_Stra_from_old_neighbor(ws_net, 3, 20, add_nodes=[2, 3, 4], add_nodes_life=[1])
    
    print('节点收益检查')
    for node in ws_net.nodes():
        print(node, ws_net.nodes[node]['select'], ws_net.nodes[node]['profit'])
    draw_small_gaming_net(ws_net, 'WS', 'WS test network')
        
    game_stra_learn( ws_net ) 
    
    draw_small_gaming_net(ws_net, 'WS', 'WS test network')
    
    calc_profit_PGG(ba_net, 3)   # 人工演算：19收益2.5，16收益6.3
    for node in ba_net.nodes():
        print(node, ba_net.nodes[node]['select'], ba_net.nodes[node]['profit'])
    
    game_stra_learn( ba_net ) 
    draw_small_gaming_net(ba_net, 'BA', 'BA test network')
    
    # _________________________________________________________________  
    # 网络演变异常处理机制测试
    
    ws_net, ba_net, inter_net = creat_profit_test_Net() 
    calc_profit_PGG(ba_net, 3)   # 人工演算：0收益6.7，7收益2.3
    game_stra_learn( ba_net )
    
    BA_grow(ba_net, 4, 3, 20, None, None)
    
    #set_Rand_Stra_2_NewNode(ws_net, 3, 20, add_nodes=[2, 3, 4], add_nodes_life=[1])
    learn_Stra_from_old_neighbor(ba_net, 3, 20, 0.5, add_nodes=[22, 23, 4], add_nodes_life=[1])
    
    
    """    