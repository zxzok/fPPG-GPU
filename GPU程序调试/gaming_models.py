"""

This is a Python script of different gaming models for evolution gaming simulation

including Public Goods Game (PGG) till 202408, 

plan to add Prison Dillema (PD), Hawk-Dove Game, Snowdrfit Game

Last revised on Aug 2024

@author: Mrh, Dan, Lty

"""

import datetime
import random as rd
import networkx as nx
import logging
import math
import pandas as pd
import numpy as np
import itertools

import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt


from net_creat import  creat_profit_test_Net
from net_visualization import draw_small_gaming_net

pC = 1    # 合作者向PGG游戏投入的资源份数
KP = 0.5  # 学习概率


#计算一个网络中的合作者比例
def calc_C_num(net):
    
    countC = 0
    for node in net.nodes():
        if net.nodes[node]['select']==1:
            countC +=1
            
    return countC


# ______________________________________
# 设置网络中节点在博弈演化中的倾向性
# 'B' 背叛倾向，一直背叛，直至被惩罚才调整策略，生成概率betray_P
# 'C' 合作倾向，一直合作，生成概率cooperate_P
# 'N' 无倾向

def set_gaming_role( net, cooperate_P, betray_P ):
    
    if cooperate_P<0 or betray_P<0:
        print('背叛或合作倾向概率不能小于0！')
        return
    
    nx.set_node_attributes(net, 'N', 'role')         #初始节点均设为无倾向
    count_C = 0
    count_B = 0
    for node in net.nodes():
        pp = rd.random()
        if pp<cooperate_P:
            net.nodes[node]['role']='C' #记录每个节点的博弈角色
            count_C += 1
        elif (pp-cooperate_P)<betray_P:
            net.nodes[node]['role']='B' #记录每个节点的博弈角色
            count_B += 1
  
    return [count_C, count_B, nx.number_of_nodes(net)-count_C-count_B]

# ——————————————————————————————————————
# Lty创立 
# 存在博弈倾向性的情况下，初始化网络中节点的策略、收益和生存回合数
# 返回随机选择策略后的合作者的数量
# ——————————————————————————————————————

def init_game_strategy_withRole( net ):
    
    nx.set_node_attributes(net, 0, 'profit')         #初始收益均为0
    nx.set_node_attributes(net, 0, 'survival_time')  #初始生存回合数均为0
    
    iniC = 0      #初始合作者为0
    for node in net.nodes():
        
        if net.nodes[node]['role']=='C':
            net.nodes[node]['select'] = 1
            iniC +=1
        elif net.nodes[node]['role']=='B':
            net.nodes[node]['select'] = 0
        else:
            status = rd.sample(range(2),1)[0]     #0表示背叛，1表示合作
            net.nodes[node]['select']=int(status) #记录每个节点的策略
            if status == 1:
                iniC +=1         #记录合作者个数

    return iniC

# ——————————————————————————————————————
# Dan创立 Lty修改
# 初始化网络中节点的策略、收益和生存回合数
# 在博弈的最初阶段和更新一个子网的构型时，才会用到此方法
# 返回随机选择策略后的合作者的数量
# ——————————————————————————————————————

def init_game_strategy( net ):
    
    nx.set_node_attributes(net, 0, 'profit')         #初始收益均为0
    nx.set_node_attributes(net, 0, 'survival_time')  #初始生存回合数均为0
    
    iniC = 0      #初始合作者为0
    for node in net.nodes():
        status = rd.sample(range(2),1)[0]     #0表示背叛，1表示合作
        net.nodes[node]['select']=int(status) #记录每个节点的策略
        if status == 1:
            iniC +=1         #记录合作者个数

    return iniC

# ——————————————————————————————————————
# 整个网络所有节点收益计算    公共品博弈(PGG)
# r为公共资源投资收益系数
# ——————————————————————————————————————

def calc_profit_PGG(net, r):

    # 清空收益
    for node in net.nodes():
        net.nodes[node]['profit'] = 0

    for node in net.nodes():
        neighbors = list(net.neighbors(node))
        play_in_PGG( net, node, neighbors, r)
    
    return

# ——————————————————————————————————————
# 计算单个节点node与内部邻居neighbors博弈的收益和，采用公共品博弈
# 返回当前节点在本轮PGG中的收益
# 遍历节点：
#        计算邻居的合作者数量
#        计算该组每人应得的收益
#        节点收益=节点收益+新收益
#        遍历邻居：
#        每个邻居节点=原收益+新受益
# ——————————————————————————————————————

def play_in_PGG(net, node, neighbors, r):

    numCN = 0 # 组内合作者数量
    numN = 1+len(neighbors)  # 组内总人数
    if numN==1:
        print('报错！邻居节点数量为0！')  
        
    # 当迭代器被提供给for循环时，最后一个StopIteration将导致它第一次退出，而试图在另一个for循环中使用相同的迭代器将立即再次导致StopIteration，因为迭代器已经被使用。
    # 解决这个问题的一个简单方法是将所有元素保存到一个列表中，可以根据需要多次遍历该列表。
    #neighbors1, neighbors2 = itertools.tee(neighbors, 2)

    if net.nodes[node]['select']==1:
        numCN = 1
    for neighbor in neighbors:
        if net.nodes[neighbor]['select']==1:
            numCN += 1
    
    # 群体合作收益与合作数量有关的情况
    #if numCN>0:
    #    r = r + round(np.log10(numCN), 4)
    
    profitALL = r * pC * numCN
    ProfitAverage = profitALL/numN

    # 分配收益
    if net.nodes[node]['select'] == 1:
        net.nodes[node]['profit'] += (ProfitAverage-pC)
    else:
        net.nodes[node]['profit'] += ProfitAverage
    
    #if node==17:
    #    print(node, (ProfitAverage-pC), net.nodes[node]['profit'])

    for neighbor in neighbors:
        # print('neighbor节点为{0}'.format(neighbor))
        
        if net.nodes[neighbor]['select']==1:
            net.nodes[neighbor]['profit'] += (ProfitAverage-pC)
        else:
            net.nodes[neighbor]['profit'] += ProfitAverage
            
        #if neighbor==17:
        #    print(node, neighbor, (ProfitAverage-pC), net.nodes[neighbor]['profit'])
     
    return ProfitAverage


# ——————————————————————————————————————
# Dan创建、lty修改
# 网络中节点与邻居比较收益后，选择一个节点学习其策略
# 通过调整学习对象选择策略，提高演化稳定速度
# ——————————————————————————————————————

def game_stra_learn( net, r=2 ):
       
    # 记录各节点本轮博弈稳定后的策略，避免策略更新先后对学习的影响
    netlast = nx.get_node_attributes(net, 'select')

    for node in net:
        fnode = net.nodes[node]['profit']
        neighborList = list(nx.all_neighbors(net, node)) 

        # 随机选出一个邻居比较收益
        if len(neighborList)>=1:          ####### 是否带=号，是否影响结果
            
            neip = rd.randint(0, len(neighborList)-1)
            studynei = neighborList[neip]
            if 'profit' not in net.nodes[studynei].keys():
                #print('profit为空节点:{0}'.format(studynei))
                logging.error('节点{0}的profit为空！'.format(studynei))
            
            # 获取邻居节点收益
            fnei = net.nodes[studynei]['profit']
            
            # 计算学习概率
            pp = learning_Pr_sigmod (fnode, fnei)
            
            # 给相应边赋值为学习概率
            net[node][studynei]['study_p']=( pp )
            
            # 进行学习，生成一个随机数，更新策略，一旦学习则终止当前节点的策略更新
            found = rd.sample(list(np.arange(0, 1, 0.00001)), 1)
            if (found[0] < pp):
                net.nodes[node]['select'] = netlast[studynei]    
    
    return


# ——————————————————————————————————————
# lty创建
# 网络中节点与邻居比较收益时，考虑政策导向的影响
# alpha_G代表政策导向（一般为集体主义倾向，alpha_G>0；如为个人主义倾向alpha_G<0），alpha_I为个体利益导向
# ——————————————————————————————————————

def game_stra_learn_withPrefer( net, r=2, alpha_G=0, alpha_I=1.0 ):
       
    # 记录各节点本轮博弈稳定后的策略，避免策略更新先后对学习的影响
    netlast = nx.get_node_attributes(net, 'select')
    
    calc_learn_Probabilty(net, r, alpha_G, alpha_I)
        
    #calc_learnP_distribution(net, r, alpha_G, alpha_I)            

    for node in net:
        neighborList = list(nx.all_neighbors(net, node)) 

        # 随机选出一个邻居比较收益
        if len(neighborList)>=1:          ####### 是否带=号，是否影响结果
            
            neip = rd.randint(0, len(neighborList)-1)
            studynei = neighborList[neip]        
            
            # 给相应边赋值为学习概率
            if net[node][studynei]['study_p'][0][0]==node:
                pp = net[node][studynei]['study_p'][0][2]
            else:
                pp = net[node][studynei]['study_p'][1][2]
            
            # 进行学习，生成一个随机数，更新策略，一旦学习则终止当前节点的策略更新
            found = rd.sample(list(np.arange(0, 1, 0.00001)), 1)
            if (found[0] < pp):
                net.nodes[node]['select'] = netlast[studynei]    
    
    return

"""
        # 遍历邻居，比较收益后，以一定概率学习
        nei_candidate = list()
        nei_candidate_p = list()
        for nei in neighborList:
            
            # 出现过一次无profit的节点，但是有一定随机性，后来跑几次，没有复现，应该是程序哪块有小问题
            if 'profit' not in net.nodes[nei].keys():
                print('profit为空节点:{0}'.format(nei))
                logging.error('profit为空节点:{0}'.format(nei))  
            
            fnei = net.nodes[nei]['profit']
            
            # 为使概率小于1，分母设为：邻居节点周围均为合作者但自身不付出（收益约为k*r），节点自身只付出无收益(收益-1)
            # p = (fnei - fnode)/(r*net.degree[nei]+1)    
            p = learning_Pr_sigmod (fnode, fnei)
            if p>0:
                if p>1:
                    p = 1
                p = np.around(p, 4)
                nei_candidate.append(nei)
                nei_candidate_p.append(p)

        # 以一定概率向每个学习概率大于0的邻居学习
        sump = sum( nei_candidate_p )
        
        # 如果总概率和大于1，则归一化
        if sump > 1:
            nei_candidate_p = list( np.divide(nei_candidate_p, sump) )
            sump = 1
            
        found = rd.sample( list(np.arange(0, 1, 0.01)), 1 )    #    从可学习的邻居中选择学习对象
              
        # 选出需要学习的邻居
        if found[0]<sump:
            for i in range(len(nei_candidate_p)):
                if found[0]<nei_candidate_p[i]:
                    studynei = nei_candidate[i]
                    net.nodes[node]['select'] = netlast[studynei]    
                    break
                else:
                    nei_candidate_p[i+1] += nei_candidate_p[i]      
        
        """


'''
# ——————————————————————————————————————
# 根据两个节点收益，测算节点间学习概率，采用Sigmod方法
    
'''

def learning_Pr_sigmod (fnode, fnei):

    # 计算学习概率，采用Sigmod方法
    if fnode - fnei <100:
        p = 1 / (1 + pow(np.around(math.e,4), np.around(((fnode - fnei) / KP), 4) ))
    else:
        p = 0
            
    pp = np.around(p, 4) # 如果自身收益太高会出现近0极小数，此时学习概率通过四舍五入置为0
    
    return pp

# ——————————————————————————————————————
# 计算网络中节点间相互学习概率
# 由于节点状态的影响，两端点互相学习的概率不相等，
# 分别计入每条边的['study_p']属性，其格式为[[edge[0], edge[1], '学习概率'], [edge[0], edge[1], '学习概率']]属性

def calc_learn_Probabilty(net, r, alpha_G, alpha_I):
    
    # 记录各节点本轮博弈稳定后的策略
    netlast = nx.get_node_attributes(net, 'select')
      
    # 遍历每条边，计算其学习概率
    for edge in net.edges():
        
        study_p = list()
        
        # 1、计算基于个体利益的学习概率，由edge[0]向edge[1]学习的概率
        pp_I = learning_Pr_sigmod (net.nodes[edge[0]]['profit'], net.nodes[edge[1]]['profit'])
        
        neighborList = list(nx.all_neighbors(net, edge[0]))
        
        if alpha_G>=0:
            fmax = (net.degree[edge[0]]+1)*r
        elif alpha_G<0:
            fmax = net.degree[node]*r/(net.degree[edge[0]]+1)
            for nei in neighborList:
                fmax += net.degree[nei]*r/(net.degree[nei]+1)
                   
        # 计算政策导向影响的学习概率，并考虑个体接受政策导向的倾向性
        pp_G = learning_Pr_sigmod (net.nodes[edge[0]]['profit'], fmax) * net.nodes[edge[0]]['preference']
            
        # 计算综合学习概率，这与政策导向方向和目标节点策略是否吻合有关，相同则叠加，不同则相减
        pp = -2
        if alpha_G>=0 and netlast[edge[1]]==1:    # 政策导向为集体主义、目标节点策略合作
            pp = alpha_G*pp_G + alpha_I*pp_I
        elif  alpha_G>=0 and netlast[edge[1]]==0:    # 政策导向为集体主义、目标节点策略背叛
            pp = - alpha_G*pp_G + alpha_I*pp_I
        elif alpha_G<0 and netlast[edge[1]]==1:    # 政策导向为个人主义、目标节点策略合作
            pp = alpha_G*pp_G + alpha_I*pp_I
        elif  alpha_G<0 and netlast[edge[1]]==0:    # 政策导向为个人主义、目标节点策略背叛
            pp = - alpha_G*pp_G + alpha_I*pp_I   
        
        if pp<-1 or pp>1:
            print('学习概率计算差错！')
        
        # 给相应边赋值为学习概率
        study_p.append( [ edge[0], edge[1], pp ] )
        
        # 2、计算基于个体利益的学习概率，由edge[1]向edge[0]学习的概率
        pp_I = learning_Pr_sigmod (net.nodes[edge[1]]['profit'], net.nodes[edge[0]]['profit'])
        
        neighborList = list(nx.all_neighbors(net, edge[1]))
        
        if alpha_G>=0:
            fmax = (net.degree[edge[1]]+1)*r
        elif alpha_G<0:
            fmax = net.degree[node]*r/(net.degree[edge[1]]+1)
            for nei in neighborList:
                fmax += net.degree[nei]*r/(net.degree[nei]+1)
                   
        # 计算政策导向影响的学习概率，并考虑个体接受政策导向的倾向性
        pp_G = learning_Pr_sigmod (net.nodes[edge[1]]['profit'], fmax) * net.nodes[edge[1]]['preference']
            
        # 计算综合学习概率，这与政策导向方向和目标节点策略是否吻合有关，相同则叠加，不同则相减
        pp = -2
        if alpha_G>=0 and netlast[edge[0]]==1:    # 政策导向为集体主义、目标节点策略合作
            pp = alpha_G*pp_G + alpha_I*pp_I
        elif  alpha_G>=0 and netlast[edge[0]]==0:    # 政策导向为集体主义、目标节点策略背叛
            pp = - alpha_G*pp_G + alpha_I*pp_I
        elif alpha_G<0 and netlast[edge[0]]==1:    # 政策导向为个人主义、目标节点策略合作
            pp = alpha_G*pp_G + alpha_I*pp_I
        elif  alpha_G<0 and netlast[edge[0]]==0:    # 政策导向为个人主义、目标节点策略背叛
            pp = - alpha_G*pp_G + alpha_I*pp_I     
        
        if pp<-1 or pp>1:
            print('学习概率计算差错！')
            
        # 给相应边赋值为学习概率
        study_p.append( [ edge[1], edge[0], pp ] )
        
        net.edges[edge[0], edge[1]]['study_p'] = study_p
        
    return

# ——————————————————————————————————————
# 计算一个网络中的合作者比例

def calc_C_num(net):
    
    countC = 0
    for node in net.nodes():
        if net.nodes[node]['select']==1:
            countC +=1
            
    return countC

# ——————————————————————————————————————
# 计算一个网络节点平均收益

def calc_ave_profit(net):
    
    sum_profit = 0
    for node in net.nodes():
        
        # 合作者比例为0或1时，进行特定情况检查
        #if(net.nodes[node]['profit']>0):
        #    print(net.nodes[node]['select'], net.nodes[node]['profit'])
        #    neighbors = nx.all_neighbors( net, node )
        #    for nei in neighbors:
        #        print(net.nodes[nei]['select'], net.nodes[nei]['profit'])
        
        sum_profit += net.nodes[node]['profit']
            
    return sum_profit/nx.number_of_nodes(net)


# ——————————————————————————————————————
# 快速调试入口

if __name__ == '__main__':
    
    #multiprocessing.freeze_support()
         
    date_time_1 = datetime.datetime.now()
    print("Gaming Started at " + str(date_time_1) + "!!\n")


    
    date_time_2 = datetime.datetime.now()
    print("Gaming Ended at " + str(date_time_2) + "!!\n")   
    
 