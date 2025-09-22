"""

This is a Python script (PGG_A) for output gaming result of PGG to Excel

Last revised on July 2024

@author: Dan, Lty

"""

import networkx as nx
import xlrd
import xlwt
#import openpyxl as pyxl    # 待采用xlsx文件输出格式再启用此包
import numpy as np
from xlutils.copy import copy
import statistics
import datetime

import logging
import math


""" —————————————————————————————————————— """

# Excel首行输出的样式
# 创建样式对象，设置文字居中  
style = xlwt.XFStyle() 
alignment = xlwt.Alignment()  
alignment.horz = xlwt.Alignment.HORZ_CENTER  # 水平居中  
alignment.vert = xlwt.Alignment.VERT_CENTER  # 垂直居中  
style.alignment = alignment
    
# 设置背景色为浅蓝色，这是xlwt中接近淡蓝色的一个选择  
pattern = xlwt.Pattern()  
pattern.pattern = xlwt.Pattern.SOLID_PATTERN  
pattern.pattern_fore_colour = 22            # 17 淡绿、22 灰 
style.pattern = pattern 

""" —————————————————————————————————————— """

# ——————————————————————————————————————
# Dan 创立 2024.07 lty 修改
# 创建一个具有制定页签的Excel
# excel_name   str   Excel名称
# sheet_name   str   页签名称
# ——————————————————————————————————————

def excel_creat(excel_name, sheet_name):
    
    workbook = xlwt.Workbook()  # 新建一个工作簿
    workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    workbook.save(excel_name)  # 保存工作簿
    
    # 获取第一行对象，并设置其高度，需要转成xlsx格式才能使用高级方法
    #workbook = pyxl.load_workbook(excel_name)
    #worksheet = workbook.active  
    #worksheet.row_dimensions[1].height = 18
    #worksheet.save(excel_name)
  
    return


# ——————————————————————————————————————
# Dan 创立 2024.07 lty 修改
# 创建一个具有制定页签的Excel
# excel_name   str   Excel名称
# sheet_name   str   页签名称
# ——————————————————————————————————————

def excel_creat_with_sheetS(excel_name, sheet_list):
    
    workbook = xlwt.Workbook()  # 新建一个工作簿
    for sheet_name in sheet_list:
        workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    workbook.save(excel_name)  # 保存工作簿
  
    return

"""
# ——————————————————————————————————————
# Dan 创立 2024.07 lty 修改
# 
# 向表格写入一行信息，当写入第一行时默认按照标题行处理
# 
# filename      str   文件名称
# lineNo        int   写入的行数
# colStartNo    int   写入的起始列数  
# columns_info  list  每个位置记录对应列的信息
# ——————————————————————————————————————"""

def excel_write_line(filename, lineNo, colStartNo, columns_info, sheet_num=0):
    
    # 行数太多，不写入结果
    if lineNo>65530:
        print('警告！写入行数超过Excel限值，不再写入！')
        #logging.warning('Result output terminated due to exeeding the line limit of Excel!')
        return
    
    workbook = xlrd.open_workbook(filename)  # 打开工作簿
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(sheet_num)  # 获取转化后工作簿中的第一个表格
    
    # 分列写入信息
    high_degree_count = 0
    for i in range(len(columns_info)):
        
        if lineNo==0:
            new_worksheet.write(lineNo, colStartNo+i, str(columns_info[i]), style)
        else:
            # Excel列数最多256，目前只有BA度分布存在此情况，超过255列时度分布值累加
            if i<245:
                new_worksheet.write(lineNo, colStartNo+i, str(columns_info[i]))
            else:
                high_degree_count += int(columns_info[i])
        
        # 对于第一行设置列宽为其字符数加2
        if lineNo==0:
            new_worksheet.col(colStartNo+i).width = (max(len(str(columns_info[i]))+2, 8))*256
            new_worksheet.col(colStartNo+i)
    
    if high_degree_count>0:
        new_worksheet.write(lineNo, 255, str(high_degree_count))
            
    new_workbook.save(filename)  # 保存结果文件
    
    return

# —————————————————————————————————————
# 输出群组博弈时，整体网络（geo_net）中不同网络构型的占位情况
# 返回写入Excel的明细信息，无标题

def save_geo_position( filename, geo_net, subnets, times, geo_title=None ):
    
    L = int(math.sqrt(nx.number_of_nodes(geo_net)))
    geo_info = []
    
    # 256//(L+2) 每行最多输出的整体网络数
    startLine = int(times//(256//(L+2)))   
    startCol = max(int(times%(256//(L+2))), 1)
    lineNo = 1 
    colNo = 1+(startCol-1)*(L+2)
    
    if lineNo+L>65535:
        print('输出行数超过Excel最大限度！')
        return
    
    # 在左上角写入标题
    if not geo_title==None:
        excel_write_line(filename, lineNo+startLine*(L+2)-1, colNo, [geo_title], 1)
    
    curLine = []
    ii = 0
    for node in geo_net.nodes():
        net_id = geo_net.nodes[node]['occupy_net']
        
        # 记满一行，写入文件
        if ii==L*lineNo:
            excel_write_line(filename, lineNo+startLine*(L+2), colNo, curLine, 1)
            geo_info += [curLine]
            lineNo += 1
            curLine = []
            
        if net_id==-1:   # 处理位置空缺的情况
            curLine += [ -1 ]
        else: 
            temp_id = []
            for subset in subnets:
                if subset[0]==net_id:
                    temp_id = [subset[1][0]+'_'+str(subset[1][len(subset[1])-1])] # 网络构型及其规模
                    break
            curLine += temp_id # 网络构型及其规模
                
            # 处理占位子网无对应明细信息的情况
            if len(temp_id)==0:
                print('错误！占位子网{0}无对应明细信息！'.format(net_id))
      
        ii += 1
        
    # 输出最后一行
    excel_write_line(filename, lineNo+startLine*(L+2), colNo, curLine, 1)
    geo_info += [curLine]
        
    return geo_info

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

def save_nets_statistics( excel_name, netType, k, smoothed_sizes, all_net_detail, log=1):

    """
    lty创建

    保存网络的统计特征，适用于单一静态和动态网络
    
    Parameters
    ----------
    smoothed_sizes : list
                    the smoothed sizes of each dynamic stage to meet the user-input target initial and ending sizes
                    the sizes are created by the method "smoothing_net_size_changes"
    
    all_net_detail : list
                    statistic details of all nets under different creating times, stages etc.
                    the details for each net are as follows: 
                    ['Net_ID', 'D_Times', 'Net_size', 'ave_degree', 'ave_diameter', 'ave_shortest_path', 'ave_clustering' ]

    """ 
    
    # 输出每一阶段网络的拓扑信息，从第二行开始，保留标题行
    if log==1:
        for excel_y in range(len(all_net_detail)):
            excel_write_line(excel_name, excel_y+1, 0, all_net_detail[excel_y])
    
    # ——————————————————————————————————————
    # 网络演变的拓扑规律分析
    
    # 统计不同规模网络的拓扑特征
    # 同规模度分布根据不同网络对应度数的节点出现概率取平均值
    collumns_title = ['Net_size', 'stat_degree', 'var_degree', 'stat_diameter', 'var_diameter', \
                      'stat_shortest_path', 'var_shortest_path', 'stat_clustering', 'var_clustering', 'ave_degree_distri' ]
    excel_write_line(excel_name, 0, 0, collumns_title, 1)   # 在第2页签写入表头从第1行、第1列开始
    
    net_stat = list()
    excel_y = 1    # 从第二行开始，保留标题行
    d_time = 0
    
    
    '''    
    # 由于特定原因，动态网络的规模可能不出现在指定动态列表里，进行检查并确定实际applied_size
    applied_sizes = []
    for i in range(len(all_net_detail)):
        if int(all_net_detail[i][2]) not in applied_sizes:
            applied_sizes += [int(all_net_detail[i][2])]
    
    # 如两者不符则按照实际网络规模输出
    if not sorted(applied_sizes)==sorted(smoothed_sizes):
        print( '\n网络规模变化列表{0}，\n与实际动态规模{1}不符.'.format(smoothed_sizes, applied_sizes) )
        logging.info( '\n网络规模变化列表{0}，\n与实际动态规模{1}不符.'.format(smoothed_sizes, applied_sizes) )
        smoothed_sizes = applied_sizes   '''
        
    for net_size in smoothed_sizes:
        
        if netType=='REG':
            L, layers = grid_netsize_to_edgelen( net_size, k )
            if k==4:
                net_size = L*L
            elif k==5:
                net_size = 2*L*L
            elif k==6:
                net_size = L*L*L
                
        net_count = 0  # 同规模网络数量
        ave_degree = list()
        ave_diameter = list()
        ave_shortest_path = list()
        ave_clustering = list()
        ave_degree_dis = [0]
        
        for i in range(len(all_net_detail)):
            if int(all_net_detail[i][2])==net_size and int(all_net_detail[i][1])==d_time:
                net_count += 1
                
                ave_degree.append( float(all_net_detail[i][3]) )
                if float(all_net_detail[i][4])>0:
                    ave_diameter.append( float(all_net_detail[i][4]) )
                    ave_shortest_path.append( float(all_net_detail[i][5]) )
                ave_clustering.append( float(all_net_detail[i][6]) )
                
                # 统计不同度值的节点数
                degree_dis = all_net_detail[i][7: ]
                j = 0
                for j in range(min(len(ave_degree_dis), len(degree_dis))):
                    ave_degree_dis[j] += degree_dis[j]/net_size
                while j<len(degree_dis)-1:
                    j += 1
                    ave_degree_dis.append(degree_dis[j]/net_size)    
                        
        # 同规模网络数量太少（<=2），不统计方差均值
        if len(ave_diameter)<=2:
            continue
        else:
            net_stat = [net_size, round(statistics.mean(ave_degree), 4), round(statistics.variance(ave_degree), 4), \
                        round(statistics.mean(ave_diameter), 4), round(statistics.variance(ave_diameter), 4), \
                        round(statistics.mean(ave_shortest_path), 4), round(statistics.variance(ave_shortest_path), 4),\
                        round(statistics.mean(ave_clustering), 4), round(statistics.variance(ave_clustering), 4)    ]
            
            for j in range(len(ave_degree_dis)):
                ave_degree_dis[j] = round(ave_degree_dis[j]/net_count, 4)
        
            excel_write_line(excel_name, excel_y, 0, net_stat+ave_degree_dis, 1)
            excel_y += 1
        
        d_time += 1
                
    collumns_title = ['Net_size_change', 'degree_fluc', 'var_degree_fluc', 'diameter_fluc', 'var_diameter_fluc', \
                      'shortest_path_fluc', 'var_shortest_path_fluc', 'clustering_fluc', 'var_clustering_fluc' ]    
    excel_write_line(excel_name, 0, 0, collumns_title, 2)   # 在第3页签写入表头从第1行、第1列开始
    
    # 以d_time（处于演变的第几阶段）为主键梳理规模变化前后各项拓扑指标的变化情况
    # 避免演化过程中出现多次同等规模的情况
    d_time = 0
    pre_degree = list()
    pre_diameter = list()
    pre_shortest_path = list()
    pre_clustering = list()
    for i in range(len(all_net_detail)):
        if int(all_net_detail[i][1])==d_time:
            pre_degree.append( float(all_net_detail[i][3]) )
            if float(all_net_detail[i][4])>0:
                pre_diameter.append( float(all_net_detail[i][4]) )
                pre_shortest_path.append( float(all_net_detail[i][5]) )
            else:
                pre_diameter.append( 0.0 )
                pre_shortest_path.append( 0.0 )
            pre_clustering.append( float(all_net_detail[i][6]) )
    
    d_time = 1
    net_stat = list()
    while d_time<len(smoothed_sizes):
                       
        # 网络演变次数太少，不进行变化统计
        if len(smoothed_sizes)<=3:
            break
        
        cur_degree = list()
        cur_diameter = list()
        cur_shortest_path = list()
        cur_clustering = list()
        for i in range(len(all_net_detail)):
            if int(all_net_detail[i][1])==d_time:
                cur_degree.append( float(all_net_detail[i][3]) )
                if float(all_net_detail[i][4])>0:
                    cur_diameter.append( float(all_net_detail[i][4]) )
                    cur_shortest_path.append( float(all_net_detail[i][5]) )
                else:
                    cur_diameter.append( 0.0 )
                    cur_shortest_path.append( 0.0 )
                cur_clustering.append( float(all_net_detail[i][6]) )
        
        diff_degree = statistics.mean(cur_degree)-statistics.mean(pre_degree)
        var_degree = statistics.variance(cur_degree)-statistics.variance(pre_degree)
        pre_degree = cur_degree
        
        diff_clustering = statistics.mean(cur_clustering)-statistics.mean(pre_clustering)
        var_clustering = statistics.variance(cur_clustering)-statistics.variance(pre_clustering)
        pre_clustering = cur_clustering
        
        diff_diameter = statistics.mean(cur_diameter)-statistics.mean(pre_diameter)
        var_diameter = statistics.variance(cur_diameter)-statistics.variance(pre_diameter)
        pre_diameter = cur_diameter
        
        diff_shortest_path = statistics.mean(cur_shortest_path)-statistics.mean(pre_shortest_path)
        var_shortest_path = statistics.variance(cur_shortest_path)-statistics.variance(pre_shortest_path)
        pre_shortest_path = cur_shortest_path
        
        net_stat = [str(smoothed_sizes[d_time-1])+'to'+str(smoothed_sizes[d_time]), \
                        round(diff_degree, 4), round(var_degree, 4), \
                        round(diff_diameter, 4), round(var_diameter, 4), \
                        round(diff_shortest_path, 4), round(var_shortest_path, 4),\
                        round(diff_clustering, 4), round(var_clustering, 4)    ]
        excel_write_line(excel_name, d_time, 0, net_stat, 2)
        d_time += 1
    
    return

# ——————————————————————————————————————

def save_HE_nets_statistics( net_excel_name, netConfigs, smoothed_sizes, all_net_detail):

    """
    lty创建

    保存异质网络的统计特征
    ['Net_ID', 'r', 'pro_ID', 'D_Times', 'Net_type', 'Net_Seq', 'Net_size', 'ave_degree', 'ave_diameter', 'ave_shortest_path', 'ave_clustering' ]
    
    """ 
   
    # ——————————————————————————————————————
    # 网络演变的拓扑规律分析
    
    # 统计不同规模网络的拓扑特征
    # 同规模度分布根据不同网络对应度数的节点出现概率取平均值
    collumns_title = ['Net_type', 'Net_size', 'D_times', 'stat_degree', 'var_degree', 'stat_diameter', 'var_diameter', \
                      'stat_shortest_path', 'var_shortest_path', 'stat_clustering', 'var_clustering', 'ave_degree_distri' ]
    excel_write_line(net_excel_name, 0, 0, collumns_title, 1)   # 在第2页签写入表头从第1行、第1列开始
    
    net_stat = list()
    excel_y = 1    # 从第二行开始，保留标题行
    D_times = -1    
    net_types = list()
    for net_config in netConfigs:
        net_types.append(net_config[0])
    net_types.append('HE')
    
    
    '''
    # 由于特定原因，动态网络的规模可能不出现在指定动态列表里，进行检查并确定实际applied_size
    applied_sizes = []
    for i in range(len(all_net_detail)):
        if int(all_net_detail[i][6]) not in applied_sizes:
            applied_sizes += [int(all_net_detail[i][6])]
    
    # 如两者不符则按照实际网络规模输出
    if not sorted(applied_sizes)==sorted(smoothed_sizes):
        print( '\n网络规模变化列表{0}，\n与实际动态规模{1}不符.'.format(smoothed_sizes, applied_sizes) )
        logging.info( '\n网络规模变化列表{0}，\n与实际动态规模{1}不符.'.format(smoothed_sizes, applied_sizes) )
        smoothed_sizes = applied_sizes  '''
    
    for net_size in smoothed_sizes:
        D_times += 1
        net_seq = -1
        
        for net_type in net_types:
            net_count = 0  # 同类型、规模、演变阶段的网络数量
            net_seq += 1
            ave_degree = list()
            ave_diameter = list()
            ave_shortest_path = list()
            ave_clustering = list()
            ave_degree_dis = [0]
            if net_type=='HE':
                net_size = net_size*len(netConfigs)
            
            # 获取同类型、规模、演变阶段的网络
            # Net_ID r pro_ID	D_Times	Net_type Net_Seq	Net_size	ave_degree	ave_diameter	ave_shortest_path ave_clustering
            for i in range(len(all_net_detail)):
                for j in range(len(all_net_detail[i])):
                    cur_net_detail = all_net_detail[i][j]
                    if int(cur_net_detail[6])==net_size and cur_net_detail[4]==net_type \
                        and int(cur_net_detail[3])==D_times and int(cur_net_detail[5])==net_seq:
                        net_count += 1
                        
                        ave_degree.append( float(cur_net_detail[7]) )
                        if float(cur_net_detail[8])>0:
                            ave_diameter.append( float(cur_net_detail[8]) )
                            ave_shortest_path.append( float(cur_net_detail[9]) )
                        ave_clustering.append( float(cur_net_detail[10]) )
                        
                        # 统计不同度值的节点数
                        degree_dis = cur_net_detail[11: ]
                        ll = 0
                        for ll in range(min(len(ave_degree_dis), len(degree_dis))):
                            ave_degree_dis[ll] += degree_dis[ll]/net_size
                        while ll<len(degree_dis)-1:
                            ll += 1
                            ave_degree_dis.append(degree_dis[ll]/net_size)    
                            
            # 同规模网络数量太少（<=2），不统计方差均值
            if len(ave_diameter)<=2:
                continue
            else:
                net_stat = [net_type, net_size, D_times, round(statistics.mean(ave_degree), 4), round(statistics.variance(ave_degree), 4), \
                            round(statistics.mean(ave_diameter), 4), round(statistics.variance(ave_diameter), 4), \
                            round(statistics.mean(ave_shortest_path), 4), round(statistics.variance(ave_shortest_path), 4),\
                            round(statistics.mean(ave_clustering), 4), round(statistics.variance(ave_clustering), 4)    ]
                
                for j in range(len(ave_degree_dis)):
                    ave_degree_dis[j] = round(ave_degree_dis[j]/net_count, 4)
            
                excel_write_line(net_excel_name, excel_y, 0, net_stat+ave_degree_dis, 1)
                excel_y += 1
    
    # 以d_time（处于演变的第几阶段）为主键梳理规模变化前后各项拓扑指标的变化情况
    # ['Net_ID', 'r', 'pro_ID', 'D_Times', 'Net_type', Net_Seq, 'Net_size', 'ave_degree', 'ave_diameter', 'ave_shortest_path', 'ave_clustering' ]
    # 避免演化过程中出现多次同等规模的情况
    collumns_title = ['Net_type', 'Net_size_change', 'degree_fluc', 'var_degree_fluc', 'diameter_fluc', 'var_diameter_fluc', \
                      'shortest_path_fluc', 'var_shortest_path_fluc', 'clustering_fluc', 'var_clustering_fluc' ]    
    excel_write_line(net_excel_name, 0, 0, collumns_title, 2)   # 在第3页签写入表头从第1行、第1列开始
    
    excel_y = 1
    net_seq = -1
    for net_type in net_types:
    
        D_time = 0
        net_seq += 1
        pre_degree = list()
        pre_diameter = list()
        pre_shortest_path = list()
        pre_clustering = list()
        for i in range(len(all_net_detail)):
            for j in range(len(all_net_detail[i])):
                cur_net_detail = all_net_detail[i][j]
                if int(cur_net_detail[3])==D_time and cur_net_detail[4]==net_type and int(cur_net_detail[5])==net_seq:
                    pre_degree.append( float(cur_net_detail[7]) )
                    if float(cur_net_detail[8])>0:
                        pre_diameter.append( float(cur_net_detail[8]) )
                        pre_shortest_path.append( float(cur_net_detail[9]) )
                    else:
                        pre_diameter.append( 0.0 )
                        pre_shortest_path.append( 0.0 )
                    pre_clustering.append( float(cur_net_detail[10]) )
            
        D_time = 1
        net_stat = list()
        while D_time<len(smoothed_sizes):
                           
            # 网络演变次数太少，不进行变化统计
            if len(smoothed_sizes)<=3:
                break
            
            cur_degree = list()
            cur_diameter = list()
            cur_shortest_path = list()
            cur_clustering = list()
            for i in range(len(all_net_detail)):
                for j in range(len(all_net_detail[i])):
                    cur_net_detail = all_net_detail[i][j]
                    if int(cur_net_detail[3])==D_time and cur_net_detail[4]==net_type and int(cur_net_detail[5])==net_seq:
                        cur_degree.append( float(cur_net_detail[7]) )
                        if float(cur_net_detail[8])>0:
                            cur_diameter.append( float(cur_net_detail[8]) )
                            cur_shortest_path.append( float(cur_net_detail[9]) )
                        else:
                            cur_diameter.append( 0.0 )
                            cur_shortest_path.append( 0.0 )
                        cur_clustering.append( float(cur_net_detail[10]) )
            
            diff_degree = [cur_degree[i]-pre_degree[i] for i in range(len(pre_degree))]
            pre_degree = cur_degree
            diff_diameter = [cur_diameter[i]-pre_diameter[i] for i in range(len(pre_diameter))]
            pre_diameter = cur_diameter
            diff_shortest_path = [cur_shortest_path[i]-pre_shortest_path[i] for i in range(len(pre_shortest_path))]
            pre_shortest_path = cur_shortest_path
            diff_clustering = [cur_clustering[i]-pre_clustering[i] for i in range(len(pre_clustering))]
            pre_clustering = cur_clustering
            
            # 相同规模变化的网络数量太少（<=2），不统计方差均值
            if len(diff_diameter)<=2:
                break
            else:
                net_stat = [net_type, str(smoothed_sizes[D_time-1])+'to'+str(smoothed_sizes[D_time]), \
                            round(statistics.mean(diff_degree), 4), round(statistics.variance(diff_degree), 4), \
                            round(statistics.mean(diff_diameter), 4), round(statistics.variance(diff_diameter), 4), \
                            round(statistics.mean(diff_shortest_path), 4), round(statistics.variance(diff_shortest_path), 4),\
                            round(statistics.mean(diff_clustering), 4), round(statistics.variance(diff_clustering), 4)    ]
            
                excel_write_line(net_excel_name, excel_y, 0, net_stat, 2)
                D_time += 1
                excel_y += 1
    
    return

# ——————————————————————————————————————
# [r, alpha, 'N'+str(net_loop), '0', numV, process_id, iterTimes, round(iniC/numV, 4), pctC]

def save_gaming_result_statistics( filename, netType, k, r_list, alpha_list, smoothed_sizes, result_list):
       
    # ——————————————————————————————————————
    # 博弈演化结果的统计分析
    
    excel_y = 1    # 从第二行开始，保留标题行
    if len(result_list)<65535:
        for cur_result in result_list:     # 多进程结果采用get获取.get()
            excel_write_line(filename, excel_y, 0, cur_result)
            excel_y += 1

    # 统计不同规模网络的拓扑特征
    collumns_title = ['r', 'alpha', 'D_time', 'Net_size', 'ave_C_Ratio', 'var_C_Ratio', 'ave_iter_times', 'var_iter_times' ]    # 择机补充演化结果的集中度，即非1即0的情况
    excel_write_line(filename, 0, 0, collumns_title, 1)   # 在第2页签写入表头从第1行、第1列开始
    
    # ['r', 'Net_ID', 'D_time', 'Net_Size', 'Pro_ID', 'Iter_Times', 'Initial_C', dynamic_flag+'_C_Ratio']
    game_stat = list()
    game_stat_list = list()
    excel_y = 1    # 从第二行开始，保留标题行
    
    '''
    # 由于特定原因，动态网络的规模可能不出现在指定动态列表里，进行检查并确定实际applied_size
    applied_sizes = []
    for i in range(len(result_list)):
        if int(result_list[i][3]) not in applied_sizes:
            applied_sizes += [int(result_list[i][3])]
    
    # 如两者不符则按照实际网络规模输出
    if not sorted(applied_sizes)==sorted(smoothed_sizes):
        print( '\n网络规模变化列表{0}，\n与实际动态规模{1}不符.'.format(smoothed_sizes, applied_sizes) )
        logging.info( '\n网络规模变化列表{0}，\n与实际动态规模{1}不符.'.format(smoothed_sizes, applied_sizes) )
        smoothed_sizes = applied_sizes '''
    
    for r in r_list:    
        for alpha in alpha_list:  
            D_time = 0
            for net_size in smoothed_sizes:
                
                if netType=='REG':
                    L, layers = grid_netsize_to_edgelen( net_size, k )
                    if k==4:
                        net_size = L*L
                    elif k==5:
                        net_size = 2*L*L
                    elif k==6:
                        net_size = L*L*L
                
                ave_C = list()
                ave_iter = list()
                
                # [r, alpha, 'N'+str(net_loop), '0', numV, process_id, iterTimes, round(iniC/numV, 4), pctC]
                for cur_result in result_list:     # 多进程结果采用get获取.get()
                    if int(cur_result[4])==net_size and float(cur_result[0])==float(r) and float(cur_result[1])==float(alpha) and int(cur_result[3])==D_time:
                        ave_C.append( float(cur_result[8]) )
                        ave_iter.append( float(cur_result[6]) )
            
                # 同规模网络数量太少（<=2），不统计方差均值
                if len(ave_C)<=2:
                    break
                else:
                    game_stat = [ r, alpha, D_time, net_size, round(statistics.mean(ave_C), 4), round(statistics.variance(ave_C), 4), \
                                round(statistics.mean(ave_iter), 4), round(statistics.variance(ave_iter), 4) ]
                    game_stat_list.append(game_stat)
            
                excel_write_line(filename, excel_y, 0, game_stat, 1)
                excel_y += 1
                D_time += 1
    
    return game_stat_list

# ——————————————————————————————————————
#[r, C_P, B_P, found_P, punish_r, '', '0', numV, '', ave_played_pct, punished_num, i+1, iniC, pctC, r_count, stable_profit]

def save_punish_game_result( result_excel_name, netType, excel_x, stage_result_list):
    
    for cur_result in stage_result_list:  
        gameresult = [cur_result[0], cur_result[1], cur_result[2], cur_result[3], cur_result[4], netType, 'D'+'0', cur_result[7], \
                      round(cur_result[9]/cur_result[14], 4), round(cur_result[10]/cur_result[14], 4), \
                          round(cur_result[11]/cur_result[14], 4), round(cur_result[12]/cur_result[14], 4), \
                              round(cur_result[13]/cur_result[14], 4), round(cur_result[15]/cur_result[14], 4)]
        excel_write_line(result_excel_name, excel_x, 0, gameresult, 0)
        excel_x += 1
    
    return

    """
    

    Parameters
    ----------
    excel_name : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.
    netType : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    smoothed_sizes : TYPE
        DESCRIPTION.
    all_net_detail : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    """
    lty创建

    保存网络的博弈演化，适用于单一静态
    
    Parameters
    ----------
    smoothed_sizes : list
                    the smoothed sizes of each dynamic stage to meet the user-input target initial and ending sizes
                    the sizes are created by the method "smoothing_net_size_changes"
    
    all_net_detail : list
                    statistic details of all nets under different creating times, stages etc.
                    the details for each net are as follows: 
                    ['Net_ID', 'D_Times', 'Net_size', 'ave_degree', 'ave_diameter', 'ave_shortest_path', 'ave_clustering' ]

    """ 

""" 
# ——————————————————————————————————————
# 异质网博弈演化结果的统计分析
""" 

def save_HEgaming_result_stat( filename, netConfigs, r_list, smoothed_sizes, result_list):
       
    # 输出演化结果的统计
    collumns_title = ['r', 'Net_size', 'D_Times', 'Net_type', 'ave_C_Ratio', 'var_C_Ratio', 'ave_iter_times', 'var_iter_times' ]    # 择机补充演化结果的集中度，即非1即0的情况
    excel_write_line(filename, 0, 0, collumns_title, 1)   # 在第2页签写入表头从第1行、第1列开始
    
    # 输出netConfig信息，例如[['WS', 4, 0.1, 30], ['ER', 3, 30]]
    excel_write_line(filename, 0, len(collumns_title)+1, ['No.', 'netType', 'K', 'WS_P/ini_Size', 'ini_Size'], 1)
    for ii in range(len(netConfigs)):
        #with lock:
        # excel_write_line(filename, lineNo, colStartNo, columns_info, sheet_num=0)
        excel_write_line(filename, ii+1, len(collumns_title)+1, [str(ii)]+netConfigs[ii], 1)   
    
    # r	Net_ID	'D_Times'	Net_Size Pro_ID	Iter_Times	WS_Initial_C	WS_C_Ratio	ER_Initial_C	ER_C_Ratio	BA_Initial_C	BA_C_Ratio	
    # Initial_C	D_C_Ratio
    game_stat = list()
    excel_y = 1    # 从第二行开始，保留标题行
    
    '''
    # 由于特定原因，动态网络的规模可能不出现在指定动态列表里，进行检查并确定实际applied_size
    applied_sizes = []
    for i in range(len(result_list)):
        if int(result_list[i][3]) not in applied_sizes:
            applied_sizes += [int(result_list[i][3])]
    
    # 如两者不符则按照实际网络规模输出
    if not sorted(applied_sizes)==sorted(smoothed_sizes):
        print( '\n网络规模变化列表{0}，\n与实际动态规模{1}不符.'.format(smoothed_sizes, applied_sizes) )
        logging.info( '\n网络规模变化列表{0}，\n与实际动态规模{1}不符.'.format(smoothed_sizes, applied_sizes) )
        smoothed_sizes = applied_sizes '''
    
    for r in r_list:
        D_times = 0
        for net_size in smoothed_sizes:
            
            '''
            netType = netConfigs[ii]
            if netType=='REG':
                L, layers = grid_netsize_to_edgelen( net_size, k )
                if k==4:
                    net_size = L*L
                elif k==5:
                    net_size = 2*L*L
                elif k==6:
                    net_size = L*L*L '''
            
            ave_C = list()
            ave_iter = list()
            ave_sub_C = list()
            for i in range(len(netConfigs)):
                ave_sub_C.append(list())
        
            for result in result_list:     # 多进程结果采用get获取.get()
                cur_list = result    #.get() 
                # 演化结果获取
                if D_times==int(cur_list[2]) and cur_list[3]==str(net_size)+'*'+str(len(netConfigs)) and float(cur_list[0])==float(r):
                    ave_C.append( float(cur_list[5+len(netConfigs)*2+2]) )
                    ave_iter.append( float(cur_list[5]) )
                    for i in range(len(netConfigs)):
                        ave_sub_C[i].append( float(cur_list[5+2*i+2]) )
        
            # 同规模网络数量太少（<=2），不统计方差均值
            if len(ave_C)<=2:
                break
            else:
                # ['r', 'Net_size', 'D_Times', 'Net_type', 'ave_C_Ratio', 'var_C_Ratio', 'ave_iter_times', 'var_iter_times' ]
                game_stat = [ r, net_size, D_times, 'HE', round(statistics.mean(ave_C), 4), round(statistics.variance(ave_C), 4), \
                            round(statistics.mean(ave_iter), 4), round(statistics.variance(ave_iter), 4) ]
                excel_write_line(filename, excel_y, 0, game_stat, 1)
                excel_y += 1
                
                for i in range(len(netConfigs)):
                    game_stat = [ r, net_size, D_times, netConfigs[i][0]+'_'+str(i), round(statistics.mean(ave_sub_C[i]), 4), round(statistics.variance(ave_sub_C[i]), 4), \
                                round(statistics.mean(ave_iter), 4), round(statistics.variance(ave_iter), 4) ]
                    excel_write_line(filename, excel_y, 0, game_stat, 1)
                    excel_y += 1
                
            D_times += 1
    
    return

""" 
# ——————————————————————————————————————
# 群组博弈演化结果统计分析
# ['r', 'Net_ID', 'Pro_ID', 'Net_Size', 'Group_Iters', 'Iter_Times', ] + collumns[构型节点比例, 构型合作者比例] + ['Initial_C', 'D_C_Ratio']
""" 

def save_GRgaming_result_stat( filename, netConfigs, r_list, L, result_list):
    
    # 1. 输出群组博弈明细稳定结果
    # 获取网络类型名称和结果表头
    nets_type = netConfigs[0][0]
    collumns = [netConfigs[0][0]+'+'+str(0)+'_Size', netConfigs[0][0]+'+'+str(0)+'_C_Ratio']
    i = 1
    while i<len(netConfigs):
        nets_type += '+'+netConfigs[i][0]
        collumns += [netConfigs[i][0]+'+'+str(i)+'_Size', netConfigs[i][0]+'+'+str(i)+'_C_Ratio']
        i += 1  
    
    # 群组博弈明细稳定结果表头
    collumns_title = ['r', 'Net_ID', 'Pro_ID', 'Net_Size', 'Group_Iters', 'Iter_Times', ] + collumns + ['1st_Stab_C', 'D_C_Ratio']
    excel_write_line(filename, 0, 0, collumns_title, 0)   # 写入表头从第1行、第1列开始
    
    # 输出netConfig信息，例如[['WS', 4, 0.1, 30], ['ER', 3, 30]]
    excel_write_line(filename, 0, len(collumns_title)+1, ['No.', 'netType', 'K', 'WS_P/ini_Size', 'ini_Size'])
    for ii in range(len(netConfigs)):
        #with lock:
        excel_write_line(filename, ii+1, len(collumns_title)+1, [str(ii)]+netConfigs[ii], 0)   
    
    # 输出明细稳定结果
    excel_y = 1
    for temp_result in result_list:
        excel_write_line( filename, excel_y, 0, temp_result[0][:-2], 0)   # 不输出迁移节点和删除节点数
        excel_y += 1
        
    # 2. 输出群组博弈稳定结果统计
    # 获取网络类型名称和结果表头
    nets_type = netConfigs[0][0]
    collumns = [netConfigs[0][0]+'+'+str(0)+'_AveSize', netConfigs[0][0]+'+'+str(0)+'_C_AveRatio']
    i = 1
    while i<len(netConfigs):
        nets_type += '+'+netConfigs[i][0]
        collumns += [netConfigs[i][0]+'+'+str(i)+'_AveSize', netConfigs[i][0]+'+'+str(i)+'_C_AveRatio']
        i += 1  
    
    # 群组博弈汇总结果表头
    collumns_title = ['r', 'Ave_Net_Size', 'Ave_Group_Iters', 'Ave_Iter_Times', ] + collumns + ['1st_Stab_AveC', 'D_C_AveRatio']
    excel_write_line(filename, 0, 0, collumns_title, 1)   # 写入表头从第1行、第1列开始
                
    # 输出netConfig信息，例如[['WS', 4, 0.1, 30], ['ER', 3, 30]]
    excel_write_line(filename, 0, len(collumns_title)+1, ['No.', 'netType', 'K', 'WS_P/ini_Size', 'ini_Size'], 1)
    for ii in range(len(netConfigs)):
        #with lock:
        excel_write_line(filename, ii+1, len(collumns_title)+1, [str(ii)]+netConfigs[ii], 1)   
    
    # ['r', 'Net_ID', 'Pro_ID', 'Net_Size', 'Group_Iters', 'Iter_Times', ] + collumns[构型节点比例, 构型合作者比例] + ['Initial_C', 'D_C_Ratio']
    excel_y = 1    # 从第二行开始，保留标题行
    for r in r_list:
        
        # 初始化当前r值的汇总结果
        game_stat = list()
        count = 0
        game_stat += [r]
        game_stat += [0, 0, 0]     # 'Ave_Net_Size', 'Sum_Group_Iters', 'Sum_Iter_Times'
        net_size_list = []          # 记录结果中不同子网规模占比高于0的次数，用于计算平均合作者比例
        for i in range(len(netConfigs)):
            game_stat += [0.0, 0.0]  # collumns[构型节点比例, 构型合作者比例]
            net_size_list += [0]
        game_stat += [0.0, 0.0]  # '1st_Stab_AveC', 'D_C_AveRatio'

        # 累加同r值的结果
        for temp_result in result_list:
            if temp_result[0][0]==r:
                count += 1
                for ii in range(1, len(game_stat), 1):
                    if ii<4 or ii>=22:   
                        game_stat[ii] += temp_result[0][ii+2]
                    elif ii<22 and ii%2==0:      # 对每种网络构型占比累加
                        game_stat[ii] += temp_result[0][ii+2]
                        net_size_list[(ii-4)//2] += temp_result[0][ii+2]*temp_result[0][3]
                        game_stat[ii+1] += temp_result[0][ii+2+1]*temp_result[0][ii+2]*temp_result[0][3]                           
        
        # 计算平均值
        for ii in range(1, len(game_stat), 1):
            if ii<4 or ii>=22: 
                game_stat[ii] = round(game_stat[ii]/count, 4)
            elif ii<22:
                if ii%2==0:      # 计算每种网络构型占比平均值
                    game_stat[ii] = round(game_stat[ii]/count, 4)
                elif ii%2==1 and net_size_list[(ii-4)//2]>0:      # 计算每种网络合作者比例平均值
                    game_stat[ii] = round(game_stat[ii]/net_size_list[(ii-4)//2], 4)        
        
        excel_write_line( filename, excel_y, 0, game_stat, 1)
        excel_y += 1
    
    # 3. 输出群组博弈稳定结果地理分布 
    times = 0
    for temp_result in result_list: # temp_result[1]为长度为L+1的二维数组
        times += 1
        
        # 256//(L+2) 每行最多输出的整体网络数
        startLine = int(times//(256//(L+2)))   
        startCol = max(int(times%(256//(L+2))), 1)
        lineNo = 1 
        colNo = 1+(startCol-1)*(L+2)
        
        if lineNo+L>65535:
            print('输出行数超过Excel最大限度！')
            return
        
        # 在左上角写入标题，temp_result[1][0]记录当前地理分布标签名称
        if not temp_result[1][0]==None:
            excel_write_line(filename, lineNo+startLine*(L+2)-1, colNo, [temp_result[1][0]], 2)
        
        for ii in range(L):   # temp_result[1][1-L+]记录L行子网地理分布
            excel_write_line(filename, lineNo+startLine*(L+2), colNo, temp_result[1][ii+1], 2)
            lineNo += 1
    
    # 4. 输出群组博弈初始时地理分布 
    times = 0
    for temp_result in result_list: # temp_result[1]为长度为L+1的二维数组
        times += 1
        
        # 256//(L+2) 每行最多输出的整体网络数
        startLine = int(times//(256//(L+2)))   
        startCol = max(int(times%(256//(L+2))), 1)
        lineNo = 1 
        colNo = 1+(startCol-1)*(L+2)
        
        if lineNo+L>65535:
            print('输出行数超过Excel最大限度！')
            return
        
        # 在左上角写入标题，temp_result[2][0]记录当前地理分布标签名称
        if not temp_result[2][0]==None:
            excel_write_line(filename, lineNo+startLine*(L+2)-1, colNo, [temp_result[2][0]], 3)
        
        for ii in range(L):   # temp_result[1][1-L+]记录L行子网地理分布
            excel_write_line(filename, lineNo+startLine*(L+2), colNo, temp_result[2][ii+1], 3)
            lineNo += 1
            
    return


# ——————————————————————————————————————
# 快速调试入口

if __name__ == '__main__':
    
    print("向Excel输出信息的方法调试开始 !\n")

