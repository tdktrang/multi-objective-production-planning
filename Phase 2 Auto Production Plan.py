# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 22:35:05 2021

@author: Th-Do-Kieu.Trang
"""
from itertools import combinations
import pandas as pd
import numpy as np
import datetime
import copy
import plotly_express as px
from plotly.offline import plot
import os



matrix_data = pd.read_excel("Changeover Data.xlsx",sheet_name="Matrix")
max_cycle = pd.read_excel("Changeover Data.xlsx",sheet_name="Max Cycle")
week = pd.read_excel("Changeover Data.xlsx",sheet_name="Input Week")
input_product = pd.read_excel("Changeover Data.xlsx",sheet_name="Input product")
input_product_data=pd.merge(input_product,max_cycle,on=['Product Group'])


a = min(week['Start'])


input_product_data['Start'] = input_product_data['Time Available'] - a
input_product_data['Start'] = input_product_data['Start']/datetime.timedelta(hours=1)
df_ = pd.read_excel("Changeover Data.xlsx",sheet_name="Machine Group")
df1 = pd.read_excel("Changeover Data.xlsx",sheet_name="Input group machine")
df2 = pd.read_excel("Changeover Data.xlsx",sheet_name="Input machine")
merge_machine = pd.merge(df1, df_, on=['Machine Group'])
merge_machine1 = pd.merge(merge_machine, df2, on= ['Machine'],how='left')
#sample1: product
#sample2: machine


def product(sample1):
    product_list = []
    for i in sample1.index:
        if sample1.loc[i,'Product Group'] not in product_list:     
            product_list.append(sample1.loc[i,'Product Group'])
    return product_list

def max_cycle(sample1):
    max_cycle_list = []
    for i in sample1.index:
        max_cycle_list.append(sample1.loc[i,'Max cycle'])
    return max_cycle_list


def generate_ps_matrix(sample1):
    product_list = product(sample1)
    column_matrix = len(matrix_data.columns)
    matrix = []
    list_ = []
    for j in product_list:
        for i in range(column_matrix):
            if matrix_data.columns[i] == j and i not in list_:
                list_.append(i)
    for i in list_:
        temporary_list = []
        for j in list_:
            #print(matrix_data.columns[i],'-',matrix_data.columns[j],'-',matrix_data.loc[i][j])
            x = matrix_data.loc[i][j]
            if pd.isnull(x):
                temporary_list.append(9999999)
                #9999999
            else:
                temporary_list.append(x)
        matrix.append(temporary_list)
    return matrix


def time_available(sample1):
    available_time = []
    for i in sample1.index:
        available_time.append(sample1.loc[i,'Start'])
    return available_time

def rank(sample1):
    rank_list = []
    for i in sample1.index:
        rank_list.append(sample1.loc[i,'Ranking'])
    return rank_list

def UHT(sample1, sample2):
    list_product = product(sample1)
    UHT_list = []
    for i in list_product:
        g = sample2[sample2['Product Group'] == i].copy()
        group = np.unique(g['UHT'].tolist()).tolist()
        list_ = []
        for j in group:
            list_.append(j)
        UHT_list.append(list_)   
    return UHT_list


def machine_group(sample1,sample2):
    list_product = product(sample1)
    list_UHT = UHT(sample1,sample2)
    machine_group_list = []
    for i in range(len(list_product)):
        machine_group = []
        for j in range(len(list_UHT[i])):
            g = sample2[(sample2['Product Group'] == list_product[i]) & (sample2['UHT'] == list_UHT[i][j])].copy()
            group = np.unique(g['Machine Group'].tolist()).tolist()
            machine_group.append(group)
        machine_group_list.append(machine_group)
    return machine_group_list


def number_machine(sample1,sample2):
    list_product = product(sample1)
    list_UHT = UHT(sample1,sample2)
    list_machine_group = machine_group(sample1, sample2)
    machine_number_list = []
    for i in range(len(list_product)):
        machine_number = []
        for j in range(len(list_UHT[i])):
            list_ = []
            for z in range(len(list_machine_group[i][j])):
                g = sample2[(sample2['Product Group'] == list_product[i]) & (sample2['UHT'] == list_UHT[i][j]) & (sample2['Machine Group'] == list_machine_group[i][j][z])].copy()
                list_.append(np.mean(g['Number']))
            machine_number.append(list_)
        machine_number_list.append(machine_number)
    return machine_number_list

def machine(sample1,sample2):
    list_product = product(sample1)
    list_UHT = UHT(sample1,sample2)
    list_machine_group = machine_group(sample1, sample2)
    machine_list = []
    for i in range(len(list_product)):
        list_1 = []
        for j in range(len(list_UHT[i])):
            list_2 = []
            for z in range(len(list_machine_group[i][j])):
                g = sample2[(sample2['Product Group'] == list_product[i]) & (sample2['UHT'] == list_UHT[i][j]) & (sample2['Machine Group'] == list_machine_group[i][j][z])].copy()
                distinct_machine_list = np.unique(g['Machine'].tolist()).tolist()
                list_2.append(distinct_machine_list)
            list_1.append(list_2)
        machine_list.append(list_1)
    return machine_list          

def time_block(sample1,sample2):
    min_date = min(week['Start'])
    sample2['Start'] = (sample2['Start Block'] - min_date)/datetime.timedelta(hours=1)
    sample2['End'] = (sample2['End Block'] - min_date)/datetime.timedelta(hours=1)
    for i in sample2.index:
        if pd.isnull(sample2.loc[i,'Start']) and pd.isnull(sample2.loc[i,'End']):
            sample2.loc[i,'Start'] = 1000000
            sample2.loc[i,'End'] = 9999999999
    list_product = product(sample1)
    list_UHT = UHT(sample1,sample2)
    list_machine_group = machine_group(sample1, sample2)
    list_machine = machine(sample1, sample2)
    block_time = []
    for i in range(len(list_product)):
        list_1 = []
        for j in range(len(list_UHT[i])):
            list_2 = []
            for z in range(len(list_machine_group[i][j])):
                list_3 = []
                for k in range(len(list_machine[i][j][z])):
                    list_4 = []
                    g = sample2[(sample2['Product Group'] == list_product[i]) & (sample2['UHT'] == list_UHT[i][j]) & (sample2['Machine Group'] == list_machine_group[i][j][z]) & (sample2['Machine'] == list_machine[i][j][z][k])]
                    for x in g.index:
                        list_4.append((g.loc[x,'Start'],g.loc[x,'End']))
                    list_3.append(list_4)
                list_2.append(list_3)
            list_1.append(list_2)
        block_time.append(list_1)
    return block_time

def good_production_time(sample1):
    good_time = []
    for i in sample1.index:
        good_time.append(sample1.loc[i,'GPT'])
    return good_time


def create_data_model(sample1,sample2):
    data = {}
    data['Product'] = product(sample1)
    data['Max Cycle'] = max_cycle(sample1)
    data['Ranking'] = rank(sample1)
    data['UHT'] = UHT(sample1,sample2)
    data['Machine Group'] = machine_group(sample1, sample2)
    data['Number Machine'] = number_machine(sample1,sample2)
    data['Machine'] = machine(sample1, sample2)
    data['Available'] = time_available(sample1)
    data['Block'] = time_block(sample1, sample2)
    data['Time_Good'] = good_production_time(sample1)
    data['PS Matrix'] = generate_ps_matrix(sample1)
    return data

def dynamic_programing(arr):
    list_1 = []
    for i in range(len(arr)):
        list_2 = []
        for j in range(len(arr[i])):
            if len(list_1) == 0:
                list_2.append([arr[i][j]])
            else:
                for k in range(len(list_1)):
                    a = copy.deepcopy(list_1[k])
                    a.append(arr[i][j])
                    list_2.append(a)
        list_1 = copy.deepcopy(list_2)    
    return list_1


def is_overlaping(a, b):
    if b[0] >= a[0] and b[0] < a[1]:
        return True
    else:
        return False
 
 
# merge the intervals
def merge(arr):
    #sort the intervals by its first value
    arr.sort(key = lambda x: x[0])
    merged_list= []
    merged_list.append(arr[0])
    for i in range(1, len(arr)):
        pop_element = merged_list.pop()
        if is_overlaping(pop_element, arr[i]):
            new_element = pop_element[0], max(pop_element[1], arr[i][1])
            merged_list.append(new_element)
        else:
            merged_list.append(pop_element)
            merged_list.append(arr[i])
    return merged_list


#Tìm ra PS lớn nhất
def find_max_PS(list_product_used,now):
    list_product = model['Product']
    matrix = model['PS Matrix']
    var = 0
    for i in range(len(list_product)):
        if i != now and i not in list_product_used and matrix[now][i] > var and matrix[now][i] != 9999999:
            var = matrix[now][i]
    return var



def generate_free_time(now,list_product_used,range_block,min_time,max_time):
    merge_block = merge(range_block)

    available_list1 = []
    for interval in merge_block:
        for t in interval:
            available_list1.append(t)
    available_list1.sort()
    
    lower_than_min_time = []
    greater_than_min_time = []
    for i in available_list1:
        if i <= min_time:
            lower_than_min_time.append(i)
        elif i > min_time:
            greater_than_min_time.append(i)
    if len(lower_than_min_time) % 2 == 0:
        greater_than_min_time.append(min_time)
    
    lower_than_max_time = []
    for j in greater_than_min_time:
        if j < max_time:
            lower_than_max_time.append(j)
    if len(lower_than_max_time) % 2 != 0:
        lower_than_max_time.append(max_time)
    
    available_list = lower_than_max_time.copy()
    available_list.sort()
    
    list_free_final = []
    for item in range(len(available_list)-1):
        if available_list[item] <= max_time and available_list[item+1] <= max_time and \
            (available_list[item],available_list[item+1]) not in merge_block and \
                available_list[item+1] - available_list[item] >= model['Time_Good'][now] + find_max_PS(list_product_used,now):
            list_free_final.append((available_list[item],available_list[item+1]))
    
    num = 0
    for i in list_free_final:
        if len(i) == 0:
           num += 1 
    if num == len(list_free_final):
        list_free_final = [[9999999,999999999]]
    #f_free = merge(list_free_final)
    return list_free_final
    



def define_earliest_start(U,list_product_used,now,min_time,max_time):
    list_product = model['Product']
    list_UHT = model['UHT']
    list_machine_group = model['Machine Group']
    list_machine = model['Machine']
    list_number = model['Number Machine']
    list_block = model['Block']
    _block = []
    _machine = []
    for z in range(len(list_machine_group[now][list_UHT[now].index(U)])):
        list_1 = []
        list_1 = list(combinations(list(range(len(list_machine[now][list_UHT[now].index(U)][z]))), int(list_number[now][list_UHT[now].index(U)][z])))
        scenario_machine = []
        scenario_block = []
        for t in list_1:
            temp_machine = []
            temp_block = []
            for t_ in t:
                temp_machine.append(list_machine[now][list_UHT[now].index(U)][z][t_])
                for t_1 in list_block[now][list_UHT[now].index(U)][z][t_]:
                    temp_block.append(t_1)
            scenario_machine.append(temp_machine)
            scenario_block.append(temp_block)
        _block.append(scenario_block)
        _machine.append(scenario_machine)
    f_machine = dynamic_programing(_machine)
    f_block = dynamic_programing(_block)
    # print(f_block)
    # print(f_machine)
    new_machine = []
    new_block = []
    for item_1 in f_machine:
        new = []
        for item_2 in item_1:
            for item_3 in item_2:    
                if item_3 not in new:
                    new.append(item_3)
        new_machine.append(new)
    #print(1,'  ',new_machine)
    
    for item_1 in f_block:
        new = []
        for item_2 in item_1:
            for item_3 in item_2:
                if item_3 not in new:    
                    new.append(item_3)
        new_block.append(new)
    #print(2,'    ',new_block)
    
    range_available = []
    
    for i in new_block:
        avai = generate_free_time(now,list_product_used,i,min_time, max_time)
        range_available.append(avai)
    #print(3,'   ',range_available)
    
    para = 99999999999999
    point = 999999999

    for k in range(len(range_available)):
        for k1 in range(len(range_available[k])):
            if range_available[k][k1][0] < para:
                para = range_available[k][k1][0]
                point = k
    #print(1,range_available)

    available = range_available[point][0]

    machine_used = new_machine[point]
    # print(available)
    # print(machine_used)

    return available,machine_used

def first_product(U,list_product_used,product_manufacture,max_time):
    product_in_U = []

    for i in product_manufacture:
        if U in model['UHT'][i]:
            product_in_U.append(i)

    check_available = []
    for i in product_in_U:
        min_time = model['Available'][i]
        available,machine_used = define_earliest_start(U, list_product_used, i, min_time, max_time)
        check_available.append(available)
    
    min_available = min([i[0] for i in check_available])
    
    disqualification_case = []    

    for item in range(len(check_available)):
        if check_available[item][0] != min_available:
            disqualification_case.append(item)
            # check_available.pop(item)
            # product_in_U.pop(item)
    check_available = [check_available[i] for i in range(len(check_available)) if i not in disqualification_case]
    product_in_U = [product_in_U[i] for i in range(len(product_in_U)) if i not in disqualification_case]
    
    min_rank_dic = {i: model['Ranking'][i] for i in product_in_U}
    first_point = min(min_rank_dic, key=lambda k: min_rank_dic[k])
    runtime = model['Time_Good'][first_point]
    available_first, machine_final = define_earliest_start(U, list_product_used, first_point, min_time, max_time)
    
    return first_point, machine_final, min_available

#product_manufacture là list những sp cần được sản xuất 
#sau khi tìm được sản phẩm hiện tại, phải loại sản phẩm đó ra khỏi list product_manufacture


#Chờ xử lý xong earlies_start
def next_product(now,T,U,product_manufacture,list_product_used,max_time):
    product_in_U = []
    product_ready = []
    product_not_ready = []
    available_not_ready = []
    for i in product_manufacture:
        if U in model['UHT'][i]:
            product_in_U.append(i)
    for i in product_in_U:
        T = T + model['PS Matrix'][now][i]
        min_time = model['Available'][i]
        available,machine_used = define_earliest_start(U, list_product_used, i, min_time, max_time)
        available_time = available[0]
        if T >= available_time and T <= max_time:
            product_ready.append(i)
        elif T < available_time and T <= max_time:
            product_not_ready.append(i)
            available_not_ready.append(available_time)
    return product_ready,product_not_ready,available_not_ready



def fexist_PS(now,product_ready):
    aa = {i: model['PS Matrix'][now][i] for i in product_ready}
    return(min(aa, key = lambda k: aa[k]))

def fexist_waiting_time(now,product_not_ready,available_not_ready):
    aa = {product_not_ready[i]: available_not_ready[i] for i in range(len(product_not_ready))}
    point_min_waiting = min(aa,key = lambda k: aa[k])
    min_ready = available_not_ready[product_not_ready.index(point_min_waiting)]
    return point_min_waiting,min_ready

def subset(alpha):
    sample1 = input_product_data[input_product_data['Week'] == alpha].copy()
    sample2 = merge_machine1[merge_machine1['Week']== alpha].copy()
    return sample1,sample2

def insert_block_time(list_machine,range_T,df2,alpha):
    #x là thời gian bắt đầu sản xuất trong tuần, tức là thời gian nhỏ nhất
    #df2 là data input machine
    max_cycle = pd.read_excel("Changeover Data.xlsx",sheet_name="Max Cycle")
    input_product = pd.read_excel("Changeover Data.xlsx",sheet_name="Input product")
    input_product_data=pd.merge(input_product,max_cycle,on=['Product Group'])
    a = min(week['Start'])
    for i in list_machine:
        new_row = {'Machine':i,'Start Block':a+datetime.timedelta(hours=range_T[0]),'End Block':a+datetime.timedelta(hours=range_T[1])}
        df2 = df2.append(new_row,ignore_index=True)
    input_product_data['Start'] = input_product_data['Time Available'] - a
    input_product_data['Start'] = input_product_data['Start']/datetime.timedelta(hours=1)
    df_ = pd.read_excel("Changeover Data.xlsx",sheet_name="Machine Group")
    df1 = pd.read_excel("Changeover Data.xlsx",sheet_name="Input group machine")
    merge_machine = pd.merge(df1, df_, on=['Machine Group'])
    merge_machine1 = pd.merge(merge_machine, df2, on= ['Machine'],how='left')
    
    sample1 = input_product_data[input_product_data['Week'] == alpha].copy()
    sample2 = merge_machine1[merge_machine1['Week']== alpha].copy()
    model = create_data_model(sample1, sample2)
    return model,df2


master_matrix = generate_ps_matrix(input_product_data)
master_product = product(input_product_data)

def fexist_PS_master(now,product_ready):
    aa = {i: master_matrix[now][i] for i in product_ready}
    return(min(aa, key = lambda k: aa[k]))

def next_product_master(now,T,U,product_manufacture,list_product_used,max_time):
    product_in_U = []
    product_ready = []
    product_not_ready = []
    available_not_ready = []
    for i in product_manufacture:
        if U in model['UHT'][i]:
            product_in_U.append(i)
    for i in product_in_U:
        next_product = master_product.index(model['Product'][i])
        T = T + master_matrix[now][next_product]
        min_time = model['Available'][i]
        available,machine_used = define_earliest_start(U, list_product_used, i, min_time, max_time)
        available_time = available[0]
        if T >= available_time and T <= max_time:
            product_ready.append(next_product)
        elif T < available_time and T <= max_time:
            product_not_ready.append(next_product)
            available_not_ready.append(available_time)
    return product_ready,product_not_ready,available_not_ready




list_U = ['U1','U2']
solution = {"UHT":[],"Product":[],"Start":[],"End":[],"Machine":[]}
#model = create_data_model(input_product_data, merge_machine1)
U1 = []
machine_U1 = []
time_U1 = []
T1 = 0
T2 = 0
M1 = []
M2 = []
P1 = None
P2 = None
for xyz in range(len(week)):
    print('==================')
    print('====> Week',xyz+1,'<====')
    print('==================')
    alpha = week.loc[xyz,'Week']
    sample1,sample2 = subset(alpha)
    model = create_data_model(sample1, sample2)
    
    product_manufacture = []
    for i in range(len(model['Product'])):
        if model['Available'][i] >= 0:
            product_manufacture.append(i)
    print(product_manufacture)
    
    if len(product_manufacture) == 0:
        print('Error! Please check data!!!!')
   
    U = None
    for i in range(len(list_U)):
        if xyz == 0:
            if i == 0:
                U = 'U1'
                T = T1
            elif i == 1:
                U = 'U2'
                T = T2
        else:
            if i == 0:
                U = 'U1'
                T = T1
                M = M1
                P = P1
            elif i == 1:
                U = 'U2'  
                T = T2
                M = M2
                P = P2
        max_time = int((week.loc[len(week)-1,'End'] - week.loc[0,'Start'])/datetime.timedelta(hours=1))
        U1 = []
        machine_U1 = []
        time_U1 = []
        while T < max_time:
            if len(product_manufacture) == 0:
                break
            else:   
                if len(U1) == 0:
                    if xyz == 0:
                        first_point, machine_final, min_available = first_product(U, U1, product_manufacture,max_time)
                        product_manufacture.pop(product_manufacture.index(first_point))
                        now = first_point
                        if model['Time_Good'][now] <= model['Max Cycle'] [now]:
                            T += min_available
                            T += model['Time_Good'][now]
                            U1.append(model['Product'][now])
                            machine_U1.append(machine_final)
                            time_U1.append([min_available,T])
                            model, df2 = insert_block_time(machine_final, [int(min_available),int(T)], df2, alpha)
                        else:
                            #Tách Cycle
                            cycle_num = int(model['Time_Good'][now]/model['Max Cycle'][now])
                            
                            #1. 1st Cycle
                            T += min_available
                            T += model['Max Cycle'][now]
                            U1.append(model['Product'][now])
                            machine_U1.append(machine_final)
                            time_U1.append([min_available,T])
                            model, df2 = insert_block_time(machine_final, [int(min_available),int(T)], df2,alpha)
                            
                            #2. PS giữa cycle
                            U1.append('PS')
                            machine_U1.append(machine_final)
                            time_U1.append([T,T+model['PS Matrix'][now][now]])
                            model, df2 = insert_block_time(machine_final, [int(T),int(T+model['PS Matrix'][now][now])], df2,alpha)
                            T += model['PS Matrix'][now][now]
                            
                            for cycle in range(cycle_num - 1):     
                                #3 next Cycle
                                U1.append(model['Product'][now])
                                time_U1.append([T,T+model['Max Cycle'][now]])
                                machine_U1.append(machine_final)
                                model, df2 = insert_block_time(machine_final, [int(T),int(T+model['Max Cycle'][now])], df2,alpha)
                                T += model['Max Cycle'][now]    
                            
                                #4. PS giữa Cycle
                                U1.append('PS')
                                machine_U1.append(machine_final)
                                time_U1.append([T,T+model['PS Matrix'][now][now]])
                                model, df2 = insert_block_time(machine_final, [int(T),int(T+model['PS Matrix'][now][now])], df2,alpha)
                                T += model['PS Matrix'][now][now]
                                
                            #5. end cycle
                            U1.append(model['Product'][now])
                            time_U1.append([T,T+(model['Time_Good'][now] - model['Max Cycle'][now])])
                            machine_U1.append(machine_final)
                            model, df2 = insert_block_time(machine_final, [int(T),int(T+(model['Time_Good'][now] - model['Max Cycle'][now]*cycle_num))], df2, alpha)
                            T += (model['Time_Good'][now] - model['Max Cycle'][now]*cycle_num)  
                    else:                      
                        now = master_product.index(P)                        
                        product_ready,product_not_ready,available_not_ready = next_product_master(now,T,U,product_manufacture,U1,max_time)
                        if len(product_ready) != 0:
                            next_1 = fexist_PS_master(now, product_ready)
                            min_time = T + master_matrix[now][next_1]
                            product_manufacture.pop(product_manufacture.index(model['Product'].index(master_product[next_1])))
                            next_ = model['Product'].index(master_product[next_1])
                            #update PS
                            U1.append('PS')
                            machine_U1.append(M)
                            time_U1.append([T,min_time])
                            model, df2 = insert_block_time(M, [int(T),int(min_time)], df2, alpha)
                            available,machine_used = define_earliest_start(U, U1, next_, min_time, max_time)
                            
                            #Tách cycle (tương tự như first product)
                            if model['Time_Good'][next_] <= model['Max Cycle'][next_]:
                                model, df2 = insert_block_time(machine_used,[int(min_time),int(min_time+model['Time_Good'][next_])], df2,alpha)
                                T = T + master_matrix[now][next_1] + model['Time_Good'][next_]
                                
                                
                                #UPDATE SOLUTION PRODUCT
                                U1.append(model['Product'][next_])
                                machine_U1.append(machine_used)
                                time_U1.append([min_time,T])
                                now = next_
                            else:
                                cycle_num = int(model['Time_Good'][next_]/model['Max Cycle'][next_])
                                T += master_matrix[now][next_1]
                                U1.append(model['Product'][next_])
                                machine_U1.append(machine_used)
                                time_U1.append([T,T+model['Max Cycle'][next_]])
                                model, df2 = insert_block_time(machine_used,[int(T),int(T+model['Max Cycle'][next_])], df2,alpha)
                                T += model['Max Cycle'][next_]
                                
                                U1.append('PS')
                                machine_U1.append(machine_used)
                                time_U1.append([T,T+model['PS Matrix'][next_][next_]])
                                model, df2 = insert_block_time(machine_used,[int(T),int(T+model['PS Matrix'][next_][next_])], df2,alpha)
                                T += model['PS Matrix'][next_][next_]
                                
                                for cycle in range(cycle_num-1):
                                    U1.append(model['Product'][next_])
                                    machine_U1.append(machine_used)
                                    time_U1.append([T,T+model['Max Cycle'][next_]])
                                    model, df2 = insert_block_time(machine_used,[int(T),int(T+model['Max Cycle'][next_])], df2,alpha)
                                    T += model['Max Cycle'][next_]
                                    
                                    U1.append('PS')
                                    machine_U1.append(machine_used)
                                    time_U1.append([T,T+model['PS Matrix'][next_][next_]])
                                    model, df2 = insert_block_time(machine_used,[int(T),int(T+model['PS Matrix'][next_][next_])], df2,alpha)
                                    T += model['PS Matrix'][next_][next_]
                                    
                                U1.append(model['Product'][next_])
                                machine_U1.append(machine_used)
                                time_U1.append([T,T+(model['Time_Good'][next_]-model['Max Cycle'][next_])])
                                model, df2 = insert_block_time(machine_used,[int(T),int(T+(model['Time_Good'][next_]-model['Max Cycle'][next_]*cycle_num))], df2,alpha)
                                T += (model['Time_Good'][next_]-model['Max Cycle'][next_]*cycle_num)
                                now = next_

                        elif len(product_not_ready) != 0:
                            next_1, min_ready = fexist_waiting_time(now, product_not_ready, available_not_ready)
                            min_time = min_ready
                            product_manufacture.pop(product_manufacture.index(model['Product'].index(master_product[next_1])))
                            next_ = model['Product'].index(master_product[next_1])
                            
                            #update PS
                            U1.append('PS')
                            machine_U1.append(M)            
                            time_U1.append([T,T + master_matrix[now][next_1]])
                            model, df2 = insert_block_time(M, [int(T),int(T + master_matrix[now][next_1])], df2,alpha)
                            available,machine_used = define_earliest_start(U, U1, next_, min_time, max_time)
                            
                            if model['Time_Good'][next_] <= model['Max Cycle'][next_]:
                                model, df2 = insert_block_time(machine_used,[int(min_time),int(min_time+model['Time_Good'][next_])], df2,alpha)
                                T = min_time + model['Time_Good'][next_]
                                
                                
                                #UPDATE SOLUTION PRODUCT
                                U1.append(model['Product'][next_])
                                machine_U1.append(machine_used)
                                time_U1.append([min_time,T])
                                now = next_
                            else:
                                cycle_num = int(model['Time_Good'][next_]/model['Max Cycle'][next_])
                                T = min_time
                                U1.append(model['Product'][next_])
                                machine_U1.append(machine_used)
                                time_U1.append([T,T+model['Max Cycle'][next_]])
                                model, df2 = insert_block_time(machine_used,[int(T),int(T+model['Max Cycle'][next_])], df2,alpha)  
                                T += model['Max Cycle'][next_]
                                
                                U1.append('PS')
                                machine_U1.append(machine_used)
                                time_U1.append([T,T+model['PS Matrix'][next_][next_]])
                                model, df2 = insert_block_time(machine_used,[int(T),int(T+model['PS Matrix'][next_][next_])], df2,alpha)  
                                T += model['PS Matrix'][next_][next_]
                                
                                for cycle in range(cycle_num - 1):
                                    U1.append(model['Product'][next_])
                                    machine_U1.append(machine_used)
                                    time_U1.append([T,T+model['Max Cycle'][next_]])
                                    model, df2 = insert_block_time(machine_used,[int(T),int(T+model['Max Cycle'][next_])], df2,alpha)  
                                    T += model['Max Cycle'][next_]
                                    
                                    U1.append('PS')
                                    machine_U1.append(machine_used)
                                    time_U1.append([T,T+model['PS Matrix'][next_][next_]])
                                    model, df2 = insert_block_time(machine_used,[int(T),int(T+model['PS Matrix'][next_][next_])], df2,alpha)  
                                    T += model['PS Matrix'][next_][next_]
                                    
                                U1.append(model['Product'][next_])
                                machine_U1.append(machine_used)
                                time_U1.append([T,T+(model['Time_Good'][next_] - model['Max Cycle'][next_]*cycle_num)])
                                model, df2 = insert_block_time(machine_used,[int(T),int(T+(model['Time_Good'][next_] - model['Max Cycle'][next_]*cycle_num))], df2,alpha)  
                                T += (model['Time_Good'][next_] - model['Max Cycle'][next_]*cycle_num)
                                now = next_
                                
                else:                    
                    product_ready,product_not_ready,available_not_ready = next_product(now,T,U,product_manufacture,U1,max_time)
                    if len(product_ready) != 0:
                        next_ = fexist_PS(now, product_ready)
                        min_time = T + model['PS Matrix'][now][next_]
                        product_manufacture.pop(product_manufacture.index(next_))
                        
                        #update PS
                        U1.append('PS')
                        machine_U1.append(machine_U1[-1])
                        time_U1.append([T,min_time])
                        model, df2 = insert_block_time(machine_U1[-1], [int(T),int(min_time)], df2, alpha)
                        available,machine_used = define_earliest_start(U, U1, next_, min_time, max_time)
                        
                        #Tách cycle (tương tự như first product)
                        if model['Time_Good'][next_] <= model['Max Cycle'][next_]:
                            model, df2 = insert_block_time(machine_used,[int(min_time),int(min_time+model['Time_Good'][next_])], df2,alpha)
                            T = T + model['PS Matrix'][now][next_] + model['Time_Good'][next_]
                            
                            
                            #UPDATE SOLUTION PRODUCT
                            U1.append(model['Product'][next_])
                            machine_U1.append(machine_used)
                            time_U1.append([min_time,T])
                            now = next_
                        else:
                            cycle_num = int(model['Time_Good'][next_]/model['Max Cycle'][next_])
                            T += model['PS Matrix'][now][next_]
                            U1.append(model['Product'][next_])
                            machine_U1.append(machine_used)
                            time_U1.append([T,T+model['Max Cycle'][next_]])
                            model, df2 = insert_block_time(machine_used,[int(T),int(T+model['Max Cycle'][next_])], df2,alpha)
                            T += model['Max Cycle'][next_]
                            
                            U1.append('PS')
                            machine_U1.append(machine_used)
                            time_U1.append([T,T+model['PS Matrix'][next_][next_]])
                            model, df2 = insert_block_time(machine_used,[int(T),int(T+model['PS Matrix'][next_][next_])], df2,alpha)
                            T += model['PS Matrix'][next_][next_]
                            
                            for cycle in range(cycle_num-1):
                                U1.append(model['Product'][next_])
                                machine_U1.append(machine_used)
                                time_U1.append([T,T+model['Max Cycle'][next_]])
                                model, df2 = insert_block_time(machine_used,[int(T),int(T+model['Max Cycle'][next_])], df2,alpha)
                                T += model['Max Cycle'][next_]
                                
                                U1.append('PS')
                                machine_U1.append(machine_used)
                                time_U1.append([T,T+model['PS Matrix'][next_][next_]])
                                model, df2 = insert_block_time(machine_used,[int(T),int(T+model['PS Matrix'][next_][next_])], df2,alpha)
                                T += model['PS Matrix'][next_][next_]
                                
                            U1.append(model['Product'][next_])
                            machine_U1.append(machine_used)
                            time_U1.append([T,T+(model['Time_Good'][next_]-model['Max Cycle'][next_])])
                            model, df2 = insert_block_time(machine_used,[int(T),int(T+(model['Time_Good'][next_]-model['Max Cycle'][next_]*cycle_num))], df2,alpha)
                            T += (model['Time_Good'][next_]-model['Max Cycle'][next_]*cycle_num)
                            now = next_
                            
                        continue
                    elif len(product_not_ready) != 0:
                        next_, min_ready = fexist_waiting_time(now, product_not_ready, available_not_ready)
                        min_time = min_ready
                        product_manufacture.pop(product_manufacture.index(next_))
                        #update PS
                        U1.append('PS')
                        machine_U1.append(machine_U1[-1])            
                        time_U1.append([T,T + model['PS Matrix'][now][next_]])
                        model, df2 = insert_block_time(machine_U1[-1], [int(T),int(T + model['PS Matrix'][now][next_])], df2,alpha)
                        
                        available,machine_used = define_earliest_start(U, U1, next_, min_time, max_time)
                        
                        if model['Time_Good'][next_] <= model['Max Cycle'][next_]:
                            model, df2 = insert_block_time(machine_used,[int(min_time),int(min_time+model['Time_Good'][next_])], df2,alpha)
                            T = min_time + model['Time_Good'][next_]
                            
                            
                            #UPDATE SOLUTION PRODUCT
                            U1.append(model['Product'][next_])
                            machine_U1.append(machine_used)
                            time_U1.append([min_time,T])
                            now = next_
                        else:
                            cycle_num = int(model['Time_Good'][next_]/model['Max Cycle'][next_])
                            T = min_time
                            U1.append(model['Product'][next_])
                            machine_U1.append(machine_used)
                            time_U1.append([T,T+model['Max Cycle'][next_]])
                            model, df2 = insert_block_time(machine_used,[int(T),int(T+model['Max Cycle'][next_])], df2,alpha)  
                            T += model['Max Cycle'][next_]
                            
                            U1.append('PS')
                            machine_U1.append(machine_used)
                            time_U1.append([T,T+model['PS Matrix'][next_][next_]])
                            model, df2 = insert_block_time(machine_used,[int(T),int(T+model['PS Matrix'][next_][next_])], df2,alpha)  
                            T += model['PS Matrix'][next_][next_]
                            
                            for cycle in range(cycle_num - 1):
                                U1.append(model['Product'][next_])
                                machine_U1.append(machine_used)
                                time_U1.append([T,T+model['Max Cycle'][next_]])
                                model, df2 = insert_block_time(machine_used,[int(T),int(T+model['Max Cycle'][next_])], df2,alpha)  
                                T += model['Max Cycle'][next_]
                                
                                U1.append('PS')
                                machine_U1.append(machine_used)
                                time_U1.append([T,T+model['PS Matrix'][next_][next_]])
                                model, df2 = insert_block_time(machine_used,[int(T),int(T+model['PS Matrix'][next_][next_])], df2,alpha)  
                                T += model['PS Matrix'][next_][next_]
                                
                            U1.append(model['Product'][next_])
                            machine_U1.append(machine_used)
                            time_U1.append([T,T+(model['Time_Good'][next_] - model['Max Cycle'][next_]*cycle_num)])
                            model, df2 = insert_block_time(machine_used,[int(T),int(T+(model['Time_Good'][next_] - model['Max Cycle'][next_]*cycle_num))], df2,alpha)  
                            T += (model['Time_Good'][next_] - model['Max Cycle'][next_]*cycle_num)
                            now = next_
                        continue
                    else:
                        break
                     
        
        if U == 'U1' and len(time_U1)!=0:
            T1 = time_U1[-1][1]
            M1 = machine_U1[-1]
            P1 = U1[-1]
        elif U == 'U2' and len(time_U1)!= 0:
            T2 = time_U1[-1][1]
            M2 = machine_U1[-1]
            P2 = U1[-1]
                
        for num in range(len(U1)):
            solution['UHT'].append(U)
            solution['Product'].append(U1[num])
            solution['Start'].append(str(a + datetime.timedelta(hours=time_U1[num][0])))
            solution['End'].append(str(a + datetime.timedelta(hours=time_U1[num][1])))
            solution['Machine'].append(machine_U1[num])
        
        print('---------------Solution',list_U[i],'---------------')
        for k in range(len(U1)):
            print(k+1,'--Point:','[',U1[k],']','--Interval:',time_U1[k],'--Machine:',machine_U1[k])
            
            
        if len(product_manufacture) > 0:
            print('-------o-------')
            print('=> Max time:',max_time)
            print('=> Current time:',T)
            print('Week',xyz+1,'-',U,'-','Sản phẩm chưa được xét tới:')
            for pro in product_manufacture:
                print('  -',model['Product'][pro])
            print('-------o-------')


solution_dataframe = pd.DataFrame(solution)
fig = px.timeline(solution_dataframe, 
                  y="UHT",
                  x_start="Start", 
                  x_end="End", 
                  color="Product",
                  text="Machine",
                  opacity = 1,
                  title="Production Plan For Milk Factory")
plot(fig)
solution_dataframe.to_excel('output1.xlsx')
os.system('output2.xlsx')