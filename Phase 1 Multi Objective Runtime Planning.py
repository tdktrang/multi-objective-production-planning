# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 20:49:31 2021

@author: Th-Do-Kieu.Trang
"""

import pandas as pd
import numpy as np
from itertools import combinations
import os
import math
import matplotlib.pyplot as plt

Data_Code = pd.read_excel('Planning Data.xlsx',sheet_name="Data Code")
Data_Product = pd.read_excel('Planning Data.xlsx',sheet_name="Data Product")
Data_Machine = pd.read_excel('Planning Data.xlsx',sheet_name="Data Machine")
Parameter = pd.read_excel('Planning Data.xlsx',sheet_name="Parameter")
Input = pd.read_excel('Planning Data.xlsx',sheet_name="Input")
Capacity = pd.read_excel('Planning Data.xlsx',sheet_name="Capacity")

Group_Product_List = list(Input['Product Group'])
Group_Product_List = list(dict.fromkeys(Group_Product_List))
Master_Output_List = []

def generate_scenario(list_):
    list_1 = []
    for i in list_:
      list_1.append(list(range(i+1)))
    list_2 = []
    for j in list_1:
      for j_ in j:
        list_2.append(j_)
    list_3 = list(combinations(list_2,len(list_)))
    #seperate scenario
    over_load = []
    for i in range(len(list_3)):
      for j in range(len(list_3[i])):
        if list_3[i][j] > list_[j]:
          over_load.append(list_3[i])
    #create tuple full 0 number
    temp = []
    for i in range(len(list_)):
      temp.append(0)
    temp = tuple(temp)
    #create scenario number of machine
    scenario_machine = []
    for i in list_3:
      if i not in over_load and i != temp and i not in scenario_machine:
        scenario_machine.append(i)  
    return scenario_machine

def export_solution(Data_Raw,scenario_machine):
    Pack_Size_List = list(Data_Raw['Pack Size'])
    Pack_Size_List = list(dict.fromkeys(Pack_Size_List))
    objective_1 = []
    objective_2 = []
    out = []
    for i in scenario_machine:
      Data_Raw['Number'] = i
      Data_Raw['Exp_Output_lh'] = Data_Raw['Pack Size']*Data_Raw['Number']*Data_Raw['Normal Speed']*Data_Raw['Line Eff']/1000
      Data_Raw['Exp_Output_kgh'] = Data_Raw['Exp_Output_lh']*Data_Raw['Density']
      exp_output_lh = sum(Data_Raw['Exp_Output_lh'])
      ex_output_kgh = sum(Data_Raw['Exp_Output_kgh'])
      atank_after = np.mean(Data_Raw['a'])/(np.mean(Data_Raw['b'])-exp_output_lh/1000)
      Data_Raw['Ratio'] = Data_Raw['Exp_Output_kgh']/Data_Raw['Exp_Output_kgh'].sum()
      for j in range(1,25):
        Batch_Size_STD = np.mean(Data_Raw['Batch Size'])*(1-np.mean(Data_Raw['TS Loss']))
        Exp_Output_STD = Batch_Size_STD*j
        
        Fixloss_QueueTank = np.mean(Data_Raw['Fix Loss'])+ 50*j*2
        if atank_after*(Exp_Output_STD-Fixloss_QueueTank)/exp_output_lh < 0.5:
          frequency_atank_full = 0
        else:
          frequency_atank_full = math.ceil((Exp_Output_STD-Fixloss_QueueTank)/exp_output_lh/atank_after)

        atank_loss_duetofull = np.mean(Data_Raw['Loss Due To Full'])*frequency_atank_full
        Exp_Output_Atank = Exp_Output_STD - Fixloss_QueueTank - atank_loss_duetofull
        
        Percent_expect_rework_FP = sum(Data_Raw['Number'])*np.mean(Data_Raw['Percent Loss_Machine'])
        Over_fill = Exp_Output_Atank*np.mean(Data_Raw['Percent Overfill'])
        if (Exp_Output_Atank - Percent_expect_rework_FP*Batch_Size_STD - Over_fill)/ex_output_kgh < 24:
          QA_sample = np.mean(Data_Raw['QA Sample'])*sum(Data_Raw['Pack Size']*Data_Raw['Number'])*np.mean(Data_Raw['Density'])/1000
        else:
          QA_sample = 2*np.mean(Data_Raw['QA Sample'])*sum(Data_Raw['Pack Size']*Data_Raw['Number'])*np.mean(Data_Raw['Density'])/1000
        #print(i,' - ',j,' - ',ex_output_kgh)
        Percent_detroy_if_rework = Percent_expect_rework_FP*Batch_Size_STD
        Percent_detroy_if_not_rework = Percent_expect_rework_FP*Batch_Size_STD*j
        if np.mean(Data_Raw['Rework or not']) == 1:
          Exp_Output_FP = Exp_Output_Atank - Over_fill - QA_sample - Percent_detroy_if_rework
        else:
          Exp_Output_FP = Exp_Output_Atank - Over_fill - QA_sample - Percent_detroy_if_not_rework 
        Exp_RT_Filling = Exp_Output_FP/ex_output_kgh
        Output = {}
        Output_List = []
        Volume_List = []
        for k in Pack_Size_List:
          Seperate_Pack_Size = Data_Raw.loc[Data_Raw['Pack Size'] == k].copy()
          Sum_ratio = sum(Seperate_Pack_Size['Ratio'])
          Exp_Output = Exp_Output_FP*Sum_ratio/(np.mean(Seperate_Pack_Size['brik_cs'])*np.mean(Seperate_Pack_Size['Pack Size'])/1000*np.mean(Seperate_Pack_Size['Density']))
          Output_List.append(Exp_Output)
          Volume_List.append(np.mean(Seperate_Pack_Size['Volume']))
        Output['Product Group'] = product
        Output['Pack Size'] = Pack_Size_List
        Output['Batch'] = j
        Output['RT'] = Exp_RT_Filling
        Output['Volume'] = Volume_List
        Output['Exp Output'] = Output_List
        Output['Cap1'] = np.mean(Data_Raw['Cap'])
        Output['lh'] = exp_output_lh
        Output_Data = pd.DataFrame.from_dict(Output)
        Output_Data['Var'] = abs(Output_Data['Volume']-Output_Data['Exp Output'])
        var = 0
        for i_ in list(Output_Data['Var']):
            var += i_
        Checklist = []
        for i_ in range(len(list(Output_Data['Volume']))):
            if Output_Data.loc[i_,'Exp Output'] >= 0.5*Output_Data.loc[i_,'Volume']:
                Checklist.append(1)
            else:
                Checklist.append(0)
        Output_Data['Check'] = Checklist
        Export_Dic = pd.merge(Data_Raw, Output_Data, on=['Product Group','Pack Size','Volume'])
        Export = Export_Dic[['Product Group','Pack Size','UHT','Machine','Number','Batch','RT','Volume','Exp Output','Var','Cap1','lh','Check']]
        objective_1.append(var)
        objective_2.append(Exp_RT_Filling)
        out.append(Export)
    return objective_1,objective_2, out
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

def fast_non_dominated_sort(values1, values2,out):
    S=[[] for i in range(0,len(values1))]
    S_=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]
    
    for p in range(0,len(values1)):
        if out[p].loc[0,'lh'] <= out[p].loc[0,'Cap1']:
            S[p]=[]
            n[p]=0
            for q in range(0, len(values1)):
                if ((values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q])):
                    if q not in S[p]:
                        S[p].append(q)
                elif ((values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p])):
                    if q not in S[p]:
                        S_[p].append(q)
                        n[p] = n[p] + 1
            if n[p]==0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)
    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                if p in S_[q] and q not in Q:
                    Q.append(q)
        i = i+1
        
        front.append(Q)
    del front[len(front)-1]
    return front

outtttt = []

for product in Group_Product_List:
  df_ = Input.loc[Input['Product Group'] == product].copy()
  df_0 = pd.merge(df_, Data_Product, on=['Product Group','Pack Size'])  
  df_1 = pd.merge(df_0,Data_Machine,on=['Machine'])
  df_2_ = pd.merge(df_1,Parameter,on=['Product Group','UHT'])
  df_2 = pd.merge(df_2_,Capacity,on=['UHT'])
  UHT_List = list(df_2['UHT'])
  UHT_List = list(dict.fromkeys(UHT_List))
  for UHT in UHT_List:
    Data_Raw = df_2.loc[df_2['UHT'] == UHT].copy()
    print(UHT," ",product)
    a = UHT + " - " +product
    list_ = list(Data_Raw['Max'])
    scenario_machine = generate_scenario(list_)
    objective_1,objective_2, out = export_solution(Data_Raw, scenario_machine)
    front = fast_non_dominated_sort(objective_1,objective_2,out)
    l = 1
    if (len(front) != 0):
        for i in front[-1]:
            Ex = out[i]
            Ex['Solution'] = 'Solution_{}'.format(l)
            l += 1
            function1 = [i for i in objective_1]
            function2 = [j for j in objective_2]
            plt.title(a)
            plt.xlabel('Runtime', fontsize=15)
            plt.ylabel('Demand Variance', fontsize=15)
            plt.scatter(objective_1, objective_2)
            plt.show()
            Master_Output_List.append(Ex)
    else:
        continue
Master_Output = pd.concat(Master_Output_List).reset_index(drop=True)
Master_Output.to_excel('output1.xlsx')
os.system('output1.xlsx')