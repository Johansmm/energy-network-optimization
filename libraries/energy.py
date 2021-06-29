# Import PuLP modeler functions
from pulp import *
import pandas as pd
import numpy as np
from itertools import combinations
import math as mt
from scipy.spatial import distance


if __name__ == "__main__":

   InputData = "../data/InputDataEnergySmallInstance.xlsx"

  # Input Data Preparation #
   def read_excel_data(filename, sheet_name):
        data = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        values = data.values
        if min(values.shape) == 1 and max(values.shape) == 1:  # This If is to make the code insensitive to column-wise or row-wise expression #
            values = values.tolist()
            return values[0]        
        else:
            data_dict = {}
            if min(values.shape) == 1:  # For signle-dimension parameters in Excel
                if values.shape[0] == 1:
                    for i in range(values.shape[1]):
                        data_dict[i+1] = values[0][i]
                else:
                    for i in range(values.shape[0]):
                        data_dict[i+1] = values[i][0]
                
            else:  # For two-dimension (matrix) parameters in Excel
                for i in range(values.shape[0]):
                    for j in range(values.shape[1]):
                        data_dict[(i+1, j+1)] = values[i][j]
            return data_dict
   # Edges Length Calculation #
   
   def edgesLengthCalculation(NodesCord):
       EdgesLength = {}
    
       for i in range(1,int(len(NodesCord)/2)+1):
           for j in range(1,int(len(NodesCord)/2)+1):
               EdgesLength[i,j] = distance.euclidean([NodesCord[i,1],NodesCord[i,2]],[NodesCord[j,1],NodesCord[j,2]]);
       return EdgesLength
   
        
   # This section contains all required data #
   
   # Create the sets
   #set_V = read_excel_data(InputData, "Nodes")
   nodeNum = read_excel_data(InputData, "Nodes")[0]
   set_V = [i for i in range(1,nodeNum+1)]


   # Source id
   SourceNum = read_excel_data(InputData, "SourceNum")
   index_V0 = SourceNum[0]
   
   
   # Coordinates of each node #
   #param_nodes_coordx = read_excel_data(InputData, "NodesCord(pxi)")
   #param_nodes_coordy = read_excel_data(InputData, "NodesCord(pyi)")
   NodesCord = read_excel_data(InputData, "NodesCord")
   
   param_L = edgesLengthCalculation(NodesCord)
   
   FixedUnitCost = read_excel_data(InputData, "FixedUnitCost")[0]

   # Fixed Investment Cost "euro/m" #
   param_cfix = {}
   for i in set_V:
       for j in set_V:
           if i != j:
               param_cfix[i,j] = param_L[i,j] * FixedUnitCost
    
   # Variable Investment Cost "euro/(m kW)" #
   param_cvar = read_excel_data(InputData, "cvar(cijvar)")
   
   # Operation & Maintenance Cost "euro/(m a)" #
   param_com = read_excel_data(InputData, "com(cijom)")
   
   # Heat Generation Cost "euro/(kW h)" #
   param_cheat = read_excel_data(InputData, "cheat(ciheat)")
   print(param_cheat)
   
   # Revenue for Delivered Heat "euro/(kW h)" #
   param_crev = read_excel_data(InputData, "crev(cijrev)")
   
   # Annuity Factor for Investment Cost "1/a" #
   param_Alpha = read_excel_data(InputData, "Alpha")
   
   # Fixed Thermal Losses "kW/m" #
   param_vfix = read_excel_data(InputData, "vfix(thetaijfix)")
   
   # Variable Thermal Losses "kW/(kW m)" #
   param_vvar = read_excel_data(InputData, "vvar(thetaijvar)")
   
   # Full Load Hours of Source Vertices "h/a" #
   param_Tflh = read_excel_data(InputData, "Tflh(Tiflh)")
   
   # Concurrence Effect "1" #
   param_Betta = read_excel_data(InputData, "Betta")
   
   
   # Connect Quota "1" #
   param_Lambda = read_excel_data(InputData, "Lambda")
   
   
   # Edge Peak Demand "kW" #
   param_d = read_excel_data(InputData, "EdgesDemandPeak(dij)")
   
   
   # Edge Peak Demand "kWh/a" #
   param_D = read_excel_data(InputData, "EdgesDemandAnnual(Dij)")
   
   
   # Maximum Pipe Capacity "kW" #
   param_Cmax = read_excel_data(InputData, "Cmax(cijmax)")
   
   
   # Penalty of Unmet Demand #
   param_pumd = read_excel_data(InputData, "pumd(pijumd)")
   
   
   # Source Vertex Capacity "kW" #
   param_Qmax = read_excel_data(InputData, "SourceMaxCap(Qimax)")

   

  
   param_Etta = {}
   for i in set_V:
       for j in set_V:
           param_Etta[(i, j)] = 1 - param_L[i, j] * param_vvar[i, j]

   param_Delta = {}
   for i in set_V:
       for j in set_V:
           param_Delta[(i, j)] = param_d[i, j] * param_Betta[0] * param_Lambda[0] + param_L[i, j] * param_vfix[i, j]

   # Create the decision variables
   #If edge (i,j) is constructed or not
   x_var = LpVariable.dicts('x', (set_V, set_V), 0, 1, cat="Binary")

   # Thermal power flow from vertex vi into edge (i,j)
   Pin_var = LpVariable.dicts('Pin', (set_V, set_V), 0)

   # Thermal power flow out of edge (i,j) into vertex vj
   Pout_var = LpVariable.dicts('Pout', (set_V, set_V), 0)

   # Auxiliary variables

   # Total Revenue variable
   TotalRevenue_var = LpVariable('TotalRevenue', lowBound=0, cat='Continuous')

   # TotalHeatGenerationCost
   TotalHeatGenerationCost_var = LpVariable('TotalHeatGenerationCost', lowBound=0, cat='Continuous')

   # TotalHeatGenerationCost
   TotalFixedInvestmentCost_var = LpVariable('TotalFixedInvestmentCost', lowBound=0, cat='Continuous')

   # TotalHeatGenerationCost
   TotalMaintenanceCost_var = LpVariable('TotalMaintenanceCost', lowBound=0, cat='Continuous')

   # TotalHeatGenerationCost
   TotalVariableInvestmentCost_var = LpVariable('TotalVariableInvestmentCost', lowBound=0, cat='Continuous')

   # TotalHeatGenerationCost
   UnmetDemandPenalty_var = LpVariable('UnmetDemandPenalty', lowBound=0, cat='Continuous')

   # TotalHeatGenerationCost
   #ObjectiveValue_var = LpVariable('ObjectiveValue', lowBound=0, cat='Continuous')

   # Create the 'District_Heating' variable to contain the problem data
   District_Heating = LpProblem("District_Heating", LpMinimize)

   # The objective function is added to 'District_Heating'
   #District_Heating += TotalHeatGenerationCost_var + TotalMaintenanceCost_var + TotalFixedInvestmentCost_var + \
   #                    TotalVariableInvestmentCost_var + UnmetDemandPenalty_var - TotalRevenue_var
   
   print(param_Tflh[0],param_cheat[index_V0],param_Betta[0])
   District_Heating += (param_Tflh[0] * param_cheat[index_V0] / param_Betta[0]) * lpSum(Pin_var[index_V0][j] for j in set_V if (j != index_V0))+ \
                        lpSum(param_com[i, j] * param_L[i, j] * x_var[i][j] for i in set_V for j in set_V if (j != i))+ \
                        lpSum(param_cfix[i, j] * param_L[i, j] * param_Alpha[0] * x_var[i][j] for i in set_V for j in set_V if (j != i))+ \
                        lpSum(param_cvar[i, j] * param_L[i, j] * param_Alpha[0] * Pin_var[i][j] for i in set_V for j in set_V if(j != i))+ \
                        lpSum(0.5 * param_pumd[i, j] * param_D[i, j] * (1 - x_var[i][j] - x_var[j][i]) for i in set_V for j in set_V if (j != i))- \
                        lpSum(param_crev[i, j] * param_D[i, j] * param_Lambda[0] * x_var[i][j] for i in set_V for j in set_V if (j != i))
   print(District_Heating)
   

   # ***--------------- Constraints ----------------- ***

   print(len(set_V))
   
   # Tree structure of the network
   District_Heating += lpSum(x_var[i][j] for i in set_V for j in set_V if (i != j)) == len(set_V) - 1
   
   # Unidirectionality
   for i in set_V:
       for j in set_V:
           if i != j: District_Heating += x_var[i][j] + x_var[j][i] <=1

   # Demand satisfaction constraints
   for i in set_V:
        for j in set_V:
            if i != j: District_Heating += param_Etta[i,j] * Pin_var[i][j] - Pout_var[i][j] == param_Delta[i,j] * x_var[i][j]

   # Flow equilibrium at each vertex ****
   for i in set_V:
        if i != index_V0: District_Heating += lpSum(Pin_var[i][m] for m in set_V if (i != m and m!= index_V0)) == lpSum(Pout_var[m][i] for m in set_V if (i != m))

#   Edge capacity constraint
   for i in set_V:
        for j in set_V:
            if i != j: District_Heating += Pin_var[i][j] <= param_Cmax[i,j] * x_var[i][j]

#   Source structural constraint
   District_Heating += lpSum(x_var[i][index_V0] for i in set_V if (i != index_V0)) == 0

   #   Source's heat generation capacity
   District_Heating += lpSum(Pin_var[index_V0][j] for j in set_V if (j != index_V0)) <= param_Qmax[index_V0]

   #   Tour elimination
   for i in set_V:
       if(i != index_V0): District_Heating += lpSum(x_var[j][i] for j in set_V if (j != i)) == 1

   # The problem is solved using PuLP's choice of Solver (the default solver is Coin Cbc)
   District_Heating.solve()

   # The status of the solution is printed to the screen
   print("Status:", LpStatus[District_Heating.status])

   # The optimal value of the decision variables and the
   # optimised objective function value is printed to the screen
   for v in District_Heating.variables():
      if v.varValue > 0:
         print(v.name, "=", v.varValue)

   print ("Objective value District_Heating = ", value(District_Heating.objective))
