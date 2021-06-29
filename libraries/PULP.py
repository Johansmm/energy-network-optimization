## Libraries ###########################################################
import pulp
import numpy as np
import random
from colorama import Fore
import colorama
import matplotlib.pyplot as plt

## Functions and Class #################################################
class PULP():
    def __init__(self, parameters):
        colorama.init()
        plt.rc('text', usetex=True)
        # Input data preparation
        self.v0 = parameters['SourceNum'][0]
        self.nodes_num = parameters['Nodes'][0]
        self.set_V = list(range(1,self.nodes_num + 1))
        
        nodes = parameters['NodesCord']
        self.nodes = {}
        for i in self.set_V: # Coordinates representation for each node
            self.nodes[i] = (nodes[i,1],nodes[i,2])
        
        cfix_unit = parameters['FixedUnitCost'][0]
        self.cij_var = parameters['cvar(cijvar)']
        self.cij_om = parameters['com(cijom)']
        self.ci_heat = parameters['cheat(ciheat)']
        self.ci_rev = parameters['crev(cijrev)']
        self.Pij_umd = parameters['pumd(pijumd)']
        self.alpha = parameters['Alpha'][0]
        self.thetaij_fix = parameters['vfix(thetaijfix)']
        self.thetaij_var = parameters['vvar(thetaijvar)']
        self.Ti_flh = parameters['Tflh(Tiflh)'][0]
        self.beta = parameters['Betta'][0]
        self.lam = parameters['Lambda'][0]
        self.dij = parameters['EdgesDemandPeak(dij)']
        self.Dij = parameters['EdgesDemandAnnual(Dij)']
        self.Cij_max = parameters['Cmax(cijmax)']
        self.Qi_max = parameters['SourceMaxCap(Qimax)']
        self.lij, self.cij_fix = {}, {}
        self.eta, self.delta = {}, {}

        for i in self.set_V:
            for j in self.set_V:
                self.lij[i,j] = np.sqrt((self.nodes[i][0] - self.nodes[j][0])**2 + (self.nodes[i][1] - self.nodes[j][1])**2)
                self.cij_fix[(i,j)] = cfix_unit # * self.lij[(i,j)]
                self.eta[(i,j)] = 1 - self.thetaij_var[i,j]*self.lij[i,j]
                self.delta[(i,j)] = self.dij[i,j]*self.beta*self.lam + self.thetaij_fix[i,j]*self.lij[i,j]
        
        # Create the decision variables
        self.Xij = pulp.LpVariable.dicts('x',(self.set_V,self.set_V),0,1,cat="Binary")
        self.Pij_in = pulp.LpVariable.dicts('Pin',(self.set_V,self.set_V),0)
        self.Pij_out = pulp.LpVariable.dicts('Pout',(self.set_V,self.set_V),0)
        
        # Create variables for the problem
        self.heat_network = pulp.LpProblem("District-Heating-Network", pulp.LpMinimize)

    # Cost function 26320105.769727014
    
    def objective_function(self):
        self.heat_network += (self.Ti_flh*self.ci_heat[self.v0]/self.beta)*pulp.lpSum(self.Pij_in[self.v0][j] for j in self.set_V if j != self.v0) +\
        pulp.lpSum(self.cij_om[(i,j)]*self.lij[(i,j)]*self.Xij[i][j] for i in self.set_V for j in self.set_V if i != j) +\
        pulp.lpSum(self.cij_fix[(i,j)]*self.lij[(i,j)]*self.alpha*self.Xij[i][j] for i in self.set_V for j in self.set_V if i != j) +\
        pulp.lpSum(self.cij_var[(i,j)]*self.lij[(i,j)]*self.alpha*self.Pij_in[i][j] for i in self.set_V for j in self.set_V if i != j) +\
        0.5*pulp.lpSum(self.Pij_umd[(i,j)]*self.Dij[(i,j)] * (1 - self.Xij[i][j] - self.Xij[j][i]) for i in self.set_V for j in self.set_V if i != j) -\
        pulp.lpSum(self.ci_rev[(i,j)]*self.Dij[(i,j)]*self.lam*self.Xij[i][j] for i in self.set_V for j in self.set_V if i != j)

    # Constraints
    def constraints(self):  
        # Tree structure
        self.heat_network += pulp.lpSum(self.Xij[i][j] for i in self.set_V for j in self.set_V if i != j) == abs(self.nodes_num) - 1

        # Unidirectionality
        for i in self.set_V:
            for j in self.set_V:
                if i != j: self.heat_network += self.Xij[i][j] + self.Xij[j][i] <= 1
        
        # Demand satisfaction
        for i in self.set_V:
            for j in self.set_V:
                if i is not j or i is not self.v0:
                    self.heat_network += self.eta[i,j]*self.Pij_in[i][j] - self.Pij_out[i][j] == self.delta[i,j]*self.Xij[i][j]

        # Flow equilibrium at each vertex
        for j in self.set_V:
            if j != self.v0: 
                self.heat_network += pulp.lpSum(self.Pij_in[j][i] for i in self.set_V if i != j) ==\
                pulp.lpSum(self.Pij_out[i][j] for i in self.set_V if (i != j))
        
        # Edge capacity
        for i in self.set_V:
            for j in self.set_V:
                if i != j: self.heat_network += self.Pij_in[i][j] <= self.Cij_max[(i,j)]*self.Xij[i][j]

        # Source structural
        self.heat_network += pulp.lpSum(self.Xij[i][self.v0] for i in self.set_V if i != self.v0) == 0

        # Source's heat generation capacity
        if len(self.Qi_max) == 1:
            self.heat_network += pulp.lpSum(self.Pij_in[self.v0][j] for j in self.set_V if j != self.v0) <= self.Qi_max[0]
        else:
            self.heat_network += pulp.lpSum(self.Pij_in[self.v0][j] for j in self.set_V if j != self.v0) <= self.Qi_max[self.v0]

        # Tour eliminitation
        for i in self.set_V:
            if i != self.v0:
                self.heat_network += pulp.lpSum(self.Xij[j][i] for j in self.set_V if j != i) >= 1

    def system_solution(self, ILS_edge_based = None):
        new_line = 4
        self.heat_network.solve()
        print("Status: {}".format(pulp.LpStatus[self.heat_network.status]))

        count = 0
        edge_based = [] # Edge based representation by tuples
        for v in self.heat_network.variables():
            if v.varValue > 0: 
                count += 1
                if 'x' in v.name: edge_based.append([eval(x) for x in v.name.split('_')[1:]])
                if count % new_line == 0: print("{} = {}".format(v.name, v.varValue))
                elif 'x' in v.name:                     
                    print("{} = {}".format(v.name, v.varValue), end ="\t\t")
                else: print("{} = {}".format(v.name, v.varValue), end ="\t")
        
        print("\nObjective value problem = {}".format(pulp.value(self.heat_network.objective)))

        # Plot network configuration
        fig = plt.figure()
        centers = np.array([list(node) for node in self.nodes.values()])
        plt.scatter(centers[:, 0], centers[:, 1], marker = 'o', c = "white", alpha = 0.8, s = 200, edgecolor = 'k')
        for id_node, coor in self.nodes.items():
            if id_node != self.v0:
                plt.scatter(coor[0], coor[1], marker = "${}$".format(id_node), alpha = 0.8, s = 60, edgecolor = 'k')
            else:
                plt.scatter(coor[0], coor[1], marker = "${}$".format(id_node), alpha = 0.8, s = 100, edgecolor = 'r')

        # Plot PULP connections
        if ILS_edge_based is not None: color = 'b'
        else: color = list(random.choices(np.linspace(0.0,1.0,256), k=3))
        arrow_params = {'length_includes_head': True, 'shape': 'right', 'head_starts_at_zero': False,
        'fc': color, 'width':0.3, 'ec':color}
        for edge in edge_based: # edge = (source, destiny) and self.nodes[i] = (i-coor_x, i-coor_y)
            x,y = self.nodes[edge[0]][0], self.nodes[edge[0]][1]
            dx,dy = self.nodes[edge[1]][0] - x, self.nodes[edge[1]][1] - y
            xc,yc = x + 0.5*dx, y + 0.5*dy

            arrowPUPL = plt.arrow(xc, yc, dx*0.001, dy*0.001, alpha=0.7, head_width=2.5, head_length=3, **arrow_params)
            plt.arrow(x , y, dx, dy, alpha=0.2, head_width=0.0, head_length=0.0, **arrow_params)
        
        # Plot ILS connections
        if ILS_edge_based is not None:
            color = 'r'
            arrow_params['fc'], arrow_params['ec'], arrow_params['shape'] = color, color, 'left'
            for edge in ILS_edge_based: # edge = (source, destiny) and self.nodes[i] = (i-coor_x, i-coor_y)
                x,y = self.nodes[edge[0]][0], self.nodes[edge[0]][1]
                dx,dy = self.nodes[edge[1]][0] - x, self.nodes[edge[1]][1] - y
                xc,yc = x + 0.5*dx, y + 0.5*dy

                arrowILS = plt.arrow(xc, yc, dx*0.001, dy*0.001, alpha=0.7, head_width=2.5, head_length=3, **arrow_params)
                plt.arrow(x , y, dx, dy, alpha=0.2, head_width=0.0, head_length=0.0, **arrow_params)

        plt.xlabel(r"$x$-coordinate")
        plt.ylabel(r"$y$-coordinate")
        fig.set_size_inches(16.0/2.54, 16.0/2.54 * 9/16)
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if ILS_edge_based is not None: 
            plt.legend((arrowPUPL, arrowILS),('PULP configuration','ILS configuration'))
            fig.savefig("outputPULP_ILS.pdf")        
        else: fig.savefig("outputPULP.pdf")
        plt.show()
        return pulp.LpStatus[self.heat_network.status]

