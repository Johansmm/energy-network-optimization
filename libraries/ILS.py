## Libraries ###########################################################
import pandas as pd
import numpy as np
import time, random, sys
from datetime import datetime
from itertools import combinations
import math as mt
from colorama import Fore
import colorama

solve_method = False # True for equation mode, False for flow estimation

## Functions and Class #################################################
class ILS():
    ''' Attributes '''
    
    ''' Constructor '''
    # Read the parameters and save them #
    def __init__(self, parameters):
        ''' Input format:
            parameters: Data from excel file
            Output format:
            edge_based = [(h1,sp1),(d2,sp2),...,(hM,sM)] with M = N + solution_num
        '''
        # Input data preparation
        colorama.init()
        random.seed(datetime.now())
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

        self.edge_based, self.spokes, self.hubs  = [], [], [] # Restart edge based representation, spokes and hubs lists.

        # Create the decision variables
        self.X, self.Pin, self.Pout = {},{},{}
        for i in self.set_V:
            for j in self.set_V:
                self.X[(i,j)] = 0
                self.Pin[(i,j)] = 0.0
                self.Pout[(i,j)] = 0.0

    ''' Methods '''
    # Evaluate the object function #
    def objectFunction(self):
        revenue, heat_gen_cost, maintenance_cost = 0,0,0
        fixed_inv_cost, variable_inv_cost, unmet_demand_pen_cost = 0,0,0
        
        for j in self.set_V:
            for i in self.set_V:
                revenue += self.ci_rev[i,j]*self.Dij[i,j]*self.X[i,j]
                maintenance_cost += self.cij_om[i,j]*self.lij[i,j]*self.X[i,j]
                fixed_inv_cost += self.cij_fix[i,j]*self.lij[i,j]*self.X[i,j]
                variable_inv_cost += self.cij_var[i,j]*self.lij[i,j]*self.Pin[i,j]
                unmet_demand_pen_cost += self.Pij_umd[i,j]*self.Dij[i,j]*(1 - self.X[i,j] - self.X[j,i])
            heat_gen_cost += self.Pin[self.v0,j]

        revenue *= self.lam
        heat_gen_cost *= (self.Ti_flh * self.ci_heat[self.v0]) / self.beta
        fixed_inv_cost *= self.alpha
        variable_inv_cost *= self.alpha
        unmet_demand_pen_cost *= 0.5
        Z = heat_gen_cost + maintenance_cost + fixed_inv_cost + variable_inv_cost + unmet_demand_pen_cost - revenue
        return Z

    # Evaluate all constraints and show if anybody is unsatisfied #
    def constrainsEval(self, warnings_off = False):
        tree_struct_sum = 0.0
        for j in self.set_V:
            flow_eq_sum_Pin, flow_eq_sum_Pout, tour_elim_sum = 0.0, 0.0, 0.0
            for i in self.set_V:
                if self.X[i,j] != 0 and self.X[i,j] != 1: # Constraint 9: Domain of variables
                    if not warnings_off: print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Constraint 9 unsatisfied. X[{},{}] != 0 or 1.".format(i,j) + Fore.RESET)
                    return False
                if self.Pin[i,j] < 0.0: # Constraint 9: Domain of variables
                    if not warnings_off: print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Constraint 9 unsatisfied. Pin[{},{}] < 0.".format(i,j) + Fore.RESET)
                    return False
                if self.Pout[i,j] < 0.0: # Constraint 9: Domain of variables
                    if not warnings_off: print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Constraint 9 unsatisfied. Pout[{},{}] < 0.".format(i,j) + Fore.RESET)
                    return False

                tree_struct_sum += self.X[i,j] # Constraint 1: Tree structure

                if i != j: 
                    if self.X[i,j] + self.X[j,i] > 1: #  Constraint 2: Unidirectionality
                        if not warnings_off: print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Constraint 2 unsatisfied in ({},{}).".format(i,j) + Fore.RESET)
                        return False

                    if np.abs(self.eta[i,j]*self.Pin[i,j] - self.Pout[i,j] - self.delta[i,j]*self.X[i,j]) > 1e-4 : # Constraint 3: Demand satisfaction
                        if not warnings_off: print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Constraint 3 unsatisfied in ({},{}).".format(i,j) + Fore.RESET)
                        return False

                    if j != self.v0: # Constraint 4 - 8
                        flow_eq_sum_Pin += self.Pin[j,i]
                        flow_eq_sum_Pout += self.Pout[i,j]
                        tour_elim_sum += self.X[i,j]
                    
                    if self.Pin[j,i] > self.Cij_max[j,i]*self.X[j,i]: # Constraint 5: Edge capacity constraint
                        if not warnings_off: 
                            print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Constraint 5 unsatisfied in ({},{}).".format(j,i) + Fore.RESET)
                            print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Pin = {}, Cmax = {}, X = {}.".format(self.Pin[j,i],self.Cij_max[j,i],self.X[j,i]) + Fore.RESET)
                        return False

            if np.abs(flow_eq_sum_Pin - flow_eq_sum_Pout) > 1e-4: # Constraint 4: Flow equilibrium at each vertex
                if not warnings_off: print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Constraint 4 unsatisfied with j = {}.".format(j) + Fore.RESET)
                return False

            if tour_elim_sum < 1 and j != self.v0: # Constraint 8: Tour elimination
                if not warnings_off: print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Constraint 8 unsatisfied with i = {}.".format(j) + Fore.RESET)
                return False
        
        if tree_struct_sum != self.nodes_num - 1: # Constraint 1: Tree structure
            if not warnings_off: print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Constraint 1 unsatisfied." + Fore.RESET)
            return False

        source_struct_sum, source_heat_gen_sum = 0.0, 0.0
        for m in self.set_V:
            if m != self.v0: 
                source_struct_sum += self.X[m,self.v0] # Constraint 6: Source structural constraint
                source_heat_gen_sum += self.Pin[self.v0,m] # Constraint 7: Source's heat generation capacity

        if source_struct_sum != 0: # Constraint 6: Source structural constraint
            if not warnings_off: print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Constraint 6 unsatisfied." + Fore.RESET)
            return False

        if len(self.Qi_max) == 1: # Constraint 7: Source's heat generation capacity
            if source_heat_gen_sum > self.Qi_max[0]: 
                if not warnings_off: print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Constraint 7 unsatisfied." + Fore.RESET)
                return False
        else:
            if source_heat_gen_sum > self.Qi_max[self.v0]: 
                if not warnings_off: print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Constraint 7 unsatisfied." + Fore.RESET)
                return False
        return True

    # Organize the edge from source to spokes #
    def flowOrganization(self, current_node, previous_none = None):
        for edge_ind in range(len(self.edge_based)):
            next_node = None
            if self.edge_based[edge_ind][0] == current_node and self.edge_based[edge_ind][1] != previous_none: 
                next_node = self.edge_based[edge_ind][1]
            if self.edge_based[edge_ind][1] == current_node and self.edge_based[edge_ind][0] != previous_none: 
                next_node = self.edge_based[edge_ind][0]
            if next_node is not None:
                self.edge_based[edge_ind] = (current_node, next_node) # Coordinates update
                if next_node not in self.spokes: self.flowOrganization(next_node, current_node)
    
    # Find the spokes and hubs lists #
    def updateHubsSpokes(self):
        xcor = [x[0] for x in self.edge_based]
        ycor = [x[1] for x in self.edge_based]
        total_conex = xcor + ycor
        total_conex = {i:total_conex.count(i) for i in total_conex}
        self.spokes = [key for key, x in total_conex.items() if x == 1] # Nodes with only one conection
        self.hubs = [x for x in self.set_V if x not in self.spokes] # The rest

    # Update the values of variables decisition # 
    # Note: The flow powers is estimated with 3 constrains: Demand, Flow equilibrium and Edge Capacity in equation option#
    def variablesUpdate(self):
        # tic = time.time() # PROOF
        self.updateHubsSpokes()
        self.flowOrganization(self.v0) # Organizate flow conexions.
        # print("Flow organizate = {}".format(self.edge_based)) # Proof
        
        ''' Decision variables update'''
        for i in self.set_V: # Reset values
            for j in self.set_V:
                self.X[(i,j)] = 0
                self.Pin[(i,j)] = 0.0
                self.Pout[(i,j)] = 0.0
        
        if solve_method: ## Equation solve using Constraints 3,4, 5 (Pin = 0 if X = 0)
            for edge in self.edge_based: self.X[edge] = 1 # Edge turn on
            num_eq = 0 
            matrix_sys = np.zeros((self.nodes_num - 1,self.nodes_num - 1))
            source_sys = np.zeros((self.nodes_num - 1,1))
            for j in self.set_V: # Estimate the self.nodes_num - 1 equations, with Pin != 0 (where X[i,j] != 0)
                if j != self.v0:
                    for i in self.set_V:
                        if self.X[(i,j)] == 1:
                            pos_var = self.edge_based.index((i,j))
                            matrix_sys[num_eq, pos_var] = self.eta[i,j]
                            source_sys[num_eq] += self.delta[i,j]
                        if self.X[(j,i)] == 1:
                            pos_var = self.edge_based.index((j,i))
                            matrix_sys[num_eq, pos_var] = -1.0
                    num_eq = num_eq + 1
            
            Power_in = np.linalg.solve(matrix_sys,source_sys) # Pin estimate with flow equilibrium constraint
            for edge_ind in range(len(self.edge_based)): # Power_in assigned
                edge = self.edge_based[edge_ind]
                self.Pin[edge] = Power_in[edge_ind][0] # Pin[i,j] != 0
                if np.abs(self.Pin[edge]) < 1e-3: self.Pin[edge] = 0.0 # Errase values near to zero
                self.Pout[edge] = self.eta[edge]*self.Pin[edge] - self.delta[edge] # Pout[i,j] != 0
                if np.abs(self.Pout[edge]) < 1e-3: self.Pout[edge] = 0.0
        else: ## Flow analysis solution
            for edge in self.edge_based: # Edge turn on only in spokes 
                if edge[1] in self.spokes: 
                    self.X[edge] = 1
                    self.Pin[edge] = self.delta[edge]/self.eta[edge]
            hub_conexion, isUpdate = 0, True
            hubs_list = self.hubs + [self.v0]
            # print(self.edge_based)
            while isUpdate: # Pin and Pout estimated
                hub_conexion += 1
                hub_dict, isUpdate = {}, False
                for hub in [h for h in self.hubs if h != self.v0]:
                    node_conex = [node for node in self.edge_based if (node[0] == hub and node[1] in hubs_list)\
                        or (node[1] == hub and node[0] in hubs_list)] # Hub's conexions
                    if hub_conexion == len(node_conex): hub_dict[hub], isUpdate = node_conex, True

                for _ in range(len(hub_dict)):
                    for hub, node_conex in hub_dict.items():
                        start_node = [node for node in node_conex if node[1] == hub][0]
                        if self.X[start_node] == 0:
                            hub_analysis = True
                            for node in [node for node in self.edge_based if node[0] == hub]:
                                if self.X[node] == 1: self.Pout[start_node] += self.Pin[node]
                                else: 
                                    self.Pout[start_node], hub_analysis = 0.0, False
                                    break
                            if hub_analysis:
                                # print(hub, node_conex)
                                if np.abs(self.Pout[start_node]) < 1e-3: self.Pout[start_node] = 0.0
                                self.Pin[start_node] = (self.Pout[start_node] + self.delta[start_node])/self.eta[start_node]
                                if np.abs(self.Pin[start_node]) < 1e-3: self.Pin[start_node] = 0.0
                                self.X[start_node] = 1 # Turn on edge
            # Source power estimation
            for edge_v0 in [node for node in self.edge_based if node[0] == self.v0 and self.X[node] == 0]:
                for node in [node for node in self.edge_based if node[0] == edge_v0[1]]:
                    self.Pout[edge_v0] += self.Pin[node]
                if np.abs(self.Pout[edge_v0]) < 1e-3: self.Pout[edge_v0] = 0.0
                self.Pin[edge_v0] = (self.Pout[edge_v0] + self.delta[edge_v0])/self.eta[edge_v0]
                if np.abs(self.Pin[edge_v0]) < 1e-3: self.Pin[edge_v0] = 0.0
                self.X[edge_v0] = 1
            # print(self.edge_based)
        # print(Fore.CYAN + "[PROOF:] Time spend in variablesUpdate = {} s".format(time.time() - tic) + Fore.RESET)
        # self.printResults()
    
    # First method to generate the random network #
    def generateGreedyNetwork(self):
        self.edge_based = [] # Restart edge based representation
        self.hubs = random.sample(self.set_V, k = random.randint(min(self.set_V),max(self.set_V)))
        self.spokes = [x for x in self.set_V if x not in self.hubs]

        # Hubs created randomly and conect them with Greedy Heuristic
        hubs = [random.choice(self.hubs)] # Initial value
        hubs_dist = {} # Save the edge with its distance
        while len(hubs) < len(self.hubs): # Greedy Heuristic
            hub_min, dist = -1, 0
            for hub in [x for x in self.hubs if x not in hubs]: # Remove hubs organice previously
                if self.lij[hubs[-1], hub] < dist or hub_min == -1: 
                    dist, hub_min = self.lij[hubs[-1], hub], hub
            hubs_dist[(hubs[-1], hub_min)] = dist
            hubs.append(hub_min)

        # Join start hub with end hub and Take len(self.hubs) - 1 conexions, at close distance.
        hubs_dist[(hubs[-1], hubs[0])] = self.lij[hubs[-1], hubs[0]]
        self.edge_based = [edge[0] for edge in sorted(hubs_dist.items(), key = lambda x: x[1])[:len(self.hubs) - 1]]

        # Spokes conect to the nearlest hub
        for spoke in self.spokes:
            hub_min, dist = -1, 0.0
            for hub in self.hubs:
                if self.lij[spoke, hub] < dist or hub_min == -1:
                    dist, hub_min = self.lij[spoke, hub], hub
            self.edge_based.append((hub_min, spoke))

    # Second method to generate the random network #
    def generateRandomNetwork(self, start_node = None):
        self.edge_based = [] # Restart edge based representation
        if start_node is None or start_node not in self.set_V: start_node = random.choice(self.set_V)
        hubs = [start_node]
        spokes = [x for x in self.set_V if x != start_node]
        while len(spokes) > 0:
            spk, hb = random.choice(spokes), random.choice(hubs)
            self.edge_based.append((hb,spk))
            spokes.remove(spk)
            hubs.append(spk)

    ''' User's methods '''
    # Get current configuration #
    def getCurrentER(self):
        return self.edge_based.copy()

    # Set a new current configuration #
    def setCurrentER(self, new_edge_based, warning_off = True): # Set a new current configuration (a valid configuration)
        edge_based = self.edge_based.copy()
        self.edge_based = new_edge_based.copy()
        self.variablesUpdate()
        result = self.constrainsEval(warning_off)
        if not result:
            self.edge_based = edge_based.copy()
            print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Impossible to update configuration. It isn't solution." + Fore.RESET)
        return result

    # Generate the first initial solution #
    def GenerateInitialSolution(self, start_node = None, choose_method = False, warnings_off = False):
        # Generate a network
        if choose_method: self.generateGreedyNetwork()
        else: self.generateRandomNetwork(start_node = start_node)
        # self.edge_based = [(2,3),(4,1),(4,2),(4,5),(4,7),(4,8),(7,6)] # Proof (pulp)
        self.variablesUpdate() # Variable value update
        return self.constrainsEval(warnings_off = warnings_off) # Return if network is a solution or not.

    # Find a minimal solution with initial value and Local Search algorithm #
    def localSearch(self, method_choose = 1, epochs = 100):
        # Input update
        method_save = method_choose
        # self.variablesUpdate()
        if self.constrainsEval():
            min_z = self.objectFunction() # Initial value of object function value
            best_edge = self.edge_based.copy()
        else:
            print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Initial solution invalid. Imposible to apply local search algorithm." + Fore.RESET)
            return self.edge_based.copy(), []

        z_history, spoke_i = [min_z], None
        # print(Fore.RED + "Initial z = {}".format(min_z) + Fore.RESET)
        for i in range(epochs): # Number of executions
            if method_save not in [1,2,3,4]: method_choose = random.choice([1,2,3]) # Random method
            self.variablesUpdate() # Update list spokes and hubs 
            edge_based = self.edge_based.copy()
            if method_choose == 1 or method_choose == 4: # Local Search 1
                index_list = [x for x in range(len(edge_based)) if edge_based[x][1] in self.spokes]
                spoke_list = [edge_based[x][1] for x in index_list]
                start_sp, final_sp = tuple(random.sample(range(len(index_list)), k = 2)) # Choose position randomly
                if final_sp < start_sp: start_sp, final_sp = final_sp, start_sp # final_sp must be bigger than start_sp
                spoke_list = spoke_list[:start_sp] + spoke_list[start_sp:final_sp + 1][::-1] + spoke_list[final_sp + 1:]
                for i in range(len(index_list)): 
                    self.edge_based[index_list[i]] = (edge_based[index_list[i]][0], spoke_list[i]) # Swap from start_sp to final_sp
                self.variablesUpdate() # Update values of decision variables
                if self.constrainsEval(warnings_off=True):
                    z = self.objectFunction() # Evaluate object function with new network
                    if z < min_z: 
                        min_z, best_edge = z, self.edge_based.copy()
                        # print(Fore.CYAN + "Nueva minima red1 = {} with z = {}".format(best_edge, z) + Fore.RESET)
            if method_choose > 1:
                spoke_pv = spoke_i # For method 2 and 3
                while spoke_i == spoke_pv or spoke_i is None: spoke_i = random.choice([x for x in self.spokes if x != self.v0])
                index_spoke_i = [sp[1] for sp in self.edge_based].index(spoke_i)
                hub_spoke = self.edge_based[index_spoke_i][0]

                # print("Choose spoke {} in pos {}".format(spoke_i,index_spoke_i))
                node_list = []
                if method_choose == 2 or method_choose == 4: node_list += [x for x in self.hubs if hub_spoke != x] # Local Search 2
                if method_choose > 2: node_list = [x for x in self.spokes if x != spoke_i] # Local Search 3
                
                # print(Fore.RED + "Random spoke = {}".format(spoke_i) + Fore.RESET)
                # print(Fore.RED + "\nHUBS = {}. SPOKES = {}".format(self.hubs, self.spokes) + Fore.RESET)
                # print(Fore.RED + "({},{}), node_list = {}".format(hub_spoke, spoke_i, node_list) + Fore.RESET)
                for new_hub in node_list: # Local Search 2-3
                    index_spoke_i = [sp[1] for sp in self.edge_based].index(spoke_i)
                    self.edge_based[index_spoke_i] = (new_hub, spoke_i)
                    self.variablesUpdate() # Update values of decision variables
                    if self.constrainsEval(warnings_off=True):
                        z = self.objectFunction() # Evaluate object function with new network
                        # print(" Intento de z = {}".format(z))
                        if z < min_z: 
                            min_z, best_edge = z, self.edge_based.copy()
                            # print(Fore.CYAN + "Nueva minima red2-3 = {} with z = {}".format(best_edge, z) + Fore.RESET)
                    self.edge_based = edge_based.copy()
            self.edge_based = best_edge.copy() # Best solution to next iteration
            z_history.append(min_z)
            
        self.variablesUpdate()
        return self.edge_based.copy(), z_history # Solution and history return

     # Shaking operator implementation as perturbation method #
    def perturbation(self, max_total_operations = 100, num_perturbation = 1):
        self.updateHubsSpokes() # Update list spokes and hubs 
        if len(self.hubs) < 2:
            print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Impossible to apply shaking operator. At least 2 hubs required." + Fore.RESET)
            return
        
        cont = 0
        for _ in range(max_total_operations):
            hub1, hub2 = tuple(random.sample(self.hubs, k = 2)) # Choose two hubs randomly
            edge_based = self.edge_based.copy()
            # print("Initial red: {} with hub1 = {} and hub2 = {}".format(edge_based,hub1,hub2))
            for i in range(len(edge_based)):
                new_edge = edge_based[i]
                if new_edge[0] == hub1: new_edge = (hub2, new_edge[1]) # Exchanbe hub1 with hub2 in hubs position
                elif new_edge[0] == hub2: new_edge = (hub1, new_edge[1])
                if new_edge[1] == hub1: new_edge = (new_edge[0], hub2) # Exchanbe hub1 with hub2 in spokes position
                elif new_edge[1] == hub2: new_edge = (new_edge[0], hub1)
                self.edge_based[i] = new_edge # Exchange save
            # print("Final red: {}".format(self.edge_based))
            self.variablesUpdate() # Update Pin, Pout and X (decision variables)
            if self.constrainsEval(warnings_off = True): 
                cont += 1
                if cont >= num_perturbation: break # Acceptable perturbation. Return
            else: self.edge_based = edge_based.copy() # Restore edge based values

        if self.edge_based == edge_based:
            print(Fore.YELLOW + "["+time.strftime("%x %I:%M:%S %p")+ "][WARNING:] Invalid shaking operator" +
                " in {} iterations. Network doesn't change.".format(max_total_operations) + Fore.RESET)
            self.variablesUpdate() # Update to previous values
    
    # Evaluate and choose the best configuration #
    def acceptanceCriterion(self, new_edge_base, previous_edge_base = None, admission_error = 0.05, **kwargs):
        # Save previous result
        edge_based = self.edge_based.copy()
        if previous_edge_base is None: previous_edge_base = edge_based.copy()

        previous_cons = self.setCurrentER(previous_edge_base.copy())
        previous_z = self.objectFunction()
        new_cons = self.setCurrentER(new_edge_base.copy())
        new_z = self.objectFunction()

        if not (previous_cons or new_cons): self.edge_based = edge_based.copy()
        elif previous_cons and not new_cons: self.edge_based = previous_edge_base.copy()
        elif not previous_cons and new_cons: self.edge_based = new_edge_base.copy()
        elif (new_z - previous_z)/max([new_z, previous_z]) < admission_error: self.edge_based = new_edge_base.copy()
        else: self.edge_based = previous_edge_base.copy()

        self.variablesUpdate()
        return self.edge_based.copy(), self.objectFunction()

    # Print the resutls #
    def printResults(self):
        newline, count = 4, 0
        for i in self.set_V:
            for j in self.set_V:
                if np.abs(self.Pin[(i,j)]) > 1e-3: 
                    count += 1
                    if count % newline == 0: print("Pin_{}_{} = {}".format(i,j,self.Pin[(i,j)]))
                    else: print("Pin_{}_{} = {}".format(i,j,self.Pin[(i,j)]), end = "\t")
        for i in self.set_V: # Print results (proof)
            for j in self.set_V:
                if np.abs(self.Pout[(i,j)]) > 1e-3: 
                    count += 1
                    if count % newline == 0: print("Pout_{}_{} = {}".format(i,j,self.Pout[(i,j)]))
                    else: print("Pout_{}_{} = {}".format(i,j,self.Pout[(i,j)]), end = "\t")
        for i in self.set_V: # Print results (proof)
            for j in self.set_V:
                if np.abs(self.X[(i,j)] - 1) < 1e-3: 
                    count += 1
                    if count % newline == 0: print("X_{}_{} = {}".format(i,j,self.X[(i,j)]))
                    else: print("X_{}_{} = {}".format(i,j,self.X[(i,j)]), end = "\t")
        print("\nZ = {}".format(self.objectFunction()))
                