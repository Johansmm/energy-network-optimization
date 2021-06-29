## Libraries ###########################################################
#try:
import sys, os, argparse, time, platform, colorama, random
import numpy as np
from colorama import Fore

import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import rc

import ILS
## Paths and global variables ##########################################
class ILSgraphics(ILS.ILS):
    ''' Construct '''
    def __init__(self, parameters, legends_off = False):
        # Input data preparation
        rc('text', usetex=True)
        super().__init__(parameters)
        self.filelist = [] # Start without any file
        self.possible_plots = ['z_history','z_history.acceptance','z_history.perturbation',\
        'z_history.best','localSearch.random','localSearch.all','total_time','best_result',\
        'localSearch','localSearch.ini','localSearch.fin','acceptance_history']
        self.legends_off = legends_off

    ''' Users methods'''
    ## Read files, choose statistics and plot them ##
    def statisticsPlots(self, file_directory, statistics = ["z_history"], imformat = 'pdf', show_plots = False):
        # Read and create the file list and get functions to plot (in 'all' case)
        self.readFileList(file_directory)
        if 'all' in statistics: statistics = self.possible_plots.copy() # Plot all statistics

        # Process
        fig, fig_cont = None, 0
        if len(self.filelist) > 1:
            for i in range(len(self.filelist)):
                print(Fore.CYAN+"[" +time.strftime("%x %I:%M:%S %p")+"][INFO]: {}".format(self.filelist[i])+
                    " file renamed as file{}".format(i)+Fore.RESET)
        for fun in [self.selectFunction(fun_id) for fun_id in statistics]:
            if fig is None or len(fig.axes) > 0: fig = plt.figure()
            for json_file in self.filelist:
                experiment_data = self.readJsonFile(json_file)
                fun['function']({'data':experiment_data, 'json_file':json_file,\
                    'fun_name':fun['name'], 'figure': fig, 'axes':fig.axes}) # Corresponding graphic plot
            # if len(fig.axes) == 0: plt.close()
            if len(fig.axes) > 0: 
                fig_cont += 1
                if fig_cont == 1 and not os.path.isdir('image_results'): os.mkdir('image_results')
                if (fun['name'] not in ['localSearch', 'best_result', 'z_history']) and len(self.filelist) == 1: 
                    fig.set_size_inches(8.0/2.54, 8.0/2.54 * 9/16)
                else: fig.set_size_inches(16.0/2.54, 16.0/2.54 * 9/16)
                if (fun['name'] not in ['localSearch', 'best_result', 'z_history']) and (len(self.filelist) > 1) and not self.legends_off:
                    plt.legend(tuple(['file {}'.format(i) for i in range(len(self.filelist))]), ncol=2)
                fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                if len(self.filelist) == 1: outname = self.filelist[0].split(os.sep)[-1].split('.')[0] + '_' + fun['name']
                else: outname = 'output_' + fun['name']
                outname += ".{}".format(imformat)
                if imformat is not None: fig.savefig(os.path.join('image_results', outname))
        
        if show_plots: 
            if fig_cont > 0: plt.show()
            else: plt.close('all')
        print(Fore.GREEN+"\n[" +time.strftime("%x %I:%M:%S %p")+"][INFO]: Total graphics done = {}.".format(fig_cont)+Fore.RESET)

    ''' File / Directory read methods '''
    ## Read a list files inside of directory, or return file if input parameter is a valid file ##
    def readFileList(self, file_directory, ext = 'json'): # Read a valid json-files inside folder / independenly file
        files_list = []
        if os.path.isdir(file_directory): # Return files with 'json' extension
            for root_path, _, files_name in os.walk(file_directory):
                for element in files_name:
                    if element.split(".")[-1].lower() == ext.lower():
                        files_list.append(os.path.join(root_path, element))
            # files_list = [os.path.join(file_directory,x) for x in os.listdir(file_directory) if x.split('.')[-1] == ext]
        elif file_directory.split('.')[-1] == ext: files_list = [file_directory] # Return file inside of list
        self.filelist = files_list.copy()
        return files_list

    ## Get statistics from a specific json file ##
    def readJsonFile(self, file_path):
        try: 
            with open(file_path,'r') as f: json_dict = json.load(f)
        except:
            print(Fore.YELLOW+"[" +time.strftime("%x %I:%M:%S %p")+"][WARNING]: {} file invalid. Ignore.".format(file_path)+Fore.RESET)
            json_dict = {}
        return json_dict
    
    ''' Plot methods '''
    def selectFunction(self, function_name):
        if function_name in self.possible_plots:
            print("[" +time.strftime("%x %I:%M:%S %p")+"][INFO]: Ploting {} ...".format(function_name))
        if function_name == 'total_time': fun = self.timeProcessingPlot
        elif 'z_history' in function_name: fun = self.z_historyPlot
        elif 'localSearch' in function_name: fun = self.localSearchPlot
        elif 'acceptance_history' == function_name: fun = self.acceptancePlot
        elif 'best_result' == function_name: fun = self.bestPlot
        else: fun = lambda *args, **kwargs:\
                print(Fore.YELLOW+"[" +time.strftime("%x %I:%M:%S %p")+"][WARNING]: {} ".format(function_name)+
                "function name invalid (must be {})".format(self.possible_plots)+Fore.RESET)
        return {'name': function_name, 'function': fun}
    
    def timeProcessingPlot(self, prop):
        time_signal = prop['data']['total_time']
        print("[" +time.strftime("%x %I:%M:%S %p")+"][INFO]: Total time processing = {}.".format(np.sum(time_signal))+
            " Averaged time processing = {}.".format(np.mean(time_signal)))
        plt.plot(time_signal)
        if len(prop['axes']) == 0:
            plt.xlabel("Epochs")
            plt.ylabel("Processing time / s")
            plt.grid('both')

    def acceptancePlot(self, prop):
        acceptance_signal = prop['data']['acceptance_history']
        plt.plot(acceptance_signal)
        if len(prop['axes']) == 0:
            plt.xlabel("Epochs")
            plt.ylabel("Acceptance value")
            plt.grid('both')

    def z_historyPlot(self, prop):
        zplot = prop['fun_name']
        zsignals = []
        if zplot == 'z_history' or "acceptance" in zplot: 
            zsignals.append(prop['data']['z_history']['acceptance'])
            color = 'b'
        if zplot == 'z_history' or "perturbation" in zplot: 
            zsignals.append(prop['data']['z_history']['perturbation'])
            color = 'r'
        if zplot == 'z_history' or "best" in zplot: 
            zsignals.append(prop['data']['z_history']['best'])
            color = 'g'
        if len(zsignals) > 0:
            if len(zsignals) == 1 and len(prop['axes']) == 0: 
                plt.plot(np.array(zsignals).T, color)
            else: 
                plt.plot(np.array(zsignals).T)
            if len(prop['axes']) == 0:
                plt.xlabel("Epochs")
                plt.ylabel(r"Total expense (\texteuro)/year")
                plt.grid('both')
                if len(zsignals) > 1 and not self.legends_off:
                    plt.legend(('Acceptance cost','Perturbation cost','Best cost'))

    def localSearchPlot(self, prop):
        ls_plot = prop['fun_name']
        ls_signals, ls_time, legends, colors = [], [], [], []
        data = prop['data']['z_history']['localSearch']
        if ls_plot == 'localSearch' or "all" in ls_plot:
            sub_data, sub_time = [], []
            for epoch in data.keys():
                data_len = len(data[epoch])
                sub_data += data[epoch]
                sub_time += list(np.array(range(data_len))/(data_len-1) + eval(epoch))
            ls_signals.append(sub_data)
            ls_time.append(sub_time)
            legends.append('LS-algor in each epoch')
            colors.append('g')
        options = [x for x in ['ini','fin','random'] if x in ls_plot or ls_plot == 'localSearch']
        if len(options) > 0:
            keys = list(data.keys())
            for str_comp in options:
                if "ini" == str_comp: 
                    index = keys[0]
                    legends.append('LS-algor in first epoch')
                    colors.append('b')
                elif "fin" == str_comp: 
                    index = keys[-1]
                    legends.append('LS-algor in last epoch')
                    colors.append('m')
                else: 
                    index = random.choice(keys)
                    legends.append('LS-algor in random epoch ({})'.format(index))
                    colors.append('r')
                ls_signals.append(data[index])
                ls_time.append(list(range(len(data[index]))))
        if len(ls_signals) > 1:
            plot_ord = {231:1,232:3,233:2,212:0}
            titles = {1:'(a)',3:'(b)',2:'(c)',0:'(d)'}
            if len(prop['axes']) == 0: self.axes = {}
            for subp_id, i in plot_ord.items():
                if len(prop['axes']) == 0:
                    self.axes[i] = plt.subplot(subp_id)
                    plt.title(titles[i])
                    plt.plot(ls_time[i],ls_signals[i], colors[i])
                    plt.grid('both')
                    if subp_id == 231 or subp_id == 212: 
                        plt.ylabel(r"Total expense (\texteuro)/year")
                    if subp_id == 212: plt.xlabel("Epochs")
                else: self.axes[i].plot(ls_time[i],ls_signals[i])
        if len(ls_signals) == 1: 
            if len(prop['axes']) == 0:
                plt.plot(ls_time[0],ls_signals[0],colors[0])
                plt.grid('both')
                plt.xlabel("LS epochs")
                plt.ylabel(r"Total expense (\texteuro)/year")
                if not self.legends_off: plt.legend(legends)
            else:
                plt.plot(ls_time[0],ls_signals[0])

    def bestPlot(self, prop):
        # Import best solution
        best_config = prop['data']['best_result']['config']
        nodes = [tuple(x) for x in best_config]
        self.setCurrentER(nodes) # Inherent function to set a new configuration

        # Print best results with inherent function
        print("[" +time.strftime("%x %I:%M:%S %p")+"][INFO]: Description of better result:\n")
        print("[" +time.strftime("%x %I:%M:%S %p")+"][INFO]: Network configuration = {}".format(nodes))
        print("[" +time.strftime("%x %I:%M:%S %p")+"][INFO]: Flow powers, edge activations and cost function to {}:".format(prop['json_file']))
        self.printResults()

        # Plot network configuration
        centers = np.array([list(node) for node in self.nodes.values()])
        plt.scatter(centers[:, 0], centers[:, 1], marker = 'o', c = "white", alpha = 0.8, s = 200, edgecolor = 'k')
        for id_node, coor in self.nodes.items():
            if id_node != self.v0:
                plt.scatter(coor[0], coor[1], marker = "${}$".format(id_node), alpha = 0.8, s = 60, edgecolor = 'k')
            else:
                plt.scatter(coor[0], coor[1], marker = "${}$".format(id_node), alpha = 0.8, s = 100, edgecolor = 'r')

        # Plot connections
        color = list(random.choices(np.linspace(0.0,1.0,256), k=3))
        arrow_params = {'length_includes_head': True, 'shape': 'right', 'head_starts_at_zero': False,
        'fc': color, 'width':0.3, 'ec':color}
        for edge in best_config: # edge = (source, destiny) and self.nodes[i] = (i-coor_x, i-coor_y)
            x,y = self.nodes[edge[0]][0], self.nodes[edge[0]][1]
            dx,dy = self.nodes[edge[1]][0] - x, self.nodes[edge[1]][1] - y
            xc,yc = x + 0.5*dx, y + 0.5*dy

            plt.arrow(xc, yc, dx*0.001, dy*0.001, alpha=0.7, head_width=2.5, head_length=3, **arrow_params)
            plt.arrow(x , y, dx, dy, alpha=0.2, head_width=0.0, head_length=0.0, **arrow_params)        
        
        if len(prop['axes']) == 0:
            plt.xlabel(r"$x$-coordinate")
            plt.ylabel(r"$y$-coordinate")
            