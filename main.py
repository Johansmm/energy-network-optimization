## Libraries ###########################################################
import pandas as pd
import numpy as np
import sys, time, os, json, argparse
import matplotlib.pyplot as plt
from colorama import Fore
from multiprocessing import Process, Queue

from libraries import ILS, PULP, ILSgraphics

## Paths and global variables ##########################################
InputData = "data/InputDataEnergyLargeInstance.xlsx"
ILS_parameters = {
    "Generation":{
        u"choose_method": False, # True: generateGreedy(), False: generateRandomly()
        u"start_node": 4 # Initial node, None = Random initial node
    },
    "Perturbation":{
        u"max_total_operations": 1000, # Total operation if doesn't found a valid perturbation
        u"num_perturbation": 1 # Number of perturbations
    },
    "LocalSearch":{
        u"method_choose": 0, # Method in [0,1,2,3,4]. 0 option to random choose method
        u"epochs": 250 # Epochs number for each LS
    },
    "AcceptanceCriterion":{
        u"admission_error": 0.001, # Percentage of criteria admitted
        u"lenght_data_comp": 20, # Data amount to be compared for the purpose of the algorithm. None = Ignore this feature. 
        u"alpha": 0.1 # Linear increasly in admission_error value
    },
    "General":{
        u"ILS_epochs": 1000, # Total Epochs
        u"lenght_data_comp": 375, # Data amount to be compared for the purpose of the algorithm. None = Ignore this feature.
        u"std_comp": 0.01 # Percentage comparison value
    }
}

## Functions ###########################################################
def read_excel_sheets_names(filename):
    data = pd.read_excel(filename, None)
    return list(data.keys())

def read_excel_data(filename, sheet_name):
    data = pd.read_excel(filename, sheet_name = sheet_name, header = None)
    values = data.values
    # This If is to make the code insensitive to column-wise or row-wise expression #
    if min(values.shape) == 1 and max(values.shape) == 1: 
        values = values.tolist()
        return values[0]
    else:
        data_dict = {}
        if min(values.shape) == 1:  # For single-dimension parameters in Excel
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

def delayProcess(iqueue): # Show 
    count, message = 0, None
    while True:
        if not iqueue.empty(): 
            if message is not None: print("\033[F" + message + ".   ")
            message, count = iqueue.get(), 0
            print() # New message
        if message is not None:
            count += 1
            print("\033[F" + message + "."*count + "   ")
            if count % 3 == 0: count = 0
            # sys.stdout.flush()
            time.sleep(1) # Wait 1 second

## Algorithms ##########################################################
def pulp_algorithm(ILS_comp = None): # Find solution using PULP method
    pulp_obj = PULP.PULP(excel_data) # PULP object creation
    pulp_obj.objective_function() # Join object function to model
    pulp_obj.constraints() # Join constraints to model
    if ILS_comp is not None:
        ILS_comp = ILSPlotResults(ILS_comp, statistics = ['best_result'], legends_off = True, imformat = None)
    pulp_obj.system_solution(ILS_edge_based = ILS_comp) # Model solve and graphics
        
def ILS_algoritm(config_file = None): # Find solution using ILS method
    # Variables initialization
    tic = time.time() # Initial time
    tic_t = tic
    local_search = ILS.ILS(excel_data) # Object with ILS functions
    z_history, total_time, z_ls = {}, [], {} # Cost function, time processing and individual LS cost function by epochs
    z_history['acceptance'], z_history['perturbation'], z_history['best'], z_history['localSearch'] = [], [], [], z_ls
    genname = 'greddy' if ILS_parameters['Generation'][u"choose_method"] else 'random'
    outname = 'results/output_{}{}_pert{}_ls{}_acep{}.json'.format(genname, ILS_parameters['Generation'][u"start_node"],\
        ILS_parameters['Perturbation'][u"num_perturbation"], ILS_parameters['LocalSearch'][u"method_choose"], \
        ILS_parameters['AcceptanceCriterion'][u"admission_error"])
    statistics = {}
    statistics['z_history'], statistics['total_time'], statistics['best_result'] = z_history, total_time, {}
    statistics['acceptance_history'] = []

    # Start subprocess
    iqueue = Queue(maxsize=1)
    p_detect = Process(target = delayProcess, args=(iqueue,))
    p_detect.daemon = True
    p_detect.start()

    # Generate initial point until it's a solution and apply LS in this solution
    if iqueue.empty(): iqueue.put("["+time.strftime("%x %I:%M:%S %p")+ "][INFO:] Creating an initial solution and applying LS")
    while True: 
        if local_search.GenerateInitialSolution(**ILS_parameters["Generation"], warnings_off = True):
            s_aster, z_ls[0] = local_search.localSearch(**ILS_parameters["LocalSearch"]) # Save initial value as better initial solution
            if local_search.objectFunction() < 85e6: break 
    
    s_best, z_min = s_aster.copy(), local_search.objectFunction()
    z_history['acceptance'].append(z_min)
    z_history['perturbation'].append(z_min)
    z_history['best'].append(z_min)
    statistics['acceptance_history'].append(ILS_parameters["AcceptanceCriterion"][u"admission_error"])
    total_time.append(time.time() - tic)
    if iqueue.empty(): 
        iqueue.put("["+time.strftime("%x %I:%M:%S %p")+ "][INFO:] Initial solution with LS created in {} s with cost function (z) = {}"\
        .format(total_time[-1], z_history['acceptance'][-1]))

    # local_search.setCurrentER([(10,6),(12,30),(13,12),(14,24),(15,22),(18,13),(19,2),(19,25),(1,26),(20,19),\
    #    (21,18),(22,5),(22,7),(23,3),(24,27),(25,23),(26,28),(27,16),(29,14),(2,17),\
    #    (2,21),(3,8),(4,15),(4,20),(4,9),(5,10),(6,29),(7,1),(9,11)], warning_off = False) # OPTIMAL SOLUTION

    if iqueue.empty(): iqueue.put("["+time.strftime("%x %I:%M:%S %p")+ "][INFO:] Iterative Local Search start..")
    accept_copy = ILS_parameters["AcceptanceCriterion"][u"admission_error"]
    for epochs in range(ILS_parameters["General"]["ILS_epochs"]):
        tic = time.time() # Time restart
        local_search.setCurrentER(s_aster)
        local_search.perturbation(**ILS_parameters["Perturbation"]) # Solution perturbation
        s_prima, z_ls[epochs + 1] = local_search.localSearch(**ILS_parameters["LocalSearch"]) # And local search algorithm
        z_history['perturbation'].append(local_search.objectFunction())
        s_aster, z_aster = local_search.acceptanceCriterion(s_prima, s_aster, **ILS_parameters["AcceptanceCriterion"]) # Comparation between previous results
        if z_aster < z_min: s_best, z_min = s_aster.copy(), z_aster # Best solution update
        z_history['acceptance'].append(z_aster)
        z_history['best'].append(z_min)
        total_time.append(time.time() - tic)
        if iqueue.empty(): 
            iqueue.put("["+time.strftime("%x %I:%M:%S %p")+ "][INFO:] Iteration {} of {}. Cost = [{}, {}, {}]. Iteration time = {} s. Accepvalue = {}"\
                .format(epochs, ILS_parameters["General"]["ILS_epochs"], round(z_history['best'][-1],2), \
                round(z_history['acceptance'][-1],2), round(z_history['perturbation'][-1],2), round(total_time[-1],2), \
                round(ILS_parameters["AcceptanceCriterion"][u"admission_error"],5)))
        
        # Stop condition
        if (ILS_parameters["General"]["lenght_data_comp"] is not None) and (len(z_history['best']) > ILS_parameters["General"]["lenght_data_comp"])\
            and np.std(z_history['best'][-ILS_parameters["General"]["lenght_data_comp"]:]) < ILS_parameters["General"]["std_comp"]*z_min: break

        # Acceptance parameter dinamics
        if (ILS_parameters["AcceptanceCriterion"]["lenght_data_comp"] is not None) and (len(z_history['acceptance']) > ILS_parameters["AcceptanceCriterion"]["lenght_data_comp"])\
            and np.std(z_history['acceptance'][-ILS_parameters["AcceptanceCriterion"]["lenght_data_comp"]:]) < 1e-3:
            ILS_parameters["AcceptanceCriterion"][u"admission_error"] *= (1 + ILS_parameters["AcceptanceCriterion"][u"alpha"])
        else:
            ILS_parameters["AcceptanceCriterion"][u"admission_error"] = accept_copy

        statistics['best_result']['cost'] = z_min
        statistics['best_result']['config'] = s_best.copy()
        statistics['acceptance_history'].append(ILS_parameters["AcceptanceCriterion"][u"admission_error"])
        with open(outname,'w') as f: json.dump(statistics, f) # Statistics save

    print(Fore.GREEN + "["+time.strftime("%x %I:%M:%S %p")+ "][INFO:] Total execution time: {} s. Best solution = {} with Z = {}."\
        .format(time.time() - tic_t, s_best, z_min) + Fore.RESET)
    local_search.setCurrentER(s_best)
    local_search.printResults()
    p_detect.terminate()

    ## Graphics
    ILSPlotResults(outname, show_plots = True) # Statistics graphics with ILSgraphics object

def ILSPlotResults(directory_file_path, statistics = ['all'], legends_off = False, show_plots = False, imformat = 'pdf'):
    ILSgraph = ILSgraphics.ILSgraphics(excel_data, legends_off = legends_off) # Object with methods to graphics results
    ILSgraph.statisticsPlots(directory_file_path, statistics=statistics, show_plots=show_plots, imformat=imformat)
    return ILSgraph.getCurrentER()

## Algorithms ##########################################################
def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                    usage="%(prog)s [-h] [-dbf DATABASE_FILENAME] {PULP,ILS,ILSgraphics}", 
                                    description='Program to execute MILP and ILS algorithms, and graphics the ILS results.',
                                    epilog='''Example: \n\t python %(prog)s PULP # Execute MILP algorithm with {} file.'''.format(InputData) + 
                                    '''\n\t python %(prog)s -dbf data.xlsx ILS # Execute ILS algorithm with parameters in data.xlsx file.''' + 
                                    '''\n\t python %(prog)s # Execute ILS algorithm with {} file.'''.format(InputData) + 
                                    '''\n\nNote: For more information, run the help of each program with the -h flag.'''+
                                    '''\nExample: python %(prog)s PULP -h # Execute PULP help.''')
    
    # General arguments and subparsers creation
    parser.add_argument('-dbf', '--database_filename', default = InputData, type = str, help = '''Data base file name with algorithms parameters (default = %(default)s).''')
    subparsers = parser.add_subparsers(help='Kind of programs to execute', dest='which')

    # PULP subparser    
    subparsePULP = subparsers.add_parser('PULP', help='MILP algorithm based on PULP library.')
    subparsePULP.add_argument('-ILS', '--ILS_compared', type=str, default=None, help='''External ILS result to compare with MILP (default = %(default)s).''')

    # ILS subparser
    subparseILS = subparsers.add_parser('ILS', help='Iterative Local Search (ILS) algorithm, whose core is LS algoritm.')
    subparseILS.add_argument('-cf', '--config_file', type=str, default=None, help='''External parameters to ILS algorithm (default = %(default)s).''')

    # ILSgraphics subparser
    subparseGraphics = subparsers.add_parser('ILSgraphics', help='Program to plot results obtained by ILS algorithm.')
    subparseGraphics.add_argument('-dfp', '--directory_file_path', type=str, help='''File or folder path, with file(s) to plot.''')
    subparseGraphics.add_argument('-s', '--statistics', nargs='+', default=['all'], help='''Plot types to be made (default = %(default)s).''')
    subparseGraphics.add_argument('-lo', '--legends_off', action='store_true',  help='''Legends turn off  (default = %(default)s).''')
    subparseGraphics.add_argument('-sp', '--show_plots', action='store_true',  help='''Show plots in console (default = %(default)s).''')
    subparseGraphics.add_argument('-if', '--image_format', type=str, default='pdf',  help='''Image format (default = %(default)s).''')
    return parser.parse_args()

def main(args):
    global sheets_list, excel_data
    sheets_list = read_excel_sheets_names(args.database_filename)
    excel_data = {}
    for col in sheets_list: excel_data[col] = read_excel_data(args.database_filename, col)
    
    if args.which == 'PULP': pulp_algorithm(args.ILS_compared) # PULP algoritm executed
    elif args.which == 'ILSgraphics': 
        ILSPlotResults(args.directory_file_path, args.statistics, args.legends_off, \
            args.show_plots, args.image_format) # Print results get with ILS (json files)
    else: ILS_algoritm(args.config_file) # PULP algoritm executed    

if __name__ == "__main__":  
    os.system("clear")
    main(parse_args())
