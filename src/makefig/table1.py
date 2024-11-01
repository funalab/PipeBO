import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys
from utils import *
sys.path.append(get_home_path())
from benchmark.bbob import *
from scipy import stats

parser = argparse.ArgumentParser(description='Make table 1 in paper')
parser.add_argument('-dir', required=True, help='Directory containing the results')
parser.add_argument('-json', required=True, help='Setting file')
args = parser.parse_args()

max = 201
json_f = args.json
batch_size, param_num = read_settings(json_f)

dim = len(param_num)
seq_time = dim + np.arange(int(max/batch_size)) * dim
pipe_time = dim + np.arange(int(max/batch_size)) * 1
time_of_all_bench = []

ave = 0
ave_c = 0
for i in range(1,25):
    dir = args.dir + '/f'+str(i).zfill(2) + '_i01'


    batch_size, param_num = read_settings(json_f)
    function_name = os.path.basename(dir)
    benchmark_class = globals()[function_name]
    problem = benchmark_class(len(param_num))

    # read files of results
    result_files = os.listdir(dir)
    method1=sorted([f for f in result_files if 'Seq' in f and'eval.txt' in f])
    method2=sorted([f for f in result_files if 'Pipe' in f and 'up1_pipelp1' in f and'eval.txt' in f])

    # number of simulations
    simu_num = len(method2)

    best_of_method1 = np.empty((simu_num, int(max/batch_size)))
    best_of_method2 = np.empty((simu_num, int(max/batch_size)))
    interquartile_of_method1 = np.empty((2, int(max/batch_size)))
    interquartile_of_method2 = np.empty((2, int(max/batch_size)))
    methods=[method1, method2]

    for metod_files in methods:
        for f_i, f in enumerate(metod_files):
            table = pd.read_table(dir +'/'+ f)
            output = table['Y']
            output = output.values
            output_mold = np.full((max), min(output))
            output_mold[:len(output)] = output
            output_mold_conv = np.empty(int(max/batch_size))
            output_mold_conv[0] = min(output_mold[:batch_size])

            for v_i, value in enumerate(output_mold):
                if (v_i + 1) % batch_size == 0 and (v_i + 1) / batch_size != 1:
                    if min(output_mold[v_i+1-batch_size:v_i+1]) > output_mold_conv[int((v_i+1)/batch_size)-2]:
                        output_mold_conv[int((v_i+1)/batch_size)-1] = output_mold_conv[int((v_i+1)/batch_size)-2]
                    else:
                        output_mold_conv[int((v_i+1)/batch_size)-1] = min(output_mold[v_i+1-batch_size:v_i+1])
            if 'Seq' in f:
                best_of_method1[f_i, ] = output_mold_conv
            if 'up1_pipelp1' in f:
                best_of_method2[f_i, ] = output_mold_conv

    # plt.scatter(seq_time, np.median(best_of_method1, axis=0))
    # plt.scatter(pipe_time, np.median(best_of_method2, axis=0))
    base_line = np.median(best_of_method1[:,int(np.floor(100 / dim) - 1)])

    time_list = []
    for one_simu in best_of_method2:
        for itr, regret in enumerate(one_simu):
            if regret < base_line:
                time_list.append(itr+dim)
                break
    print('benchmark ' + str(i) ,end='')
    time_of_all_bench.append(time_list)
    if len(time_list) > 25:
        print(' : \t' + str((sorted(time_list)[25]+sorted(time_list)[24])/2),end='')
        ave += (sorted(time_list)[25]+sorted(time_list)[24])/2
        ave_c += 1
        if len(time_list) >= 37:
            print(' (' + str((-sorted(time_list)[12]+sorted(time_list)[37])) + ')' + '\n',end='')
        else:
            print(' (-)')


    else:
        print(' : \t-')

print('average : ' + str(ave / ave_c))