# Run in the directory where plot_optprocess.py is located
# # Need pandas import (not in requirements.txt)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys

from utils import *
sys.path.append(get_home_path())
from benchmark.bbob import *

parser = argparse.ArgumentParser(description='plotting optimization process of seaquential BO and PipeBO')
parser.add_argument('-dir', required=True, help='Directory containing the results')
parser.add_argument('-json', required=True, help='Setting file')
parser.add_argument('-max',type=int, required=True, help='Max. number of data')
parser.add_argument('-sdir', required=True, help='Destination directory for output')
args = parser.parse_args()

# read json
batch_size, param_num = read_settings(args.json)

# benchmark function
function_name = os.path.basename(args.dir)
benchmark_class = globals()[function_name]
problem = benchmark_class(len(param_num))

# read file of results
result_files = os.listdir(args.dir)

seq_files = [f for f in result_files if 'Seq' in f and 'eval' in f]
pipe_up0_pipelp0_files=[f for f in result_files if 'Pipe' in f and 'up0_pipelp0' in f and'eval.txt' in f]
pipe_up0_pipelp1_files=[f for f in result_files if 'Pipe' in f and 'up0_pipelp1' in f and'eval.txt' in f]
pipe_up1_pipelp0_files=[f for f in result_files if 'Pipe' in f and 'up1_pipelp0' in f and'eval.txt' in f]
pipe_up1_pipelp1_files=[f for f in result_files if 'Pipe' in f and 'up1_pipelp1' in f and'eval.txt' in f]

# number of simulations
simu_num = len(seq_files)

best_value_seq = np.empty((simu_num, int(args.max/batch_size)))
best_value_pipe_up0_pipelp0 = np.empty((simu_num, int(args.max/batch_size)))
best_value_pipe_up0_pipelp1 = np.empty((simu_num, int(args.max/batch_size)))
best_value_pipe_up1_pipelp0 = np.empty((simu_num, int(args.max/batch_size)))
best_value_pipe_up1_pipelp1 = np.empty((simu_num, int(args.max/batch_size)))

interquartile_seq = np.empty((2, int(args.max/batch_size)))
interquartile_pipe_up0_pipelp0 = np.empty((2, int(args.max/batch_size)))
interquartile_pipe_up0_pipelp1 = np.empty((2, int(args.max/batch_size)))
interquartile_pipe_up1_pipelp0 = np.empty((2, int(args.max/batch_size)))
interquartile_pipe_up1_pipelp1 = np.empty((2, int(args.max/batch_size)))

methods=[seq_files, pipe_up0_pipelp0_files, pipe_up0_pipelp1_files, pipe_up1_pipelp0_files, pipe_up1_pipelp1_files]
for metod_files in methods:
    for f_i, f in enumerate(metod_files):
        table = pd.read_table(args.dir +'/'+ f)
        output = table['Y']
        output = output.values
        output_mold = np.full((args.max), min(output))
        output_mold[:len(output)] = output
        output_mold_conv = np.empty(int(args.max/batch_size))
        output_mold_conv[0] = min(output_mold[:batch_size])

        for v_i, value in enumerate(output_mold):
            if (v_i + 1) % batch_size == 0 and (v_i + 1) / batch_size != 1:
                if min(output_mold[v_i+1-batch_size:v_i+1]) > output_mold_conv[int((v_i+1)/batch_size)-2]:
                    output_mold_conv[int((v_i+1)/batch_size)-1] = output_mold_conv[int((v_i+1)/batch_size)-2]
                else:
                    output_mold_conv[int((v_i+1)/batch_size)-1] = min(output_mold[v_i+1-batch_size:v_i+1])
        if 'Seq' in f:
            best_value_seq[f_i, ] = output_mold_conv
        if 'up0_pipelp0' in f:
            best_value_pipe_up0_pipelp0[f_i, ] = output_mold_conv
        if 'up0_pipelp1' in f:
            best_value_pipe_up0_pipelp1[f_i, ] = output_mold_conv
        if 'up1_pipelp0' in f:
            best_value_pipe_up1_pipelp0[f_i, ] = output_mold_conv
        if 'up1_pipelp1' in f:
            best_value_pipe_up1_pipelp1[f_i, ] = output_mold_conv

for i in range(0,int(args.max/batch_size)):
    interquartile_seq[0, i] = np.percentile(best_value_seq[:, i], 25)
    interquartile_seq[1, i] = np.percentile(best_value_seq[:, i], 75)
    interquartile_pipe_up0_pipelp0[0, i] = np.percentile(best_value_pipe_up0_pipelp0[:, i], 25)
    interquartile_pipe_up0_pipelp0[1, i] = np.percentile(best_value_pipe_up0_pipelp0[:, i], 75)
    interquartile_pipe_up0_pipelp1[0, i] = np.percentile(best_value_pipe_up0_pipelp1[:, i], 25)
    interquartile_pipe_up0_pipelp1[1, i] = np.percentile(best_value_pipe_up0_pipelp1[:, i], 75)
    interquartile_pipe_up1_pipelp0[0, i] = np.percentile(best_value_pipe_up1_pipelp0[:, i], 25)
    interquartile_pipe_up1_pipelp0[1, i] = np.percentile(best_value_pipe_up1_pipelp0[:, i], 75)
    interquartile_pipe_up1_pipelp1[0, i] = np.percentile(best_value_pipe_up1_pipelp1[:, i], 25)
    interquartile_pipe_up1_pipelp1[1, i] = np.percentile(best_value_pipe_up1_pipelp1[:, i], 75)

dim = len(param_num)
seq_time = dim + np.arange(int(args.max/batch_size)) * dim
pipe_time = dim + np.arange(int(args.max/batch_size)) * 1

plt.rcParams["font.size"] = 40
plt.rcParams["font.family"] = "Arial"
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax.grid()

alpha_val = 0.1
ax.fill_between(seq_time, problem.best + interquartile_seq[0, :], problem.best + interquartile_seq[1, :] , color='blue', alpha=alpha_val)
# ax.fill_between(pipe_time, problem.best + interquartile_pipe_up0_pipelp0[0, :], problem.best + interquartile_pipe_up0_pipelp0[1, :] , color='r', alpha=alpha_val)
# ax.fill_between(pipe_time, problem.best + interquartile_pipe_up0_pipelp0[0, :], problem.best + interquartile_pipe_up0_pipelp0[1, :] , color='hotpink', alpha=alpha_val)
# ax.fill_between(pipe_time, problem.best + interquartile_pipe_up0_pipelp1[0, :], problem.best + interquartile_pipe_up0_pipelp1[1, :] , color='m', alpha=alpha_val)
# ax.fill_between(pipe_time, problem.best + interquartile_pipe_up0_pipelp1[0, :], problem.best + interquartile_pipe_up0_pipelp1[1, :] , color='sienna', alpha=alpha_val)
# ax.fill_between(pipe_time, problem.best + interquartile_pipe_up1_pipelp0[0, :], problem.best + interquartile_pipe_up1_pipelp0[1, :] , color='orange', alpha=alpha_val)
ax.fill_between(pipe_time, problem.best + interquartile_pipe_up1_pipelp1[0, :], problem.best + interquartile_pipe_up1_pipelp1[1, :] , color='g', alpha=alpha_val)

ax.scatter(seq_time, problem.best + np.median(best_value_seq, axis=0), c='b', marker='o', label=('Sequential BO'))
# ax.scatter(pipe_time, problem.best + np.median(best_value_pipe_up0_pipelp0, axis=0), c='r', marker='o', label=('only batch'))
# ax.scatter(pipe_time, problem.best + np.median(best_value_pipe_up0_pipelp0, axis=0), c='hotpink', marker='o', label=('only batch'))
# ax.scatter(pipe_time, problem.best + np.median(best_value_pipe_up0_pipelp1, axis=0), c='m', marker='o', label=('PLAyBOOK-L'))
# ax.scatter(pipe_time, problem.best + np.median(best_value_pipe_up0_pipelp1, axis=0), c='sienna', marker='o', label=('brouwn'))
# ax.scatter(pipe_time, problem.best + np.median(best_value_pipe_up1_pipelp0, axis=0), c='orange', marker='o', label=('Pipelining BO'))
ax.scatter(pipe_time, problem.best + np.median(best_value_pipe_up1_pipelp1, axis=0), c='g', marker='o', label=('PipeBO'))

ax.set_yscale('log')
ax.minorticks_off()
ax.set_xlim(0, int(args.max/batch_size)-1+dim)
# ax.hlines(0.01 * abs(problem.best), 0, 1000, colors='r', linewidth=3)

plt.savefig(args.sdir + '/' + function_name + '_process_median.pdf')