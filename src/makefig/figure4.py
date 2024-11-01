import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys
from utils import *
sys.path.append(get_home_path())
from benchmark.bbob import *
import gc

parser = argparse.ArgumentParser(description='Make table figure 4 in paper')
parser.add_argument('-dir', required=True, help='Directory containing the results')
parser.add_argument('-sdir', required=True, help='Destination directory for output')

args = parser.parse_args()

max = 201
json_f_l = [
    args.dir + '/K03_case01/K03_case01.json',
    args.dir + '/K03_case02/K03_case02.json',
    args.dir + '/K03_case03/K03_case03.json',
    args.dir + '/K03_case04/K03_case04.json',
    args.dir + '/K03_case05/K03_case05.json',
    ]
dir_l = [
    args.dir + '/K03_case01/f',
    args.dir + '/K03_case02/f',
    args.dir + '/K03_case03/f',
    args.dir + '/K03_case04/f',
    args.dir + '/K03_case05/f',
]
data = []

for s in range(5):
    batch_size, param_num = read_settings(json_f_l[s])

    dim = len(param_num)
    seq_time = dim + np.arange(int(max/batch_size)) * dim
    pipe_time = dim + np.arange(int(max/batch_size)) * 1
    time_of_all_bench = []

    battle_result_all = []
    for i in range(1,25):
        dir = dir_l[s] +str(i).zfill(2)+'_i01'


        batch_size, param_num = read_settings(json_f_l[s])
        function_name = os.path.basename(dir)
        benchmark_class = globals()[function_name]
        problem = benchmark_class(len(param_num))

        # read files of results
        result_files = os.listdir(dir)
        method1=sorted([f for f in result_files if 'Pipe' in f and 'up0_pipelp1' in f and'eval.txt' in f])
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
                if 'up0_pipelp1' in f:
                    best_of_method1[f_i, ] = output_mold_conv
                if 'up1_pipelp1' in f:
                    best_of_method2[f_i, ] = output_mold_conv
        battle_result_each = []
        for i in range(1, len(np.median(best_of_method1, axis=0))):
            if np.median(best_of_method1, axis=0)[i] > np.median(best_of_method2, axis=0)[i]:
                battle_result_each.append(1)
            elif np.median(best_of_method1, axis=0)[i] == np.median(best_of_method2, axis=0)[i]:
                battle_result_each.append(0)
            else:
                battle_result_each.append(-1)
        battle_result_all.append(battle_result_each)

    wariai = []
    for i, battle_each_bench in enumerate(battle_result_all):
        count = 0
        for itr in battle_each_bench:
            if itr == 1:
                count += 1
        wariai.append(count/len(battle_each_bench)* 100)
    data.append(wariai)

    del battle_result_all
    del best_of_method1
    del best_of_method2
    gc.collect()

# create data frame
df = pd.DataFrame(data).transpose()
df.columns = ['D=(8,1,1)', 'D=(5,3,2)', 'D=(3,4,3)', 'D=(2,3,5)', 'D=(1,1,8)']

fig, ax = plt.subplots(figsize=(6, 6))

df.boxplot(ax=ax, positions=np.arange(len(df.columns))*1.0,
            boxprops=dict(facecolor='#1E90FF80',
            color='black', linewidth=1),
            widths=0.4, patch_artist=True, medianprops=dict(color='black'), zorder=1,whis=3,)
np.random.seed(1293103)
for i, col in enumerate(df.columns):
    ax.scatter(np.random.normal(i*1.0, 0.07, len(df)), df[col], c="black", zorder=2)

ax.set_xticks(np.arange(len(df.columns))*1.0)
ax.set_xticklabels(df.columns)
ax.set_ylabel('')
ax.set_axisbelow(True)
plt.savefig(args.sdir + '/figure4.pdf')