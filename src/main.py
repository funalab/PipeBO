# directory management for input/output
import os

# import benchmark function
from benchmark.bbob import *

# for json
import json

# import of GPyOpt system
import GPyOpt
from GPyOpt.methods.pipelining_bayesian_optimization import PipeliningBayesianOptimization
from numpy.random import seed

# import others
import numpy as np
from distutils.util import strtobool

import argparse
parser = argparse.ArgumentParser(description='Run PipeBO')
parser.add_argument('-bf', required=True, help='benchmark function')
parser.add_argument('-json', required=True, help='json file for problem setting')
parser.add_argument('-rs', required=True, type=int,help='seed')
parser.add_argument('-iter', required=True, type=int, help='number of iterations')
parser.add_argument('-exec', help='How to run: sequential BO or PipeBO (if none both)')
parser.add_argument('--up', action='store_false', help='Whether to perform intermadiate update')
parser.add_argument('--pipelp', action='store_false', help='Whether to use local penaliser')
args = parser.parse_args()

# about -exec
if args.exec == None:
    exec_seq = True
    exec_pipe = True
elif args.exec == 'seq':
    exec_seq = True
    exec_pipe = False
elif args.exec == 'pipe':
    exec_seq = False
    exec_pipe = True
else :
    exec_seq = False
    exec_pipe = False

# read json
bounds = []
with open(args.json, 'r') as f:
    jsn = json.load(f)
    batch_size = jsn['batch']
    num_cores = jsn['num_cores']
    param_num = []
    domain = []
    for process_key in jsn['parameter'].keys():
        param_num.append(len(jsn['parameter'][process_key]))
        for parameter in jsn['parameter'][process_key]:
            domain.append({'name' : parameter['name'], 'type': parameter['type'], 'domain' : tuple(parameter['domain'])})
            bounds.append(tuple(parameter['domain']))
    param_num = tuple(param_num)

# Creating a file to store the results
dimension = sum(param_num)
output_dir = '../data/' + os.path.splitext(os.path.basename(args.json))[0] + '/' + args.bf
os.makedirs(output_dir, exist_ok=True)

# Creation of benchmark function classes and setting of condition ranges, etc.
benchmark_class = globals()[args.bf]
black_box_function = benchmark_class(dimension)

max_iter = args.iter
id = 'rs'+str(args.rs).zfill(3) + '_' + 'batch' + str(batch_size).zfill(2)
id_pipe = 'rs'+str(args.rs).zfill(3) + '_' + 'batch' + str(batch_size).zfill(2) + '_up' + str(int(args.up)) + '_pipelp' + str(int(args.pipelp))

def random_initial_design(bounds, points_count):
    """
    Function to randomly generate initial condition settings
    """
    dim = len(bounds)
    Z_rand = np.zeros(shape=(points_count, dim))
    for k in range(0,points_count):
        for l in range(0,dim):
            Z_rand[k,l] = np.random.uniform(low=bounds[l][0], high=bounds[l][1])
    return Z_rand

seed(args.rs)
pre_initial_design = random_initial_design(bounds = bounds, points_count= len(param_num) * batch_size)
initial_design = pre_initial_design.reshape(len(param_num) * batch_size, sum(param_num))


if batch_size > 1:
    if exec_pipe:
        seed(args.rs)
        BO_supersclar = PipeliningBayesianOptimization(f=black_box_function.bbob_f,
                                        domain = domain,
                                        acquisition_type = 'LCB',
                                        X = initial_design,
                                        normalize_Y = True,
                                        evaluator_type = 'local_penalization_pipelining',
                                        batch_size = batch_size,
                                        num_cores = num_cores,
                                        acquisition_jitter = 0,
                                        maximize=True,
                                        process_setting=param_num,
                                        intermediate_update=args.up,
                                        other_pipeline_LP=args.pipelp)

        BO_supersclar.run_pipelining_optimization(max_iter,
                                        report_file= output_dir + '/Pipelinig_' + id_pipe + '_report.txt',
                                        evaluations_file= output_dir + '/Pipelinig_' + id_pipe + '_eval.txt',
                                        models_file= output_dir + '/Pipelinig_' + id_pipe + '_model.txt')
    if exec_seq:
        seed(args.rs)
        BO_seq = GPyOpt.methods.BayesianOptimization(f=black_box_function.bbob_f,
                                                    domain = domain,
                                                    acquisition_type = 'LCB',
                                                    X = initial_design[0, :].reshape(1, sum(param_num)),
                                                    normalize_Y = True,
                                                    evaluator_type = 'local_penalization',
                                                    batch_size = batch_size,
                                                    num_cores = num_cores,
                                                    acquisition_jitter = 0,
                                                    maximize=True)

        BO_seq.run_optimization(max_iter,
                                        report_file= output_dir + '/Sequential_' + id + '_report.txt',
                                        evaluations_file= output_dir + '/Sequential_' + id + '_eval.txt',
                                        models_file= output_dir + '/Sequential_' + id + '_model.txt')
else:
    if exec_pipe:
        seed(args.rs)
        BO_pipe = PipeliningBayesianOptimization(f=black_box_function.bbob_f,
                                                    domain = domain,
                                                    acquisition_type = 'LCB',
                                                    X = initial_design,
                                                    normalize_Y = True,
                                                    evaluator_type = 'local_penalization_pipelining',
                                                    batch_size = batch_size,
                                                    num_cores = num_cores,
                                                    acquisition_jitter = 0,
                                                    maximize=True,
                                                    process_setting=param_num,
                                                    intermediate_update=args.up,
                                                    other_pipeline_LP=args.pipelp)

        BO_pipe.run_pipelining_optimization(max_iter,
                                        report_file= output_dir + '/Pipelinig_' + id_pipe + '_report.txt',
                                        evaluations_file= output_dir + '/Pipelinig_' + id_pipe + '_eval.txt',
                                        models_file= output_dir + '/Pipelinig_' + id_pipe + '_model.txt')
    if exec_seq:
        seed(args.rs)
        BO_seq = GPyOpt.methods.BayesianOptimization(f=black_box_function.bbob_f,
                                                    domain = domain,
                                                    acquisition_type = 'LCB',
                                                    X = initial_design[0, :].reshape(1, sum(param_num)),
                                                    normalize_Y = True,
                                                    evaluator_type = 'local_penalization',
                                                    batch_size = batch_size,
                                                    num_cores = num_cores,
                                                    acquisition_jitter = 0,
                                                    maximize=True)

        BO_seq.run_optimization(max_iter,
                                        report_file= output_dir + '/Sequential_' + id + '_report.txt',
                                        evaluations_file= output_dir + '/Sequential_' + id + '_eval.txt',
                                        models_file= output_dir + '/Sequential_' + id + '_model.txt')