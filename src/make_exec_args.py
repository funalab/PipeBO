exec_args = open('exec_args.txt', 'w')

import argparse
parser = argparse.ArgumentParser(description='Run PipeBO')
parser.add_argument('-json', required=True, help='json file for problem setting')
parser.add_argument('-rs', required=True, type=int,help='seed')
parser.add_argument('-iter', required=True, type=int, help='number of iterations')
parser.add_argument('-exec', help='How to run: sequential BO[seq] or PipeBO[pipe] (if none both)')
parser.add_argument('--up', action='store_false', help='Do not use intermadiate update in pipelining')
parser.add_argument('--pipelp', action='store_false', help='Do not use local penaliser in pipelining')

methods = ['', ' --up', ' --pipelp', ' --up --pipelp']

args = parser.parse_args()
for j in range(0, args.num):
    for i in range (1, 25):
        exec_args.write('-bf f'+str(i).zfill(2)+'_i01'
                        + ' -rs ' + str(args.rs+j)
                        + ' -iter ' + str(args.iter)
                        + ' -json ' + str(args.json)
                        )
        exec_args.write(' -exec seq')
        exec_args.write('\n')
        for m in methods:
            exec_args.write('-bf f'+str(i).zfill(2)+'_i01'
                            + ' -rs ' + str(args.rs+j)
                            + ' -iter ' + str(args.iter)
                            + ' -json ' + str(args.json)
                            )
            exec_args.write(' -exec pipe')
            exec_args.write(m)
            exec_args.write('\n')