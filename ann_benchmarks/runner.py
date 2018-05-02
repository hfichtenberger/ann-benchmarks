import argparse
import datetime
import docker
import json
import multiprocessing.pool
import numpy
import os
import psutil
import requests
import sys
import sklearn.neighbors
import threading
import time

from ann_benchmarks.datasets import get_dataset, DATASETS
from ann_benchmarks.algorithms.definitions import Definition, instantiate_algorithm
from ann_benchmarks.distance import metrics
from ann_benchmarks.results import store_results
import KNNTest as kg


def run(definition, dataset, count, run_count=3, force_single=False, use_batch_query=False):
    algo = instantiate_algorithm(definition)

    D = get_dataset(dataset)
    X_train = numpy.array(D['train'])
    X_test = numpy.array(D['test'])
    distance = D.attrs['distance']
    neighbors = D['neighbors']
    
    # X_test ist Teilmenge von X_train, wir haben die neighbors fuer X_test, daher testen wir X_test

    print('Got a train set of size (%d * %d)' % X_train.shape)
    print('Got %d queries' % len(X_test))
    try:
        t0 = time.time()
        index_size_before = algo.get_index_size("self")
        algo.fit(X_test)
        build_time = time.time() - t0
        index_size = algo.get_index_size("self") - index_size_before
        print('Built index in', build_time)
        print('Index size: ', index_size)

        best_search_time = float('inf')
        for i in range(run_count):
            print('Run %d/%d...' % (i+1, run_count))

            print('  Calculating distance...')
                            
            n = X_test.shape[0]
            wrong_edges = 0
            for i in range(n):
                algonghs = algo.query(X_test[i], count)
                brutenghs = neighbors[i, : count]
                wrong_edges += numpy.setdiff1d(brutenghs, algonghs).shape[0]

            print('  -> Distance: {:.{prec}f}'.format(wrong_edges / (n*count), prec=3))

            print('  Testing...')
            
            def query(i):
                return X_test[algo.query(X_test[i], count), :].astype('float')
                
            ga = kg.KNN_Graph(count)
            ga.build(X_test.astype('float'))
            
            toa = kg.KNN_Tester_Oracle(kg.Query_Oracle(query))
            toa.c1_auto_calculation = False
            
            c1_set = [0.05, 0.5, 5]
            c2_set = [0.001, 0.01, 0.1]
            for x in c1_set:
                for y in c2_set:
                    toa.c1 = x
                    toa.c2 = y
                    result = toa.test(ga, count, 0.5)
                    print('  -> Decision {} in time {:.{prec}f}, query time of that {:.{prec}f}, with c1={:.{prec}f}, c2={:.{prec}f}'
                          .format(result.decision, result.total_time, result.query_time, x, y, prec=5))

        #store_results(dataset, count, definition, attrs, results)
    finally:
        algo.done()


def run_from_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    parser.add_argument(
        '--algorithm',
        required=True)
    parser.add_argument(
        '--module',
        required=True)
    parser.add_argument(
        '--constructor',
        required=True)
    parser.add_argument(
        '--count',
        required=True,
        type=int)
    parser.add_argument(
        '--json-args',
        action='store_true')
    parser.add_argument(
        '-a', '--arg',
        dest='args', action='append')
    args = parser.parse_args()
    if args.json_args:
        algo_args = [json.loads(arg) for arg in args.args]
    else:
        algo_args = args.args

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None, # not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args
    )
    run(definition, args.dataset, args.count)


def run_docker(definition, dataset, count, runs, timeout=5*3600, mem_limit=None):
    cmd = ['--dataset', dataset,
           '--algorithm', definition.algorithm,
           '--module', definition.module,
           '--constructor', definition.constructor,
           '--count', str(count),
           '--json-args']
    for arg in definition.arguments:
        cmd += ['--arg', json.dumps(arg)]
    print('Running command', cmd)
    client = docker.from_env()
    if mem_limit is None:
        mem_limit = psutil.virtual_memory().available
    print('Memory limit:', mem_limit)
    container = client.containers.run(
        definition.docker_tag,
        cmd,
        volumes={
            os.path.abspath('ann_benchmarks'): {'bind': '/home/app/ann_benchmarks', 'mode': 'ro'},
            os.path.abspath('data'): {'bind': '/home/app/data', 'mode': 'ro'},
            os.path.abspath('results'): {'bind': '/home/app/results', 'mode': 'rw'},
        },
        mem_limit=mem_limit,
        detach=True)

    def stream_logs():
        import colors
        with open("results/log.txt", "a") as f:
            f.write("{} {} {}\n".format(dataset, definition.algorithm, str(count)))
            for line in container.logs(stream=True, stdout=True, stderr=True):
                text = colors.color(line.decode().rstrip(), fg='yellow')
                print(text)
                f.write("{}\n".format(text))
                f.flush()

    t = threading.Thread(target=stream_logs)#, daemon=True)
    t.start()
    try:
        exit_code = container.wait(timeout=timeout)

        # Exit if exit code
        if exit_code == 0:
            return
        elif exit_code is not None:
            raise Exception('Child process raised exception %d' % exit_code)

    finally:
        container.remove(force=True)
