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
    print('Got a train set of size (%d * %d)' % X_train.shape)
    print('Got %d queries' % len(X_test))

    try:
        t0 = time.time()
        index_size_before = algo.get_index_size("self")
        algo.fit(X_train)
        build_time = time.time() - t0
        index_size = algo.get_index_size("self") - index_size_before
        print('Built index in', build_time)
        print('Index size: ', index_size)

        best_search_time = float('inf')
        for i in range(run_count):
            print('Run %d/%d...' % (i+1, run_count))

            print('  Calculating distance...')
            bf_nn = sklearn.neighbors.NearestNeighbors(algorithm='brute', metric='l2')
            bf_nn.fit(X_train)
            n = X_train.shape[0]
            wrong_edges = 0
            for i in range(n):
                algonghs = algo.query(X_train[i, :], count)
                brutenghs = bf_nn.kneighbors([X_train[i, :]], return_distance=False, n_neighbors=count)[0]
                wrong_edges += numpy.setdiff1d(brutenghs, algonghs).shape[0]
            print('  -> Distance: {:.{prec}f}'.format(wrong_edges / n*count, prec=3))

            print('  Testing...')
            d = X_train.shape[1]
            def query(i):
                return X_train[algo.query(X_train[i, :], count), :].astype('float')
            ga = kg.KNN_Graph(count)
            ga.build(kg.Relation(X_train.astype('float')))
            oa = kg.Query_Oracle(query)
            toa = kg.KNN_Tester_Oracle(oa)
            result = toa.test(ga, count, d, 0.5, 0.001, 0.5)
            print('  -> Decision {} in time {:.{prec}f}, query time of that {:.{prec}f}'
                  .format(result.decision, result.total_time, result.query_time, prec=3))

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


def run_docker(definition, dataset, count, runs, timeout=3*3600, mem_limit=None):
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
        for line in container.logs(stream=True):
            print(colors.color(line.decode().rstrip(), fg='yellow'))

    t = threading.Thread(target=stream_logs, daemon=True)
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
