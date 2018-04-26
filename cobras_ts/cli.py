import argparse
import sys
from pathlib import Path
import logging
import time
import subprocess

import numpy as np
from sklearn import metrics


logger = logging.getLogger("cobra_ts")


description = """COBRAS-TS Time Series Active Semi-Supervised Clustering."""
epilog = """
Copyright 2018 KU Leuven, DTAI Research Group.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog)

    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')

    dist_group = parser.add_argument_group("distance arguments")
    dist_group.add_argument('--dist', choices=['dtw', 'kshape'], default='kshape',
                            help='Distance computation')
    dist_group.add_argument('--dtw-window', metavar='INT', dest='dtw_window', type=int, default=10,
                            help='Window size for DTW')
    dist_group.add_argument('--dtw-alpha', metavar='FLOAT', dest='dtw_alpha', type=float, default=0.5,
                            help='Compute affinity from distance: affinity = exp(-dist * alpha)')

    data_group = parser.add_argument_group("dataset arguments")
    data_group.add_argument('--format', choices=['csv'], help='Dataset format')
    data_group.add_argument('--labelcol', metavar='INT', type=int,
                            help='Column with labels')
    data_group.add_argument('--visual', action='store_true',
                            help='Use visual interface to query constraints if no labels are given')


    parser.add_argument('--budget', type=int, default=100,
                        help='Number of constraints to ask maximally')
    parser.add_argument('input', nargs=1, help='Dataset file')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.WARNING - 10 * args.verbose, logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    budget = args.budget

    data_fn = Path(args.input[0])
    data_format = None
    if args.format is None:
        if data_fn.suffix == '.csv':
            data_format = 'csv'
    else:
        data_format = args.format

    if data_format == 'csv':
        data = np.loadtxt(str(data_fn), delimiter=',')
    else:
        raise Exception("Unknown file format (use the --format argument)")

    if args.labelcol is None:
        from cobras_ts.commandlinequerier import CommandLineQuerier
        series = data
        labels = None
        querier = CommandLineQuerier()
    else:
        nonlabelcols = list(idx for idx in range(data.shape[1]) if idx != args.labelcol)
        series = data[:, nonlabelcols]
        labels = data[:, args.labelcol]
        if args.visual:
            subprocess.call(["bokeh serve cobras_ts/webapp"], shell=True)
            sys.exit(1)
        else:
            from cobras_ts.labelquerier import LabelQuerier
            querier = LabelQuerier(labels)

    if args.dist == 'dtw':
        from dtaidistance import dtw
        from cobras_ts.cobras_dtw import COBRAS_DTW
        window = args.dtw_window
        alpha = args.dtw_alpha
        dists = dtw.distance_matrix(series, window=int(0.01 * window * series.shape[1]))
        dists[dists == np.inf] = 0
        dists = dists + dists.T - np.diag(np.diag(dists))
        affinities = np.exp(-dists * alpha)
        clusterer = COBRAS_DTW(affinities, querier, budget)
    elif args.dist == 'kshape':
        from cobras_ts.cobras_kshape import COBRAS_kShape
        clusterer = COBRAS_kShape(series, querier, budget)
    else:
        raise Exception("Unknown distance type: {}".format(args.dist))

    logger.info("Start clustering ...")
    start_time = time.time()
    clusterings, runtimes, ml, cl = clusterer.cluster()
    end_time = time.time()
    logger.info("... done clustering in {} seconds".format(end_time - start_time))
    print("Clustering:")
    print(clusterings)
    if args.labelcol is not None:
        print("ARI score = " + str(metrics.adjusted_rand_score(clusterings[-1], labels)))
