import sys
import os
import json
import pandas as pd
from sklearn.decomposition import PCA
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', \
        required=True, \
        dest = 'input_matrix',
        help='The input matrix'
    )
    parser.add_argument('-s', '--samples', \
        required=False, \
        dest = 'samples',
        help=('A comma-delimited list of the samples to run PCA on. Without'
            ' this argument, PCA is run on all samples.'
        )
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # read the input matrix:
    working_dir = os.path.dirname(args.input_matrix)
    f = os.path.join(working_dir, args.input_matrix)
    if os.path.exists(f):
        # the passed file should always be tab-delimited.
        df = pd.read_table(f, index_col=0, sep='\t')
    else:
        sys.stderr.write('Could not find file: %s' % f)
        sys.exit(1)

    # if a subset of samples was requested, subset the matrix:
    if args.samples:
        samples_from_mtx = set(df.columns.tolist())
        requested_sample_list = [x.strip() for x in args.samples.split(',')]
        requested_sample_set = set(requested_sample_list)
        difference_set = requested_sample_set.difference(samples_from_mtx)
        if len(difference_set) > 0:
            sys.stderr.write('Requested samples differed from those in matrix: {csv}'.format(
                csv = ','.join(difference_set)
            ))
            sys.exit(1)
        df = df[requested_sample_list]

    # check the size of the matrix. If either the number of samples or features is less than 2,
    # we can't use the solver. Practically, there is no utility in running such a PCA anyway.
    df_shape = df.shape
    # if the number of features is less than 2, there's no point to PCA anyway
    if (df_shape[0] < 2):
        sys.stderr.write('The number of clustering features is less than 2, and a 2-D PCA would' 
             ' not make sense here. Typically, PCA is used to visualize high-dimensional data with many'
             ' more clustering features than 2.'
        )
        sys.exit(1)
    if (df_shape[1] < 2):
        sys.stderr.write('The number of samples is less than 2, and a 2-D PCA would' 
             ' not make sense here. Typically, PCA is used to visualize high-dimensional data'
             ' such that relationships between samples can be explored.'
        )
        sys.exit(1)


    # now run the PCA
    pca = PCA(n_components=2)

    # fill any NAs with zeros
    df = df.fillna(0)

    try:
        # the fit_transform method expects a (samples, features) orientation
        transformed = pca.fit_transform(df.T)
    except Exception as ex:
        sys.stderr.write('Encountered an exception while calculating the princpal components. Exiting.')
        sys.exit(1)

    t_df = pd.DataFrame(
        transformed.T, # note the transform so the resulting matrix matches our convention (Samples in cols)
        columns=df.columns, 
        index=['pc1', 'pc2']
    )
    fout = os.path.join(working_dir, 'pca_output.tsv')
    t_df.to_csv(fout, sep='\t')

    outputs = {
        'pca_coordinates': fout,
        'pc1_explained_variance':pca.explained_variance_ratio_[0],
        'pc2_explained_variance': pca.explained_variance_ratio_[1]
    }
    json.dump(outputs, open(os.path.join(working_dir, 'outputs.json'), 'w'))
