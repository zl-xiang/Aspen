import os
import sys
import pandas as pd
import networkx
from networkx import draw, Graph

from pyjedai.datamodel import Data
from pyjedai.utils import print_clusters, print_blocks, print_candidate_pairs
from pyjedai.evaluation import Evaluation
from pyjedai.joins import EJoin, TopKJoin
from pyjedai.matching import EntityMatching
from pyjedai.block_cleaning import BlockFiltering
from pyjedai.clustering import ConnectedComponentsClustering

d1 = pd.read_csv("../../dataset/dblp/dblp.csv", sep=',')
d2 =  pd.read_csv("../../dataset/dblp/acm.csv", sep=',')
gt = pd.read_csv("../../dataset/dblp/DBLP-ACMpm.csv", sep=',')
#attr = list(d1.columns)
attr = ['title','authors']

data =Data(dataset_1=d1,
    id_column_name_1='id',
    ground_truth=gt,
    attributes_1=attr,
    dataset_name_1="dblp",id_column_name_2='id',attributes_2=attr,dataset_2=d2,dataset_name_2='acm')

# delta = [0.7,0.75,0.8,0.85,0.9]
delta = [0.75]
for d in delta:
    print(f'===== Evaluating threshold of {d} =====')
    join = EJoin(similarity_threshold = d,
                metric = 'jaccard',
                tokenization = 'qgrams_multiset',
                qgrams = 2)

    g = join.fit(data)
    _ = join.evaluate(g)

    ec = ConnectedComponentsClustering()
    clusters = ec.process(g, data, similarity_threshold=0.3)
    # print(clusters)
    _ = ec.evaluate(clusters)