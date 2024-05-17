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



def title_baisics():
    d1 = pd.read_csv("../../dataset/imdb/title_basics.csv", sep=',')
    gt = pd.read_csv("../../dataset/imdb/title_basics_dups.csv", sep=',')
    #attr = list(d1.columns)
    attr = ['primaryTitle','originalTitle']

    data =Data(dataset_1=d1,
        id_column_name_1='tconst',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="title_basics")

    # delta = [0.7,0.75,0.8,0.85,0.9]
    delta = [0.7]
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
        
def name_basics():
    d1 = pd.read_csv("../../dataset/imdb/name_basics.csv", sep=',')
    gt = pd.read_csv("../../dataset/imdb/name_basics_dups.csv", sep=',')
    attr = ['primaryName']

    data =Data(dataset_1=d1,
        id_column_name_1='nconst',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="name_basics")

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
        results = ec.evaluate(clusters,export_to_dict=True,with_classification_report=True)
        return (len(gt),results['True Positives'],results['False Positives'],results['False Negatives'])
        
        
if __name__ == "__main__":
    name_basics()

