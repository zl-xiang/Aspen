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



def ability():
    d1 = pd.read_csv("../../dataset/pokemon/py-em/ability.csv", sep=',')
    gt = pd.read_csv("../../dataset/pokemon/py-em/ability_dups.csv", sep=',')
    attr = ['generation','is_main_series']

    data =Data(dataset_1=d1,
        id_column_name_1='ability',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="ability")

    # delta = [0.7,0.75,0.8,0.85,0.9]
    delta = [1]
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
    
def item():
    d1 = pd.read_csv("../../dataset/pokemon/py-em/item.csv", sep=',')
    gt = pd.read_csv("../../dataset/pokemon/py-em/item_dups.csv", sep=',')
    attr = ['category','cost','fling_power']

    data =Data(dataset_1=d1,
        id_column_name_1='item',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="item")

    # delta = [0.7,0.75,0.8,0.85,0.9]
    delta = [1]
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
    
    
def move():
    d1 = pd.read_csv("../../dataset/pokemon/py-em/move.csv", sep=',')
    gt = pd.read_csv("../../dataset/pokemon/py-em/move_dups.csv", sep=',')
    attr = ['type','power','accuracy','priority','effect','damage_class']

    data =Data(dataset_1=d1,
        id_column_name_1='move',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="move")

    # delta = [0.7,0.75,0.8,0.85,0.9]
    delta = [1]
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
    
    
def pokemon():
    d1 = pd.read_csv("../../dataset/pokemon/py-em/pokemon.csv", sep=',')
    gt = pd.read_csv("../../dataset/pokemon/py-em/pokemon_dups.csv", sep=',')
    attr = ['name']

    data =Data(dataset_1=d1,
        id_column_name_1='pokemon',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="pokemon")

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
    
def species():
    d1 = pd.read_csv("../../dataset/pokemon/py-em/species.csv", sep=',')
    gt = pd.read_csv("../../dataset/pokemon/py-em/species_dups.csv", sep=',')
    attr = ['evolves_from_species','evolution_chain','color','shape','habitat','gender_rate','capture_rate']

    data =Data(dataset_1=d1,
        id_column_name_1='species',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="species")

    # delta = [0.7,0.75,0.8,0.85,0.9]
    delta = [1]
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
    


def precision(tp,fp):
    return float(tp/(tp+fp))

def recall(tp,fn):
    return float(tp/(tp+fn))

def f1(tp,fp,fn):
    pre = precision(tp,fp)
    re = recall(tp,fn)
    if pre == 0 or re == 0:
        return 0
    else:
        return float(2*(pre*re/(pre+re)))
if __name__ == "__main__":
   results = [ability(),move(),item(),pokemon(),species()]
   
   sum = 0
   tp = 0
   fp = 0
   fn = 0
   for t in results:
       sum+=t[0]
       tp+=t[1]
       fp+=t[2]
       fn+=t[3]
   print(f'p:{precision(tp,fp)}, r:{recall(tp,fn)}, f1:{f1(tp,fp,fn)}')

