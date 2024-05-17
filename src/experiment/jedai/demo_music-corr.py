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



def area():
    d1 = pd.read_csv("../../dataset/music/50-corr/area.csv", sep=',')
    gt = pd.read_csv("../../dataset/music/50-corr/area_dups.csv", sep=',')
    attr = ['name']

    data =Data(dataset_1=d1,
        id_column_name_1='area',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="area")

    # delta = [0.7,0.75,0.75,0.755,0.9]
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
    
def artist():
    d1 = pd.read_csv("../../dataset/music/50-corr/artist.csv", sep=',')
    gt = pd.read_csv("../../dataset/music/50-corr/artist_dups.csv", sep=',')
    attr = ['name','sort_name']

    data =Data(dataset_1=d1,
        id_column_name_1='artist',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="artist")

    # delta = [0.7,0.75,0.75,0.755,0.9]
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
    
    
def artist_credit():
    d1 = pd.read_csv("../../dataset/music/50-corr/artist_credit.csv", sep=',')
    gt = pd.read_csv("../../dataset/music/50-corr/artist_credit_dups.csv", sep=',')
    attr = ['name']

    data =Data(dataset_1=d1,
        id_column_name_1='artist_credit',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="artist_credit")

    # delta = [0.7,0.75,0.75,0.755,0.9]
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
    
    
def label():
    d1 = pd.read_csv("../../dataset/music/50-corr/label.csv", sep=',')
    gt = pd.read_csv("../../dataset/music/50-corr/label_dups.csv", sep=',')
    attr = ['name']

    data =Data(dataset_1=d1,
        id_column_name_1='label',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="label")

    # delta = [0.7,0.75,0.75,0.755,0.9]
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
    
def medium():
    d1 = pd.read_csv("../../dataset/music/50-corr/medium.csv", sep=',')
    gt = pd.read_csv("../../dataset/music/50-corr/medium_dups.csv", sep=',')
    attr = ['release']

    data =Data(dataset_1=d1,
        id_column_name_1='medium',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="medium")

    # delta = [0.7,0.75,0.75,0.755,0.9]
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
    
def place():
    d1 = pd.read_csv("../../dataset/music/50-corr/place.csv", sep=',')
    gt = pd.read_csv("../../dataset/music/50-corr/place_dups.csv", sep=',')
    attr = ['name','address']

    data =Data(dataset_1=d1,
        id_column_name_1='place',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="place")

    # delta = [0.7,0.75,0.75,0.755,0.9]
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
    

    
def recording():
    d1 = pd.read_csv("../../dataset/music/50-corr/recording.csv", sep=',')
    gt = pd.read_csv("../../dataset/music/50-corr/recording_dups.csv", sep=',')
    attr = ['name']

    data =Data(dataset_1=d1,
        id_column_name_1='recording',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="recording")

    # delta = [0.7,0.75,0.75,0.755,0.9]
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
    
def release():
    d1 = pd.read_csv("../../dataset/music/50-corr/release.csv", sep=',')
    gt = pd.read_csv("../../dataset/music/50-corr/release_dups.csv", sep=',')
    attr = ['name']

    data =Data(dataset_1=d1,
        id_column_name_1='release',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="release")

    # delta = [0.7,0.75,0.75,0.755,0.9]
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
    
def release_group():
    d1 = pd.read_csv("../../dataset/music/50-corr/release_group.csv", sep=',')
    gt = pd.read_csv("../../dataset/music/50-corr/release_group_dups.csv", sep=',')
    attr = ['name']

    data =Data(dataset_1=d1,
        id_column_name_1='release_group',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="release_group")

    # delta = [0.7,0.75,0.75,0.755,0.9]
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
    

def track():
    d1 = pd.read_csv("../../dataset/music/50-corr/track.csv", sep=',')
    gt = pd.read_csv("../../dataset/music/50-corr/track_dups.csv", sep=',')
    attr = ['name']

    data =Data(dataset_1=d1,
        id_column_name_1='track',
        ground_truth=gt,
        attributes_1=attr,
        dataset_name_1="track")

    # delta = [0.7,0.75,0.75,0.755,0.9]
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
   results = [area(),artist(),artist_credit(),label(),medium(),place(),recording(),release(),release_group(),track()]
   
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

