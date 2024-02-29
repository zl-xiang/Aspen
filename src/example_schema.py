from dataloader import Dataloader, drop_columns, atom2df, DBLP_ACM, ISO, UTF8,remove_records_and_save_copy, modify_and_save_column
from dataloader import  ID, PUBLICATION, TITLE, VENUE, YEAR, STRING, NUM, LST, CORA_NAME, load_csv
from metainfo import Schema
import contextc_
import utils
from utils import DF_EMPTY, ATOM_PAT, REL_PAT
from pandas import DataFrame
import pandas as pd
from clingo.control import Control
from clingo import  Symbol, Number, String
from clingo import Model
import pickle
import os
import re
import random

ACTIVE_DOM = "adom"
SEP_COMMA = ','
SEP_AND = ' and '
SEP_AMP = ' & '

SEP_LST = [SEP_COMMA,SEP_AMP,SEP_AND] 

def sampling_df( nrecords:int, df:DataFrame = None,):
        return df.sample(n=nrecords)
    

def save_split(token, df:DataFrame):
    id = 0
    pub_recs = [[f'pid_{token[0]}',f'title_{token[0]}',f'venue_{token[0]}']]
    venue_recs = [[f'vid_{token[0]}',f'name_{token[0]}', f'year_{token[0]}']]
    author_recs = [[f'aid_{token[0]}',f'name_{token[0]}']]
    wrote_recs = [[f'aid_{token[0]}',f'pid_{token[0]}',f'position_{token[0]}']]
    #print(venue_dom.keys())
    for _, row in df.iterrows():
        # p_id = self.t_idx()
        pub_id = row[ID]
        pub = [pub_id,row[TITLE]]

        if str(row['authors']) != DF_EMPTY:
            venue = [id,row[VENUE].strip(),row[YEAR]]
            venue_recs.append(venue)
            pub.append(id)
            id+=1
        else:
            pub.append('')
        pub_recs.append(pub)
        if str(row['authors']) != DF_EMPTY:
            authors = row['authors'].split(',')
            for i,a  in enumerate(authors):
                a = a.strip()
                author = [id,a]
                author_recs.append(author)
                wrote = [id, pub_id, i]
                wrote_recs.append(wrote)
                id+=1
    pub_df = pd.DataFrame(pub_recs[1:], columns=pub_recs[0])
    venue_df = pd.DataFrame(venue_recs[1:], columns=venue_recs[0])
    author_df = pd.DataFrame(author_recs[1:], columns=author_recs[0])
    wrote_df = pd.DataFrame(wrote_recs[1:], columns=wrote_recs[0])
    dfs = {f'pub_{token}':pub_df,f'venue_{token}':venue_df,f'author_{token}':author_df,f'wrote_{token}':wrote_df}
    out_dir = './dataset/benchmarks/DBLP-ACM'
    for k,d in dfs.items():
        d.to_csv(f'{out_dir}/{k}.csv', index=False, header=True,encoding=ISO) 
    return pub_recs, venue_recs, author_recs, wrote_recs 

def save_split_cora( df:DataFrame,ver = 0):
    id = df.shape[0]
    split1 = {'pub':[5,6,9,10,12,13,14,15],'author':[1],'venue':[0,2,3,7,8,11,16,4]}
    split2 = {'pub':[5,10,13,15],'author':[1],'venue':[0,2,3,7,11,16,4]}
    split3 = {'pub':[5,10,13,15],'author':[1],'venue':[0,2,7,11,16]}
    split4 = {'pub':[5,13],'author':[1],'venue':[2,7,11,16]}
    splits = [split1,split2,split3,split4]
    sp = splits[ver]
    pub_recs = [[df.columns[c] for c in sp['pub']]]
    pub_recs[0].append('vid')
    pub_recs[0].append('aid')
    venue_recs = [[df.columns[c] for c in sp['venue']]]
    venue_recs[0].insert(0,'vid')
    author_recs = [['aid','name']]
    # wrote_recs = [[f'aid',f'pid',f'position']]
    # pub_recs = [[f'pid_{token[0]}',f'title_{token[0]}',f'venue_{token[0]}']]
    #venue_recs = [[f'vid_{token[0]}',f'name_{token[0]}', f'year_{token[0]}']]
    #author_recs = [[f'aid_{token[0]}',f'name_{token[0]}']]
    #wrote_recs = [[f'aid_{token[0]}',f'pid_{token[0]}',f'position_{token[0]}']]
    #print(venue_dom.keys())
    for _, row in df.iterrows():
        # p_id = self.t_idx()
        pub = []
        for c in pub_recs[0][:-2]:
            pub.append(row[c])
        pub_id = pub[0]
        #pub_id = row[ID]
        #pub = [pub_id,row[TITLE]]
        venue = []
        for c in venue_recs[0][1:]:
            venue.append(row[c])
        venue.insert(0,str(id))
        venue_recs.append(venue)
        pub.append(str(id))
        id+=1
        if str(row['authors']) != DF_EMPTY:
            author_recs.append([str(id),row['authors']])
            pub.append(id)
            id+=1
        pub_recs.append(pub)
        """
        if str(row['authors']) != DF_EMPTY:
            authors = utils.split(sep_lst=SEP_LST, string=row['authors']) 
            # row['authors'].split(',')
            for i,a  in enumerate(authors):
                a = a.strip()
                if not utils.is_empty(a):                 
                    author = [id,a]
                    author_recs.append(author)
                    wrote = [id, pub_id, i]
                    wrote_recs.append(wrote)
                    id+=1
        """
    pub_df = pd.DataFrame(pub_recs[1:], columns=pub_recs[0])
    venue_df = pd.DataFrame(venue_recs[1:], columns=venue_recs[0])
    author_df = pd.DataFrame(author_recs[1:], columns=author_recs[0])
    #wrote_df = pd.DataFrame(wrote_recs[1:], columns=wrote_recs[0])
    dfs = {f'pub_{str(ver)}':pub_df,f'venue_{str(ver)}':venue_df,f'authors_{str(ver)}':author_df}
    out_dir = './dataset/benchmarks/cora-ref'
    for k,d in dfs.items():
        d.to_csv(f'{out_dir}/{k}.csv', index=False, header=True,encoding=UTF8,float_format='%.0f') 
    return pub_recs, venue_recs, author_recs 

def split_cora():
    CORA = 'cora'
    dl = Dataloader(name = 'cora_tsv',encoding=UTF8)
    tbls = dl.load_data()
    save_split_cora(tbls,)


def split_dblp():
    DBLP = 'dblp'
    ACM = 'acm'
    DATA_PATH = './dataset/benchmarks/DBLP-ACM'
    #dup_rel = {DBLP,ACM}
    path_list = [DATA_PATH+f'/{DBLP}.csv',DATA_PATH+f'/{ACM}.csv',]
    dl = Dataloader(name = 'dblp-acm', path_list= path_list, ground_truth=[DATA_PATH+f'/DBLP-ACM_perfectMapping.csv'],encoding=ISO)
    tbls = dl.load_data()
    for tbl in tbls:
        save_split(tbl[0],tbl[1])
        
  
    

def dblp_non_split_schema(files)->tuple[Schema,Dataloader]:
    file = files[0]
    #sim_attrs = get_reduced_spec(file)[1]
    DBLP = 'dblp'
    ACM = 'acm'
    DATA_PATH = './dataset/benchmarks/DBLP-ACM'
    dup_rel = {DBLP,ACM}
    path_list = [DATA_PATH+f'/{DBLP}.csv',DATA_PATH+f'/{ACM}.csv',]
    dl = Dataloader(name = 'dblp', path_list= path_list, ground_truth=[DATA_PATH+f'/DBLP-ACM_perfectMapping.csv'],encoding=ISO)
    tbls = dl.load_data()
    tbls_dict = {t[0]:(0,t[1]) for t in tbls}
    ref_dict =    {
    ('dblp','id'):('acm','id')}
    schema = Schema('1',f'dblp-acm',tbls_dict,dup_rels=dup_rel,spec_dir=file,refs=ref_dict)
    #schema.set_sim_attrs(sim_attrs)
    return schema, dl

def dblp_split_schema(name:str='',version=0)->tuple[Schema,Dataloader]:
    DBLP = 'dblp'
    ACM = 'acm'
    PUB_DBLP = f'pub_{DBLP}'
    VENUE_DBLP = f'venue_{DBLP}'
    AUTHOR_DBLP = f'author_{DBLP}'
    WROTE_DBLP = f'wrote_{DBLP}'
    PUB_ACM = f'pub_{ACM}'
    VENUE_ACM = f'venue_{ACM}'
    AUTHOR_ACM = f'author_{ACM}'
    WROTE_ACM = f'wrote_{ACM}'
    
    DATA_PATH = './dataset/benchmarks/DBLP-ACM'
    dup_rel = {PUB_DBLP,VENUE_DBLP,AUTHOR_DBLP,PUB_ACM,VENUE_ACM,AUTHOR_ACM}
    path_list = [DATA_PATH+f'/{PUB_DBLP}.csv',DATA_PATH+f'/{VENUE_DBLP}.csv',
                 DATA_PATH+f'/{AUTHOR_DBLP}.csv',DATA_PATH+f'/{WROTE_DBLP}.csv',
                 DATA_PATH+f'/{PUB_ACM}.csv',DATA_PATH+f'/{VENUE_ACM}.csv',
                 DATA_PATH+f'/{AUTHOR_ACM}.csv',DATA_PATH+f'/{WROTE_ACM}.csv']

    ref_dict =    {
     (PUB_DBLP,'venue_d'):(VENUE_DBLP,'vid_d'),
     (WROTE_DBLP,'aid_d'):(AUTHOR_DBLP,'aid_d'),
     (WROTE_DBLP,'pid_d'):(PUB_DBLP,'pid_d'),
     (PUB_ACM,'venue_a'):(VENUE_ACM,'vid_a'),
     (WROTE_ACM,'aid_a'):(AUTHOR_ACM,'aid_a'),
     (WROTE_ACM,'pid_a'):(PUB_ACM,'pid_a')
     }
    
    dl = Dataloader(name = 'dblp-split', path_list= path_list, ground_truth=[DATA_PATH+f'/DBLP-ACM_perfectMapping.csv'],encoding=ISO)
    tbls = dl.load_data()
    tbls_dict = {t[0]:(0,t[1]) for t in tbls}
    schema = Schema('1',f'dblp-split',tbls_dict,dup_rels=dup_rel,refs=ref_dict)
    return schema, dl
    
def cora_schema_(name:str='',version=0)->Schema:
    dl = Dataloader(name = 'cora_tsv')
    tbl = dl.load_data()
    schema = Schema('1',f'cora-{version}',{f'cora':(5,tbl)},dup_rels={'cora'})
    for a in schema.attrs:
        if a.name == 'authors':
            a.is_list = True
    split1 = {'pub':[5,6,9,10,12,13,14,15],'author':[1],'venue':[0,2,3,7,8,11,16,4]}
    split2 = {'pub':[5,10,13,15],'author':[1],'venue':[0,2,3,7,11,16,4]}
    split3 = {'pub':[5,10,13,15],'author':[1],'venue':[0,2,7,11,16]}
    split4 = {'pub':[5,13],'author':[1],'venue':[2,7,11,16]}
    splits = [split1,split2,split3,split4]
    schema.entity_split(schema.rel_index('cora').id,splits[version])
    return schema, dl

#def cora_non_split_schema(name:str='',version=0)->Schema:
 #   dl = Dataloader(name = 'cora_tsv')
  #  tbl = dl.load_data()
   # schema = Schema('1',f'cora-nonsplit-{version}',{f'pub':(5,tbl)})
    #return schema, dl

def cora_non_split_schema(files)->tuple[Schema,Dataloader]:
    CORA = 'cora'
    DATA_PATH = './dataset/benchmarks/cora-ref'
    file = files[0]
    #sim_attrs = get_reduced_spec(file)[1]
    dup_rel = {CORA}
    path_list = [DATA_PATH+f'/{CORA}.tsv']
    dl = Dataloader(name = 'cora-tsv', path_list= path_list, ground_truth=[DATA_PATH+f'/cora_DPL.tsv'],encoding=UTF8)
    tbl = dl.load_data()
    tbls_dict = {'cora':(0,tbl)}
    #print(tbls_dict)
    schema = Schema('1',f'cora',tbls_dict,dup_rels=dup_rel,spec_dir=file)#
    #schema.set_sim_attrs(sim_attrs)
    return schema, dl

def cora_schema(name:str='',version=0)->Schema:
    PUB = f'pub'
    VENUE = f'venue'
    AUTHOR = f'authors'
    #WROTE = f'wrote'
    DATA_PATH = './dataset/benchmarks/cora-ref'
    dup_rel = {PUB,VENUE,AUTHOR}
    path_list = [DATA_PATH+f'/{PUB}.csv',DATA_PATH+f'/{VENUE}.csv',
                 DATA_PATH+f'/{AUTHOR}.csv']
    ref_dict =    {
     (PUB,'vid'):(VENUE,'vid'),
     (PUB,'aid'):(AUTHOR,'aid'),
     #(WROTE,'pid'):(PUB,'id')
     }
    dl = Dataloader(name = 'cora_split',path_list=path_list,ground_truth=[DATA_PATH+f'/cora_DPL.tsv'],encoding=UTF8)
    tbls = dl.load_data()
    tbls_dict = {t[0]:(0,t[1]) for t in tbls}
    schema = Schema('1',f'cora-{version}',tbls_dict,dup_rels=dup_rel,refs=ref_dict)
    return schema, dl

def imdb_schema(split='',files=[])->tuple[Schema,Dataloader]:
    TEST = '' if len(split) == 0 else '_'+split
    T_BASIC = f'title_basics{TEST}' 
    N_BASIC = f'name_basics{TEST}' 
    T_AKA = f'title_akas{TEST}' 
    T_PRINCIPLE = f'title_principals{TEST}' 
    T_RATINGS = f'title_ratings{TEST}' 
    ref_dict =    {
    (T_AKA,'titleId'):(T_BASIC,'tconst'),
     (T_RATINGS,'tconst'):(T_BASIC,'tconst'),
     (T_PRINCIPLE,'tconst'):(T_BASIC,'tconst'),
     (T_PRINCIPLE,'nconst'):(N_BASIC,'nconst'),}
    IMDB_PATH = './dataset/imdb'
    file = files[0]
    #sim_attrs = get_reduced_spec(file)[1]
    dup_rel = {T_BASIC,N_BASIC}
    path_list = [IMDB_PATH+f'/title_akas{TEST}.csv',IMDB_PATH+f'/name_basics{TEST}.csv',IMDB_PATH+f'/title_basics{TEST}.csv',IMDB_PATH+f'/title_principals{TEST}.csv',IMDB_PATH+f'/title_ratings{TEST}.csv']
    dl_imdb = Dataloader(name = 'imdb',path_list=path_list,ground_truth=
                         [IMDB_PATH+f'/name_basics{TEST}_dups.csv',IMDB_PATH+f'/title_basics{TEST}_dups.csv'])
                         #[IMDB_PATH+f'/name_ans{TEST}.csv',IMDB_PATH+f'/title_ans{TEST}.csv'])
                       
    
    tbls = dl_imdb.load_data()   
    tbls_dict = {t[0]:(0,t[1]) for t in tbls}
    dtype_cfg = IMDB_PATH+f'/domain{TEST}.yml'
    order = [N_BASIC,T_AKA,T_BASIC,T_PRINCIPLE,T_RATINGS]
    imdb = Schema(id = '2',name =f'imdb{TEST}',tbls=tbls_dict,refs=ref_dict,dup_rels=dup_rel,attr_types=utils.load_config(dtype_cfg),order=order,spec_dir=file)
    #imdb.set_sim_attrs(sim_attrs)
    return imdb,dl_imdb



def music_schema(split='',files=[])->tuple[Schema,Dataloader]:
    #TEST = '' if len(split) == 0 else '_'+split
    # A = ')'
    # A = ''
    TRACK = f'track' 
    RECORDING = f'recording' 
    MEDIUM = f'medium'
    ARTIST_CREDIT = f'artist_credit' 
    RELEASE_GROUP = f'release_group' 
    RELEASE = f'release' 
    ARTIST_CREDIT_NAME = f'artist_credit_name' 
    ARTIST = f'artist'
    AREA = f'area'
    PLACE = f'place'
    LABEL = f'label'
    # T_RATINGS = f'title_ratings{TEST}' 
    ref_dict =    {
    (TRACK,'artist_credit'):(ARTIST_CREDIT,'artist_credit'),
     (TRACK,'recording'):(RECORDING,'recording'),
     (RECORDING,'artist_credit'):(ARTIST_CREDIT,'artist_credit'),
     (TRACK,'medium'):(MEDIUM,'medium'),
     (MEDIUM,'release'):(RELEASE,'release'),
     (RELEASE,'artist_credit'):(ARTIST_CREDIT,'artist_credit'),
     (RELEASE,'release_group'):(RELEASE_GROUP,'release_group'),
     (RELEASE_GROUP,'artist_credit'):(ARTIST_CREDIT,'artist_credit'),
     (ARTIST_CREDIT_NAME,'artist_credit'):(ARTIST_CREDIT,'artist_credit'),
     (ARTIST_CREDIT_NAME,'artist'):(ARTIST,'artist'),
     (ARTIST,'area'):(AREA,'area'),
     (PLACE,'area'):(AREA,'area'),
     (LABEL,'area'):(AREA,'area'),
     }
    file = files[0]
    #sim_attrs = get_reduced_spec(file)[1]
    # MUSIC_PATH = '/scratch/c.c2028447/project/entity-resolution/dataset/musicbrainz'
    MUSIC_PATH = f'./dataset/music/{split}/'
    dup_rel = {TRACK,MEDIUM,RECORDING,RELEASE,RELEASE_GROUP,ARTIST,AREA,PLACE,LABEL,ARTIST_CREDIT}
    path_list = [MUSIC_PATH+f'/{TRACK}.csv',MUSIC_PATH+f'/{RECORDING}.csv',MUSIC_PATH+f'/{MEDIUM}.csv',
                 MUSIC_PATH+f'/{ARTIST_CREDIT}.csv',MUSIC_PATH+f'/{RELEASE_GROUP}.csv', 
                 MUSIC_PATH+f'/{RELEASE}.csv',MUSIC_PATH+f'/{ARTIST_CREDIT_NAME}.csv',MUSIC_PATH+f'/{ARTIST}.csv',
                 MUSIC_PATH+f'/{AREA}.csv',MUSIC_PATH+f'/{PLACE}.csv',MUSIC_PATH+f'/{LABEL}.csv'
                 ]
    # u = 'u' if uniq else ''
    #print(u,'---------------======================')
    dl_music = Dataloader(name = f'musicbrainz_{split}',path_list=path_list,ground_truth=[MUSIC_PATH+f'/{TRACK}_dups.csv',
                                                                                 MUSIC_PATH+f'/{RECORDING}_dups.csv',
                                                                                 MUSIC_PATH+f'/{MEDIUM}_dups.csv',
                                                                                 MUSIC_PATH+f'/{RELEASE}_dups.csv',
                                                                                 MUSIC_PATH+f'/{ARTIST_CREDIT}_dups.csv',
                                                                                 MUSIC_PATH+f'/{RELEASE_GROUP}_dups.csv',
                                                                                 MUSIC_PATH+f'/{AREA}_dups.csv',
                                                                                 MUSIC_PATH+f'/{ARTIST}_dups.csv',
                                                                                 MUSIC_PATH+f'/{PLACE}_dups.csv',
                                                                                 MUSIC_PATH+f'/{LABEL}_dups.csv'])
    tbls = dl_music.load_data()  
    tbls_dict = {t[0]:(0,t[1]) for t in tbls}
    del tbls 
    dtype_cfg = MUSIC_PATH+f'/domain.yml'
    order = [TRACK,ARTIST_CREDIT,RECORDING,MEDIUM,RELEASE,RELEASE_GROUP,ARTIST_CREDIT_NAME,ARTIST,AREA,PLACE,LABEL]
    music = Schema(id = '2',name =f'musicbrainz_ds_{split}',tbls=tbls_dict,refs=ref_dict,dup_rels=dup_rel,attr_types=utils.load_config(dtype_cfg),order=order,spec_dir=file)
    #music.set_sim_attrs(sim_attrs)
    return music,dl_music


def pokemon_schema(split='',files=[])->tuple[Schema,Dataloader]:
    TEST = '' if len(split) == 0 else '_'+split
    # A = ')'
    # A = ''
    # 0 out degree
    # ability -> species -> items -> moves
    # ability -> 
    # generation variants
    # main_series variants
    file = files[0]
    #sim_attrs = get_reduced_spec(file)[1]
    POKEMON = f'pokemon' 
    SPECIES = f'species' 
    SPECIES_NAME = f'spec_name'
    SPECIES_DESC = f'spec_desc'
    POKEMON_ITEM = f'poke_item' 
    POKEMON_MOVE = f'poke_move' 
    POKEMON_STATS = f'poke_stats'
    POKEMON_TYPE = f'poke_type'
    ITEM = f'item' 
    ITEM_NAME = f'item_name'
    ITEM_DESC = f'item_desc'
    # MACHINE = f'machine'
    ABILITY = f'ability' 
    ABILITY_NAME = f'ability_name'
    ABILITY_DESC = f'ability_desc'
    POKEMON_ABILITY = f'poke_ability' 
    MOVE = f'move' 
    MOVE_NAME = f'move_name'
    MOVE_DESC = f'move_desc'
    STATS = f'stats'
    TYPE = f'type'
    
    name_list = [ POKEMON ,
    SPECIES ,
    SPECIES_NAME,
    SPECIES_DESC,
    POKEMON_ITEM ,
    POKEMON_MOVE,
    POKEMON_STATS ,
    POKEMON_TYPE,
    ITEM ,
    ITEM_NAME ,
    ITEM_DESC,
    # MACHINE ,
    ABILITY,
    ABILITY_NAME,
    ABILITY_DESC,
    POKEMON_ABILITY,
    MOVE,
    MOVE_NAME,
    MOVE_DESC,
    STATS,
    TYPE]
    # T_RATINGS = f'title_ratings{TEST}' 
    ref_dict =    {
     (SPECIES,'evolves_from_species'):(SPECIES,'species'), # self join
     (POKEMON,'species'):(SPECIES,'species'),
     (SPECIES_NAME,'species'):(SPECIES,'species'),
     (SPECIES_DESC,'species'):(SPECIES,'species'),
     (POKEMON_ITEM,'pokemon'):(POKEMON,'pokemon'),
     (POKEMON_ITEM,'item'):(ITEM,'item'),
     (ITEM_NAME,'item'):(ITEM,'item'),
     (ITEM_DESC,'item'):(ITEM,'item'),
     (POKEMON_MOVE,'move'):(MOVE,'move'),
     (POKEMON_MOVE,'pokemon'):(POKEMON,'pokemon'),
     (MOVE_NAME,'move'):(MOVE,'move'),
     (MOVE_DESC,'move'):(MOVE,'move'),
     (POKEMON_ABILITY,'pokemon'):(POKEMON,'pokemon'),
     (POKEMON_ABILITY,'ability'):(ABILITY,'ability'),
     (ABILITY_NAME,'ability'):(ABILITY,'ability'),
     (ABILITY_DESC,'ability'):(ABILITY,'ability'),
     # (MACHINE,'item'):(ITEM,'item'),
     # (MACHINE,'move'):(MOVE,'move'),
     (POKEMON_TYPE,'pokemon'):(POKEMON,'pokemon'),
     (POKEMON_TYPE,'type'):(TYPE,'type'),
     (POKEMON_STATS,'pokemon'):(POKEMON,'pokemon'),
     (POKEMON_STATS,'stats'):(STATS,'stats'),
     }
    # MUSIC_PATH = '/scratch/c.c2028447/project/entity-resolution/dataset/musicbrainz'
    POKEMON_PATH = f'./dataset/pokemon/{split}'
    dup_rel = {POKEMON,SPECIES,ITEM,ABILITY,MOVE}
    path_list = [POKEMON_PATH + f'/{name}.csv' for name in name_list]
    ground_truth_list = [POKEMON_PATH + f'/{name}_dups.csv' for name in dup_rel]
    
    dl_poke = Dataloader(name = f'pokemon_{split}',path_list=path_list,ground_truth=ground_truth_list)
    tbls = dl_poke.load_data()  
    tbls_dict = {t[0]:(0,t[1]) for t in tbls}
    del tbls 
    dtype_cfg = POKEMON_PATH+f'/domain.yml'
    order = name_list
    pokemon = Schema(id = '3',name =f'pokemon{TEST}',tbls=tbls_dict,refs=ref_dict,dup_rels=dup_rel,attr_types=utils.load_config(dtype_cfg),order=order,spec_dir=file)
    #pokemon.set_sim_attrs(sim_attrs)
    return pokemon,dl_poke


#def on_model(model:Model):
        # atoms = 
        #return [str(a) for a in model.symbols(atoms=True) if a.name.endswith('_')]
def sound_track(split='')->tuple[Schema,Dataloader]:
    TEST = '' if len(split) == 0 else '_'+split
    TRACK = f'track{TEST}' 
    RECORDING = f'recording{TEST}' 
    # T_RATINGS = f'title_ratings{TEST}' 
    ref_dict =    {
     (TRACK,'recording'):(RECORDING,'recording')
     }
    MUSIC_PATH = '/scratch/c.c2028447/project/entity-resolution/dataset/musicbrainz'
    # MUSIC_PATH = './dataset/music'
    dup_rel = {TRACK,RECORDING}
    path_list = [MUSIC_PATH+f'/{TRACK}.csv',MUSIC_PATH+f'/{RECORDING}.csv'
                 ]
    dl_music = Dataloader(name = 'musicbrainz',path_list=path_list,ground_truth=[MUSIC_PATH+f'/title_ans{TEST}.csv',MUSIC_PATH+f'/name_ans{TEST}.csv'])
    tbls = dl_music.load_data()  
    tbls_dict = {t[0]:(0,t[1]) for t in tbls}
    del tbls 
    dtype_cfg = MUSIC_PATH+f'/domain{TEST}.yml'
    order = [TRACK,RECORDING]
    music = Schema(id = '2',name =f'musicbrainz{TEST}',tbls=tbls_dict,refs=ref_dict,dup_rels=dup_rel,attr_types=utils.load_config(dtype_cfg),order=order)
    return music,dl_music

def filtered_track():
    s_track,dl = sound_track()
    cache_path = './cache' 
    cachefile = os.path.join(cache_path,f"sound-track.pkl")
    if os.path.isfile(cachefile):
        with open(cachefile, 'rb') as filep:
             sm_facts = pickle.load(filep)
    else:
        atom_base = s_track.load_domain()
        atom_base = ''.join(atom_base)
        query_path = "./music/filter_track.lp"
        ctrl = Control()
        ctrl.load(query_path)
        ctrl.add('base',[],atom_base)
        ctrl.ground([('base', [])])
        del atom_base
        sm_facts = []
        with ctrl.solve(yield_=True) as solution_iterator:
            for model in solution_iterator:
                sm_facts = [str(a) for a in model.symbols(atoms=True) if a.name.endswith('_')]
                break
        if not os.path.isfile(cachefile):
            with open(cachefile, 'wb') as filep:
                pickle.dump(sm_facts, filep) 
    out_dir = './dataset/music'
    # [print(a) for a in sm_facts]
    atom2df(df=s_track.tbls['track'][1],atoms= sm_facts,token='',outdir=out_dir)   

def add_ac_name(version=''):
    ARTIST_CREDIT = f'artist_credit' 
    ARTIST_CREDIT_NAME = f'artist_credit_name' 
    ARTIST = f'artist'
    # T_RATINGS = f'title_ratings' 
    ref_dict =    {
     (ARTIST_CREDIT_NAME,ARTIST_CREDIT):(ARTIST_CREDIT,ARTIST_CREDIT),
     (ARTIST_CREDIT_NAME,'artist'):(ARTIST,'artist'),
     }
    # MUSIC_PATH = '/scratch/c.c2028447/project/entity-resolution/dataset/musicbrainz'
    MUSIC_PATH = f'./dataset/music/{version}'
    dup_rel = {ARTIST_CREDIT}
    path_list = [MUSIC_PATH+f'/{ARTIST_CREDIT}.csv',MUSIC_PATH+f'/{ARTIST_CREDIT_NAME}.csv',MUSIC_PATH+f'/{ARTIST}.csv',
                 ]
    dl_music = Dataloader(name = 'musicbrainz',path_list=path_list,ground_truth=[MUSIC_PATH+f'/title_ans.csv',MUSIC_PATH+f'/name_ans.csv'])
    tbls = dl_music.load_data()  
    tbls_dict = {t[0]:(0,t[1]) for t in tbls}
    del tbls 
    dtype_cfg = MUSIC_PATH+f'/domain.yml'
    order = [ARTIST_CREDIT,ARTIST,ARTIST_CREDIT_NAME]
    music = Schema(id = '2',name =f'musicbrainz',tbls=tbls_dict,refs=ref_dict,dup_rels=dup_rel,attr_types=utils.load_config(dtype_cfg),order=order)
    #cache_path = './cache' 
    #cachefile = os.path.join(cache_path,f"sound-track.pkl")
    #if os.path.isfile(cachefile):
     #   with open(cachefile, 'rb') as filep:
        #     sm_facts = pickle.load(filep)
    #else:
    atom_base = music.load_domain()
    p_ab = set()
    for a in atom_base:
        if '"rec-' in a:
            rel_name = a.split('(')[0]
            a_split = a.split('(')
            a_split[0] = f'{rel_name}_d'
            new_a = '('.join(a_split)
            p_ab.add(new_a)
            # print(new_a)
        else:
            p_ab.add(a)
    atom_base = ''.join(p_ab)
    query_path = "./music/add_ac_name.lp"
    ctrl = Control()
    ctrl.load(query_path)
    ctrl.add('base',[],atom_base)
    ctrl.ground([('base', [])])
    del atom_base
    sm_facts = []
    with ctrl.solve(yield_=True) as solution_iterator:
        for model in solution_iterator:
            sm_facts = [str(a) for a in model.symbols(atoms=True) if a.name.endswith('_')]
            break
 
    out_dir = f'./dataset/music/{version}'
    # [print(a) for a in sm_facts]
    atom2df(df=music.tbls[ARTIST_CREDIT_NAME][1],atoms= sm_facts,token='',outdir=out_dir)   

def add_poke_mapping(version=''):
    pokemon = pokemon_schema(version)[0]
    POKEMON = f'pokemon' 
    SPECIES = f'species' 
    SPECIES_NAME = f'spec_name'
    SPECIES_DESC = f'spec_desc'
    POKEMON_ITEM = f'poke_item' 
    POKEMON_MOVE = f'poke_move' 
    POKEMON_STATS = f'poke_stats'
    POKEMON_TYPE = f'poke_type'
    ITEM = f'item' 
    ITEM_NAME = f'item_name'
    ITEM_DESC = f'item_desc'
    # MACHINE = f'machine'
    ABILITY = f'ability' 
    ABILITY_NAME = f'ability_name'
    ABILITY_DESC = f'ability_desc'
    POKEMON_ABILITY = f'poke_ability' 
    MOVE = f'move' 
    MOVE_NAME = f'move_name'
    MOVE_DESC = f'move_desc'
    STATS = f'stats'
    TYPE = f'type'
    # music = Schema(id = '2',name =f'musicbrainz',tbls=tbls_dict,refs=ref_dict,dup_rels=dup_rel,attr_types=utils.load_config(dtype_cfg),order=order)
    #cache_path = './cache' 
    #cachefile = os.path.join(cache_path,f"sound-track.pkl")
    #if os.path.isfile(cachefile):
     #   with open(cachefile, 'rb') as filep:
        #     sm_facts = pickle.load(filep)
    #else:
    dup_rel = {POKEMON,SPECIES,ITEM,ABILITY,MOVE}
    atom_base = pokemon.load_domain()
    p_ab = set()
    p_ndup = set()
    for a in atom_base:
        rel_name = REL_PAT.findall(a)[0]
        if rel_name in dup_rel:
            rel_name = a.split('(')[0]
            a_split = a.split('(')
            if '"rec-' in a :
                a_split[0] = f'{rel_name}_d'
            oid = a_split[1].split(',')[1]
            # print(a)
            oid = oid.split('-')[1]
            oid = oid.replace('"','')
            a_split[-1] = a_split[-1][:-2]+f',"{oid}").'
            new_a = '('.join(a_split)
            p_ab.add(new_a)
        else:
            p_ndup.add(a)
    p_ab.update(p_ndup)
    #[print(a) for a in p_ab]
    atom_base = ''.join(p_ab)
    query_path = "./pokemon/add-pm.lp"
    ctrl = Control()
    ctrl.load(query_path)
    ctrl.add('base',[],atom_base)
    ctrl.ground([('base', [])])
    del atom_base
    sm_facts = {}
    with ctrl.solve(yield_=True) as solution_iterator:
        for model in solution_iterator:
            for a in model.symbols(atoms=True):
                r_name = REL_PAT.findall(str(a))[0]
                if r_name.endswith('_') and r_name[-1]:
                    if r_name[:-1] not in sm_facts:
                        sm_facts[r_name[:-1]] = []
                    sm_facts[r_name[:-1]].append(str(a))
            # sm_facts = [str(a) for a in model.symbols(atoms=True) if a.name.endswith('_')]
            break
 
    out_dir = f'./dataset/pokemon/{version}'
    # [print(a) for a in sm_facts]
    for k,v in sm_facts.items():
        atom2df(df=pokemon.tbls[k][1],atoms=v,token='',outdir=out_dir)       

def add_poke_spec_decs(version=''):
    pokemon = pokemon_schema(version)[0]
    #POKEMON = f'pokemon' 
    SPECIES = f'species' 
    SPECIES_NAME = f'spec_name'
    SPECIES_DESC = f'spec_desc'
    POKEMON_ITEM = f'poke_item' 
    POKEMON_MOVE = f'poke_move' 
    POKEMON_STATS = f'poke_stats'
    POKEMON_TYPE = f'poke_type'
    #ITEM = f'item' 
    ITEM_NAME = f'item_name'
    ITEM_DESC = f'item_desc'
    # MACHINE = f'machine'
    #ABILITY = f'ability' 
    ABILITY_NAME = f'ability_name'
    ABILITY_DESC = f'ability_desc'
    POKEMON_ABILITY = f'poke_ability' 
    MOVE = f'move' 
    MOVE_NAME = f'move_name'
    MOVE_DESC = f'move_desc'
    STATS = f'stats'
    TYPE = f'type'
    # music = Schema(id = '2',name =f'musicbrainz',tbls=tbls_dict,refs=ref_dict,dup_rels=dup_rel,attr_types=utils.load_config(dtype_cfg),order=order)
    #cache_path = './cache' 
    #cachefile = os.path.join(cache_path,f"sound-track.pkl")
    #if os.path.isfile(cachefile):
     #   with open(cachefile, 'rb') as filep:
        #     sm_facts = pickle.load(filep)
    #else:
    # dup_rel = {POKEMON,SPECIES,ITEM,ABILITY,MOVE}
    atom_base = pokemon.load_domain()
    atom_base = ''.join(atom_base)
    #[print(a) for a in p_ab]
    query_path = "./pokemon/add-move-desc.lp"
    ctrl = Control()
    ctrl.load(query_path)
    ctrl.add('base',[],atom_base)
    ctrl.ground([('base', [])])
    del atom_base
    sm_facts = set()
    with ctrl.solve(yield_=True) as solution_iterator:
        for model in solution_iterator:
            for a in model.symbols(atoms=True):
                r_name = REL_PAT.findall(str(a))[0]
                if r_name.endswith('_'):
                    sm_facts.add(str(a).replace(r_name,'')[1:-1])
            # sm_facts = [str(a) for a in model.symbols(atoms=True) if a.name.endswith('_')]
            break
 
    out_dir = f'./dataset/pokemon/{version}'
    df = pd.DataFrame({'1': list(sm_facts)})

    # Save the DataFrame to a CSV file
    df.to_csv(f'{out_dir}/mis-item-desc.csv', index=False)
    # [print(a) for a in sm_facts]
    # atom2df(df=pokemon.tbls[k][1],atoms=v,token='',outdir=out_dir)       
   
def music_sampling():
    schema = music_schema()
    music = schema[0]
    dl = schema[1]
    cache_path = './cache' 
    cachefile = os.path.join(cache_path,f"sampled-music.pkl")
    if os.path.isfile(cachefile):
        with open(cachefile, 'rb') as filep:
             sm_facts = pickle.load(filep)
    else:
        track_df = music.tbls['track'][1]
        place_df = music.tbls['place'][1]
        label_df = music.tbls['label'][1]
        # sample track, sample place, sample label
        #sampled_track = sampling_df(10000,track_df)
        sampled_track = sampling_df(3000,track_df)
        music.tbls['track'] = (0,sampled_track)
        del track_df
        #sampled_place = sampling_df(1000,place_df)
        sampled_place = sampling_df(500,place_df)
        music.tbls['place'] = (0,sampled_place)
        del place_df
        #sampled_label = sampling_df(4000,label_df)
        sampled_label = sampling_df(1000,label_df)
        music.tbls['label'] = (0,sampled_label)
        del label_df
        # track place label loaded as sampled facts
        atom_base = music.load_domain()
        sample_lst= ['track','place','label']
        def replace_pred(a:str):
            rel = utils.REL_PAT.findall(a)[0]
            if rel in sample_lst:
                a = a.replace(rel,rel+'_',1)
            return a
            
        # [print(a) for a in atom_base]
        # atom_base = set(map(replace_pred,atom_base))
        atom_base = ''.join(atom_base)
        #print(atom_base)
        # query_path = "./music/sampling.lp"
        query_path = "./music/down-sampling.lp"
        ctrl = Control()
        ctrl.load(query_path)
        ctrl.add('base',[],atom_base)
        ctrl.ground([('base', [])])
        del atom_base
        sm_facts = []
        with ctrl.solve(yield_=True) as solution_iterator:
            for model in solution_iterator:
                sm_facts = [str(a) for a in model.symbols(atoms=True) if a.name.endswith('_')]
                break
        if not os.path.isfile(cachefile):
            with open(cachefile, 'wb') as filep:
                pickle.dump(sm_facts, filep) 
    #with ctrl.solve(yield_=True) as models:
    [print(a) for a in sm_facts]
    out_dir = './dataset/music'
    # [print(a) for a in sm_facts]
    dl.atom2pd(sm_facts,'',outdir=out_dir)



def music_clean_sampling(dup):
    ori_schema = music_schema('90k-id-processed')
    
    music_ori = ori_schema[0]
    atom_ori = music_ori.load_domain() 
    
    dup_schema = music_schema(dup)
    music_dup = dup_schema[0]
    dl = dup_schema[1]
    atom_dup = music_dup.load_domain(token='d')
    atom_base = atom_ori.union(atom_dup)
    atom_base = ''.join(atom_base)
    #print(atom_base)
    # query_path = "./music/sampling.lp"
    query_path = "./music/sample-clean.lp"
    query_path2 = "./music/down-sampling.lp"
    ctrl = Control()
    ctrl.load(query_path)
    ctrl.add('base',[],atom_base)
    ctrl.ground([('base', [])])
    # print(atom_base)
    del atom_base
    with ctrl.solve(yield_=True) as solution_iterator:
        for model in solution_iterator:
            sm_facts = [str(a) for a in model.symbols(atoms=True) if a.name.endswith('_c')]
            break
    """
    to_be_sampled = {'track','label','place'}
    clean_model_dict = {}
    with ctrl.solve(yield_=True) as solution_iterator:
        for model in solution_iterator:
            for a in model.symbols(atoms=True):
                if a.name.endswith('_'):
                    pred = a.name[:-1]
                    # if pred == 'recording': print(a)
                    if pred in to_be_sampled:
                        _pred = f'{a.name}c'
                        values = [str(arg) for arg in a.arguments]
                        atom = utils.get_atom_(_pred,values)
                        # str(a)+'.'
                        if pred not in clean_model_dict:
                            clean_model_dict[pred] = []
                        clean_model_dict[pred].append(atom)
                    else:
                        atom = str(a)+'.'
                        if 'others' not in clean_model_dict:
                            clean_model_dict['others'] = []
                        clean_model_dict['others'].append(atom)
            break
    #with ctrl.solve(yield_=True) as models:
    track_indices = range(len(clean_model_dict['track']))
    sampled_track = [clean_model_dict['track'][i] for i in random.sample(track_indices,900)]
    
    label_indices = range(len(clean_model_dict['label']))
    sampled_label = [clean_model_dict['label'][i] for i in random.sample(label_indices,120)]
    
    place_indices = range(len(clean_model_dict['place']))
    sampled_place = [clean_model_dict['place'][i] for i in random.sample(place_indices,60)]
    
    atom_base = set(sampled_track + sampled_label + sampled_place + clean_model_dict['others'])
    atom_base = ''.join(atom_base)
    
    ctrl = Control()
    ctrl.load(query_path)
    ctrl.add('clean',[],atom_base)
    ctrl.ground([('clean', [])])
    with ctrl.solve(yield_=True) as solution_iterator:
        for model in solution_iterator:
            sm_facts = [str(a) for a in model.symbols(atoms=True) if a.name.endswith('_c')]
            break
    """
    out_dir = f'./dataset/music/{dup}'
    # [print(a) for a in sm_facts]
    dl.atom2pd(sm_facts,'',outdir=out_dir)

def build_lookup_tbl(name,dir,group_key:str = 'gid',version = ''):
    # Group the data by a specific column
    df = pd.read_csv(dir,dtype=str,encoding='utf-8')
    grouped = df.groupby(group_key)
    # Create a dictionary to store the grouped results
    group_dict = {}
    # Iterate over the groups and store the records in the dictionary
    for name, group in grouped:
        group_dict[name] = group.to_dict(orient='records')
    # Print the dictionary
    # print(group_dict)
    dup_mapping = list()
    for k,v in group_dict.items():
        id_lst = list()
        ori_id = ''
        v = [_v['id'] for _v in v]
        for id in v:
            if not 'dup' in id: ori_id = id
            else: id_lst.append(id)
        if len(id_lst)>0:
            for i in id_lst:
                dup_mapping.append([ori_id,i])
    # Create a DataFrame from the 2D list
    dup_mapping_df = pd.DataFrame(dup_mapping)
    out_dir = f'./dataset/music/{version}/{name}_dups.csv'
    # Write the DataFrame to a new CSV file
    dup_mapping_df.to_csv(out_dir, index=False, header=False,encoding='utf-8') 
    
def get_spec_desc(spe_lst:list,decs_lst:DataFrame):
    spec_l_dict = {s:['1', '2', '3', '4', '5', '6', '7', '8', '9','10','11','12'] for s in spe_lst}
    desc_lst = []
    for _,de in decs_lst.iterrows():
        if de['ability'] in spec_l_dict and de['language'] in spec_l_dict[de['ability']]:
            print(de['ability'])
            de['flavor_text'] = de['flavor_text'].str.replace('\n', ' ')
            desc_lst.append(de)
            spec_l_dict[de['ability']].remove(de['language'])
    empty_df = pd.DataFrame(columns=decs_lst.columns) 
    result_df = pd.concat([empty_df] + [pd.DataFrame([row]) for row in desc_lst], ignore_index=True)
    out_dir = f'./dataset/pokemon/50'
    # df = pd.DataFrame({'1': list(sm_facts)})
    # Save the DataFrame to a CSV file
    result_df.to_csv(f'{out_dir}/omitted-spec-desc.csv', index=False)

      
    
 

def corrupting():
    ##### cofigurations:
    # # num_duplicates_distribution = ’zipf’: then this parameter governs the overall distribution of how many ‘duplicate’ records 
    # will be generated per ‘original’ record (i.e. the likelihood that one ‘original’ record will have one, two or more ‘duplicate’ records generated for it).
    
    ## zipf distribution
    # Zipf distribution or the power law distribution, is a statistical distribution 
    # that describes the frequency of occurrence of different items in a dataset, such 
    # that the frequency of the k-th most common item is proportional to 1/k, where k is the rank of the item. 
    
    
    # [TODO] corruption of attributes (overall_prob) is individual events or not ?
    # iterating dataframe
    # for a given record
    # use CorruptCategoricalValue
    
    # 1. get sampled data from dependencies graph program, starting from relations with no in-degree
    # 2. collect result as a clean subsets of original instance, D_sub
    # 3. create duplications on top of D_sub, scenarios where duplications of referenced relations are not referenced are undesirable
         # where we look at dependencies graph again
         # polluting first those with the highest in-degrees, e.g. artist credit
         # then traversing backwards
            # 1) create duplications in referenced relation R
            # 2) create look-up table (ground truth) for R, where each id of R-entity is mapped to
            # a set of ids of dup-R-entity
            # 3) for a referencing relation R', when duplicating an R'-entity,
            #  the attributes subjected to referential constraints will check the look-up table and randomly select 
            #  a candidate id of R-entity
 
    pass 
def processing_music():
    TRACK = f'track' 
    RECORDING = f'recording' 
    MEDIUM = f'medium'
    ARTIST_CREDIT = f'artist_credit' 
    RELEASE_GROUP = f'release_group' 
    RELEASE = f'release' 
    ARTIST_CREDIT_NAME = f'artist_credit_name' 
    ARTIST = f'artist'
    AREA = f'area'
    PLACE = f'place'
    LABEL = f'label'
    
    music = music_schema()[0]
    comment = ['comment']
    general = ['edits_pending','last_updated']
    medium = ['name']
    artist = ['begin_area','end_area']
    artist_credit_name = ['join_phrase']
    area_place_label = ['begin_date_year','begin_date_month','begin_date_day','end_date_year','end_date_month','end_date_day']
    place = ['comments']
    label = ['label_code']
    dropping_dict = {TRACK:general,RECORDING:general+comment,
                     MEDIUM:medium+general,RELEASE:general+comment, 
                     RELEASE_GROUP:general+comment,
                     ARTIST_CREDIT:[general[0]],
                     ARTIST:general+artist,
                     ARTIST_CREDIT_NAME:artist_credit_name,
                     AREA:area_place_label+general+comment,PLACE:place+area_place_label+general+comment, 
                     LABEL:label+area_place_label+general+comment}
    for r_name,tbl in music.tbls.items():
        df = tbl[1]
        out_dir = f'./dataset/music/{r_name}_.csv'
        columns = dropping_dict[r_name]
        drop_columns(df,columns=columns,out_dir=out_dir)
        
def process_id(ver,self_id=False):
    schema = music_schema(ver)[0]
    ids = schema.tbls.keys()
    for k, v in schema.tbls.items():
        for c in v[1].columns:
            if c in ids and (self_id or c!=k):
                v[1][c] = 'id-'+v[1][c]
        v[1].to_csv(f'./dataset/music/90k-id-processed/{k}_.csv', index=False)
        
def process_id_(schema,ds_name,ver,self_id=False):
    ids = schema.tbls.keys()
    for k, v in schema.tbls.items():
        for c in v[1].columns:
            if c in ids and (self_id or c!=k):
                v[1][c] = 'id-'+v[1][c]
        v[1].to_csv(f'./dataset/{ds_name}/{ver}/{k}_.csv', index=False)
        
def process_id_remove(schema:Schema,out_dir):
    # schema = music_schema(ver)[0]
    # ids = schema.tbls.keys()
    for k, v in schema.tbls.items():
        for c in v[1].columns:
            if c == k:
                print(c,k)
                v[1][c] = v[1][c].str.replace('id-','')
                v[1][c] = v[1][c].str.replace('rec-','')
                v[1][c] = v[1][c].str.replace('-dup-','/dup/')
                print(v[1][c])
        v[1].to_csv(f'{out_dir}/{k}_.csv', index=False)
        
def process_id_remove_(schema:Schema,out_dir,rel_name,col_name):
    # schema = music_schema(ver)[0]
    # ids = schema.tbls.keys()
    for k, v in schema.tbls.items():
        if k == rel_name:
            for c in v[1].columns:
                if c == col_name:
                    print(c,k)
                    v[1][c] = v[1][c].str.replace('id-','')
                    v[1][c] = v[1][c].str.replace('rec-','')
                    v[1][c] = v[1][c].str.replace('-dup-','/dup/')
                    print(v[1][c])
            v[1].to_csv(f'{out_dir}/{k}_.csv', index=False)
        
def process_id_remove_dups(schema:Schema,out_dir):
    # schema = music_schema(ver)[0]
    # ids = schema.tbls.keys()
    for k, v in schema.tbls.items():
        if schema.rel_index(k).is_dup:
            for index, row in v[1].iterrows():
                if row[k].startswith('id-'):
                        v[1].drop(index, inplace=True)
                elif row[k].startswith('rec-'):
                        v[1].at[index,k] = v[1].at[index,k].replace('rec-','').replace('-dup-','/dup/')
            v[1].to_csv(f'{out_dir}/{k}_.csv', index=False)
            
def  process_filter_dups(schema:Schema,out_dir,rel_name:str, column_name:str):
    # schema = music_schema(ver)[0]
    # ids = schema.tbls.keys()
    for k, v in schema.tbls.items():
        if k == rel_name:
            for index, row in v[1].iterrows():
                if row[column_name].startswith('id-'):
                        v[1].drop(index, inplace=True)
            v[1].to_csv(f'{out_dir}/{k}_d.csv', index=False)
            
def  process_filter_cleans(schema:Schema,out_dir,rel_name:str, column_name:str):
    # schema = music_schema(ver)[0]
    # ids = schema.tbls.keys()
    for k, v in schema.tbls.items():
        if k == rel_name:
            for index, row in v[1].iterrows():
                if row[column_name].startswith('rec-'):
                        v[1].drop(index, inplace=True)
            v[1].to_csv(f'{out_dir}/{k}_d.csv', index=False)

def  process_drop_id(df,out_dir,rel_name:str):
    # schema = music_schema(ver)[0]
    # ids = schema.tbls.keys()
    for index, row in df.iterrows():
        if row['id'].startswith('rec-'):
                df.drop(index, inplace=True)
    # df = df.drop('id', axis=1)
    df.to_csv(f'{out_dir}/{rel_name}_did.csv', index=False)
    
def process_drop_id_(df:DataFrame,out_dir,rel_name):
    def should_drop_row(row, dataframe):
        id_value = row['id']
        if id_value.startswith('id-'):
            id_string = id_value[3:]  # Extract the string after 'id-'
            # Check if there exists a row starting with 'rec-' and containing the id_string
            return dataframe[(dataframe['id'].str.startswith('rec-')) & (dataframe['id'].str.contains(id_string))].shape[0] > 0
        return False

    # Iterate over the DataFrame and drop rows as per the condition
    rows_to_drop = []
    for index, row in df.iterrows():
        if should_drop_row(row, df):
            rows_to_drop.append(index)

    # Drop the rows identified for deletion
    df.drop(rows_to_drop, inplace=True)
    df = df.drop('id',axis=1)
    df.to_csv(f'{out_dir}/{rel_name}_did_.csv', index=False)
                
            
def process_id_concat(schema:Schema,out_dir):
    # schema = music_schema(ver)[0]
    # ids = schema.tbls.keys()
    def format_numeric(val):
        if isinstance(val, (int, float)):
            return f'{val:.0f}'
        return val
    for k, v in schema.tbls.items():
        if schema.rel_index(k).is_dup:
            for index, row in v[1].iterrows():
                if row[k].startswith('rec-'):
                        v[1].drop(index, inplace=True)
            concat_df = pd.concat([v[1],load_csv(path_list=[f'{out_dir}/{k}_.csv'])])
            concat_df = concat_df.applymap(format_numeric)
            concat_df.to_csv(f'{out_dir}/{k}_c.csv', index=False, float_format='%.0f')   
            
def process_drop_col(schema:Schema,out_dir,col_name:str):
    def format_numeric(val):
        if isinstance(val, (int, float)):
            return f'{val:.0f}'
        return val
    for k, v in schema.tbls.items():
        df = v[1]
        if col_name in v[1]:
          df = v[1].drop(col_name,axis=1)
        df.to_csv(f'{out_dir}/{k}.csv', index=False, float_format='%.0f')   
          

def process_concat_df(df1,df2,name,out_dir):
    # schema = music_schema(ver)[0]
    # ids = schema.tbls.keys()
    def format_numeric(val):
        if isinstance(val, (int, float)):
            return f'{val:.0f}'
        return val

    concat_df = pd.concat([df1,df2])
    concat_df = concat_df.applymap(format_numeric)
    concat_df.to_csv(f'{out_dir}/{name}_cd.csv', index=False, float_format='%.0f')
    
def process_concat(schema:Schema,out_dir):
    for rel in schema.relations:
        rname = rel.name
        df1 = schema.tbls[rname][1]
        df2 = load_csv(path_list=[f'{out_dir}/{rname}_.csv'])
        process_concat_df(df1=df1,df2=df2, name = rname,out_dir=out_dir)
            
def process_add_index(df,name,out_dir):
    # schema = music_schema(ver)[0]
    # ids = schema.tbls.keys()
    # df = v[1].astype(str)
    df.to_csv(f'{out_dir}/{name}_id.csv', index=True)     

     
def process_id_restore(schema:Schema,out_dir):
    dup_pat = r'\d+/dup/\d+'
    for k, v in schema.tbls.items():
        if k in v[1].columns:
            # print(k)
            for index, row in v[1].iterrows():
                if row[k].startswith('id-'):
                    v[1].drop(index, inplace=True)
                elif row[k].startswith('rec-'):
                    v[1].at[index,k] = v[1].at[index,k].replace('-dup-0','')
                    
                    dup_num = re.findall(dup_pat, v[1].at[index,k])
                    if len(dup_num)>0:
                        split = v[1].at[index,k].split('/')
                        # print(split)
                        v[1].at[index,k] = f'{split[0]}-dup-{split[2]}'
                        #print(row[k])
                    else:
                        v[1].at[index,k] = v[1].at[index,k].replace('rec-','')
                        v[1].at[index,k] = f'id-{v[1].at[index,k]}'
                        #print(row[k])
            v[1].to_csv(f'{out_dir}/{k}_.csv', index=False)
    
def find_clusters(name,dir,version='',out_dir="./dataset/music"):
    df = pd.read_csv(dir,dtype=str,encoding='utf-8')
    clusters = set()
    ids = df[name].tolist()
    for i, id in enumerate(ids):
        v = id.split('-')[1]
        for j, other_id in enumerate(ids):
                if other_id!=id and other_id.split('-')[1] == v and (other_id,id) not in clusters:
                    if id.startswith('id-'):
                        clusters.add((id,other_id))
                    else:
                        clusters.add((other_id,id))

        """
        if id.startswith('id-'):
            v = id[3:]
            #cluster = [v]
            for j, other_id in enumerate(ids):
                if other_id.startswith('rec-') and other_id.split('-')[1] == v:
                    clusters.append(['id-'+v,other_id])
        else:
            v = id.split('-')[1]
            for j, other_id in enumerate(ids):
                if other_id.startswith('rec-') and other_id.split('-')[1] == v:
                    clusters.append(['id-'+v,other_id])
        """
            #clusters.append(cluster)
    dup_mapping_df = pd.DataFrame(clusters)
    out_dir = f'{out_dir}/{version}/{name}_dups.csv'
    # Write the DataFrame to a new CSV file
    dup_mapping_df.to_csv(out_dir, index=False, header=False,encoding='utf-8') 
    # return clusters

def expand_gts(context:contextc_.ERContext,name,dir)->dict:
    gts = context.ldr.load_ground_truth()
    if isinstance(gts,dict):
        cluster_dict = {}
        for r, gt_set in gts.items():
            for t in gt_set:
                context.cluster(t)
            cluster_dict[r] = context.rep_set
            
            context.rep_set = {}  
        #for r, cluster in cluster_dict.items():
           # print(r,cluster)
        expanded_gts_dict = dict()
        for r, clusters in cluster_dict.items():
            pairs = list()
            for k,c in clusters.items():
                cluster_lst = list(c)
                print(cluster_lst)
                for i,t in enumerate(cluster_lst):
                    for j in range(i+1,len(cluster_lst)):
                        if (cluster_lst[j],cluster_lst[i]) not in pairs:
                            pairs.append((cluster_lst[i],cluster_lst[j]))
            expanded_gts_dict[r] = set(pairs)
    
        gt_frame = {k:pd.DataFrame(columns=[1,2]) for k,v in gts.items()}
        print(expanded_gts_dict.keys())
        for k,v in expanded_gts_dict.items():
            splits_dicts = list()
            for _v in v:
                splits_dicts.append({1:_v[0],2:_v[1]})
            pd.concat([gt_frame[k],pd.DataFrame(splits_dicts)],ignore_index=True).to_csv(f"{dir}{str(k)}_{name}.csv",sep=",",encoding="utf-8",index=False)
    return cluster_dict

def expand_gt(context:contextc_.ERContext,name,dir)->dict:
    gts = context.ldr.load_ground_truth()
    if isinstance(gts,dict):
        cluster_dict = {}
        print(gts)
        gt_set = gts[name]
        for t in gt_set:
            context.cluster(t)
        cluster_dict[name] = context.rep_set

        #for r, cluster in cluster_dict.items():
           # print(r,cluster)
        expanded_gts_dict = dict()
        clusters = cluster_dict[name]
        pairs = list()
        for k,c in clusters.items():
            cluster_lst = list(c)
            print(cluster_lst)
            for i,t in enumerate(cluster_lst):
                for j in range(i+1,len(cluster_lst)):
                    if (cluster_lst[j],cluster_lst[i]) not in pairs:
                        pairs.append((cluster_lst[i],cluster_lst[j]))
        expanded_gts_dict[name] = set(pairs)
    
        gt_frame = {name:pd.DataFrame(columns=[1,2])}
        print(expanded_gts_dict.keys())
        for k,v in expanded_gts_dict.items():
            splits_dicts = list()
            for _v in v:
                splits_dicts.append({1:_v[0],2:_v[1]})
            pd.concat([gt_frame[k],pd.DataFrame(splits_dicts)],ignore_index=True).to_csv(f"{dir}{str(k)}_{name}.csv",sep=",",encoding="utf-8",index=False)
    return cluster_dict


def music_dup():
    TEST = '' 
    TRACK = f'track{TEST}' 
    RECORDING = f'recording{TEST}' 
    MEDIUM = f'medium{TEST}'
    ARTIST_CREDIT = f'artist_credit{TEST}' 
    RELEASE_GROUP = f'release_group{TEST}' 
    RELEASE = f'release{TEST}' 
    ARTIST_CREDIT_NAME = f'artist_credit_name{TEST}' 
    ARTIST = f'artist{TEST}'
    AREA = f'area{TEST}'
    PLACE = f'place{TEST}'
    LABEL = f'label{TEST}'
    MUSIC_PATH = './dataset/music'
    path_list = [
                MUSIC_PATH+f'/{ARTIST}.csv',
                MUSIC_PATH+f'/{AREA}.csv',MUSIC_PATH+f'/{PLACE}.csv',MUSIC_PATH+f'/{LABEL}.csv'
                ]
    for p in path_list:
        name = p.split('/')[3].replace('.csv','')
        find_clusters(name,p)
        
import os
import pandas as pd

def remove_line_breaks_in_csv_values(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Remove line breaks in values
                df = df.replace('\n', '', regex=True)

                # Save the modified DataFrame back to the CSV file
                df.to_csv(file_path, index=False)
                
def remove_line_breaks_in_csv_values(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Remove line breaks in values
                df = df.replace('\n', '', regex=True)

                # Save the modified DataFrame back to the CSV file
                df.to_csv(file_path, index=False)
                
def remove_rows_and_save_csv(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("_dups.csv"):
                file_path = os.path.join(root, file)
                output_file_path = os.path.join(root, f"{os.path.splitext(file)[0]}_.csv")

                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)

                # Remove rows where the first column contains "rec-"
                df = df[~df.iloc[:, 0].str.contains("rec-")]

                # Save the modified DataFrame to the output file
                df.to_csv(output_file_path, index=False)
                
def get_schema(name)->tuple[Schema,Dataloader]:
    if name== 'dblp':
        return dblp_non_split_schema()
    elif name== 'cora':
        return cora_non_split_schema()
    elif name== 'imdb':
        return imdb_schema()
    elif name== 'music':
        return music_schema('50')
    elif name== 'pokemon':
        return pokemon_schema('50')

def extract_body_literals_ungrounded(rule_string):
    # Use regex to find the body part of the rule
    match = re.search(r":-([^\.]*)\.", rule_string)
    
    if match:
        # Extract and clean the body part
        body_string = match.group(1).strip()
        # Split the body into individual literals
        body_literals = [literal.strip() for literal in body_string.split(",")]
        return body_literals
        #return pokemon_schema('50',files=['./experiment/5-uni/pokemon/pokemon.lp'])
    
def mod_cache(dir):
    cache = utils.load_cache(dir)
    cache_reduced = set()
    for a in cache:
        match = re.search(utils.SIM_FACT_PAT,a)
        #print(match)
        score = match.group(1)
        if int(score)>=85:
            cache_reduced.add(a)
    utils.cache(dir,cache_reduced)
        
def load_cache(dir):
    cache = utils.load_cache(dir)
    #cache_reduced = set()
    for a in cache:
        #if a.startswith('eq'):
            print(a)
if __name__ == "__main__":
    slist =  ['dblp', 'cora','imdb','music','pokemon']
    #schema = get_schema('pokemon')[0]
    #process_drop_col(schema,'./dataset/pokemon/no-idf','identifier')
    load_cache('./cache/atoms_musicbrainz_ds_50-corr-ter.pkl')
