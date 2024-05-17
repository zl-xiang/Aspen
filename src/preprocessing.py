from pandas import DataFrame
import utils
from metainfo import Schema, Relation, Attribute
from program_transformer import program_transformer
from example_schema import get_schema
import os
from dataloader import Dataloader
import pandas as pd
from clingo.control import Control
from clingo import  Symbol, Number, String
from clingo import Model
import numpy as np


def drop_columns(df:DataFrame, columns:list[str],out_dir):
    df = df.drop(columns=columns,axis=1)
    df.to_csv(out_dir,index=False,sep=",",encoding="utf-8",)

def music_schema(split='',files=[],data_dir='./dataset/music',surfix = '')->tuple[Schema,Dataloader]:
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
    MUSIC_PATH = data_dir if utils.is_empty(split) or split=='ex' else f'{data_dir}/{split}'
    dup_rel = {TRACK,MEDIUM,RECORDING,RELEASE,RELEASE_GROUP,ARTIST,AREA,PLACE,LABEL,ARTIST_CREDIT}
    path_list = [MUSIC_PATH+f'/{TRACK}{surfix}.csv',MUSIC_PATH+f'/{RECORDING}{surfix}.csv',MUSIC_PATH+f'/{MEDIUM}{surfix}.csv',
                 MUSIC_PATH+f'/{ARTIST_CREDIT}{surfix}.csv',MUSIC_PATH+f'/{RELEASE_GROUP}{surfix}.csv', 
                 MUSIC_PATH+f'/{RELEASE}{surfix}.csv',MUSIC_PATH+f'/{ARTIST_CREDIT_NAME}{surfix}.csv',MUSIC_PATH+f'/{ARTIST}{surfix}.csv',
                 MUSIC_PATH+f'/{AREA}{surfix}.csv',MUSIC_PATH+f'/{PLACE}{surfix}.csv',MUSIC_PATH+f'/{LABEL}{surfix}.csv'
                 ]
    # u = 'u' if uniq else ''
    #print(u,'---------------======================')
    dl_music = Dataloader(name = f'musicbrainz_{split}',path_list=path_list)
    tbls = dl_music.load_data()  
    tbls_dict = {t[0]:(0,t[1]) for t in tbls}
    del tbls 
    dtype_cfg = MUSIC_PATH+f'/domain.yml'
    order = [TRACK,ARTIST_CREDIT,RECORDING,MEDIUM,RELEASE,RELEASE_GROUP,ARTIST_CREDIT_NAME,ARTIST,AREA,PLACE,LABEL]
    music = Schema(id = '2',name =f'musicbrainz_ds_{split}',tbls=tbls_dict,refs=ref_dict,dup_rels=dup_rel,attr_types=utils.load_config(dtype_cfg),order=order,spec_dir=file)
    #music.set_sim_attrs(sim_attrs)
    return music,dl_music


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
    music_spec = f'./experiment/5-uni/music/music.lp'
    full_dir = f'../../dataset/music-full-origin'
    out_dir =f'../../dataset/music-full'
    music =  music_schema('full',files=[music_spec],data_dir=full_dir)[0]
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
        out_file = f'{out_dir}/{r_name}.csv'
        columns = dropping_dict[r_name]
        drop_columns(df,columns=columns,out_dir=out_file)
        
def drop_column(directory,name):
    # Get list of CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Iterate over CSV files in the directory
    for file in csv_files:
        # Read the CSV file
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        
        # Check if 'gid' column exists
        if 'gid' in df.columns:
            # Drop the 'gid' column
            df.drop(columns=[name], inplace=True)
            
            # Save the modified DataFrame to a new CSV file
            df.to_csv(file_path, index=False)
            print(f"Removed 'gid' column and saved {file} in {directory}")
        else:
            print(f"'gid' column not found in {file}")      

def sampling_df( nrecords:int, df:DataFrame = None,):
        return df.sample(n=nrecords)

def sample_independents(name,source_dir,out_dir,number):
    file = f'{source_dir}/{name}.csv'
    full = pd.read_csv(file)
    samples = sampling_df(number,full)
    del(full)
    samples.to_csv(f'{out_dir}/{name}.csv',index=False)
    
    
def merge_csv_files(dir1, dir2):
    # Get list of CSV files in both directories
    dir1_files = [f for f in os.listdir(dir1) if f.endswith('.csv')]
    dir2_files = [f for f in os.listdir(dir2) if f.endswith('.csv')]

    # Iterate over CSV files in the first directory
    for file in dir1_files:
        # Check if the file exists in the second directory
        if file in dir2_files:
            # Read CSV files from both directories
            df1 = pd.read_csv(os.path.join(dir1, file))
            df2 = pd.read_csv(os.path.join(dir2, file))
            
            # Merge the CSV files
            merged_df = pd.concat([df1, df2], ignore_index=True)
            
            # Write the merged DataFrame to a new CSV file in the first directory
            merged_file_path = os.path.join(dir1, file)
            merged_df.to_csv(merged_file_path, index=False)
            print(f"Merged and saved {file} to {dir1}")
        else:
            print(f"File {file} not found in {dir2}")


def convert_float_to_integer(directory):
    # Get list of CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Iterate over CSV files in the directory
    for file in csv_files:
        # Read the CSV file
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)

        # Iterate over columns in the DataFrame
        for col in df.columns:
            # Check if the column contains float values
            if df[col].dtype == 'float64':
                # Convert float values to integers
                df[col] = df[col].astype('int64')

        # Save the modified DataFrame to a new CSV file
        df.to_csv(file_path, index=False)
        print(f"Converted floating values to integers and saved {file} in {directory}")



def remove_floats(directory):
    # Get list of CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Iterate over CSV files in the directory
    for file in csv_files:
        # Read the CSV file
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)

        # Iterate over columns in the DataFrame
        for col in df.columns:
            # Check if the column contains numeric values
            if pd.api.types.is_numeric_dtype(df[col]):
                # Convert non-finite values to NaN
                
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Convert remaining floating-point numbers to integers
                df[col] = df[col].astype(pd.Int64Dtype(), errors='ignore')
                

        # Save the modified DataFrame to a new CSV file
        df.to_csv(file_path, index=False)
        print(f"Removed floating-point values and saved {file} in {directory}")








def modify_csv_column(csv_file_path, column_name):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Modify the values in the given column
    df[column_name] = df[column_name].apply(lambda x: 1 if x == -1 else x)

    # Get the directory and file name without extension
    directory, file_name = os.path.split(csv_file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]

    # Save the modified DataFrame to a new CSV file with the same name in the same directory
    new_csv_file_path = os.path.join(directory, file_name_without_extension + "_modified.csv")
    df.to_csv(new_csv_file_path, index=False)

    print(f"Modified {column_name} values and saved {file_name} as {file_name_without_extension}_modified.csv in {directory}")




def remove_prefix(directory):
    # Get list of CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Iterate over CSV files in the directory
    for file in csv_files:
        # Read the CSV file
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)

        # Iterate over columns in the DataFrame
        for col in df.columns:
            # Remove the prefixes 'id-' or 'rec-' from column values
            df[col] = df[col].apply(lambda x: x.split('-', 1)[1] if isinstance(x, str) and (x.startswith('id-') or x.startswith('rec-')) else x)

        # Save the modified DataFrame to a new CSV file
        df.to_csv(file_path, index=False)
        print(f"Removed prefixes and saved {file} in {directory}")


def insert_id_prefix(directory):
    # Get list of CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Iterate over CSV files in the directory
    for file in csv_files:
        # Read the CSV file
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)

        # Get the file name without extension
        file_name = os.path.splitext(file)[0]

        # Check if the column named the same as the file name exists
        if file_name in df.columns:
            # Insert 'id-' at the beginning of column values
            df[file_name] = 'id-' + df[file_name].astype(str)

            # Save the modified DataFrame to a new CSV file
            df.to_csv(file_path, index=False)
            print(f"Inserted 'id-' prefix and saved {file} in {directory}")
        else:
            print(f"Column named {file_name} not found in {file}")

    
def music_sample(query_path,superset_dir,sampled_dir,existing_dir='',):
    music_spec = f'./experiment/5-uni/music/music.lp'
    sampled_dir = sampled_dir
    full_dir = superset_dir
    existing_dir = existing_dir
    music_full = music_schema(full_dir[0],files=[music_spec],data_dir=full_dir[1])

    # sample clean tuples of relations with 0 in-degree
    # datasources
        # 1 sampled 0 in-degree tuples 
        # 2 existing data from cache
        # 3 other full tables that weren't sampled 
    #  
    prg_trans1 = program_transformer(music_full[0])
    atom_base = prg_trans1.get_atombase(ter=True)
    atom_base_str = ''.join(atom_base)
    #[print(a) for i,a in enumerate(full_ab) if i<2000]
    #[print(a) for i,a in enumerate(ex_ab) if i<2000]
    if not utils.is_empty(existing_dir):
        music_ex = music_schema('ex',files=[music_spec],data_dir=existing_dir,surfix='_d')
        prg_trans2 = program_transformer(music_ex[0])
        ex_ab = prg_trans2.get_atombase(ter=True)
        atom_base_str = atom_base_str + ''.join(ex_ab)
        #print(atom_base)
    # query_path = "./music/sample-clean2.lp"
    # query_path = "./music/down-sampling.lp"
    del atom_base
    ctrl = Control()
    ctrl.load(query_path)
    ctrl.add('base',[],atom_base_str)
    ctrl.ground([('base', [])])
    
    sm_facts = []
    with ctrl.solve(yield_=True) as solution_iterator:
        for model in solution_iterator:
            sm_facts = [str(a) for a in model.symbols(atoms=True) if a.name.endswith('_')]
            break

    #with ctrl.solve(yield_=True) as models:
    # [print(a) for a in sm_facts]
    # [print(a) for a in sm_facts]
    music_full[1].atom2pd(sm_facts,'',outdir=sampled_dir)


def insert_id_prefix(csv_file_path, column_names):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Iterate over the columns in the DataFrame
    for col in df.columns:
        # Check if the column name is in the given list of column names
        if col in column_names:
            # Insert 'id-' at the beginning of the column values
            df[col] = 'id-' + df[col].astype(str)

    # Get the directory and file name without extension
    directory, file_name = os.path.split(csv_file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]

    # Save the modified DataFrame to a new CSV file with the same name in the same directory
    new_csv_file_path = os.path.join(directory, file_name_without_extension + "_modified.csv")
    df.to_csv(new_csv_file_path, index=False)

    print(f"Inserted 'id-' prefix and saved {file_name} as {file_name_without_extension}_modified.csv in {directory}")


def drop_gid():
    pass
    
def add_artist_credit_name (clean_dir,dup_dir,out_dir,split=''):
  #clean_dir =f'../dataset/m10+4'
  #dup_dir = f'./dataset/music'
  #out_dir = f'./dataset/music/m10+4'
  music_dup = music_schema(split,files=[music_spec],data_dir=dup_dir,surfix='_d')
  music_ex = music_schema(split+'_',files=[music_spec],data_dir=clean_dir)
  prg_trans1 = program_transformer(music_dup[0])
  dup_ab = prg_trans1.get_atombase(ter=True)
  prg_trans2 = program_transformer(music_ex[0])
  ex_ab = prg_trans2.get_atombase(ter=True)
  atom_base = ''.join(dup_ab) + ''.join(ex_ab)
  #print(atom_base)
  query_path = "./music/add_ac_name.lp"
  ctrl = Control()
  ctrl.load(query_path)
  ctrl.add('base',[],atom_base)
  ctrl.ground([('base', [])])
  # del atom_base
  sm_facts = []
  with ctrl.solve(yield_=True) as solution_iterator:
    for model in solution_iterator:
        sm_facts = [str(a) for a in model.symbols(atoms=True) if a.name.endswith('_')]
        print(sm_facts)
        break
  music_ex[1].atom2pd(sm_facts,'',outdir=out_dir)
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
if __name__ == "__main__":
    base_dir = './mallegan-results'
    music_path = [f'{base_dir}/music/{a}-{a}-match.csv' for a in ['area','artist','artist_credit','label','medium','place','recording','release','release_group','track']]
    music_corr_path = [f'{base_dir}/music-corr/{a}-{a}-match.csv' for a in ['area','artist','artist_credit','label','medium','place','recording','release','release_group','track']]
    pokemon_path = [f'{base_dir}/pokemon/{a}-{a}-match.csv' for a in ['ability','item','move','pokemon','species']]
    imdb_path = [f'{base_dir}/imdb/{a}-{a}-match.csv' for a in ['title_basics','name_basics']]
    dblp_path = [f'{base_dir}/dblp-acm-match.csv']
    cora_path = [f'{base_dir}/cora-cora-match.csv']
    #out_dir =f'./dataset/music/'
    dir2 = './dataset/music/10'
    music_spec = f'./experiment/5-uni/music/music.lp'
    out_dir =f'../dataset/m10+4/id-processed'
    atom_dir = f'./music/sample/2c2d/'
    mu_dir = f'./dataset/music/m10+4'
    superset_dir = [('id-processed','../dataset/m10+4'),('m10+3','../dataset'),('m10+2','../dataset')]
    sampled_dir = ['../dataset/m10+1','../dataset/m10+2','../dataset/m10+3',]
    dup_dirs = ['./dataset/music/m10+1', './dataset/music/m10+2','./dataset/music/m10+3',]
    query_dirs = [f'./music/sample-clean3.lp',f'./music/sample-clean2.lp',f'./music/sample-clean1.lp']
    # add_artist_credit_name(f'../dataset','./dataset/music',dup_dirs[0],split='m10+1')
    #for i,d in enumerate(sampled_dir):
        #remove_floats(dup_dirs[i])
        #modify_csv_column(f'{dup_dirs[i]}/release.csv','quality')
    #s = get_schema(f'music',spec_dir=music_spec,data_dir='../dataset',split=f'm10+3')
    #process_id_remove(s[0],sampled_dir[0])
    #remove_floats(sampled_dir[0])
    #for i,d in enumerate(sampled_dir[1:]):
        #schema = get_schema(f'music',spec_dir=music_spec,data_dir='../dataset',split=f'm10+{str(i+1)}')
        #process_id_remove(schema[0],d)
        #remove_floats(d)
    #for i in range(len(query_dirs)):
     #   music_sample(superset_dir=superset_dir[i],sampled_dir=sampled_dir[i],query_path=query_dirs[i])
    drop_columns(pd.read_csv(f'{dup_dirs[0]}/artist_credit.csv'),columns=['gid'],out_dir=f'{dup_dirs[0]}/artist_credit.csv')
    # remove_floats(mu_dir)
    
    # remove_prefix('./dataset/music/10/gid')
    # s[1].atom2pd
    # merge_csv_files(dup_dirs[2],dir2)
    #insert_id_prefix(out_dir)
    #insert_id_prefix('./dataset/music/m10+4/artist_credit_name_d.csv',['artist_credit','artist'])