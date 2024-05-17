from importlib.resources import path
from typing import Sequence
import pandas as pd
from pandas import DataFrame
import os
import xml.etree.ElementTree as et
import re
import random
import utils
import trans_utils


# datasets for 2 schemas 2 tables
DBLP_ACM = 'DBLP-ACM'
DBLP_SCHOLAR = 'DBLP-Scholar'
AMAZON_GOOGLE = 'Amazon-GoogleProducts'
ABT_BUY = 'Abt-Buy'
DEFAULT_PATH = './dataset/benchmarks/'

UTF8 = 'utf-8'
ISO = 'ISO-8859-1'

# datasets for 1 schema multiple tables
CORA_NAME = "cora"
CORA_PATH = "cora-ref/cora-all-id.xml"

# datasets for 1 schema multiple tables
CORA_TSV_NAME = "cora_tsv"
CORA_TSV_PATH = "cora-ref/cora.tsv"


DEFAULT_REL_1 = "rel1"
DEFAULT_REL_2 = "rel2"
DEFAULT_REL_TUPLE = (DEFAULT_REL_1,DEFAULT_REL_2)

STRING = 0
NUM = 1
LST = 2

# col:val
DBLP_ACM_COL_TYPING = [{0:STRING,1:STRING,2:LST,3:STRING,4:NUM},
                       {0:STRING,1:STRING,2:LST,3:STRING,4:NUM}
                       ]
# 1 publications id, title
# 2 venue id, name, vol, year
# 3 author id, name
PUBLICATION = 'publication'
AUTHOR = 'author'
TITLE = 'title'
VENUE = 'venue'
NAME = 'name'
VENUE_YEAR = 'date'
VENUE_VOL = 'vol'
ID = 'id'
PUB_ID = 'pubid'
YEAR = 'year'

CORA_TYPING = [{0:STRING,1:STRING},
               {0:STRING,1:STRING},
                {0:STRING,1:STRING,2:STRING,3:STRING,4:STRING}
                ]


def load_csv( encoding = 'utf-8', path_list = [], name = ''):
    assert len(path_list) >0
    #print(path_list[0])
    sep = ',' if '.tsv' not in path_list[0] else '\t'
    #print(sep)
    tbls = []
    
    if len(path_list) >1:
        if not utils.is_empty(name):
            path_list = [p for p in path_list if name in p]
        for path in path_list:
            t_name = path.replace('.tsv','').replace('.csv','').split('/')[-1]
            t_name = t_name[:-2] if t_name.endswith('_c') else t_name
            tbls.append((t_name, pd.read_csv(path,encoding=encoding, sep=sep,dtype=str)))
        return tbls
    else: 
        return pd.read_csv(path_list[0],encoding=encoding,dtype=str,sep=sep)  

def save_gts(splits:dict,name:str)->None:
    gt_frame = {k:pd.DataFrame(columns=[1,2]) for k,v in splits.items()}
    for k,v in splits.items():
        splits_dicts = list()
        v = utils.remove_symmetry(v) # [2023-05-21] generated name GTs are now transtivly and symmertically closed
        for _v in v:
            if not _v[0] == _v[1]:
                splits_dicts.append({1:_v[0],2:_v[1]})
        pd.concat([gt_frame[k],pd.DataFrame(splits_dicts)],ignore_index=True).to_csv(f"./dataset/imdb/{name}_{str(k)}.csv",sep=",",encoding="utf-8",index=False)

def drop_columns(df:DataFrame, columns:Sequence[str],out_dir):
    df = df.drop(columns=columns,axis=1)
    df.to_csv(out_dir,index=False,sep=",",encoding="utf-8",)
    
    
def atom2df(atoms, df: DataFrame, token, outdir='./dataset/imdb'):
        frame = pd.DataFrame(columns=df.columns)     
        # partion_dict = {k:pd.DataFrame(columns=v.columns) for k,v in data}
        parts = list()
        pred = ''
        for a in atoms:
            # print(a)
            pred = utils.REL_PAT.findall(a)[0]
            # print(pred,token)
            if utils.is_empty(token):
                pred = pred[:-1]    
            else: pred = pred.replace(f'_{token}','').replace(f'_c{token}','')
            #print(pred)
            val_lst = utils.VAR_PAT.findall(a)[0]
            # print(val_lst)
            val_lst = val_lst.split('","')
            val_lst = [utils.process_atom_val(v) for v in val_lst]
            # print(val_lst)
            col = frame.columns
            row = dict()
            for i,c in enumerate(col):
                # print(pred,c,i,len(val_lst),val_lst)
                row[c] = val_lst[i]
            parts.append(row)
        # print(partion_dict.keys())
        frame = pd.concat([frame,pd.DataFrame(parts)],ignore_index=True)
            # print(v)
        frame.to_csv(f"{outdir}/{pred}_{token}.csv",sep=",",encoding="utf-8",index=False)
    

class Dataloader:
    def __init__(self, name=None, path_list=None, rel_tuple = DEFAULT_REL_TUPLE, ground_truth = [],encoding='utf-8'):
        # name of default dataset
        self.name = name
        # table names/relation names of the input pair
        self.rel_tuple = rel_tuple
        self.encoding = encoding
        if self.name:
            # directories of the pair of tables
            self.path_list = []
            self.rel_tuple = self.name.split('-')
            self.ground_truth = ''
            if self.name == DBLP_ACM:
                folder = os.path.join(DEFAULT_PATH,self.name)
                self.rel_tuple = self.name.split('-')
                self.path_list.append(os.path.join(folder,self.rel_tuple [0]+"2.csv")) 
                self.col_type_dict = DBLP_ACM_COL_TYPING
                self.path_list.append(os.path.join(folder,self.rel_tuple [1]+".csv")) 
                self.ground_truth = os.path.join(folder,'DBLP-ACM_perfectMapping.csv')
            elif self.name == DBLP_SCHOLAR:
                folder = os.path.join(DEFAULT_PATH,self.name)
                self.rel_tuple = self.name.split('-')
                self.path_list.append(os.path.join(folder,self.rel_tuple [0]+"1.csv")) 
                self.path_list.append(os.path.join(folder,self.rel_tuple [1]+".csv")) 
            elif self.name == CORA_NAME:
                folder = DEFAULT_PATH
                self.col_type_dict = CORA_TYPING
                self.path_list.append(os.path.join(folder,CORA_PATH))
            elif self.name == CORA_TSV_NAME:
                folder = DEFAULT_PATH
                self.path_list.append(os.path.join(folder,CORA_TSV_PATH))
                self.ground_truth = os.path.join(folder,'cora-ref/cora_DPL.tsv')
            else:
                self.path_list = path_list
                self.ground_truth = ground_truth
                #TODO: to be modified
        else:
            self.path_list = path_list

    def load(self,):
        if self.name == CORA_NAME:
            return self.load_xml()
        else: 
            csv = self.load_data()
            return csv

    
    def load_data (self,):
        assert self.name != None or self.path_list !=None     
        
        # encoding = 'ISO-8859-1'if self.name == DBLP_ACM else 'utf-8'
        return load_csv(path_list=self.path_list,encoding=self.encoding) 
    
    def load_ground_truth(self, encoding = 'utf-8'):
        assert self.name != None or len(self.ground_truth) >0
        # if self.name:
            # encoding = 'ISO-8859-1'
        paths = [self.ground_truth] if isinstance(self.ground_truth,str) else self.ground_truth
        tbl = load_csv(path_list=paths,encoding=encoding)
        def add_truth(_tbl):
            truth_lst = list()
            for row_idx, _ in _tbl.iterrows():
                t = []
                for c_idx, col in enumerate(_tbl.columns):
                    t.append(str(_tbl.iat[row_idx,c_idx]))
                t = *t,
                #_t = (t[1],t[0])
                truth_lst.append(tuple(t))
            return truth_lst       
        if not isinstance(tbl,list):
            return add_truth(tbl)
        else:
            gt_set = dict()
            for t in tbl:
                gt_set[ t[0].replace('_dups','')] = add_truth(_tbl=t[1])
            #reversed_truth_set.add(tuple(_t))
            return gt_set #,reversed_truth_set
    
    def ground_truth_stats(self):
        gt = self.load_ground_truth()
        if isinstance(gt,list):
            print(" * Ground truth size:",len(gt))
        else:
            size = 0
            for k,v  in gt.items():
                size+= len(v)
            print(" * Ground truth size:",size)

    
    def split_ground_truth (self,dup_num=None,token=None,prev_split=None):
        ground_truth = self.load_ground_truth()
        if isinstance(ground_truth, dict):
            weight_lst = []
            total = 0
            prev_total = 0
            if prev_split!=None:
                for k,v in prev_split.items():
                    #print(k)
                    prev_total += len(v)
                    for _v in v:
                        # print(_v)
                        ground_truth[k].remove((_v[0],_v[1]))
                dup_num -=prev_total
                    
            for k,v in ground_truth.items():
                total += len(v)
                weight_lst.append((k,len(v)))
            #print(total)
            weight_dict = dict()
            for k,v in weight_lst:
                w = v/total
                weight_dict[k] = w
            #print(weight_dict)
            split_gt = dict()
            # print(k)
            gt_frame = {k:pd.DataFrame(columns=[1,2]) for k,v in ground_truth.items()}
            # get weighted ground truth split
            for k,v in ground_truth.items():
                weighted_dup_num = int(dup_num*weight_dict[k])
                indices = list(range(len(v)))
                rd_idx = random.sample(indices,weighted_dup_num)
                split = list()
                sp_dicts = list()
                for idx in rd_idx:
                    split.append(list(v[idx]))
                    sp_dicts.append({1:v[idx][0],2:v[idx][1]})
                if prev_split!=None:
                    split+=prev_split[k]
                    for _pv in prev_split[k]:
                        sp_dicts.append({1:_pv[0],2:_pv[1]})
                pd.concat([gt_frame[k],pd.DataFrame(sp_dicts)],ignore_index=True).to_csv(f"./dataset/imdb/{k}_{str(dup_num+prev_total)}.csv",sep=",",encoding="utf-8",index=False)
                split_gt[k] = split
                # finding relevant records thru the split
            return split_gt
        
    def get_clean_recs(self,pair_num, current_gt, rel_names=[], token=None, prev_clean_list =None):
        ground_truth = self.load_ground_truth()
        data = self.load_data()
        data = [(k,v) for k,v in data if k in rel_names]
        clean_pairs = dict()
        if isinstance(ground_truth, dict):
            for k,v in ground_truth.items():
                pair_num -= len(current_gt[k])
            for k,v in ground_truth.items():
                k_data = None
                for d in data:
                    if k.split('_')[0] in d[0]:
                        k_data = d[1]
                        break
                obj_lst = k_data[k_data.columns[0]].values.tolist()
                indices = list(range(len(obj_lst)))
                # sample two records from [0,len(D_1)]
                while pair_num>0:
                    rd_idx = random.sample(indices,2)
                    # storing if the pair not in ground truth
                    if (obj_lst[rd_idx[0]],obj_lst[rd_idx[1]]) not in v and (obj_lst[rd_idx[1]],obj_lst[rd_idx[0]]) not in v:
                        if k not in clean_pairs: 
                            clean_pairs[k] = list()
                        if obj_lst[rd_idx[1]] not in clean_pairs[k] and obj_lst[rd_idx[0]] not in clean_pairs[k]:
                            if prev_clean_list !=None:
                                if obj_lst[rd_idx[1]] not in prev_clean_list and obj_lst[rd_idx[0]] not in prev_clean_list:                        
                                    clean_pairs[k].append( obj_lst[rd_idx[1]])
                                    clean_pairs[k].append( obj_lst[rd_idx[0]])
                                    pair_num-=2
                                    # print(pair_num)
                            else:
                                clean_pairs[k].append( obj_lst[rd_idx[1]])
                                clean_pairs[k].append( obj_lst[rd_idx[0]])
                                pair_num-=2
                                # print(pair_num)
                clean_df = pd.DataFrame(clean_pairs[k])
                clean_df.to_csv(f"./dataset/imdb/{k}_clean_200k_{token}.csv",sep=",",encoding="utf-8",index=False)
                
        # for the dataset with smaller amount of gt
        # checking the pair has not stored in previous split as well
        
           
    def gt_to_atom(self,gts=None):
        gt_atoms = set()
        gt_pred = 'gt'
        for k,v in gts.items():
            for tup in v:
                t1 = f'"{tup[0]}"'
                t2 = f'"{tup[1]}"'
                atom_1 = utils.get_atom_(pred_name=gt_pred,tup=[k,t1])
                atom_2 = utils.get_atom_(pred_name=gt_pred,tup=[k,t2])
                gt_atoms.add(atom_1)
                gt_atoms.add(atom_2)
        return gt_atoms
    
    def fgt_to_atom(self,cs=None,k=None):
        c_atoms = set()
        c_pred = 'gt'
        # print(cs)
        for tup in cs:
            t1 = f'"{tup[0]}","{tup[1]}"'
            atom_1 = utils.get_atom_(pred_name=c_pred,tup=[k,t1])
            c_atoms.add(atom_1)
        return c_atoms
    
    # TODO to write another atoms 2 pd function and vice versa
    # where the input takes dataframe structure for generality
    def atom2pd(self, atoms, token, outdir='./dataset/imdb'):
        data = self.load_data()          
        partion_dict = {k:pd.DataFrame(columns=v.columns) for k,v in data}
        del data
        parts = dict()
        for a in atoms:
            #print(a)
            pred = utils.REL_PAT.findall(a)[0]
            #print(pred) if pred !='artist_credit_name_' else 0
            #print(pred,token)
            if utils.is_empty(token):
                pred = pred[:-1]    
            else: pred = pred.replace(f'_c{token}','').replace(f'_d{token}','')
            #print(pred)
            #val_lst = utils.VAR_PAT.findall(a)[0]
            # print(val_lst)
            #val_lst = val_lst.split('","')
            #val_lst = [utils.process_atom_val(v) for v in val_lst]
            val_lst = trans_utils.get_ground_atom_args(a)
            # print(val_lst)
            frame = partion_dict[pred]
            col = frame.columns
            row = dict()
            for i,c in enumerate(col):
                # print(pred,c,i,len(val_lst),val_lst)
                row[c] = val_lst[i]
            if pred not in parts:
                parts[pred] = []
            parts[pred].append(row)
        for k in partion_dict.keys():
            if k in parts:
                partion_dict[k]=pd.concat([partion_dict[k],pd.DataFrame(parts[k])],ignore_index=True)
        for k,v in partion_dict.items():
            # print(v)
            v.to_csv(f"{outdir}/{k}_.csv",sep=",",encoding="utf-8",index=False)
    
def count_records_with_value(csv_file_path, column_name, value):
    df = pd.read_csv(csv_file_path)
    count = len(df[df[column_name] == value])
    return count

def remove_records_and_save_copy(csv_file_path, column_name, values):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Remove records where the specified column matches any of the given values
    df_filtered = df[df[column_name].isin(values)]

    # Create a new filename for the modified CSV file
    file_name, file_extension = os.path.splitext(csv_file_path)
    new_file_path = f"{file_name}_.csv"

    # Save the modified DataFrame to the new CSV file
    df_filtered.to_csv(new_file_path, index=False)
    
def modify_and_save_column(csv_file_path, column_name):
    def format_numeric(val):
        if isinstance(val, (int, float)):
            return f'{val:.0f}'
        return val
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path,dtype=str)
   
    # Convert the DataFrame to string data type
   # df = df.astype(str)

    # Modify values in the specified column
    df[column_name] = df[column_name].apply(lambda x: 'id-' + x if pd.notnull(x) and x != 'nan' else x)
    df.applymap(format_numeric) 
    # Save the modified DataFrame back to the original CSV file
    df.to_csv(csv_file_path, index=False,float_format='%.0f')


def modify_csv(df1_file, col_name, df2_file, dir):
    # Read the CSV files into pandas DataFrames
    df1 = pd.read_csv(df1_file)
    df2 = pd.read_csv(df2_file)
    
    # Merge df1 with df2 based on the 'id' column
    merged_df = pd.merge(df1, df2, left_on=col_name, right_on='id', how='left')
    print(merged_df.columns)
    # Replace the values in col_name with the corresponding values from the 'name' column of df2
    merged_df[col_name] = merged_df['name'].fillna(merged_df[col_name])
    
    # Drop the 'id' and 'name' columns as they are no longer needed
    merged_df.drop(columns=df2.columns, inplace=True)
    
    # Save the resulting DataFrame to a CSV file in the specified directory
    output_file = os.path.join(dir, 'result.csv')
    merged_df.to_csv(output_file, index=False)
    

def find_joins_and_save(df1_file, df2_file, attr1, attr2, output_file):
    # Read the CSV files into pandas DataFrames
    df1 = pd.read_csv(df1_file)
    df2 = pd.read_csv(df2_file)
    rel_name1 = df1.columns[0]
    rel_name2 = df2.columns[0]
    
    # Merge df1 with df2 based on attr1 and attr2
    merged_df = pd.merge(df1, df2, left_on=attr1, right_on=attr2, how='left', suffixes=('1', '2'))
    print(merged_df.columns)
    # Filter out rows where the join attributes have the same value
    if rel_name1 == rel_name2:
        merged_df = merged_df[merged_df[f'{rel_name1}1'] != merged_df[f'{rel_name2}2']]
        
     
    # Save the resulting DataFrame to a CSV file
    merged_df.to_csv(output_file, index=False)
    
def find_joins_and_save(df1_file, df2_file, attr1, attr2, output_file):
    # Read the CSV files into pandas DataFrames
    df1 = pd.read_csv(df1_file)
    df2 = pd.read_csv(df2_file)
    rel_name1 = df1.columns[0]
    rel_name2 = df2.columns[0]
    # Forward merge df1 with df2 based on attr1 and attr2
    merged_df_forward = pd.merge(df1, df2, left_on=attr1, right_on=attr2, how='inner', suffixes=('1', '2'))
    merged_df_forward.columns = [f'{df1_file[:-4]},{rel_name1}1', attr1, f'{df2_file[:-4]},{rel_name2}2', attr2]
    
    # Backward merge df2 with df1 based on attr2 and attr1
    merged_df_backward = pd.merge(df2, df1, left_on=attr2, right_on=attr1, how='inner', suffixes=('1', '2'))
    merged_df_backward.columns = [f'{df2_file[:-4]},{rel_name2}2', attr2, f'{df1_file[:-4]},{rel_name1}1', attr1]
    
    # Concatenate both DataFrames
    result_df = pd.concat([merged_df_forward, merged_df_backward])
    
    # Drop duplicate rows
    result_df.drop_duplicates(inplace=True)
    
    # Save the resulting DataFrame to a CSV file
    result_df.to_csv(output_file, index=False)
    
    
def add_index_column(input_file, output_dir):
    # Read the CSV file into pandas DataFrame
    df = pd.read_csv(input_file)
    
    # Add an extra column named 'index' with integer-valued index starting from 1
    df['index'] = range(1, len(df) + 1)
    
    # Save the new DataFrame to a CSV file in the specified directory
    # output_file = os.path.join(output_dir, 'output.csv')
    df.to_csv(out_dir, index=False)
    
    print(f"DataFrame with index column saved to {out_dir}")

if __name__ == "__main__":
    #rel_name_1 = 'area'
    #rel_name_2 = 'area'
    #df1 =  f'./dataset/music/50/{rel_name_1}_c.csv'
    #df2 = f'./dataset/music/50/{rel_name_2}_c.csv'
    #attr1 = 'area_type'
    #attr2 = 'area_type'
    in_dir = f'./dataset/music/50/artist_credit_name_c.csv'
    out_dir = f'./dataset/music/50/artist_credit_name_cid.csv'
    add_index_column(in_dir,out_dir)