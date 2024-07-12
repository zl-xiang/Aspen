
import py_stringmatching as sm
import pandas as pd
import yaml
import re
import os
import pickle
from typing import Iterable,Sequence
import numpy as np
from clingo.ast import parse_string


from clingo import Symbol
# from pymemcache.client import base
TP_CACHE = './cache/tps' 
FP_CACHE = './cache/fps' 
FN_CACHE = './cache/fns' 
EQ_CACHE = './cache/eqs' 

DF_EMPTY = 'nan'
SIM_PRED = 'sim'

ATOM_PAT =  re.compile(r"([a-zA-Z0-9_]+\([^)]*\))", re.IGNORECASE)
VAR_PAT = re.compile(r"(?<=\().+?(?=\)$)", re.IGNORECASE)
VAl_PAT = re.compile(r"(?<=\().+?(?=\)$)", re.IGNORECASE)
REL_PAT = re.compile(r"(^[a-zA-Z0-9_]+)\([^)]*\)", re.IGNORECASE)
EQUATE_PAT = re.compile(r"[A-Z0-9]+\s*=\s*[A-Z]+[A-Z0-9']?", re.IGNORECASE)
COUNT_PAT = re.compile(r"([A-Z0-9]+)?(?:\s*=|>=|<=|>|<\s*)?#count\{[^}]*\}(?:\s*=|>=|<=|>|<\s*)?([A-Z0-9]+)?",re.IGNORECASE)
SIM_THRESH_PAT = re.compile(r"sim[_]*[a-zA-Z0-9]*\([^)]*\),[A-Z0-9]+(?:>=|>)[\d]+",re.IGNORECASE)
SIM_THRESH_VAL_PAT = re.compile(r"sim[_]*[a-zA-Z0-9]*\([^)]*\),[A-Z0-9]+(?:>=|>)([\d]+)",re.IGNORECASE)
SIM_JOIN_PAT = re.compile(r"sim[_]*[a-zA-Z0-9]*\(([A-Z0-9']+)\s*,(\s*[A-Z0-9']+),100\)",re.IGNORECASE)
SIM_FACT_PAT = re.compile(r'sim\("(?:[\w\W\s\d]*)","(?:[\w\W\s\d]*)",([0-9]+)\)\.',re.IGNORECASE)
SPEC_CHAR_PAT = re.compile(r",(?=[-.!\"`%&,:;<>=@\{\}~\$\(\)\*\+\/\\\?\[\]\^\|])",re.IGNORECASE)
EQ_AC_PAT = re.compile(r"eq(?=\([^)]*\))|active(?=\([^)]*\))")
EQ_AC_AT_PAT = re.compile(r"eq\([^)]*\)|active\([^)]*\)")

PROG_DIRC = "#program"

EQ_PRED = "eq"
ANTD_SHOW_TRACE = '%!show_trace'
ANTD_TRACE_RULE = '%!trace_rule'

def split(sep_lst:list,string:str) -> list:
    for sep in sep_lst:
        if sep in string:
            return string.strip().split(sep)
    
    return [string]


def get_atom(pred_name:str,tup)->str:
    atom = f'{pred_name}('
    for v in tup[:-1]:
        atom+=str(v)+','
    atom+= str(tup[-1])+')'
    return atom

def get_atom_(pred_name:str,tup)->str:
    tup = [str(t) for t in tup]
    tup = ','.join(tup)
    atom = f'{pred_name}({tup}).'
    return atom    
     
def partition(set_, size):
    set_ = list(set_)
    set_ = [set_[i:i+size] for i in range(0, len(set_), size)]
    last = set_[-1].copy()
    if len(last)/size <0.5:
        last_2 = set_[-2].copy()
        _last=last+last_2
        set_.remove(last)
        set_.remove(last_2)
        set_.append(_last)
    return [set(s) for s in set_]

def parti_by_num(lst, part_number):
    size_each_part = int(len(lst)/part_number)
    remainder = len(lst) % part_number
    remained = lst[-remainder:].copy()
    lst = [lst[i:i+size_each_part] for i in range(0, len(lst), size_each_part)]
    lst[-1]+= remained
    return [set(s) for s in lst]
        

def is_empty(val:str)->bool:
    val = str(val) if None != val else val
    return (val == None or val == DF_EMPTY or len(val)<1)

def is_integer(s:str) :
    try:
        return int(s)
    except ValueError:
        return s

def escape(val:str,cat_seq=False):
    if isinstance(val,str):
        remove_chars = r'[’\!"#%&\'\(\)\*:;<=>?@，。?★、…【】《》？“”‘’！\[\]\^\`\{|\}~]+'
        val = re.sub(remove_chars, '', val)
        line_breaks = r'[\n]+'
        val = re.sub(line_breaks, ' ', val)
        if cat_seq:
            val = re.sub(r'[,]+','-',val)
        else: 
            val = re.sub(r'[,]+','',val)


        return val.replace('"', '\\"').replace("'", "\\'").replace('\\','')
    #.replace('\\','\\\\')
    else:
        return val
    
#def semantic_sim(X:str,Y:str):
    #os.environ['KMP_DUPLICATE_LIB_OK']='True'
 #   model = SentenceTransformer('all-MiniLM-L6-v2')
  #  embed1 = model.encode(X)
   # embed2 = model.encode(Y)
    #cosine_score = util.cos_sim(embed1, embed2).item()
    #return  cosine_score
    
    
def sim(X,Y) -> int:
    score = 0
    short_indicator = 200
    
    # string simiarities
    # if isinstance(X,String) and isinstance(Y,String):
    if  isinstance(X,str) and isinstance(Y,str):
        X= X.lower().strip()
        Y= Y.lower().strip()
        if X == DF_EMPTY or Y == DF_EMPTY:
            return int(score)
        x_len = len(X)
        y_len = len(Y)
        # treating short text-valued entries as sequences of characters
        # measure the editing distance of sequences
        if x_len + y_len <= short_indicator*2:
            measure = sm.JaroWinkler()
            score = measure.get_sim_score(X,Y)
        # TODO: numeric ids?
        else:
        # treating long text-valued entries as sets of tokens
        # measure the TF-IDF cosine score between token set
            tokeniser = sm.AlphabeticTokenizer(return_set=True)
            x_set = tokeniser.tokenize(X)
            y_set = tokeniser.tokenize(Y)
            measure = sm.TfIdf()
            score = measure.get_sim_score(x_set,y_set) 

    # numeric similarities
    # measure the Levenshtein distance for numeric values
    # two numbers is based on the minimum number of operations required to transform one into the other.
    # elif isinstance(X,Number) and isinstance(Y,Number):
    elif isinstance(X,int) and isinstance(Y,int):
        measure = sm.Levenshtein()
        score = measure.get_sim_score(str(X),str(Y))

    return int(score*100)
    
def get_atoms(source_func, cache_dir:str, fname:str, **kwargs)->set:
    cached_path = os.path.join(cache_dir,f"atoms_{fname}.pkl")
    #print(cached_path)
    if os.path.isfile(cached_path):
        with open(cached_path, 'rb') as file:
            atoms = pickle.load(file)
            return set(atoms)
    atoms = source_func(**kwargs)
    with open(os.path.join(cache_dir,f"atoms_{fname}.pkl"), 'wb') as fp:
        pickle.dump(atoms, fp)
    return set(atoms)

def get_cache(source_func, cache_dir:str, fname:str, **kwargs):
    cached_path = os.path.join(cache_dir,f"atoms_{fname}.pkl")
    #print(cached_path)
    if os.path.isfile(cached_path):
        with open(cached_path, 'rb') as file:
            atoms = pickle.load(file)
            return atoms
    atoms = source_func(**kwargs)
    with open(os.path.join(cache_dir,f"atoms_{fname}.pkl"), 'wb') as fp:
        pickle.dump(atoms, fp)
    return atoms
    
def compute_sim(tobe_sim:dict, atoms:set) -> set:
    sim_facts = set()
    for x in atoms:
        if not x.startswith('adom('):
            # print(x)
            x_tup = VAR_PAT.findall(x)[0].split(',')
            idx_x = x_tup[0] #TODO: to take care of symmertic pairs
            for y in atoms:
                if not y.startswith('adom('):
                    y_tup = VAR_PAT.findall(y)[0].split(',')
                    idx_y = y_tup[0] #TODO: to take care of symmertic pairs
                    idx = (idx_x,idx_y)
                    if idx in tobe_sim:
                        pos_set = tobe_sim[idx]
                        for p in pos_set:
                            p_0 = int(p[0])
                            p_1 = int(p[1])
                            tup = ['','','']
                            score = sim(x_tup[p_0],y_tup[p_1]) if idx_x !=idx_y else 100
                            tup[0] = x_tup[p_0]
                            tup[1] = y_tup[p_1]
                            tup[2] = score
                            if score>=80:
                                sim_atom = get_atom(SIM_PRED,tup)
                                sim_facts.add(sim_atom)
    return sim_facts
                        


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

# loading
def load_csv( encoding = 'utf-8', path_list = []):
    assert len(path_list) >0
    sep = ',' if '.tsv' not in path_list[0] else '\t'
    tbls = []
    if len(path_list) >1:
        for path in path_list:
            t_name = path.replace('.tsv','').replace('.csv','').split('/')[-1]
            tbls.append((t_name, pd.read_csv(path,encoding=encoding, sep=sep)))
        return tbls
    else: 
        return pd.read_csv(path_list[0],encoding=encoding, sep=sep)

def sim_compute(dom_values:dict,sim_threshold:int):
    sim_pairs = {}
    sim_facts = set()
    for _,dom in dom_values.items():
        for v1 in dom:
            for v2 in dom:      
                if (v1,v2) or (v2,v1) not in sim_pairs:             
                    sim_score = sim(v1,v2)
                    if sim_score>sim_threshold:
                        sim_pairs[(v1,v2)] = sim_score
    # TODO: some attributes can be ignored for sim calculation, for example, the original ids                 
    for k,s in sim_pairs.items():
        sim_fact1 = f'{SIM_PRED}("{k[0]}","{k[1]}",{s}).'
        sim_fact2 = f'{SIM_PRED}("{k[1]}","{k[0]}",{s}).'
        sim_facts.add(sim_fact1)
        sim_facts.add(sim_fact2)
    return sim_facts
        
def load_config(dir):
    with open(dir, 'r') as file:
        return yaml.safe_load(file)
    
def readLP(dir,extra: str= None, process_dc = False,merge=None):
    with open(dir,'r', encoding='utf-8',
                 errors='ignore') as f:
        # file_str=''
        prg = ''
        prgs = dict()
        dc_idx = 0
        
        if process_dc:
            atom_extra = f'{extra}(X,Y,I)'
            mock_predname = extra
            mock_fact = f'{extra}({merge[0]},{merge[1]},I)'
            dc_mock = ':- #count{I:'+ mock_fact+'}<1.' 
            extra = "%!show_trace {" + atom_extra +"}.\n"
            extra+=dc_mock
        for line in  f.readlines():
            if not line.startswith("%") and PROG_DIRC in line:
                prg = line.strip('\n')
                prgs[prg] = ''
            elif not is_empty(prg):
                if process_dc and line.startswith(':-') and not 'neq(' in line :
                    # print(line,process_dc)    
                    mock_head = mock_predname+f'(X,Y,{str(dc_idx)})'   
                    annotation = '%!trace {'+atom_extra+'," DC % was triggered and falsified eq(%,%)", I, X, Y} :-' + atom_extra +f", I={str(dc_idx)}.\n"          
                    line = annotation+'\n'+mock_head+line
                    dc_idx+=1
                    # print(line)
                prgs[prg] += line+'\n' # this is important for xclingo
        prgs = [(k.replace(PROG_DIRC,'').replace(' ','').replace('.',''),v+extra) for k,v in prgs.items()]
        # TODO to be changed in the future, currently assume only one program is given
        return prgs

def flatten_list(nested_list):
    """
    Takes a 2D list and merges all its elements into a single 1D list.
    """
    flattened_list = []
    for sublist in nested_list:
        for item in sublist:
            flattened_list.append(item)
    return flattened_list

def format_string(s:str, args, placeholder='%'):
    if args is None or len(args)<1:
        return s
    else:
        for a in args:
            s = s.replace(placeholder,str(a),1)
        return s

def atom2str(atom:Symbol):
    return str(atom)+'.'

def sym2tup(sym:Symbol):
    sym = str(sym).strip('(').strip(')').split(',')
    sym = *[str(s).strip('"') for s in sym],
    return sym  

def atom2tup(sym:Symbol):
    sym_args = [str(a) for a in sym.arguments]
    
    return tuple(sym_args)  

def get_atom_tup(atom):
    """
    Parse an atom in Answer Set Programming format and return a tuple of its arguments.

    Parameters:
    - atom (str): Atom in ASP format.

    Returns:
    - tuple: Tuple of arguments.
    """
    # Using a regular expression to extract arguments from the atom
    match = re.match(r'([^\(]+)\((.*)\)', atom)
    
    if match:
        predicate = match.group(1).strip()
        arguments = list(arg.strip() for arg in match.group(2).split(','))
        return predicate, arguments
    else:
        # Return None or raise an exception based on your requirement
        return None

def remove_symmetry(tups:Iterable[Sequence])->set:
    tups = list(map(lambda x: tuple(x),tups))
    sym_free = list()
    for t in tups:
        if t not in sym_free and (t[1],t[0]) not in sym_free:
            sym_free.append(t)
    return set(sym_free)
    
def pair2tuple(pair:str)->tuple:
    token = ','
    tup = tuple(pair.split(token))
    return tup

def pairs2tups(pairs:Sequence[str]):
    return list(map(pair2tuple,pairs))

def process_atom_val(v:str):
    v = v.replace('"','')
    if DF_EMPTY in v and len(v) == 3:
        v = np.nan
    return v

def cache(dir,data):
     with open(dir, 'wb') as fp:
        pickle.dump(data, fp)
        
def load_cache(dir):
    # print("========",dir)
    with open(dir, 'rb') as file:
        return pickle.load(file)
    
def load_result(dir = None, sol= None, a_models = None,triple=True):
    #print("=========",dir, is_empty(dir))
    #print('0000000000000000000000',triple)
    if not is_empty(dir):
        with open(dir, 'rb') as file:
            #acts = pickle.load(file)
            sol, a_models = pickle.load(file)
            if triple : sol = set(map(lambda t:(t[0].replace('(',''),t[1].replace('"',''),t[2].replace('"','').replace(')','')) ,pairs2tups(sol)))
            else: 
                sol = set(map(lambda t:(t[0].replace('(','').replace('"',''),t[1].replace('"','').replace(')','')) ,pairs2tups(sol)))
            all_models = []
            #for model in a_models:
             #   model_sym = []
              #  for a in model:
               #     parse_string(
                #        a,
                 #   lambda ast: model_sym.append(ast),
                  #  )
                #all_models.append(model_sym)
            return sol, all_models
    else:
        if triple : sol = set(map(lambda t:(t[0].replace('(',''),t[1].replace('"',''),t[2].replace('"','').replace(')','')) ,pairs2tups(sol)))
        else: sol = set(map(lambda t:(t[0].replace('(','').replace('"',''),t[1].replace('"','').replace(')','')) ,pairs2tups(sol)))
        return sol, a_models
    
def tup_expand(tups:Iterable[tuple]):
    return tups.union({(x[1],x[0]) for x in tups})

def flatten_element(original_list:list)->list:
    for element in original_list[:]:
        if isinstance(element, list):
            original_list.remove(element)
            original_list.extend(element)
    return original_list

def compare_cache(fname1,fname2):

    tp_path1 = os.path.join('./cache',fname1)
    tp1 = load_cache(tp_path1)
    #print(tp1)
    print(len(tp1))
    
    tp_path2 = os.path.join('./cache',fname2)
    tp2 = load_cache(tp_path2)
    #print(tp2)
    print(len(tp2))
    #print(tup_expand(tp1).difference(tup_expand(tp2)))
    
    tp_diff = remove_symmetry(tup_expand(tp1).difference(tup_expand(tp2)))

    print(f"* difference {fname1} \- {fname2} : {str(tp_diff)}")
    
def compare(fname1,fname2):
    fname1=fname1.replace('.lp','').split('/')[-1]
    fname2=fname2.replace('.lp','').split('/')[-1]
    tp_path1 = os.path.join(TP_CACHE,f"tps-{fname1}.pkl")
    tp1 = load_cache(tp_path1)
    fp_path1 = os.path.join(FP_CACHE,f"fps-{fname1}.pkl")
    fp1 = load_cache(fp_path1)
    fn_path1 = os.path.join(FN_CACHE,f"fns-{fname1}.pkl")
    fn1 = load_cache(fn_path1)
    
    tp_path2 = os.path.join(TP_CACHE,f"tps-{fname2}.pkl")
    tp2 = load_cache(tp_path2)
    fp_path2 = os.path.join(FP_CACHE,f"fps-{fname2}.pkl")
    fp2 = load_cache(fp_path2)
    fn_path2 = os.path.join(FN_CACHE,f"fns-{fname2}.pkl")
    fn2 = load_cache(fn_path2)
    
    tp_diff = remove_symmetry(tup_expand(tp1).difference(tup_expand(tp2)))
    fp_diff = remove_symmetry(tup_expand(fp1).difference(tup_expand(fp2)))
    fn_diff = remove_symmetry(tup_expand(fn1).difference(tup_expand(fn2)))
    print(f"* TP difference {fname1} \- {fname2} : {str(tp_diff)}")
    print(f"* FP difference {fname1} \- {fname2} : {str(fp_diff)}")
    print(f"* FN difference {fname1} \- {fname2} : {str(fn_diff)}")
    
    
def find_intersection(set_obj):
        if len(set_obj) == 0:
            return set()  # Return an empty set if the input set is empty

        # Take the intersection of all sets using the first set as a starting point
        result = set_obj.pop()
        result.intersection_update(*set_obj)
        return result
    
    
import csv

def generate_variants():
    years = range(1920, 2025)
    variants = []
    for year in years:
        # Generate variants for each year
        yy1 = str(year)[:2]
        yy2 = str(year)[2:]
        yyyy = str(year)
        variant1 = f"{yy1}({yy2})"
        variant2 = f"{yyyy}."
        variant3 = f"{yy1} {yy2}"
        variant4 = f"({yyyy})"
        variant5 = f"{yyyy}-"
        variants.append((year, variant1))
        variants.append((year, variant2))
        variants.append((year, variant3))
        variants.append((year, variant4))
        variants.append((year, variant5))

    # Write variants to a CSV file
    with open("year_variants.csv", "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Original Year", "Variant"])
        csvwriter.writerows(variants)

# Call the function to generate variants and write to CSV file

def replace_pred(to_be_replaced:str, replacement:str, atom_set:Sequence[str]) -> Sequence[str]:
    return set([a.replace(to_be_replaced,replacement,1) for a in atom_set if a.startswith(to_be_replaced)])

if __name__ == "__main__":
    # generate_variants()
    music_facts = load_cache('./cache/sim-music50.pkl')
    [print(e) for e in music_facts]
    #fname1 = 'imdb-4k'
    #fname2 = 'imdb-4k-no-eq'
    #rels = ['name_basics_4k','title_basics_4k']
    #music_rels = ['track','recording','medium','release','release_group','area','artist','label','artist_credit','place']
    #for r in rels:
     #   compare(f"{fname1}_{r}",f"{fname2}_{r}")
      #  compare(f"{fname2}_{r}",f"{fname1}_{r}")
    #sum = 653 + 2053 + 3203 + 813 + 432 + 638 + 1998 + 2170 + 2180 + 3247
    #print(str(sum))
    # example 1
    # print(semantic_sim('German classical composer',"German clajsical composer"))
    #print(f"""
    #{ANTD_TRACE_RULE}{{"(%,%) is symmetrically closed by (%,%)", X,Y,Y,X}}.
    #{EQ_PRED}(X,Y):-{EQ_PRED}(Y,X).
    #{ANTD_TRACE_RULE}{{"(%,%) is transitivly closed by (%,%) - (%,%) ", X,Z,X,Y,Y,Z}}.
    #{EQ_PRED}(X,Z):-{EQ_PRED}(X,Y),{EQ_PRED}(Y,Z).
    #""")
   
    # example 2
    #print(semantic_sim('Association of Tennis Professionals',"ATP"))
   #print(sim('$Hz222',"$H2222"))
   
   # example 3
   #print(semantic_sim('Novak Djokovic is a player of the Association of Tennis Professionals',"Novak Djokovic is a player of the ATP"))
   #print(sim('Novak Djokovic is a player of the Association of Tennis Professionals',"Novak Djokovic is a player of the ATP"))
   
   # example 4
   #print(semantic_sim('Novak Djokovic plays at the Association of Tennis Professionals',"Novak Djokovic is a player of the ATP"))
   #print(sim('Novak Djokovic plays at the Association of Tennis Professionals',"Novak Djokovic is a player of the ATP"))
   
   # example 5
   #print(semantic_sim('Novak Djokovic plays at the ATP',"Novak Djokovic is a player of the ATP"))
   #print(sim('Novak Djokovic plays at the ATP',"Novak Djokovic is a player of the ATP"))
   
   # example 6
   #print(semantic_sim('Novak Djokovic is a US player',"Novak Djokovic is a player from the United States"))
   #print(sim('Novak Djokovic is a US player',"Novak Djokovic is a player from the United States"))
   
   # example 7
   #print(semantic_sim('Novak Djokovic is a NBA player',"Novak Djokovic is a from the National Basketball Association"))
   #print(sim('Novak Djokovic is a NBA player',"Novak Djokovic is a from the National Basketball Association"))
   
    # example 8
   #print(semantic_sim('Big Waider Horton,Horton Bit Walte,1921,4,6,1981,12,8,t,,Male,id-222,blues harkonica player',"Big Walter Horton,Horton Big Walter,1921,4,6,1981,12,8,t,Person,Male,id-222,blues harmonica player"))
   #print(sim('Big Waider Horton,Horton Bit Walte,1921,4,6,1981,12,8,t,,Male,id-222,blues harkonica player',"Big Walter Horton,Horton Big Walter,1921,4,6,1981,12,8,t,Person,Male,id-222,blues harmonica player"))
   #fname1 = 'fns/fns-music_medium.pkl'
   #fname2 = 'fns/fns-music-ub_medium.pkl'
   #compare_cache(fname2,fname1)
   