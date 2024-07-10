from argparse import ArgumentError
from dataloader import Dataloader
import utils
from logger import logger
from trans_utils import get_merge_attributes, get_sim_pairs, get_merge_attributes_local

import os
# from mpi import MPIWrapper


DISTINCT = 0
PARTIAL = 1

REL_OBJ = 'obj'
REL_MAPPING = 'mapping'


VAL_DOMAIN = 'v'
OBJ_DOMAIN = 'o'
RELATION = 'rel'

# type of relation
SUB_REL = 'subrel' # views
O_REL = "o-rel"


STRING = 'string'
NUMBER = 'number'
TIME = 'time_stamp'

SEP_COMMA = ','
SEP_AND = ' and '
SEP_AMP = ' & '

SEP_LST = [SEP_COMMA,SEP_AMP,SEP_AND] 

DF_EMPTY = "nan"

# predicate name
SIM_PRED = 'sim'
ACTIVE_DOM = 'adom'
EQ_PRED = 'eq'
ANONYM_VAL = '_'
TIDX_FUNC = '@t_idx(0)'
TIDX = 'TIDX'
META = "meta_"
REL_PRED = f'{META}rel'
SCHEMA_PRED = f'{META}schema'
ATTR_PRED = f'{META}attr'
REL_ATTR_MAP = f'{META}rel_attr_map'
TUPLE_PRED = 't'
CONST_PRED = 'c'



## stats
REL_NUM ="#rel"
ATTR_NUM ="#attr"
EMPTY_RATIO = "empty_ratio"
DISTINCT_VAL = "distinct_val"
REF_NUM = "#ref"
REC_NUM = "#record"
TOTAL_REC = '#total_records'
TOTAL_CONSTANT = '#total_constants'
TOTAL_ATTR = '#total_attr'
DUP_NUM = '#ground_truth'


LACE = 0
LACE_P = 1
VLOG_LB = 2
VLOG_UB = 3


class Attribute:
    # categorical values
    CAT = 'cat'
    # sequence of categorical values
    CAT_SEQ = 'cat_seq'
    # numeric values
    NUM = 'numeric'
    # numeric sequence
    NUM_SEQ = 'numeric_seq'
    # date
    DATE = 'date'
    # string values
    STR = 'string'
    # numeric sequence
    STR_SEQ = 'string_seq'
    # ids/referenced ids
    IDS = 'id'
    
    DTYPE_LST = [CAT,CAT_SEQ,STR,STR_SEQ,NUM,NUM_SEQ,IDS]
    
    NON_SIM_LST = [CAT,IDS]
    
    # LACE type
    MERGE = 0
    SIM = 1
    
    ##LACE+ type
    O_MERGE = 2
    V_MERGE = 3
    SINGLETON = 4

    
    def __init__(self, id, name:str, type:int= SIM,data_type:str = STR, is_list:bool = False, rel_name= None) -> None:
        # assert data_type in self.DTYPE_LST
        self.id = id # id will be unique under a schema
        # self.pos = pos
        self.name = name # could be a set of names
        self.type = type
        self.is_list = is_list
        #self.comparable = comparable
        # a list of comparable partially to A attribute of R relation occurring in the same schema
        # self.plist = plist
        # [2023-1-16] adding data type to columns of schema and compute the similarities accordingly
        self.data_type = data_type
        self.rel_name = rel_name
 
        
    def __eq__(self, __o: object) -> bool:
        return __o is not None and __o.id == self.id 
    
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    
    def __hash__(self) -> int:
        return hash(self.id)

    def __str__(self) -> str:
        return f"{{id:{self.id}, name:{self.name}, rel_name:{self.rel_name} type:{self.type}, data_type: {self.data_type}, is_list: {self.is_list}}}"

# will be a set of meta information
class Relation:
    def __init__(self,s_id,id,name,type = REL_OBJ,attrs=None, main_idx=0, is_dup = False) -> None:
        self.s_id = s_id
        self.name = name 
        self.id = id
        # sequence of attribute position and the attribute domain of a schema 
        # TODO: write an add function to avoid duplication
        # expected to be ordered, thus we use list here instead
        self.attrs:list[Attribute] = attrs if attrs is not None else list()
        #self.tbl = tbl
        self.type = type
        self.main_idx = main_idx
        # [2023-01-20] added to distinguish is corrupted or not (if known)
        self.is_dup = is_dup

    def __eq__(self, __o: object) -> bool:
        #if __o is None:
            # return False
        o_keys = set([a for a in __o.attrs])
        self_keys = set([a for a in self.attrs])
        flag = False if o_keys.intersection(self_keys) != self_keys else True
        #False in [__o.attrs[k]==self.attrs[k] for k in self_keys]
        # require r_id and all attributes are the same
        return __o.id == self.id and flag
    
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    
    def __hash__(self) -> int:
        # IMPORTANT: hashing attrs makes the hash value mutable (not static), 
        # thus we hash id only, and leave __eq__ as it be
        # + sum([hash(a[1]) for a in self.attrs])
        return hash(self.id) 
    
    def __str__(self) -> str:
        # print(len(self.attrs))
        return f"{{id:{self.id}, name:{self.name}, attrs: {list(map(str,self.attrs))}}}"
    
    
    def attr_index(self,attr_id:str)->int:
        for i, a in enumerate(self.attrs):
            if a.id == attr_id:
                return i
    # relation could give a function as a template to instantiate object of the relation
    #def get_attr_by_name(self,name):
     #   attrs_dict =  {a.name:a for a in self.attrs}
      #  if name in attrs_dict:
       #     return attrs_dict[name]
        #else:
         #   return None
    
    # taking set of attributes and initialising them as meta objects
    # leaving csv as original, operating on the csv objects to get values
    # splitting functions will be defined in Relation class
    # updating the relation set of associated schema after splitting

class Schema: 
    schema_log = logger('metainfo.Schema')
    # set of tuples?
    """_summary_
    
    @refs: referential dependencies of the schema, of the shape 
            dictionary {(rel1,ref_attr_1):(rel2,refed_attr_1),...,(rel_n,ref_attr_m),(rel_j,refed_attr_k)},
            circular references are not expected here.
    """
    def __init__(self,  id, name:str='', tbls:dict = None, 
                 dup_rels:set = None,refs:dict = None, attr_types = None,order = None, spec_dir = str, ver = LACE) -> None:
        assert id is not None
        self.name = name if len(name)> 0 else f'def_schema_{id}'
        self.id = id
        #self.uniq = uniq
        self.spec_dir = spec_dir
        # id formatted as s_id-r_id
        self.relations:set[Relation] = set()
        # self.dup_rels = dup_rels
        # self.attr_types = attr_types
        # TODO: store as a dict
        # TODO: take relation position to retrieval attributes?
        self.attrs:set[Attribute] = set()
        self.attrs_lst = list() # added to test generalised tuple representation
        self.i_attrs = set ()
        self.attr_num = 0
        self.i_attr_num = 0
        self.rel_num = 0
        self.tbls = tbls
        self.refs = refs
        self.ver = ver
        # global uid
        self.u_id = 0
        if self.ver == LACE_P:
            mergeo_attrs, mergev_attrs = get_merge_attributes_local(spec_dir)
            #print('object')
            #[print(k,v) for k,v in mergeo_attrs.items()]
            #print('value')
            #[print(k,v) for k,v in mergev_attrs.items()]
        else:
            merge_attrs = get_merge_attributes(spec_dir)
        # load the set of all attributes
        for k,v in tbls.items():
            tbl = tbls[k]
            k = k[:-2] if k.endswith('_c') else k
            rel = Relation(s_id=self.id,id = self.rel_num, name=k)    
            columns = list(tbl[1].columns.values) 
            if self.ver == LACE_P:
                for i,c in enumerate(columns):
                    attr_type = Attribute.SINGLETON # [2024-4-13] those do not participate any merges
                    if k in mergeo_attrs and i in mergeo_attrs[k]:
                        #print(c,'o')
                        attr_type = Attribute.O_MERGE
                    elif k in mergev_attrs and i in mergev_attrs[k]:
                        #print(c,'v')
                        attr_type = Attribute.V_MERGE
                    # a_type = Attribute.MERGE if ver == LACE else Attribute.O_MERGE
                    attr = Attribute(id = self.attr_num, name = f'{c}',type=attr_type, \
                        # assumption here is merge attributes are not some kind of textual string
                        data_type= Attribute.STR \
                        ,rel_name=k)
                    self.add_val_attr(attr=attr)
                    rel.attrs.append(attr)
                    self.add_rel(rel=rel)
            # initialise attributes
            else:
                for i,c in enumerate(columns):
                    is_merge = False
                    for _k,_v in merge_attrs.items():
                        if k == _k:
                            is_merge = i in _v
                    # a_type = Attribute.MERGE if ver == LACE else Attribute.O_MERGE
                    attr = Attribute(id = self.attr_num, name = f'{c}',type=Attribute.MERGE if is_merge else Attribute.SIM, \
                        # assumption here is merge attributes are not some kind of textual string
                        data_type=Attribute.IDS if is_merge else Attribute.STR \
                        ,rel_name=k)
                    self.add_val_attr(attr=attr)
                    rel.attrs.append(attr)
                    self.add_rel(rel=rel)

        to_be_updated = dict()
        # iterate relations, check attributes
        # identify compatible attributes and consider them as one
        for r in self.relations:
            for i,a in enumerate(r.attrs):
                if self.refs != None and (r.name,a.name) in self.refs:
                    refed_r_name = self.refs[(r.name,a.name)][0]
                    refed_attr_name = self.refs[(r.name,a.name)][1]
                    refed_r = self.rel_index(refed_r_name) #TODO, replace attr list by string of ids
                    for _a in refed_r.attrs:
                        if _a.name == refed_attr_name:
                            to_be_updated[(r.name,i)] = _a
                            break
        # update attributes of relations
        for t,a in to_be_updated.items():
            r = self.rel_index(t[0])
            r.attrs[t[1]] = a
        # create a new attribute list for the schema
        self.attrs = set()
        for r in self.relations:
            self.attrs.update(set(r.attrs))
        self.attr_num = len(self.attrs)
        # remove redundant attributes
        # update the relations      
                       
    def get_uidx(self,):
        self.u_id += 1
        return str(self.u_id)
    
        
    def add_val_attr(self,attr):
        if isinstance(attr,set):
            self.attrs.update(attr)
            self.attr_num += len(attr)
        else:
            self.attrs.add(attr)
            self.attrs_lst.append(attr)
            self.attr_num +=1
            
    def add_obj_attr(self,attr):
        if isinstance(attr,set):
            self.i_attrs.update(attr)
            self.i_attr_num += len(attr)
        else:
            self.i_attrs.add(attr)
            self.i_attr_num +=1
            
    def add_rel(self,rel):
        if isinstance(rel,set):
            self.relations.update(rel)
            self.rel_num += len(rel)
        else:
            self.relations.add(rel)
            self.rel_num +=1
            

    def attr_index(self,id):
        return self.get_attr_dict()[id]
    
    def index_attr(self,attr_name,rel_name)-> Attribute:
        attr = None
        for a in self.attrs:
            if a.name == attr_name and a.rel_name == rel_name:
                attr = a
                break
        return attr
    
    def attr_index_name(self,name):
        a_dict = self.get_attr_name_dict()
        if name in a_dict: 
            return a_dict[name]
        else: None
    
    def iattr_index(self,id):
        return self.get_iattr_dict()[id]
    
    def rel_index(self,r_name) -> Relation:
        rel_dict = {r.name:r for r in self.relations}
        if r_name in rel_dict:
            return rel_dict[r_name]
        else:
            return None
    
    #def view_index(self,r_name):
       # rel_dict = {r.name:r for r in self.views}
        #return rel_dict[r_name]
    
    def get_attr_dict(self,): 
        return {a.id:a for a in self.attrs}
    
    def get_attr_name_dict(self,): 
        return {a.name:a for a in self.attrs}
    
    def get_rel_dict(self,):
        return {r.id:r for r in self.relations}
    
    def get_iattr_dict(self,):
        return {a.id:a for a in self.i_attrs}

    def id_template(self,pred_type,p_id = None,id= None):
        if pred_type == RELATION:
           return f'{self.id}-{pred_type}-{id}'
        elif pred_type == SUB_REL:
            return  f'{p_id}_{id}'
        else:
            return f'{p_id}-{pred_type}-{id}'
    
    def get_rel_domains(self,r_id):
        return set(a for a in self.attrs.union(self.i_attrs) if r_id in a.id)
    
    def stats (self,) -> dict:
        print("* Processing reocrd stats of the schema ...")
        stats = {'domain_attr':0, REL_NUM:0, TOTAL_REC:0 , TOTAL_CONSTANT:0, TOTAL_ATTR:0, REC_NUM:{}, ATTR_NUM:{},EMPTY_RATIO:{},DISTINCT_VAL:{},REF_NUM:{}}
        stats[REL_NUM] = len(self.tbls.items())
        for r_name,tbl in self.tbls.items():
            # print(tbl[0])
            stats[ATTR_NUM][r_name] = len(tbl[1].columns)
            stats[TOTAL_ATTR]+= len(tbl[1].columns)
            stats['domain_attr'] = len(self.attrs)
            stats[REC_NUM][r_name] = len(tbl[1])
            stats[TOTAL_REC] += len(tbl[1]) 
            # dataframe tolist for value counting
            for col in tbl[1].columns:
                distinct_cnt = len(tbl[1][col].value_counts(dropna=True))
                stats[DISTINCT_VAL][r_name+'.'+col] = distinct_cnt
                stats[TOTAL_CONSTANT] += distinct_cnt
            emp_ratio = {c:0 for c in tbl[1].columns}
            for _, row in tbl[1].iterrows():
                for col in tbl[1].columns:
                    v = row[col]
                    if utils.is_empty(v):
                        emp_ratio[col]+=1
            stats[EMPTY_RATIO][r_name] = {k:round(e/stats[REC_NUM][r_name],3) for k,e in emp_ratio.items()}            
        return stats        
    
    def schema_info(self,):
        self.schema_log.info('Schema Name: %', [self.name])
        # [self.schema_log.info('Relation % : %', [r.name,r]) for r in self.relations] 
        sum = 0
        for k,v in self.tbls.items():
            sum+= len(v[1])
            self.schema_log.info(f'Relation {k} has {str(len(v[1]))} of tuples.')
        self.schema_log.info(f'#Tuples of schema {self.name} is {str(sum)}.')



def schema_init(name:str,spec_dir:str, data_paths:list[str], ground_truth_path:list[str]=[],ref_dict:dict[tuple,tuple]={})->tuple[Schema,Dataloader]:
    path_list = data_paths
    dl = Dataloader(name = f'{name}',path_list=path_list,ground_truth=ground_truth_path)
    tbls = dl.load_data()  
    tbls_dict = {t[0]:(0,t[1]) for t in tbls}
    del tbls 
    schema = Schema(id = '1',name =f'{name}',tbls=tbls_dict,refs=ref_dict,spec_dir=spec_dir)
    return schema,dl

if __name__ == "__main__":
    #dl = Dataloader(name = 'cora_tsv')
    #tbl = dl.load_data()
    #schema = Schema('1','cora',{'cora':(5,tbl)},)
    #print([[a.name for a in r.attrs ] for r in schema.relations])
    #for a in schema.attrs:
     #   if a.name == 'authors' or a.name == 'editor':
      #      a.is_list = True
    TEST = '_10k'
    T_BASIC = 'title_basics' 
    N_BASIC = 'name_basics' 
    T_AKA = 'title_akas' 
    T_PRINCIPLE = 'title_principals' 
    T_RATINGS = 'title_ratings' 
    ref_dict =    {(T_AKA,'titleId'):(T_BASIC,'tconst'),
     (T_RATINGS,'tconst'):(T_BASIC,'tconst'),
     (T_PRINCIPLE,'tconst'):(T_BASIC,'tconst'),
     (T_PRINCIPLE,'nconst'):(N_BASIC,'nconst'),}
    IMDB_PATH = './dataset/imdb'
    dup_rel = {T_BASIC,N_BASIC}
    path_list = [IMDB_PATH+'/name_basics.csv',IMDB_PATH+'/title_akas.csv',IMDB_PATH+'/title_basics.csv',IMDB_PATH+'/title_principals.csv',IMDB_PATH+'/title_ratings.csv']
    dl_imdb = Dataloader(name = 'imdb',path_list=path_list)
    tbls = dl_imdb.load_data()   
    tbls_dict = {t[0]:(0,t[1]) for t in tbls}
    dtype_cfg = IMDB_PATH+'/domain.yml'
    imdb = Schema(id = '2',name ='imdb',tbls=tbls_dict,refs=ref_dict,dup_rels=dup_rel,attr_types=utils.load_config(dtype_cfg))
    # print(imdb.stats())
    # [print(a) for a in imdb.attrs]
    print(imdb.stats())
    #[[print(r.name,a) for a in r.attrs] for r in imdb.relations]
    # [print(atom) for atom in imdb.load_domain_mpi(sim_threshold=70)]
    #(akas,tid):(t\_basics,tid);(ratings,tid):(t\_basics,tid);
#(principals,tid):(t\_basics,tid)
#;(principals,nid):(n\_basics,nid)
    # [print(r) for r in schema.entity_split(schema.rel_index('cora').id,{'pub':[5,6,10,12,13,14],'author':[1],'editor':[4],'venue':[0,2,3,7,8,11,15,16]})]
    #print(schema.attr_num,schema.rel_num,schema.i_attr_num,schema.view_num)
    #[print(a) for a in schema.i_attrs]
    #[print(a) for a in schema.attrs]
    # print(schema.load_domain())
    # print([a for a in schema.load_domain() if a.startswith('sim(')])
    # print(schema.stats())
    #print(schema.facts_gen_rules())
    #s = schema.load_domain(sim_threshold=50)
    #print([ a for a in s if a.startswith('author')])
    #re = schema.entity_split(schema.rel_index('dblp').id,{'pub':[0,1],'au':[2],'ven':[3,4]})
    #r = Relation (s_id=schema.id,id='1-rel-0_0',name='pub',attrs={(  0, '1-rel-0-attr-0'),(1, '1-rel-0-attr-1'),(0, '1-rel-0_0-iattr-0')})
    # print(re.remove(r))
    # print(r in re)
    #r_list = [r for r in re]
    #print(len(r_list))
    #r_set = set(r_list)
    #print(len(r_set))

