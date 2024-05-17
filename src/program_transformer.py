
from metainfo import Schema, Attribute, Relation, LACE, LACE_P, VLOG_LB, VLOG_UB
import trans_utils

from typing import Sequence
from trans_utils import  get_sim_pairs, get_atom_vars, locate_body_var,get_atom_pred, get_atoms, DEF_OPERA_PAT
import pickle
import os
import re
import clingo
import utils
from logger import logger, START, END
import sys


CACHE_DIR = './cache'


### matching patterns ### 
ATOM_PAT =  utils.ATOM_PAT
VAR_PAT = utils.VAR_PAT
REL_PAT = utils.REL_PAT
EQUATE_PAT = utils.EQUATE_PAT
# if any variables in equate pattern have #count coocurrance, shall be exluded
# COUNT_PAT = re.compile(r"(([A-Z0-9]+)(?:\s*=|>=|<=|>|<\s*))?#count\{[^}]*\}((?:\s*=|>=|<=|>|<\s*)([A-Z0-9]+))?", re.IGNORECASE)
COUNT_PAT = utils.COUNT_PAT
# inside of brackets of count pattern should be separated by ";", counjuncts are the rest of strings concat by ":"
# TODO:
SIM_THRESH_PAT = utils.SIM_THRESH_PAT
SIM_THRESH_VAL_PAT =utils.SIM_THRESH_VAL_PAT
SIM_JOIN_PAT = utils.SIM_JOIN_PAT
SPEC_CHAR_PAT = utils.SPEC_CHAR_PAT
EQ_AC_PAT = utils.EQ_AC_PAT
EQ_AC_AT_PAT = utils.EQ_AC_AT_PAT
### matching patterns ### 

### data processing ####
SEP_COMMA = ','
SEP_AND = ' and '
SEP_AMP = ' & '
DF_EMPTY = 'nan'
SEP_LST = [SEP_COMMA,SEP_AMP,SEP_AND] 

META = "meta_"
REL_PRED = f'{META}rel'
SCHEMA_PRED = f'{META}schema'
ATTR_PRED = f'{META}attr'
REL_ATTR_MAP = f'{META}rel_attr_map'
TUPLE_PRED = 't'
CONST_PRED = 'c'
### data processing ####

### asp symbols ###
### predicates
SIM_PRED = "sim"
SIM_FUNC_PRED = '@' + SIM_PRED
EQ_PRED = "eq"
UP_EQ = trans_utils.UP_EQ

### LACE+ 
EQO_PRED = "eqo"
EQV_PRED = "eqv"
ACTIVE_O_PRED = "activeo"
ACTIVE_V_PRED = "activev"
PROJ_PRED = "proj"
VAL_PRED = "val"


MOCK_DC_PRED = 'falsify'
ACTIVE_PRED = "active"
TO_BE_SIM ='sim_attr'
TO_SIM ='to_sim'
ACTIVE_DOM = 'adom'
UP_EQ = 'up_eq'
NEQ_PRED = 'neq'
COUNT_DIRC = '#count'
EXTERNAL_DIRC = '#external'


### default symbols
DEFAULT_NEG = 'not'
IMPLY = ':-'
SHOW = '#show'
PROGRAM = '#program'
DOT = '.'
ANOMY_VAR = "_"
NOT_EQUAL = '!='
EQUAL = '='
LEQ = '<='
LESS = '<'

### default rules
EQ_AXIOMS = f"""
{EQ_PRED}(X,Y) {IMPLY} {EQ_PRED}(X,Z),{EQ_PRED}(Z,Y).
{EQ_PRED}(X,Y) {IMPLY} {EQ_PRED}(Y,X).
"""
EQ_AXIOMS_TER = f"""
{EQ_PRED}(X,Y,I) {IMPLY} {EQ_PRED}(X,Z,I),{EQ_PRED}(Z,Y,I).
{EQ_PRED}(X,Y,I) {IMPLY} {EQ_PRED}(Y,X,I).
"""

EQ_AXIOMS_VLOG_TRANS_TER = f"""
{EQ_PRED}(?X,?Y,?I) {IMPLY} {EQ_PRED}(?X,?Z,?I),{EQ_PRED}(?Z,?Y,?I).
"""

EQ_AXIOMS_VLOG_SYM_TER = f"""
{EQ_PRED}(?X,?Y,?I) {IMPLY} {EQ_PRED}(?Y,?X,?I).
"""


ACTIVE_CHOICE = f"""
{{{EQ_PRED}(X,Y)}} {IMPLY} {ACTIVE_PRED}(X,Y).
"""

ACTIVE_CHOICE_TER = f"""
{{{EQ_PRED}(X,Y,I)}} {IMPLY} {ACTIVE_PRED}(X,Y,I).
"""

SIM_TGRS = """sim(X,Y,S) :- sim(Y,X,S). 
sim(X,X,100) :- sim(X,X,_)."""
ADOM_TGRS = """eq(X,X) :- adom(X)."""
EMPTY_TGRS = """empty(nan). empty("nan").empty("ーーー")."""
### asp symbols ###


### trace annotations ###
ANTD_SHOW_TRACE = '%!show_trace'
ANTD_TRACE_ATOM = '%!trace'
ANTD_TRACE_RULE = '%!trace_rule'

SYMM_DESC = 'is symmetrically closed by'
TRANS_DESC = 'is transitivly closed by'

EQ_AXIOMS_TRACE = f"""
{ANTD_TRACE_RULE}{{"(%,%) {SYMM_DESC} (%,%)", X,Y,X,Y,Y,X}}.
{EQ_PRED}(X,Y){IMPLY}{EQ_PRED}(Y,X).
{ANTD_TRACE_RULE}{{"(%,%) {TRANS_DESC} (%,%) - (%,%) ", X,Z,X,Y,Y,Z}}.
{EQ_PRED}(X,Z){IMPLY}{EQ_PRED}(X,Y),{EQ_PRED}(Y,Z).
"""

EQ_AXIOMS_TER_TRACE = f"""
{ANTD_TRACE_RULE}{{"(%,%) {TRANS_DESC} (%,%) - (%,%)", X,Z,X,Y,Z,Y}}.
{EQ_PRED}(X,Y,I) {IMPLY} {EQ_PRED}(X,Z,I),{EQ_PRED}(Z,Y,I).
{ANTD_TRACE_RULE}{{"(%,%) {SYMM_DESC} (%,%) ", X,Y,X,Y,Y,X}}.
{EQ_PRED}(X,Y,I) {IMPLY} {EQ_PRED}(Y,X,I).
"""
# 1
EQ_AXIOMS_REC = f"""
{EQ_PRED}(X,Y,I) {IMPLY} {EQ_PRED}(X,Z,I),{EQ_PRED}(Z,Y,I1), I1<=I, I<i.
{EQ_PRED}(X,Y,I) {IMPLY} {EQ_PRED}(Y,X,I).
"""
# 2
EQ_AXIOMS_REC_ = f"""
{EQ_PRED}(X,Y,i) {IMPLY} {EQ_PRED}(X,Z,i-1),{EQ_PRED}(Z,Y,I), I<i.
{EQ_PRED}(X,Y,I) {IMPLY} {EQ_PRED}(Y,X,I).
"""
#3
_EQ_AXIOMS_REC = f"""
{EQ_PRED}(X,Y,i-1) {IMPLY} {EQ_PRED}(X,Z,i-1),{EQ_PRED}(Z,Y,I), I<i.
{EQ_PRED}(X,Y,I) {IMPLY} {EQ_PRED}(Y,X,I).
"""
#1
EQ_AXIOMS_REC_TER = f"""
{EQ_PRED}(X,Y,A,I) {IMPLY} {EQ_PRED}(X,Z,A,I),{EQ_PRED}(Z,Y,A,I1), I1<=I, I<i.
{EQ_PRED}(X,Y,A,I) {IMPLY} {EQ_PRED}(Y,X,A,I).
"""
#2
EQ_AXIOMS_REC_TER_ = f"""
{EQ_PRED}(X,Y,A,i) {IMPLY} {EQ_PRED}(X,Z,A,i-1),{EQ_PRED}(Z,Y,A,I), I<i.
{EQ_PRED}(X,Y,A,I) {IMPLY} {EQ_PRED}(Y,X,A,I).
"""
#3
_EQ_AXIOMS_REC_TER = f"""
{EQ_PRED}(X,Y,A,i) {IMPLY} {EQ_PRED}(X,Z,A,i),{EQ_PRED}(Z,Y,A,I),I<i.
{EQ_PRED}(X,Y,A,I) {IMPLY} {EQ_PRED}(Y,X,A,I).
"""

MEQ_REC = f'm{EQ_PRED}(X,Y,i) {IMPLY} {EQ_PRED}(X,Y,i), {EQ_PRED}(X,Y).'
MEQ_REC_TER = f'm{EQ_PRED}(X,Y,A,i) {IMPLY} {EQ_PRED}(X,Y,A,i), {EQ_PRED}(X,Y,A).'

ITER_READOFF = 'eqm(X,Y,A,I) :- eq(X,Y,A,I), eq(X,Y,A).' 


# TODO: to modify here
TRACE_EQ = f'{ANTD_SHOW_TRACE} {{{EQ_PRED}(X,Y)}}.'
# TODO: to modify here
TRACE_EQ_TER = f'{ANTD_SHOW_TRACE} {{{EQ_PRED}(X,Y,I)}}.'
TRACE_PAIR = f'{ANTD_SHOW_TRACE} {{{EQ_PRED}(^,^)}}.'
TRACE_PAIR_TER = f'{ANTD_SHOW_TRACE} {{{EQ_PRED}(^,^,^)}}.'
### trace annotations ###



### LACE+ axioms

EQUIV_CELL_SET = f'{VAL_PRED}(T1,I2,V) {IMPLY} {EQV_PRED}(T1,I1,T2,I2), {PROJ_PRED}(T2,I2,V){DOT}'

EQV_SYMMETRY = f'{EQV_PRED}(T1,I1,T2,I2) {IMPLY} {EQV_PRED}(T2,I2,T1,I1){DOT}'
EQV_TRANSITIVITY = f'{EQV_PRED}(T1,I1,T2,I2) {IMPLY} {EQV_PRED}(T1,I1,T3,I3), {EQV_PRED}(T3,I3,T2,I2){DOT}'

EQO_SYMMETRY = f'{EQO_PRED}(X,Y) {IMPLY} {EQO_PRED}(Y,X){DOT}'
EQO_TRANSITIVITY = f'{EQO_PRED}(X,Y) {IMPLY} {EQO_PRED}(X,Z),{EQO_PRED}(Z,Y){DOT}'

ACTIVE_EQV = f'{{{EQV_PRED}(T1,I1,T2,I2)}} {IMPLY} {ACTIVE_V_PRED}(T1,I1,T2,I2){DOT}'
ACTIVE_EQO = f'{{{EQO_PRED}(X,Y)}} {IMPLY} {ACTIVE_O_PRED}(X,Y){DOT}'
### LACE+ axioms


HEAD = 'head'
BODIES = 'BODIES'


def get_sim_atom(x:str,y:str,s:int) -> str:
    sim_tup = (f'"{x}"',f'"{y}"',s)
    return utils.get_atom_(SIM_PRED,sim_tup)

def get_default_tup(rel:Relation)-> list:
    # print(rel.name,len(rel.attrs))
    return ['_' for a in rel.attrs]

def add_dom_val(attribute:Attribute,val,ter = False):
    val = utils.escape(val,attribute.data_type == Attribute.CAT_SEQ) if not utils.is_empty(val) else DF_EMPTY 
    const_type = [Attribute.NUM]
    if attribute.data_type not in const_type and val!=DF_EMPTY:
        val = f'"{val}"' if ter else f'"{attribute.id}:{val}"'
    if not isinstance(val,str):
        val = str(int(val))
    return val

def add_dom_val_local(attribute:Attribute,val,ter = False):
    val = utils.escape(val,attribute.data_type == Attribute.CAT_SEQ) if not utils.is_empty(val) else DF_EMPTY 
    const_type = [Attribute.NUM]
    if attribute.data_type not in const_type and val!=DF_EMPTY:
        val = f'"{val}"' if ter or not attribute.type == Attribute.O_MERGE else f'"{attribute.id}:{val}"'
    if not isinstance(val,str):
        val = str(int(val))
    return val

# TODO wrapping schema into a factory function
class program_transformer:
    ORIGIN = 1
    UPERBOUND = 2
    SIM_PRG = 3
    LOWERBOUND = 4
    def __init__(self, schema:Schema) -> None:
        self.schema = schema
        self.spec_dir = schema.spec_dir
        self.sim_pairs = get_sim_pairs(self.spec_dir)
        self.log = logger('program_transformer')
        spec = trans_utils.get_rule_list(self.spec_dir)
        self.rules:list[str] = spec[0] 
        #if self.schema.ver == LACE else self.transform_local(spec[0])
        self.constraints:list[str] = spec[1] 
        #if self.schema.ver == LACE else self.transform_local(spec[1])
        self.annotations:list[tuple[int,str]] = spec[2] # list of tuples [(r_idx, annotation)]
        self.show_list = spec[3]
        
        
    def __load_domain(self, sep_lst = [SEP_AND,SEP_AMP,SEP_COMMA],token='',multi_lvl = False, ter = False):
        print("* Starting loading domain and processing atom base ...")
        # val_dom_dict = self.get_attr_dict()
        # dictionary
        # key attribute id
        # val set of references
        # dom_values = {}
        atom_base = set()
        multi_lvl_token = ',0' if multi_lvl else ''
        #[fixed]: avoid calculating the similarity scores of IDs
        # TODO: [2023-01-23] iterating attr set of the schema instead of instance records?
        for r_name,tbl in self.schema.tbls.items():
            rel = self.schema.rel_index(r_name)
            for _,row in tbl[1].iterrows():
                r_tup = get_default_tup(rel)
                #r_tup = r_tup[1:]
                # [2023-11-23] modified, canel tuple id
                for a_idx,attr_ in enumerate(tbl[1].columns):
                    if self.schema.refs !=None and (r_name,attr_) in self.schema.refs:
                        attr = self.schema.index_attr(self.schema.refs[(r_name,attr_)][1],self.schema.refs[(r_name,attr_)][0])
                    else: 
                        attr = self.schema.index_attr(attr_,r_name)
                    # TODO: remove condition
                    if attr!=None:
                        val = add_dom_val(attr,row[attr_],ter=ter)
                        if attr.type == Attribute.MERGE and val != DF_EMPTY:
                            reflect_eq = ''
                                # reflect_eq = f'{EQ_PRED}({val},{val},0).'
                            if ter:
                                reflect_eq = f'{EQ_PRED}({val},{val},{attr.id}{multi_lvl_token}).'
                            else:
                                reflect_eq = f'{EQ_PRED}({val},{val}{multi_lvl_token}).'
                            atom_base.add(reflect_eq)
                        #print(a_idx)
                        r_tup[a_idx] = val
                r_tup_str = ','.join(r_tup)
                pred = f'{r_name}' if utils.is_empty(token) else f'{r_name}_{token}'
                r_atom =f'{pred}({r_tup_str}).'
                atom_base.add(r_atom)
                            
        return atom_base 
    
    def __load_local_domain(self, token='',multi_lvl = False, ter = False):
        atom_base = set()
        multi_lvl_token = ',0' if multi_lvl else ''
        # TODO: [2023-01-23] iterating attr set of the schema instead of instance records?
        for r_name,tbl in self.schema.tbls.items():
            rel = self.schema.rel_index(r_name)
            for _,row in tbl[1].iterrows():
                r_tup = get_default_tup(rel)
                tid = self.schema.get_uidx()
                r_tup.insert(0,str(tid))
                # [2023-11-23] modified, canel tuple id
                for a_idx,attr_ in enumerate(tbl[1].columns):
                    if self.schema.refs !=None and (r_name,attr_) in self.schema.refs:
                        attr = self.schema.index_attr(self.schema.refs[(r_name,attr_)][1],self.schema.refs[(r_name,attr_)][0])
                    else: 
                        attr = self.schema.index_attr(attr_,r_name)
                    # TODO: remove condition
                    if attr!=None:
                        val = add_dom_val_local(attr,row[attr_],ter=ter)
                        reflect_eq = ''
                        if attr.type == Attribute.O_MERGE and val != DF_EMPTY:
                            if ter:
                                reflect_eq = f'{EQO_PRED}({val},{val},{attr.id}{multi_lvl_token}).'
                            else:
                                reflect_eq = f'{EQO_PRED}({val},{val}{multi_lvl_token}).'
                            
                        else:
                            v_pos = f'{str(tid)},{str(a_idx+1)}'
                            reflect_eq = f'{EQV_PRED}({v_pos},{v_pos}).'
                            proj_fact = f'{PROJ_PRED}({v_pos},{val}).'
                            atom_base.add(proj_fact)
                        atom_base.add(reflect_eq)

                        r_tup[a_idx+1] = val
                r_tup_str = ','.join(r_tup)
                pred = f'{r_name}' if utils.is_empty(token) else f'{r_name}_{token}'
                r_atom =f'{pred}({r_tup_str}).'
                atom_base.add(r_atom)
        return atom_base 
    
    def __load_domain_col(self, sep_lst = [SEP_AND,SEP_AMP,SEP_COMMA],sim_cols:list = None):
        load_semi_col_log = logger('load_domain_col')
        def add_dom_val(attribute:Attribute,val):
            val = utils.escape(val) if not utils.is_empty(val) else DF_EMPTY
            return f'"{val}"'
        sim_val_idx = 0
        c_temp = 'c('
        atom_base = set()
        c_dict = dict()
        #[fixed]: avoid calculating the similarity scores of IDs
        # TODO: [2023-01-23] iterating attr set of the schema instead of instance records?
        load_semi_col_log.debug(' Start loading domain and processing semi-colbased atoms ...')
        for r_name,tbl in self.schema.tbls.items():
            rel = self.schema.rel_index(r_name)
            for _,row in tbl[1].iterrows():
                r_tup = get_default_tup(rel)
                r_tup[0] = self.schema.__get_uidx()
                for a_idx,attr_ in enumerate(tbl[1].columns):
                    if self.schema.refs !=None and (r_name,attr_) in self.refs:
                        attr = self.attr_index_name(self.refs[(r_name,attr_)][1])
                    else: attr = self.attr_index_name(attr_)
                    val = add_dom_val(attr,row[attr_]) 
                        #if a_idx != tbl[0]: # TODO: might need to be removed in local semantics
                    off_set = a_idx+1
                    r_tup[off_set] = val
                    if sim_cols!=None and len(sim_cols)>0 and (r_name,off_set) in sim_cols:
                        if val not in c_dict:
                            r_tup[off_set] = str(sim_val_idx)
                            c_dict[val] = str(sim_val_idx)
                            sim_val_idx+=1 
                        else:
                            r_tup[off_set] = c_dict[val]
                r_tup_str = ','.join(r_tup)
                r_atom =f'{r_name}({r_tup_str}).'
                atom_base.add(r_atom)
                #if rel.is_dup: atom_base.add(f'{ACTIVE_DOM}({r_tup[0]}).')  # [2023-02-01] active domain added only for dup relations          
                if rel.is_dup: atom_base.add(f'eq({r_tup[0]},{r_tup[0]}).')
        for k,v in c_dict.items():
            atom_base.add(f'{c_temp}{v},{k}).')
        return atom_base 
    
    
    def __load_colbase_domain(self, sep_lst = [SEP_AND,SEP_AMP,SEP_COMMA]):
        print("* Starting loading domain and processing atom base ...")
        def add_dom_val(attribute:Attribute,val):
            val = utils.escape(val) if not utils.is_empty(val) else DF_EMPTY
            return val
        c_temp = 'c('
        atom_base = set()
        c_dict = dict()
        # schema facts
        schema_atom_tup = list()
        schema_atom_tup.append(str(self.id))
        schema_atom_tup.append(f'"{self.name}"')
        atom_base.add(utils.get_atom_(SCHEMA_PRED,schema_atom_tup))
        # attribute facts
        for a in self.attrs_lst:
            attr_atom_tup = list()
            attr_atom_tup.append(str(a.id))
            attr_atom_tup.append(f'"{a.name}"')
            # print(a.name)
            atom_base.add(utils.get_atom_(ATTR_PRED,attr_atom_tup))
        tup_idx = 0
        const_idx = 0
        # rel facts
        for r_name,tbl in self.schema.tbls.items():
            rel = self.schema.rel_index(r_name)
            rel_atom_tup = list()
            rel_atom_tup.append(str(rel.id))
            rel_atom_tup.append(f'"{rel.name}"')
            atom_base.add(utils.get_atom_(REL_PRED,rel_atom_tup))

            for a_indx,atr_ in enumerate(tbl[1].columns):
                if self.refs !=None and (r_name,atr_) in self.refs:
                    atr = self.attr_index_name(self.refs[(r_name,atr_)][1])
                else: atr = self.attr_index_name(atr_)
                rel_attr_tup = list()
                rel_attr_tup.append(str(rel.id))
                rel_attr_tup.append(str(atr.id))
                rel_attr_tup.append(str(a_indx))
                atom_base.add(utils.get_atom_(REL_ATTR_MAP,rel_attr_tup))
            for _,row in tbl[1].iterrows():
                for _,attr_ in enumerate(tbl[1].columns):
                    if self.refs !=None and (r_name,attr_) in self.refs:
                        attr = self.attr_index_name(self.refs[(r_name,attr_)][1])
                    else: attr = self.attr_index_name(attr_)
                    # put constant into dict
                    c = add_dom_val(attr,row[attr_]) 
                    cid = const_idx
                    if c not in c_dict:
                        c_dict[c] = cid
                        const_idx += 1
                    else:
                        cid = c_dict[c]
                    # create tuple atoms
                    tu_atom_tup = [str(tup_idx),str(rel.id),str(attr.id),str(cid)]
                    atom_base.add(utils.get_atom_(TUPLE_PRED,tu_atom_tup))
                tup_idx+=1
                if rel.is_dup: atom_base.add(f'{ACTIVE_DOM}({tup_idx}).')  # [2023-02-01] active domain added only for dup relations          
        for k,v in c_dict.items():
            atom_base.add(f'{c_temp}{v},{k}).')
        return atom_base 
    """
    def generating_facts(self,lower_bound:int=0, cached = True, is_symbol = False):
        lower_bound = 50
        def parse(atom:str):
            return clingo.parse_term(atom[:-1])
        # if cached
        cached_path = os.path.join(CACHE_DIR,f"facts_{self.schema.name}.pkl")
        if cached and os.path.isfile(cached_path):
             with open(cached_path, 'rb') as file:
                facts = pickle.load(file)
                if is_symbol:
                    facts = map(parse,facts)
                return set(facts)
        
        facts = self.schema.facts_gen(sim_threshold=lower_bound)
        with open(os.path.join(CACHE_DIR,f"facts_{self.schema.name}.pkl"), 'wb') as fp:
            pickle.dump(facts, fp)
        if is_symbol:
            facts = map(parse,facts)
        return set(facts)
    """

    def get_db_atombase_(self,cached:bool=True, multi_lvl=False,ter=False, ):
        # lower_bound = 50
        # if cached
        if multi_lvl:
            name = f'{self.schema.name}-multi' 
        elif ter: 
            name = f'{self.schema.name}-ter' 
        else:
            name = f'{self.schema.name}'
        atoms = utils.get_atoms(source_func = self.__load_domain,cache_dir= CACHE_DIR,fname = name, multi_lvl = multi_lvl,ter=ter)
        return set(atoms)
    
    def get_db_atombase_local(self,cached:bool=True, multi_lvl=False,ter=False, ):
        # lower_bound = 50
        # if cached
        if multi_lvl:
            name = f'{self.schema.name}-multi-p' 
        elif ter: 
            name = f'{self.schema.name}-ter-p' 
        else:
            name = f'{self.schema.name}-p'
        atoms = utils.get_atoms(source_func = self.__load_local_domain,cache_dir= CACHE_DIR,fname = name, multi_lvl = multi_lvl,ter=ter)
        return set(atoms)
    
    def get_db_atombase_col(self,cached:bool=True,version=""):
        atoms = utils.get_atoms(source_func = self.__load_colbase_domain,cache_dir= CACHE_DIR,fname= self.schema.name+version+'-col')
        return set(atoms)        
    
    """
    def generating_facts_(self,lower_bound:int=0, cached = True, is_symbol = False):
        lower_bound = 50
        def parse(atom:str):
            return clingo.parse_term(atom[:-1])
        # if cached
        cached_path = os.path.join(CACHE_DIR,f"facts_{self.schema.name}.pkl")
        if cached and os.path.isfile(cached_path):
             with open(cached_path, 'rb') as file:
                facts = pickle.load(file)
                if is_symbol:
                    facts = map(parse,facts)
                return set(facts)

        tbl_facts = self.__load_domain(sim_threshold=lower_bound)
        with open(os.path.join(CACHE_DIR,f"facts_{self.schema.name}.pkl"), 'wb') as fp:
            pickle.dump(tbl_facts, fp)
        if is_symbol:
            facts = map(parse,tbl_facts)
        return set(tbl_facts)
    """
    def get_atombase(self,multi_lvl = False,ter=False,ver = LACE):
        if ver == LACE:
            atom_base= self.get_db_atombase_(multi_lvl=multi_lvl,ter=ter)
        elif ver == LACE_P:
            atom_base = self.get_db_atombase_local(multi_lvl=multi_lvl,ter=ter)
        return atom_base
    
    def get_hard_rules(self, )-> Sequence[str]:
        hard_rules = [r for r in self.rules if r.startswith(EQ_PRED)]
        return hard_rules
    
    def get_ub_spec(self,rule_list):
        # dir = self.spec_dir
        transformed_rules = list()
        for r in rule_list:  
            if not r.startswith('%'):      
                r = r.split(IMPLY,1)
                h = r[0]
                b = r[1]
                if h.startswith(ACTIVE_PRED):
                    h = h.replace(ACTIVE_PRED,EQ_PRED)
                    transformed_rules.append(f'{h}{IMPLY}{b}')
                elif h.startswith(EQ_PRED):
                    transformed_rules.append(f'{h}{IMPLY}{b}')
            else:
                transformed_rules.append(r)
        return transformed_rules
    
    def __get_ternary_merge(self,atom:str,body)->str:
        pred = get_atom_pred(atom)
        # print(pred)
        vars = get_atom_vars(atom)
        # print(vars)
        attr = locate_body_var(vars[0],body) # here assume already only compatible attributes are merged
        #print(attr)
        # locate head variables: relation: attributes
        if attr != None:
            # print(attr[0])
            [print(r) for r in self.schema.relations]
            rel = self.schema.rel_index(attr[0])
            # print(rel)
            attr_id = rel.attrs[attr[1]].id
            atom = f'{pred}({vars[0]},{vars[1]},{attr_id})'
        else:
            atom = f'{pred}({vars[0]},{vars[1]},I)'
        #print(atom)
        return atom 
                   
    def transform_ternary(self, rule_list)-> list[str]:
        # rule_list = self.rules + self.constraints
        transformed_list = []
        # iterating rule list
        for r in rule_list:
            r = r.split(IMPLY,1)
            h = r[0]
            b = r[1]
            b = b[:-1]
            if not utils.is_empty(h):
                h = self.__get_ternary_merge(h,b)
            b_literals = trans_utils.get_body_literals(b)
            b_prime = []
            #print(b_literals)
            for b_literal in b_literals:
                b_literal = b_literal.strip()
                if b_literal.startswith(DEFAULT_NEG):
                    b_atom = b_literal.split(' ',1)[1]
                else:
                    b_atom = b_literal
                a_pred = get_atom_pred(b_atom)
                a_pred = a_pred.strip()
                if EQ_PRED == a_pred:
                    b_atom = self.__get_ternary_merge(b_atom,b)
                if b_literal.startswith(DEFAULT_NEG):
                    b_literal = f'{DEFAULT_NEG} {b_atom}'
                else:
                    b_literal = b_atom
                b_prime.append(b_literal)
            b_prime = ','.join(b_prime)   
            r = f'{h}{IMPLY}{b_prime}.'        
            transformed_list.append(r)
        return transformed_list
    
    def transform_local(self,rule_list:list[str],anonym=True) -> list[str]:
        transformed_list = []
        # iterate rule head, collecting the two sets of attributes participating eqo and eqv respectively
        for r in rule_list:
            r = r.split(IMPLY,1)
            h = r[0]
            # get rule bodies
            b = r[1]
            b = b[:-1]
            b_literals = trans_utils.get_body_literals(b)
            b_literals_cp = b_literals.copy()
            # find joins in the rule bodies
            body_joins_dict = trans_utils.get_body_joins(b_literals)
            for var, count in body_joins_dict.items():
                var_attr_pos = trans_utils.locate_body_var(var,b)
                #print(var_attr_pos)
                rel = self.schema.rel_index(var_attr_pos[0])
                #print(len(rel.attrs))
                var_attr = rel.attrs[var_attr_pos[1]-1]
                # new a fresh variable starting from the second occurance,
                fresh_vars = [f'{var}{str(i+1)}' for i in range(count)]
                fresh_vars[0] = var # keep a copy as the same variable as before, in case the variable occurs in inequality atoms
                fresh_vars_index = 0  
                # replace the var with fresh vars
                for i,bl in enumerate(b_literals_cp):
                    if re.match(REL_PAT,bl):
                        b_pred = get_atom_pred(bl)
                        b_atom_vars = trans_utils.get_atom_vars(b_literals[i])
                        if var in b_atom_vars:
                            # print(len(fresh_vars),fresh_vars_index)
                            b_atom_vars[b_atom_vars.index(var)] = f'{fresh_vars[fresh_vars_index]}'
                            fresh_vars_index+=1
                            new_bl = utils.get_atom(pred_name=b_pred,tup=tuple(b_atom_vars))
                            # print(i,new_bl)
                            b_literals[i] = new_bl
                
                # if it is a join on eqo merge variables
                if var_attr.type == Attribute.O_MERGE:
                    # otherwise, new another copy as common variable VC, then for each of the variables newed previously, append eq(X',XC) to the rule body.
                    if count>2:
                        common_var = f'{var}{str(count+1)}'
                        for fv in fresh_vars:
                            eqo = f'{EQO_PRED}({fv},{common_var})'
                            b_literals.append(eqo)
                    # if no more than 2, append directly a eqo(X,X'), 
                    elif count == 2:
                        eqo = f'{EQO_PRED}({fresh_vars[0]},{fresh_vars[1]})'
                        b_literals.append(eqo)
                # if it is a join on eqv variables
                elif var_attr.type == Attribute.V_MERGE:
                    # locates all occurances of v in the rule body (tid,position), 
                    occurances = trans_utils.locate_body_var_pos_all(var,b)
                    prj_idx = len(b_literals)
                    proj_var = f'PRJ{str(prj_idx)}'
                    for j, (t,p) in enumerate(occurances):
                        # eq_pos = f'{t}{str(j)},POS{str(j)}'
                        # equivalently
                        valv = f'{VAL_PRED}({t},{str(p)},{proj_var})'
                        b_literals.append(valv)
                        # generate new pair of variables (tid',i') and make a new atom eqv(tid,i, tid',i')
                        #eqv = f'{EQV_PRED}({t},{str(p)},{eq_pos})'
                        #b_literals.append(eqv)
                        # new a common intersection variable v
                        # generate for the tuple location a proj value atom proj(tid',i', v)
                        #projv = f'{PROJ_PRED}({eq_pos},{proj_var})'
                        #b_literals.append(projv)
                    b_literals.append(f'{DEFAULT_NEG} empty({proj_var})') # intersect cannot be empty

            # considered as singleton set for those do not participate merges, and keep same variable join unchanged
            # make changes on similarity atoms
            for i,bl in enumerate(b_literals_cp):
                if re.match(REL_PAT,bl) and trans_utils.get_atom_pred(bl) == SIM_PRED:
                    sim_vars = trans_utils.get_atom_vars(bl)
                    # print(sim_vars)
                    sim_x_pos = trans_utils.locate_body_var_pos(sim_vars[0],b)
                    sim_y_pos = trans_utils.locate_body_var_pos(sim_vars[1],b)
                    fresh_sim_x = f'{sim_vars[0]}{str(i)}'
                    fresh_sim_y = f'{sim_vars[1]}{str(i)}'
                    sim_vars[0] = fresh_sim_x
                    sim_vars[1] = fresh_sim_y
                    fresh_sim_atom = utils.get_atom(SIM_PRED,tuple(sim_vars))
                    b_literals[i] = fresh_sim_atom
                    valvx = f'{VAL_PRED}({sim_x_pos[0]},{sim_x_pos[1]},{fresh_sim_x})'
                    valvy =  f'{VAL_PRED}({sim_y_pos[0]},{sim_y_pos[1]},{fresh_sim_y})'
                    b_literals.append(valvx)
                    b_literals.append(valvy)
                    # print(b_literals)
                
                elif utils.is_empty(h) and NOT_EQUAL in bl:
                    inequated_vars = [v.strip() for v in bl.split(NOT_EQUAL)]
                    #print(inequated_vars)
                    # check attribute type
                    rel_pos = trans_utils.locate_body_var(inequated_vars[0],b)
                    ineq_attr = self.schema.rel_index(rel_pos[0]).attrs[rel_pos[1]-1]
                    #print(ineq_attr.name,ineq_attr.type)
                    if ineq_attr.type == Attribute.O_MERGE:
                        ineqo = f'{DEFAULT_NEG} {EQO_PRED}({inequated_vars[0]},{inequated_vars[1]})'
                        b_literals[i] = ineqo
                    elif ineq_attr.type == Attribute.V_MERGE:
                        eq_pos_x = trans_utils.locate_body_var_pos(inequated_vars[0],b)
                        eq_pos_y = trans_utils.locate_body_var_pos(inequated_vars[1],b)
                        #fresh_pos_x = f'{eq_pos_x[0]}X{str(i)}, POSX{str(i)}'
                        #fresh_pos_y = f'{eq_pos_y[0]}Y{str(i)}, POSY{str(i)}'
                        fresh_neq_x = f'PRJX{str(i)}'
                        fresh_neq_y = f'PRJY{str(i)}'
                        #eqvx = f'{EQV_PRED}({eq_pos_x[0]}, {str(eq_pos_x[1])}, {eq_pos_x[0]}X{str(i)}, POSX{str(i)})'
                        #eqvy = f'{EQV_PRED}({eq_pos_y[0]}, {str(eq_pos_y[1])}, {eq_pos_y[0]}Y{str(i)}, POSY{str(i)})'
                        #projx = f'{PROJ_PRED}({fresh_pos_x}, PRJX{str(i)})'
                        #projy = f'{PROJ_PRED}({fresh_pos_y}, PRJY{str(i)})'
                        eqvalvx = f'{VAL_PRED}({eq_pos_x[0]}, {str(eq_pos_x[1])},{fresh_neq_x})'
                        eqvalvy =  f'{VAL_PRED}({eq_pos_y[0]},{eq_pos_y[1]},{fresh_neq_y})'
                        new_literals = [eqvalvx, eqvalvy]
                        # new_literals = [eqvx,eqvy,projx,projy]
                        b_literals+=new_literals
                        b_literals[i] = f'PRJX{str(i)} {NOT_EQUAL} PRJY{str(i)}'
            if anonym:
                if utils.is_empty(h):
                    distinguished_vars = trans_utils.get_distinguished_vars(b_literals)
                else:
                    distinguished_vars = trans_utils.get_distinguished_vars([h]+b_literals)

                new_body_lit_cp = b_literals.copy()
                for i,bl in enumerate(new_body_lit_cp):
                    if bl.startswith(DEFAULT_NEG) :
                        # print(b_pred)
                        bl = bl.replace(DEFAULT_NEG,'').strip()
                    b_pred = trans_utils.get_atom_pred(bl)
                   
                    if not utils.is_empty(b_pred):
                        b_vars = trans_utils.get_atom_vars(bl)
                        for j,bv in enumerate(b_vars.copy()):
                            if bv in distinguished_vars:
                                b_vars[j] = ANOMY_VAR
                        anonmyed_bl = utils.get_atom(b_pred,tuple(b_vars))
                        if b_literals[i].startswith(DEFAULT_NEG): anonmyed_bl= f'{DEFAULT_NEG} {anonmyed_bl}'
                        b_literals[i] = anonmyed_bl
                            
            new_b = ','.join(b_literals)
            new_r = f'{h}{IMPLY}{new_b}{DOT}'
            transformed_list.append(new_r)
        return transformed_list  

    def transform_vlog(self,rule_list:list[str])-> list[str]:
        transformed_list = []
        def add_uq(atom:str)->str:
            is_relational = re.match(REL_PAT,atom)
            if is_relational:
                vars = trans_utils.get_atom_vars(atom)
                pred = trans_utils.get_atom_pred(atom)
            else:
                vars = re.split(DEF_OPERA_PAT, atom)                

            for j,v in enumerate(vars.copy()):
                if v[0].isupper():
                    vars[j] = f'?{v}'
            return utils.get_atom(pred,vars) if is_relational else f'{vars[0]}{vars[1]}{vars[2]}'
        
        # take input lb or ub
        for r in rule_list:
            r = r.split(IMPLY,1)
            if len(r) <2:
                transformed_list.append(r[0])
                continue
            h = r[0].strip()
            b = r[1].strip()
            h = add_uq(h)
            b = b[:-1].strip()
            b_literals = trans_utils.get_body_literals(b)
            
            pop_list = []
        # for each rule, for each variable, and universal quantifier
            for i,bl in enumerate(b_literals.copy()):
                if not bl.startswith(DEFAULT_NEG):
                    if NOT_EQUAL in bl:
                        pop_list.append(i)
                    else:
                        b_literals[i] = add_uq(bl.strip())  
                else:
                    bl = bl.replace(DEFAULT_NEG,'',1).strip()
                    b_literals[i] = f'~{add_uq(bl)}'
            for i,p in enumerate(pop_list):
                b_literals.pop(p-i)
            new_b = ','.join(b_literals)
            new_r = f'{h}{IMPLY}{new_b}{DOT}'
            transformed_list.append(new_r)
        return transformed_list
    

    def get_reduced_spec(self,sim_predname=SIM_PRED,ter=False,show=True)-> list:
        #dir = self.spec_dir
        # rule_list = get_rule_list(dir)
        if ter:
            rule_list = self.spec_construct_ter(version=program_transformer.ORIGIN,trace=False,show=show)
        else:
             rule_list = self.spec_construct(version=program_transformer.ORIGIN,trace=False,show=show)
        rule_list = [r for r in rule_list if not r.strip().startswith(IMPLY) and not r.strip().startswith('{')]
        # print('============',rule_list)
        # TODO add rules of weakened FDs for sim computation 
        sim_rules_map = dict()
        transformed_rules = list()
        commas = re.compile(r"(,)\1+", re.IGNORECASE)
        # step 2 of new sim algorithm [2023-11-18]
        to_be_sim_rules = set()
        for i, r in enumerate(rule_list):        
            # 1) find sim atoms and map sim atoms to rules
            atoms = ATOM_PAT.findall(r)
            sims = [a for a in atoms if a.startswith(SIM_PRED) or a.startswith(sim_predname)]
            # simed relation name, sim occurring position
            if len(sims)>0:
                sim_joins = list()
                sim_scripts = list() # 
                sim_threshs = SIM_THRESH_PAT.findall(r)
                #print(sim_threshs)
                if sim_threshs == None or len(sim_threshs)<1:
                    continue
                sim_threshs = {ATOM_PAT.findall(s)[0]:int(SIM_THRESH_VAL_PAT.findall(s)[0]) for s in sim_threshs}
                for s in sims:
                    svs = get_atom_vars(s)
                    if not SIM_JOIN_PAT.match(s):
                        # print(s)
                        thresh = sim_threshs[s]
                        # Start step 2 of new sim algorithm [2023-11-18]
                        # to syntheise to be sim rules for step 2 here
                        # 1) remove all the sim literals in rule body
                        r_prime = re.sub(SIM_THRESH_PAT,'',r)
                        r_prime = re.sub(commas,',',r_prime)
                        # 2) move rule head eq to rule body
                        r_prime = r_prime.split(IMPLY)
                        r_prime_h = r_prime[0]
                        if r_prime_h.startswith(ACTIVE_PRED):
                            r_prime_h = r_prime_h.replace(ACTIVE_PRED,EQ_PRED)
                        r_prime_bs = r_prime[1][:-1]
                        if r_prime_bs.endswith(','):
                            r_prime_bs = f'{r_prime_bs}{r_prime_h}.'
                        else:
                            r_prime_bs = f'{r_prime_bs},{r_prime_h}.'
                        # 3) add sim of the iteration to rule head
                        s_tup = VAR_PAT.findall(s)[0].split(',')[:-1]
                        r_prime_h = f'{SIM_PRED}({s_tup[0]},{s_tup[1]})'
                        r_prime = f'{r_prime_h}{IMPLY}{r_prime_bs}'
                        # 4) store in the set 
                        to_be_sim_rules.add(r_prime)
                        # End step 2 of new sim algorithm [2023-11-18]
                        sim_func_literals = f'{svs[2]} = {SIM_FUNC_PRED}({svs[0]},{svs[1]}), {svs[2]}>={thresh}'
                        sim_scripts.append(sim_func_literals)
                        vs_lst = list()
                        for a in atoms:
                            #print(a)
                            pred_name = REL_PAT.findall(a)[0]
                            if pred_name != SIM_PRED and pred_name != sim_predname and pred_name!=EQ_PRED and pred_name!=ACTIVE_PRED and pred_name!='c':
                                a_vars = VAR_PAT.findall(a)[0].split(',')
                                #print(a_vars, svs)
                                [vs_lst.append((pred_name,a_vars.index(sv))) for sv in svs if sv in a_vars]
                        #print(vs_lst)                   
                        vs_lst = vs_lst[0] + vs_lst[1]
                        # sim_pairs.add(vs_lst)
                        if vs_lst not in sim_rules_map:
                            sim_rules_map[vs_lst] = list()
                        sim_rules_map[vs_lst].append(i)
                    else:
                        sim_joins.append(s)  
                # 2) reshape original rules
                ## removing sim atoms where threshold <100
                r = re.sub(SIM_THRESH_PAT,'',r)
                r = re.sub(commas,',',r)
                # TODO: empty values treatments
                if len(sim_scripts)>0:
                    r = r[:-1]
                    for sl in sim_scripts:
                        r = r+f', {sl}'
                if len(sim_joins)>0:
                    ## transforming sim atoms where threshold = 100 to joins
                    for jo in sim_joins:
                        joined_vars = VAR_PAT.findall(jo)[0].split(',')
                        join = f'{joined_vars[0]}={joined_vars[1]}, not empty({joined_vars[0]})'
                        r = r.replace(jo,join) 
                    ## transforming active/2, eq/2 to up_eq/2
                r = re.sub(commas,',',r)
                r = re.sub(SPEC_CHAR_PAT,'',r)
                r += '.'  
            r = re.sub(EQ_AC_PAT,EQ_PRED,r)

            if not r.startswith('{eq'):
                transformed_rules.append(r)    

        # transformed_rules.append(EQ_PROP)
        # transformed_rules.append(EMPTY_TGRS)
        # update step 2 of new sim algorithm [2023-11-18]
        to_be_sim_rules.add(f'{TO_SIM}(X,Y) {IMPLY} {SIM_PRED}(X,Y), not {SIM_PRED}(X,Y,_), not {SIM_PRED}(Y,X,_).')
        # update step 2 of new sim algorithm [2023-11-18]
        return transformed_rules,to_be_sim_rules
    
    def get_sim_cat_sum (self,) -> tuple:
        # ('track', 5, 'track', 5): 98
        # ('dblp', 3, 'acm', 3): 90
        self.log.info("* Calculate CAT sum of sim attributes")
        sim_group = {i:k for i,k in enumerate(self.sim_pairs.keys())}
        sim_group_const_num = {i:[] for i in sim_group.keys()}    
        sim_group_const = {i:[] for i in sim_group.keys()}
        for r_name,tbl in self.schema.tbls.items():
            rel = self.schema.rel_index(r_name=r_name)
            for i,g in sim_group.items():
                if r_name in g:
                    first_index = g.index(r_name)
                    if first_index == 0:
                        attr_name = rel.attrs[g[1]].name
                        distinct_cnt = len(tbl[1][attr_name].value_counts(dropna=True))
                        sim_group_const_num[i].append(distinct_cnt)
                        # unique_const = tbl[1][attr_name].value_counts(dropna=True)
                        filtered_column = tbl[1][attr_name].dropna()
                        unique_consts = filtered_column.unique()
                        sim_group_const[i].append(unique_consts)
                        if r_name in g[2:]:
                            attr_name_2 = rel.attrs[g[3]].name
                            distinct_cnt_2 = len(tbl[1][attr_name_2].value_counts(dropna=True))
                            sim_group_const_num[i].append(distinct_cnt_2)
                            filtered_column2 = tbl[1][attr_name_2].dropna()
                            unique_consts2 = filtered_column2.unique()
                            sim_group_const[i].append(unique_consts2)
                    else:
                        attr_name_2 = rel.attrs[g[3]].name
                        distinct_cnt_2 = len(tbl[1][attr_name_2].value_counts(dropna=True))
                        sim_group_const_num[i].append(distinct_cnt_2)
                        filtered_column2 = tbl[1][attr_name_2].dropna()
                        unique_consts2 = filtered_column2.unique()
                        sim_group_const[i].append(unique_consts2)
        
        cat_sum = sum([num[0]*num[1] for _,num in sim_group_const_num.items()])
        sim_const_sum = sum([sum(num) for _,num in sim_group_const_num.items()])
        start = self.log.timer(proc_name='Computing #CatSum constants similarities', state=START)
        sim_facts = set()
        #print(len(sim_facts))
        for k, v in sim_group_const.items():
            for c1 in v[0]:
                for c2 in v[1]:
                    sim_facts.add((c1,c2,utils.sim(c1,c2)))
        self.log.timer(proc_name='Computing #CatSum constants similarities', state=END, start= start )
        naive_sim_mem_size = sys.getsizeof(sim_facts) / (1024*1024)
        print(len(sim_facts))
        self.log.info(f'Naive sim computation obtained memory size of {str(naive_sim_mem_size)} MB .')
        return cat_sum, sim_const_sum, naive_sim_mem_size
    
    
    def generate_show(self, ter=False, rec= False, rec_readoff = False, show = True)-> list[str]:
        merge_attrs = [a for a in self.schema.attrs if a.type == Attribute.MERGE]
        show_list:list[str] = [] 
        show_list.append(f'{SHOW}{DOT}')
        iter_var = ',R' if rec_readoff else ''
        eq_pred_rec = f'm{EQ_PRED}' if rec_readoff else EQ_PRED
        if show:
            for a in merge_attrs:
                v_idx = 0
                pred_name = a.rel_name
                rel = self.schema.rel_index(pred_name)
                merge_idx = rel.attr_index(a.id)
                tup_1 = get_default_tup(rel)
                for i in range(len(tup_1)):
                    tup_1[i] = f'V{v_idx}'
                    v_idx+=1
                tup_2 = get_default_tup(rel)
                for i in range(len(tup_2)):    
                    tup_2[i] = f'V{v_idx}'
                    v_idx+=1
                merge_var_X = tup_1[merge_idx]
                merge_var_Y = tup_2[merge_idx]
                tup_1_str = ','.join(tup_1)
                tup_2_str = ','.join(tup_2)
                uaid = '' if not ter else f',{a.id}'
                if rec:
                    show_directive = f'{SHOW}({pred_name},{merge_var_X},{merge_var_Y}{iter_var}):{pred_name}({tup_1_str}),{pred_name}({tup_2_str}),{eq_pred_rec}({merge_var_X},{merge_var_Y}{uaid},R),{merge_var_X}!={merge_var_Y}{DOT}'
                else:
                    show_directive = f'{SHOW}({pred_name},{merge_var_X},{merge_var_Y}):{pred_name}({tup_1_str}),{pred_name}({tup_2_str}),{EQ_PRED}({merge_var_X},{merge_var_Y}{uaid}),{merge_var_X}!={merge_var_Y}{DOT}'
                #show =  '#show(release,X,Y): release(X,RGI,AC,NA,B,S,PA,LA,SC,Q),release(Y,RGI1,AC1,NA1,B1,S1,PA1,LA1,SC1,Q1),eq(X,Y,28),not eq(RGI,RGI1,24).'
                #show_list.append(show)
                show_list.append(show_directive)
        else:
            # check self relation numbers
            is_pair = len(self.schema.relations) > 1
            # if greater than 1 e.g. dblp
            srels = list(self.schema.relations)
            rel1 = srels[0]
            rel2 = srels[1] if is_pair else srels[0]
            pred_name1 = rel1.name
            pred_name2 = rel2.name
            tup_1 = get_default_tup(rel1)
            v_idx = 0
            for i in range(len(tup_1)):
                tup_1[i] = f'V{v_idx}'
                v_idx+=1
            tup_2 = get_default_tup(rel2)
            for i in range(len(tup_2)):    
                tup_2[i] = f'V{v_idx}'
                v_idx+=1
            a = merge_attrs[0]
            merge_idx = rel1.attr_index(a.id)
            merge_var_X = tup_1[merge_idx]
            merge_var_Y = tup_2[merge_idx]
            tup_1_str = ','.join(tup_1)
            tup_2_str = ','.join(tup_2)
            uaid = '' if not ter else f',{a.id}'
            if rec:
                show_directive = f'{SHOW}({merge_var_X},{merge_var_Y}{iter_var}):{pred_name1}({tup_1_str}),{pred_name2}({tup_2_str}),{eq_pred_rec}({merge_var_X},{merge_var_Y}{uaid},R),{merge_var_X}!={merge_var_Y}{DOT}'
            else:
                show_directive = f'{SHOW}({merge_var_X},{merge_var_Y}):{pred_name1}({tup_1_str}),{pred_name2}({tup_2_str}),{EQ_PRED}({merge_var_X},{merge_var_Y}{uaid}),{merge_var_X}!={merge_var_Y}{DOT}'
            
            show_list.append(show_directive)
            # else e.g. cora
            
        #show_list.append('#show(dummy,X,Y):release_group(X,NA2,ACI,TY),release_group(Y,NA3,ACI1,TY1), dummy(X,Y).')

        return show_list
    
    # (version,ter,trace-|traced rels)
    def spec_construct(self,version,trace = False,show=True,ub_guarded=False, is_ol_sim = False)-> list[str]:
        if version == program_transformer.ORIGIN:
            rules = self.rules.copy()
            if self.schema.ver == LACE:
                if ub_guarded:
                    rules = self.transform_ub_grauded(rules)
                rules.append(ACTIVE_CHOICE)
                rules += self.constraints
            elif self.schema.ver == LACE_P:
                rules.append(ACTIVE_EQO)
                rules.append(ACTIVE_EQV)  
        #elif version == program_transformer.SIM_PRG:
            # rules = self.get_reduced_spec(rule_list=rules)
        elif version == program_transformer.UPERBOUND:
            rules = self.rules.copy()
            rules = self.get_ub_spec(rule_list=rules)
        elif version == program_transformer.LOWERBOUND:
            rules = self.get_hard_rules()
            if ub_guarded:
                rules = self.transform_ub_grauded(rules)
        
        if is_ol_sim:
            rules = [self.transform_online_sim(r) for r in rules]
        
        if self.schema.ver == LACE_P:
            rules.append(EQUIV_CELL_SET) 
        
        if trace:
            for i,l in self.annotations:
                rules[i] = f"""
                {l}
                {rules[i]}"""    
            rules.append(EQ_AXIOMS_TRACE)
            rules.append(TRACE_EQ)
        else:
            rules.append(EQ_AXIOMS)
        rules.append(EMPTY_TGRS)
        rules+=self.generate_show(show=show)
        return rules
    
    
    def spec_construct_local(self,version,trace = False,show=True)-> list[str]:
        if version == program_transformer.ORIGIN:
            rules = self.rules.copy()
            rules += self.constraints
            rules.append(ACTIVE_EQO)
            rules.append(ACTIVE_EQV)  
        #elif version == program_transformer.SIM_PRG:
            # rules = self.get_reduced_spec(rule_list=rules)
        elif version == program_transformer.UPERBOUND:
            rules = self.rules.copy()
            rules = self.get_ub_spec(rule_list=rules)
        elif version == program_transformer.LOWERBOUND:
            rules = self.get_hard_rules()
        
        if trace:
            for i,l in self.annotations:
                rules[i] = f"""
                {l}
                {rules[i]}"""    
            rules.append(EQ_AXIOMS_TRACE)
            rules.append(TRACE_EQ)
        else:
            rules.append(EQO_TRANSITIVITY)
            rules.append(EQO_SYMMETRY)
            rules.append(EQV_TRANSITIVITY)
            rules.append(EQV_SYMMETRY)
        rules.append(EMPTY_TGRS)
        # rules+=self.generate_show(show=show)
        return rules
    
    
    def spec_construct_ter(self,version,trace = False,show=True,ub_guarded = False, is_ol_sim = False)-> list[str]:
        rules = self.rules.copy()
        
        if version == program_transformer.ORIGIN:
            ter_rules = self.transform_ternary(rules)
            ter_constraints = self.transform_ternary(self.constraints)
            if ub_guarded:
                # print(ter_rules)
                ter_rules = self.transform_ub_grauded(ter_rules)
            ter_spec = ter_rules+ ter_constraints
            ter_spec.append(ACTIVE_CHOICE_TER)
            #rules += self.constraints      
        #elif version == program_transformer.SIM_PRG:
           #  ter_spec = self.get_reduced_spec(rule_list=ter_spec)
        elif version == program_transformer.UPERBOUND:
            ter_spec = self.transform_ternary(rules)
            ter_spec = self.get_ub_spec(rule_list=ter_spec)
        elif version == program_transformer.LOWERBOUND:
            rules = self.get_hard_rules()
            ter_spec = self.transform_ternary(rules)
            if ub_guarded:
                    ter_spec = self.transform_ub_grauded(ter_spec)
            

        
        if is_ol_sim:
                ter_spec = [self.transform_online_sim(r) for r in ter_spec]    
        ter_spec.append(EMPTY_TGRS)
        if trace:
            for i,l in self.annotations:
                ter_spec[i] = f"""
                                {l}
                                {ter_spec[i]}"""    
            ter_spec.append(EQ_AXIOMS_TER_TRACE)
            ter_spec.append(TRACE_EQ_TER)
        if self.schema.ver == LACE:
            ter_spec.append(EQ_AXIOMS_TER)
            ter_spec+=self.generate_show(ter=True,show=show)
        else:
            ter_spec.append(EQ_AXIOMS_VLOG_TRANS_TER)
            ter_spec.append(EQ_AXIOMS_VLOG_SYM_TER)
        return ter_spec
    
    

    
    def get_spec(self,ter=False,spec_ver=ORIGIN,trace=False,show=True,ub_guarded=False, is_ol_sim = False)->list[str]:
        if ter:
            if self.schema.ver == LACE:
                program = self.spec_construct_ter(version=spec_ver,trace=trace,show=show,ub_guarded=ub_guarded, is_ol_sim=is_ol_sim)
            elif self.schema.ver == VLOG_LB:
                program = self.spec_construct_ter(version=program_transformer.LOWERBOUND)
                program = self.transform_vlog(program)
                # if 'dblp' in self.schema.name : print('================LB===============','\n'.join(program))
            elif self.schema.ver == VLOG_UB:
                program = self.spec_construct_ter(version=program_transformer.UPERBOUND)
                program = self.transform_vlog(program)
                
        else:
            if self.schema.ver == LACE:
                program = self.spec_construct(version=spec_ver,trace=trace,show=show,ub_guarded=ub_guarded, is_ol_sim = is_ol_sim)
            elif self.schema.ver == LACE_P:
                program = self.spec_construct_local(version=spec_ver,trace=False,show=True)

        return program
    
    

    
    

    
    
    def get_merge_constraint(self,merge:list,neg:bool=False,trace=False,ter=False) -> list[str]:
        merge_constraint = []
        if trace:
            trace_option = utils.format_string(TRACE_PAIR,merge,'^')
            trace_option = f'{trace_option}'
            merge_constraint.append(trace_option)
        neg_str = DEFAULT_NEG if neg else ''
        
        if not ter:
            pm_ic = f'{IMPLY}{neg_str} {EQ_PRED}({merge[0]},{merge[1]}).'
        else:
            # TODO: adapting to ternary merge, specifying relation and attribute
            pm_ic = f'{IMPLY}{neg_str} {EQ_PRED}({merge[0]},{merge[1]},{merge[2]}).'
        merge_constraint.append(pm_ic)
        return merge_constraint
    
    def get_merge_vars(self, body:str, ter=False) -> tuple[dict,dict]:
        eq_attr_dict = {}
        neq_attr_dict = {}
        
        body_literals = trans_utils.get_body_literals(body)
        merge_literals = [l for l in body_literals if l.startswith(EQ_PRED)]
        for m in merge_literals:
            m_vars = trans_utils.get_atom_vars(atom=m)
            if ter:
                attr_id = m_vars[2]
                m_vars = m_vars[:-1]
            else:
                ### index the var in the predicate, 
                pred,index = locate_body_var(m_vars[0],body)
                ##### find the relation with predicate name 
                rel = self.schema.rel_index(r_name=pred)
                attr_id = rel.attrs[index].id

            if attr_id not in eq_attr_dict:
                    eq_attr_dict[attr_id] = m_vars 
        
        not_merge_literals = [m for m in body_literals if m.startswith(DEFAULT_NEG) and f'{EQ_PRED}(' in m]
        for m in not_merge_literals:
            m_vars = trans_utils.get_atom_vars(atom=m)
            if ter:
                attr_id = m_vars[2]
                m_vars = m_vars[:-1]
            else:
                ### index the var in the predicate, 
                pred,index = locate_body_var(m_vars[0],body)
                ##### find the relation with predicate name 
                rel = self.schema.rel_index(r_name=pred)
                attr_id = rel.attrs[index].id

            if attr_id not in neq_attr_dict:
                    neq_attr_dict[attr_id] = m_vars 
        
        return eq_attr_dict,neq_attr_dict
                      
        
        
        
   
    def get_trace_rule_desc(self,body:str,ter=False)-> tuple[str,str]:
        eq_desc_list = {}
        # get all body literals
        #print(body)
        body_literals = trans_utils.get_body_literals(body)
        #print(body_literals)
        # equality descriptions
            ## relation.attribute is
                ### the same: 1) all equalities of two different vars, 2)same var join if not eq atom
                ### merged: all eqs
        merge_literals = [l for l in body_literals if l.startswith(EQ_PRED)]
        # get relation and attributes
        # for eqs
        ## if ternary, take attribute id and its relation
        for m in merge_literals:
            m_vars = trans_utils.get_atom_vars(atom=m)
            if ter:
                attr_id = m_vars[2]
                attr = self.schema.attr_index(id=int(attr_id))
                m_desc = f'{attr.rel_name}.{attr.name} (%,%) are merged'
                m_vars_str = ','.join(m_vars[:-1]) # do not join the compatiable attribute id
              
            else:    ## else locate the vars,       
                ### index the var in the predicate, 
                pred,index = locate_body_var(m_vars[0],body)
                ##### find the relation with predicate name 
                rel = self.schema.rel_index(r_name=pred)
                ##### and use the index to get attribute
                attr_name = rel.attrs[index].name
                m_desc = f'{rel.name}.{attr_name} (%,%) are merged'
                m_vars_str = ','.join(m_vars)
                
            if m_vars_str not in eq_desc_list:
                    eq_desc_list[m_vars_str] = m_desc
                    
         # for equalities and the same var
        equality_literals = [l for l in body_literals if utils.EQUATE_PAT.match(l)]
        for el in equality_literals:
            el_vars = el.split('=')
            ### index the var in the predicate, 
            pred,index = locate_body_var(el_vars[0],body)
            # TODO: to allow equality on different relations
            ##### find the relation with predicate name 
            rel = self.schema.rel_index(r_name=pred)
            ##### and use the index to get attribute
            attr_name = rel.attrs[index].name
            el_desc = f'{rel.name}.{attr_name} (%,%) are equal'
            el_vars_str = ','.join(el_vars)
            if el_vars_str not in eq_desc_list:
                    eq_desc_list[el_vars_str] = el_desc
        # the same var
        vars = []
        #print(body_literals)
        for bl in body_literals:
            #print(bl)
            if not bl.startswith(EQ_PRED) and not bl.startswith('empty') and not bl.startswith(DEFAULT_NEG) and not '=' in bl and not '!=' in bl:
                #print(bl)
                bl_vars = trans_utils.get_atom_vars(bl)
                vars+= bl_vars
        join_vars = set()
        dup = set()
        for v in vars:
            if v in dup:
                join_vars.add(v)
            else:
                dup.add(v)
        
        for jv in join_vars:
            pred,index = locate_body_var(jv,body)
            rel = self.schema.rel_index(r_name=pred)
            attr_name = rel.attrs[index].name
            jv_desc = f'{rel.name}.{attr_name} (%) is joined'
            if jv not in eq_desc_list:
                    eq_desc_list[jv] = jv_desc
        
        # inequality descriptions (only in denials)
        ineq_desc_list = {}
        not_merge_literals = [m for m in body_literals if m.startswith(DEFAULT_NEG) and f'{EQ_PRED}(' in m]
        for m in not_merge_literals:
            m_vars = trans_utils.get_atom_vars(atom=m)
            if ter:
                attr_id = m_vars[2]
                attr = self.schema.attr_index(id=int(attr_id))
                nm_desc = f'{attr.rel_name}.{attr.name} (%,%) are not merged'
                nm_vars_str = ','.join(m_vars[:-1]) # do not join the compatiable attribute id
              
            else:    ## else locate the vars,       
                ### index the var in the predicate, 
                pred,index = locate_body_var(m_vars[0],body)
                ##### find the relation with predicate name 
                rel = self.schema.rel_index(r_name=pred)
                ##### and use the index to get attribute
                attr_name = rel.attrs[index].name
                nm_desc = f'{rel.name}.{attr_name} (%,%) are not merged'
                nm_vars_str = ','.join(m_vars)
                
            if nm_vars_str not in ineq_desc_list:
                    ineq_desc_list[nm_vars_str] = nm_desc
             
             
        neq_literals = [m for m in body_literals if len(m.split('!=')) == 2 ]    
        for nel in neq_literals:
            nel_vars = nel.split('!=')
            ### index the var in the predicate, 
            pred,index = locate_body_var(nel_vars[0],body)
            # TODO: to allow equality on different relations
            ##### find the relation with predicate name 
            rel = self.schema.rel_index(r_name=pred)
            ##### and use the index to get attribute
            attr_name = rel.attrs[index].name
            nel_desc = f'{rel.name}.{attr_name} (%,%) are not equal'
            nel_vars_str = ','.join(nel_vars)
            if nel_vars_str not in ineq_desc_list:
                    ineq_desc_list[nel_vars_str] = nel_desc
                    
        eq_desc = [v for _,v in eq_desc_list.items()]
        ineq_desc = [v for _,v in ineq_desc_list.items()]
        desc = ','.join(eq_desc)+' but ' + ','.join(ineq_desc) if len(ineq_desc) >0 else ','.join(eq_desc)
        desc_vars = ','.join([','.join(list(eq_desc_list.keys())), ','.join(list(ineq_desc_list.keys()))])
        return (desc,desc_vars)

            

    """
    def __gen_denial_atoms(self,merge:list,ter=False) -> list[str]:
        if not ter:
            denial_atom_ng = f'{MOCK_DC_PRED}(X,Y,D)'
            denial_atom = f'{MOCK_DC_PRED}({merge[0]},{merge[1]},D)'
        else:
            denial_atom_ng = f'{MOCK_DC_PRED}(X,Y,I,D)'
            denial_atom = f'{MOCK_DC_PRED}({merge[0]},{merge[1]},{merge[2]},D)'
        return [denial_atom_ng,denial_atom]   
    
    def __gen_denials(self,merge:list,ter=False) -> list[str]:
        if not ter:
            denial_atom_ng = f'{MOCK_DC_PRED}(X,Y,D)'
            denial_atom = f'{MOCK_DC_PRED}({merge[0]},{merge[1]},D)'
        else:
            denial_atom_ng = f'{MOCK_DC_PRED}(X,Y,I,D)'
            denial_atom = f'{MOCK_DC_PRED}({merge[0]},{merge[1]},{merge[2]},D)'
        return [denial_atom_ng,denial_atom]   
         
    def get_trace_denial(self,merge:list, ter:False,):
        denial_atoms = self.__gen_denial_atoms(merge,ter)
        denial_atom_ng =denial_atoms[0]
        denial_atom = denial_atoms[1]
        not_found = f'{IMPLY} {COUNT_DIRC}{{D:{denial_atom}}}<1.' 
        denial_trace = f"{ANTD_SHOW_TRACE} {{{denial_atom_ng}}}.\n"
        
        #for idx,d in enumerate(self.constraints):
            #d_head = f'{MOCK_DC_PRED}(X,Y,{str(idx)})' if not ter else f'{MOCK_DC_PRED}(X,Y,I,{str(idx)})'
           # d_anntd = f'{ANTD_TRACE_ATOM} {{{denial_heads}," DC % was triggered and falsified eq(%,%)", D, X, Y}} :-' + atom_extra +f", I={str(dc_idx)}.\n"
    """
    
    def get_merge_conditions(self,merge:list, ter=False) -> list[str]:
        var_collection = [[f'{merge[0]}','X'], [f'{merge[1]}','Y'], ['X','Y'],] if not ter else [[f'{merge[0]}','X',f'{str(merge[2])}'], [f'{merge[1]}','Y',f'{str(merge[2])}'], ['X','Y',f'{str(merge[2])}'],] 
        atoms = []
        for var_lst in var_collection:
            m_atom = utils.get_atom_(EQ_PRED,tuple(var_lst))[:-1]
            atoms.append(m_atom)
        return atoms
    
    def _get_merge_conditions(self,merge:list, ter=False) -> list[str]:
        var_collection = []
        atoms = []
        for var_lst in var_collection:
            m_atom = utils.get_atom_(EQ_PRED,tuple(var_lst))[:-1]
            atoms.append(m_atom)
        return atoms

    def get_trace_denials(self,ter=False,conditions:list=[])->list[str]:
        # iterating constraints
        trace_denials = list()
        # for each constraint i
        constraints = self.transform_ternary(self.constraints) if ter else self.constraints
        # adding head falsify(i)
        for idx,d in enumerate(constraints):
            d_body = d.split(IMPLY,1)[1][:-1]
            d_trace_desc = self.get_trace_rule_desc(d_body,ter)
            merge_vars = self.get_merge_vars(d_body,ter)
            
            denial_label = f'{ANTD_TRACE_RULE} {{"{idx} is violated because {d_trace_desc[0]}",{d_trace_desc[1]}}}.'
            eq_vars = ','.join([','.join(vars)+','+k for k,vars in merge_vars[0].items()]) if ter else ','.join([','.join(vars) for _,vars in merge_vars[0].items()])
            ineq_vars = ','.join([','.join(vars)+','+k for k,vars in merge_vars[1].items()]) if ter else ','.join([','.join(vars) for _,vars in merge_vars[1].items()])
            denial_head = f'{MOCK_DC_PRED}({idx},{EQ_PRED}s({eq_vars}),{DEFAULT_NEG}{EQ_PRED}s({ineq_vars}))'
            trace_denials.append(denial_label)
            trace_denials.append(f'{denial_head}{d}')
        # antd_denials = f'{ANTD_TRACE_ATOM} {{{MOCK_DC_PRED}(I), "% is violated.", I}} {IMPLY} {MOCK_DC_PRED}(I).'
        show_denials = f'{ANTD_SHOW_TRACE} {{{MOCK_DC_PRED}(I,E,NE)}}.'
        #atoms = self.get_merge_conditions(merge,ter)
        show_denials = self.__add_trace_condition(trace_atom=f'{MOCK_DC_PRED}(I,E,NE)',atoms=[],options=show_denials)
        # trace_denials.append(antd_denials)
        trace_denials.append(show_denials)
        return trace_denials
        # TODO: to expand the explaination of falsification by showing: 
        ## what attributes of what relations are merged/equal: eq(R[a],R'[b]) or R[a] = R'[b]
        ## but what attributes of what relations are not merged/equal: not eq(R[a],R'[b]) or R[a] != R'[b]
        
        # adding annotation tracing atom falsify(i) except for \bot <- not eq(a,b,i)
        # where a,b is the possible merge to be checked  
        
    def get_weak_denials(self,atoms = Sequence[str],ter=False)->Sequence[str]:
        # constraints = self.transform_ternary(self.constraints) if ter else self.constraints
        weak_denials = self.get_trace_denials(ter=ter,conditions=atoms)
        # [f'{MOCK_DC_PRED}({i}){d}' for i,d in enumerate(constraints)] 
        # trace_option = self.add_traced_atom(trace_atom=f'{MOCK_DC_PRED}(I)',atoms=atoms,desc='the % denial constraints is violated!')
        # weak_denials.append(trace_option) 
        return weak_denials
    
    def set_show_trace_condition_w_rel(self,):
        pass
    
    def set_show_condition_w_atom(self,trace_atom,trace_option:str,atoms:Sequence[str])->str:
        #trace_option = TRACE_EQ if not ter else TRACE_EQ_TER
        #eq_atom = f'{EQ_PRED}(X,Y)' if not ter else f'{EQ_PRED}(X,Y,I)'
        return self.__add_trace_condition(trace_atom=trace_atom,atoms=atoms,options=trace_option)
    
    def set_show_trace_condition_w_atom(self,atoms:Sequence[str],ter=False)->str:
        trace_option = TRACE_EQ if not ter else TRACE_EQ_TER
        eq_atom = f'{EQ_PRED}(X,Y)' if not ter else f'{EQ_PRED}(X,Y,I)'
        return self.__add_trace_condition(trace_atom=eq_atom,atoms=atoms,options=trace_option)
    
    def add_traced_atom(self,trace_atom:str,atoms:Sequence[str],desc='')->str:
        vars = ','.join(trans_utils.get_atom_vars(trace_atom))
        trace_head = f'{{{trace_atom},"{desc}",{vars}}}'
        show_head = f'{{{trace_atom}}}'
        atom_option = f'{ANTD_TRACE_ATOM}{trace_head}.'
        show_option =f'{ANTD_SHOW_TRACE}{show_head}.'
        atom_option = self.__add_trace_condition(trace_atom=trace_atom,atoms=atoms,options=atom_option)
        show_option = self.__add_trace_condition(trace_atom=trace_atom,atoms=atoms,options=show_option)
        return atom_option+'\n'+show_option
    
    def __add_trace_condition(self,trace_atom:str,atoms:Sequence[str],options:str)->str:
        #print(atoms)
        if len(atoms)>0:
            conditions = ','.join(atoms)
            options = options.replace('.','',len(options)-1)
            #print(options)
            options = f'{options}{IMPLY}{conditions},{trace_atom}.'
            return options
        return f'{options}{IMPLY}{trace_atom}.'
    
    
    def get_rec_spec(self,ter=False,show=False,max=False)->Sequence[str]:
        # get ub
        rules = self.get_spec(ter=ter)
        # 1 check whether its undefined or violated
        hard_rules = [r for r in rules if r.startswith(EQ_PRED)]
        soft_rules = [r for r in rules if r.startswith(ACTIVE_PRED)]
        harden_soft_rules = self.get_ub_spec(rule_list=soft_rules)
        # atom_base = self.get_atombase(ter=ter)
        shows = '\n'.join(self.generate_show(ter=ter,rec=True,show=show))
        base = [EMPTY_TGRS + shows]
        # get base program
        # iterate rules
        rules = hard_rules + harden_soft_rules
        transformed_rules = []
        # for each r in Pi
        iter_var = 'ITER'
        for r in rules:
            r = r.split(f'{IMPLY}',1)
            #print(r)
            r_head = r[0]
            #print(r_head)
            r_h_pred, r_h_tup = utils.get_atom_tup(r_head)
            # change rule head from eq(x,y) to eq(x,y,i)
            r_h_tup = list(r_h_tup)
            r_h_tup.append('i')
            r_head = utils.get_atom_(r_h_pred,r_h_tup)[:-1]
            bodies = trans_utils.get_body_literals(r[1]) 
            #print(bodies)
            for i,b in enumerate(bodies[:]):
                # print(bodies[i])
                if b.startswith(EQ_PRED): # no negative eq in rule bod
                    b_pred, b_tup = utils.get_atom_tup(b)
                    curr_iter_var = f'{iter_var}{str(i)}'
                    b_tup = list(b_tup)
                    b_tup.append(f'{iter_var}{str(i)}')
                    # bodies[i] = utils.get_atom_(b_pred,)
                    cond = f'{curr_iter_var}{LESS}i' 
                    # for each eq in rule bodies
                    # replace eq with eq(x,y,it_idx), it_idx != i,
                    #print(bodies[i])
                    bodies[i] = utils.get_atom_(b_pred,b_tup)[:-1] + ',' + cond
            if max:
                # trans_utils.get
                r_h_tup_b = r_h_tup[:-1]
                # r_h_tup_b.append(f'{ANOMY_VAR}')
                r_h_tup_b_str = ','.join(r_h_tup_b)
                bodies.append(f'{EQ_PRED}({r_h_tup_b_str})')
                #print(bodies)
            bodies = ','.join(bodies)
            rule = f'{r_head} {IMPLY} {bodies}' 
            rule = rule if rule.endswith(DOT) else f'{r_head} {IMPLY} {bodies}{DOT}' 
            transformed_rules.append(rule)
            
        rec_eq_axioms = _EQ_AXIOMS_REC_TER if ter else _EQ_AXIOMS_REC
        transformed_rules.append(rec_eq_axioms)
        if max:
            maxsol_rec = MEQ_REC_TER if ter else MEQ_REC
            transformed_rules.append(maxsol_rec)
        transformed_rules.insert(0,f'{PROGRAM} spec(i).')
        #[print(r) for r in transformed_rules]
        return base + transformed_rules
    
    
    def transform_ub_grauded(self,rules: list[str])-> list[str]:
        transformed = []
        for r in rules:
            # get head variables
            h, b_literals = trans_utils.get_rule_parts(r)
            head_vars = trans_utils.get_atom_vars(h)
            # create up_eq atom with the variables
            up_eq = utils.get_atom(UP_EQ,tup=head_vars)
            # append in the rule body
            b_literals.append(up_eq)
            transformed.append(trans_utils.make_normal_rule(h,b_literals))
        return transformed
    
    def transform_online_sim(self,r:str,sim_pred_name = SIM_PRED)-> str:
        h, b_literals = trans_utils.get_rule_parts(r)
        pop_list =[]
        for i,bl in enumerate(b_literals.copy()):
            # TODO: for testing reson here assume sim atoms only occur positively, which is not the case as in dblp we do allow negating sim atoms in constraint bodies
            if bl.startswith(sim_pred_name):
                # remake a online sim atom
                sim_vars = trans_utils.get_atom_vars(bl)
                # moving thresholds
                threshold = b_literals[i+1].split('>=')[1].strip() 

                ol_sim_atom = utils.get_atom(SIM_FUNC_PRED,tuple(sim_vars[:-1]))
                ol_sim_literal = f'{ol_sim_atom}>={threshold}'

                b_literals.append(ol_sim_literal)
                pop_list.append(i)
                pop_list.append(i+1)

        for i,p in enumerate(pop_list):
            b_literals.pop(p-i)
        return trans_utils.make_normal_rule(h,b_literals)
    

    def get_datalog_sim(self,ter=False)->list[str]:
        if ter:
           rules = self.rules.copy()
           ub = self.get_ub_spec(rules)
           ub = self.transform_ternary(ub)
           ub1 = ub.copy()
           ub2 = ub.copy()
           up_eq_rules = []
           for i,r in enumerate(ub1):
               h,b_literals = trans_utils.get_rule_parts(r)
               pop_list = []
               for j,b in enumerate(b_literals):
                if b.startswith(SIM_PRED):
                    pop_list.append(j)
                    pop_list.append(j+1)
               for j,p in enumerate(pop_list):
                   b_literals.pop(p-j)
                
               up_r = trans_utils.make_normal_rule(h,b_literals)  
               up_eq_rules.append(up_r)
            
           to_be_simed_rules = []
           for i,r in enumerate(ub2):
               h,b_literals = trans_utils.get_rule_parts(r)
               sim_index = []
               sim_bodies = []
               for j,b in enumerate(b_literals):
                   if b.startswith(SIM_PRED):
                       sim_index.append(j)
                       sim_index.append(j+1)
                       sim_bodies.append(b)
               # remove sim in body literals
               for j,p in enumerate(sim_index):
                   b_literals.pop(p-j)
               b_literals.append(h)
               #print(sim_bodies)
               for s in sim_bodies:
                   s_vars = trans_utils.get_atom_vars(s)
                   sim_head = utils.get_atom(SIM_PRED,tuple(s_vars[:-1]))
                   to_be_simed_rules.append(trans_utils.make_normal_rule(sim_head, b_literals))
           
           return self.transform_vlog(up_eq_rules), self.transform_vlog(to_be_simed_rules)
       