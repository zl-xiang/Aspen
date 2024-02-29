
import re
import utils

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


SIM_PRED = "sim"
SIM_FUNC_PRED = '@' + SIM_PRED
EQ_PRED = "eq"
ACTIVE_PRED = "active"
ANNO_SYM = "%!"


DF_EMPTY = 'nan'

CACHE_DIR = './cache' 

ANOMY_VAR = "_"


ACTIVE_DOM = 'adom'
UP_EQ = 'up_eq'

SIM_TGRS = """sim(X,Y,S) :- sim(Y,X,S). 
sim(X,X,100) :- sim(X,X,_)."""

ADOM_TGRS = """eq(X,X) :- adom(X)."""

TO_BE_SIM ='sim_attr'
TO_SIM ='to_sim'

HEAD = 'head'
BODIES = 'BODIES'
IMPLY = ':-'

EMPTY_TGRS = """empty(nan). empty("nan")."""


def get_rule_list(dir:str) -> list:
    rule_list:list[str] = []
    constraint_list:list[str] = []
    label_list:list[tuple[int,str]] = []
    show_list:list[str] = []
    rule = ""
    comment_pat = re.compile(r"\s*%.*\s*", re.IGNORECASE)
    with open(dir,'r', encoding='utf-8',
                 errors='ignore') as f:
        for line in  f.readlines():
            line = line.strip('\n')
            if line not in ['','\n','\t','\r'] and not comment_pat.match(line): # TODO: update to symbol API
                line = comment_pat.sub('', line)
                # print(line)
                #if (line.startswith("eq(") or line.startswith("active(")) or (rule.startswith("eq(") or rule.startswith("active(")): 
                line = line.replace('\r', '').replace('\n', '').replace('\t','')
                if not line.startswith("#"):
                    line = re.sub(r"(?<!not)\s+",'',line)
                rule += line
                if line.endswith("."):
                    if not rule.startswith(IMPLY) and not rule.startswith('#') and IMPLY in rule:
                        rule_list.append(rule)
                    elif not rule.startswith('#') and rule.startswith(IMPLY):
                        constraint_list.append(rule)
                    elif rule.startswith('#show'):
                        show_list.append(rule)
                    rule = ""
            elif comment_pat.match(line) and line.strip().startswith('%!'): # labels are only single-lines
                # if its rule label, check how many rules collected already
                # assign the label with the index of the rule
                label_list.append((len(rule_list),line.strip()))
        return rule_list, constraint_list, label_list, show_list

def get_sim_pairs(dir:str, sim_predname=SIM_PRED)-> dict:
    spec =  get_rule_list(dir)
    rules = spec[0]
    constraints = spec[1]
    rule_list = rules + constraints
    sim_pairs = dict()
    for i, r in enumerate(rule_list): 
        atoms = ATOM_PAT.findall(r)
        sims = [a for a in atoms if a.startswith(SIM_PRED) or a.startswith(sim_predname)]
        # simed relation name, sim occurring position
        if len(sims)>0:
            sim_threshs = SIM_THRESH_PAT.findall(r)
            if sim_threshs == None or len(sim_threshs)<1:
                continue
            sim_threshs = {ATOM_PAT.findall(s)[0]:int(SIM_THRESH_VAL_PAT.findall(s)[0]) for s in sim_threshs}
            for s in sims:
                svs = VAR_PAT.findall(s)[0].split(',')
                thresh = sim_threshs[s]
                vs_lst = list()
                for a in atoms:
                    pred_name = REL_PAT.findall(a)[0]
                    if pred_name != SIM_PRED and pred_name != sim_predname and pred_name!=EQ_PRED and pred_name!=ACTIVE_PRED and pred_name!='c':
                        a_vars = VAR_PAT.findall(a)[0].split(',')
                        [vs_lst.append((pred_name,a_vars.index(sv))) for sv in svs if sv in a_vars]                
                vs_lst = vs_lst[0] + vs_lst[1]
                if vs_lst not in sim_pairs:
                    sim_pairs[vs_lst] = thresh
                elif sim_pairs[vs_lst] > int(thresh):
                    sim_pairs[vs_lst] = int(thresh)
    return sim_pairs


def get_atom_vars(atom:str)->list[str]:
    return VAR_PAT.findall(atom)[0].split(',')

def get_atom_pred(atom:str)->str:
    _atom = REL_PAT.findall(atom)
    if len(_atom)>0:
        return _atom[0]
    else: 
        return ''

def get_atoms(cnj:str,)->list[str]:
    return ATOM_PAT.findall(cnj)

def locate_body_var(vname:str,body:str)-> tuple[str,int]:
    atoms = get_atoms(body)
    for a in atoms:
        a_pred = REL_PAT.findall(a)[0]
        a_vars:list = get_atom_vars(a)
        if vname in a_vars and a_pred != EQ_PRED:
            return a_pred,a_vars.index(vname)

def get_merge_attributes(dir:str):
    rule_list:list[str] = get_rule_list(dir)[0]
    merge_attrs = dict()
    # check rule head variables
    for r in rule_list:
        r = r.split(IMPLY,1)
        h = r[0]
        b = r[1]
        if h.startswith(EQ_PRED) or h.startswith(ACTIVE_PRED):
            merge_vars = get_atom_vars(h)
            #print(merge_vars)
            # iterate rule body to locate occurring relation
            x = locate_body_var(merge_vars[0],b)
            #print(x)
            y = locate_body_var(merge_vars[1],b)
            #print(y)
            if x != None and y != None:
                if x[0] not in merge_attrs:
                    merge_attrs[x[0]] = set()
                merge_attrs[x[0]].add(x[1])
                if y[0] not in merge_attrs:
                    merge_attrs[y[0]] = set()
                merge_attrs[y[0]].add(y[1])
    # merge_attrs.pop(EQ_PRED)
    return merge_attrs     
    # store in the dictionary as {R:{p1,...,pi},...,R':{pj,...,pm}}
    # 

def remove_eqs(body_str:str) -> str:
    p_1 = r'(\s*)?eq\(([^,]+),([^,]+)\)(,\s*)?'
    p_2 = r'(\s*)?(not)(\s*)eq\(([^,]+),([^,]+)\)(,\s*)?'
    removed = re.sub(p_1,'',body_str)
    if removed.endswith(',.'):
        removed = removed[:-2]+'.'
    return removed

def get_body_literals(body_str:str) -> list[str]:
    body_literals = re.split(r',\s*(?![^()]*\))', body_str)
    literals = [literal.strip() for literal in body_literals]
    literals[-1] = literals[-1][:-1] if literals[-1].endswith('.') else literals[-1]
    return literals



if __name__ == "__main__":
    # Example usage:
    rule_example_ungrounded = "not body_literal1(X,Y), body_literal2(Y,Y), X!=Y."

  
  #print('eq(X,Y),release(A,B,C).'.replace(EQ_AC_AT_PAT,''))