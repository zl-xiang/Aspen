import py_stringmatching as sm

from clingo.symbol import  Symbol, SymbolType, Number, String
import utils
from _xclingo2 import XclingoContext

from dataloader import  STRING, NUM


SYM_TYPING = {STRING: String,
               NUM: Number}
DF_EMPTY = 'nan'

EMPTY_TGRS = """empty(nan). empty("nan")."""

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
EQ_PROP = """
eq(X,X):-adom(X).
eq(X,Y):-eq(X,Z),eq(Z,Y).
eq(X,Y):-eq(Y,X).
"""

def type_checking(X:Symbol):
    if X.type == SymbolType.String: return (X.string,String)
    elif X.type == SymbolType.Number: return (X.number,Number)
    
"""[ER context, defines the mappings and evaluations to the external raw data]
"""
class ERContext(XclingoContext):
    def __init__(self,tokeniser=None, ter = False) -> None:
        if tokeniser == None:
            self.tokeniser = sm.AlphabeticTokenizer(return_set=True)
        else: self.tokeniser = tokeniser
        self.rep_set = {}
        # UID for relational tuples
        self.tuple_idx = 0
        # set of dictionaries
        # for every given pair of (X,Y) map to X
        self.rep_set_ = {}
        # third version of hard rules
        self._rep_set = {}
        #self.schema = schema
        self.ter = ter
        self.active_dom = set()
        self.sim_counter = 0
        # self.spec_dir = spec_dir
        # stores the sim pairs compared
        self.sim_pairs = dict()

        
    def t_idx(self,x:Number) -> Symbol:
        self.tuple_idx += 1
        return x.number + self.tuple_idx
    
    def lesser(self,x:Number,y:Number) -> Symbol:
        #print(x.number,y.number)
        if x.number == 0 or y.number ==0:
            return Number(1)
        return x if x.number<y.number else y
        
    def single_search_rep(self,_x:str)-> str:
        # x itself is not rep 
        # special cases
        presented_keys_x = list()
                
        # checking every set in dict
        for k,v in self.rep_set.items():
            # if x is found
            if _x == k or _x in v:
                # leave a mark
                presented_keys_x.append(k)
                    
        len_pkx = len(presented_keys_x) 
        if len_pkx>0 :
            cluster = self.rep_set[presented_keys_x[0]]
            if len_pkx>1:
                for i in range(1,len_pkx):
                    # take union of the cluster and update rep_dict_set 
                    cluster = cluster.union(self.rep_set[presented_keys_x[i]])
                    self.rep_set.pop(presented_keys_x[i])
            self.rep_set[presented_keys_x[0]] = cluster
            return presented_keys_x[0]           
        else:  
            newset = set()
            newset.add(_x)
            self.rep_set[_x] = newset
            return _x
        
    def add_single(self,_x:str)-> str:
        # x itself is not rep 
        # special cases
        presented_keys_x = list()
                
        # checking every set in dict
        for k,v in self.rep_set.items():
            # if x is found
            if _x == k or _x in v:
                # leave a mark
                presented_keys_x.append(k)
                    
        len_pkx = len(presented_keys_x) 
        if len_pkx>0 :
            cluster = self.rep_set[presented_keys_x[0]]
            if len_pkx>1:
                for i in range(1,len_pkx):
                    # take union of the cluster and update rep_dict_set 
                    cluster = cluster.union(self.rep_set[presented_keys_x[i]])
                    self.rep_set.pop(presented_keys_x[i])
            self.rep_set[presented_keys_x[0]] = cluster
            return presented_keys_x[0]           
        else:  
            newset = set()
            newset.add(_x)
            self.rep_set[_x] = newset
            return _x
        
    def search_rep(self,_x:str,_y:str)-> str:
        # x itself is not rep 
        # special cases
        presented_keys_x = list()
        presented_keys_y = list()
            
        # checking every set in dict
        for k,v in self.rep_set.items():
            # if x is found
            if _x == k or _x in v:
                # leave a mark
                presented_keys_x.append(k)
                
            if _y == k or _y in v:
                presented_keys_y.append(k)
        
        len_pkx = len(presented_keys_x)
        len_pky = len(presented_keys_y)
        
        if len_pkx>0 and len_pky>0:
            cluster = self.rep_set[presented_keys_x[0]]
            if len_pkx>1:
                for i in range(1,len_pkx):
                    # take union of the cluster and update rep_dict_set 
                    cluster = cluster.union(self.rep_set[presented_keys_x[i]])
                    self.rep_set.pop(presented_keys_x[i])
            # if y is included in rep_x^i, update directly
            # if not, find rep_y^j
            if _y not in cluster:
                if len_pky>1:
                    for j in range(1,len_pky):
                        # take union of the cluster and update rep_dict_set 
                        cluster = cluster.union(self.rep_set[presented_keys_y[j]])
                        self.rep_set.pop(presented_keys_y[j])
                else: cluster = cluster.union(self.rep_set[presented_keys_y[0]])
            self.rep_set[presented_keys_x[0]] = cluster  
            return presented_keys_x[0]
                
        elif len_pkx>0 and len_pky<1:
            cluster = self.rep_set[presented_keys_x[0]]
            if len_pkx >1:
                for i in range(1,len_pkx):
                    cluster = cluster.union(self.rep_set[presented_keys_x[i]])
                    self.rep_set.pop(presented_keys_x[i])
            if _y not in cluster:
                cluster.add(_y)
            self.rep_set[presented_keys_x[0]] = cluster  
            return presented_keys_x[0]
            
        elif len_pky>0 and len_pkx<1:
            cluster = self.rep_set[presented_keys_y[0]]
            if len_pky>1:
                for i in range(1,len_pky):
                    cluster = cluster.union(self.rep_set[presented_keys_y[i]])
                    self.rep_set.pop(presented_keys_y[i])
            if _x not in cluster:
                cluster.add(_x)
            self.rep_set[presented_keys_y[0]] = cluster
            return presented_keys_y[0]
        
        else:  
            newset = set()
            newset.add(_x)
            newset.add(_y)
            self.rep_set[_x] = newset
            return _x

    def add_pair(self,_x:str,_y:str)-> str:
        # x itself is not rep 
        # special cases
        presented_keys_x = list()
        presented_keys_y = list()
            
        # checking every set in dict
        for k,v in self.rep_set.items():
            # if x is found
            if _x == k or _x in v:
                # leave a mark
                presented_keys_x.append(k)
                
            if _y == k or _y in v:
                presented_keys_y.append(k)
        #print(f"====presented {_x}:",presented_keys_x)
        #print(f"====presented {_y}:",presented_keys_y)
        len_pkx = len(presented_keys_x)
        len_pky = len(presented_keys_y)
        
        if len_pkx>0 and len_pky>0:
            cluster = self.rep_set[presented_keys_x[0]]
            if len_pkx>1:
                for i in range(1,len_pkx):
                    # take union of the cluster and update rep_dict_set 
                    cluster = cluster.union(self.rep_set[presented_keys_x[i]])
                    self.rep_set.pop(presented_keys_x[i])
            #else: cluster = cluster.union(self.rep_set[presented_keys_x[0]])
            # if y is included in rep_x^i, update directly
            # if not, find rep_y^j
            if _y not in cluster:
                if len_pky>1:
                    for j in range(0,len_pky):
                        # take union of the cluster and update rep_dict_set 
                        cluster = cluster.union(self.rep_set[presented_keys_y[j]])
                        self.rep_set.pop(presented_keys_y[j])
                else: 
                    cluster = cluster.union(self.rep_set[presented_keys_y[0]])
                    self.rep_set.pop(presented_keys_y[0])
            self.rep_set[presented_keys_x[0]] = cluster  
                
        elif len_pkx>0 and len_pky<1:
            cluster = self.rep_set[presented_keys_x[0]]
            if len_pkx >1:
                for i in range(1,len_pkx):
                    cluster = cluster.union(self.rep_set[presented_keys_x[i]])
                    self.rep_set.pop(presented_keys_x[i])
            if _y not in cluster:
                cluster.add(_y)
            self.rep_set[presented_keys_x[0]] = cluster  
            
        elif len_pky>0 and len_pkx<1:
            cluster = self.rep_set[presented_keys_y[0]]
            if len_pky>1:
                for i in range(1,len_pky):
                    cluster = cluster.union(self.rep_set[presented_keys_y[i]])
                    self.rep_set.pop(presented_keys_y[i])
            if _x not in cluster:
                cluster.add(_x)
            self.rep_set[presented_keys_y[0]] = cluster
        
        else:  
            newset = set()
            newset.add(_x)
            newset.add(_y)
            self.rep_set[_x] = newset

    
    def rep(self,X:Symbol,Y:Symbol)-> Symbol:
        
        x = type_checking(X)
        y = type_checking(Y)
        # print(self.rep_set)
        # initially empty
        if len(self.rep_set.keys())<1:
            newset = set()
            newset.add(x[0])
            newset.add(y[0])
            self.rep_set[x[0]] = newset
            return X 
        # not empty, search clusters for x and y
        else:     
            if x[0]==y[0]:
               return x[1](self.single_search_rep(x[0]))
            else:       
               return x[1](self.search_rep(x[0],y[0]))
           
    def cluster(self,truth_pair)-> dict:
        # print(self.rep_set)
        # initially empty
        if len(self.rep_set.keys())<1:
            newset = set()
            newset.add(truth_pair[0])
            newset.add(truth_pair[1])
            self.rep_set[truth_pair[0]] = newset
        # not empty, search clusters for x and y
        else:     
            if truth_pair[0]==truth_pair[1]:
                self.add_single(truth_pair[0])
            else:       
                self.add_pair(truth_pair[0],truth_pair[1])
           
    def rep_ (self,X:Symbol,Y:Symbol) -> Symbol:
        x = type_checking(X)
        y = type_checking(Y)
        if x[0] != y[0]:
            if len(self.rep_set_) > 0:
                key_1 = (x[0],y[0])
                key_2 = (y[0],x[0])
                for k,v in self.rep_set_.items():
                    if k == key_1 or k == key_2:
                        return x[1](v) 
            self.rep_set_[(x[0],y[0])]=x[0]
            self.rep_set_[(y[0],x[0])]=x[0]
        return X 
    
    def repp (self,X:Symbol,Y:Symbol) -> Symbol:
        x = type_checking(X.arguments[0])
        y = type_checking(Y.arguments[0])
        if x[0] != y[0]:
            x_attr = X.name
            y_attr = Y.name
            if len(self._rep_set) > 0:
                if y_attr == x_attr:
                    if x_attr in self._rep_set.keys():
                        attr_dom = self._rep_set[x_attr]
                        key_1 = (x[0],y[0])
                        key_2 = (y[0],x[0])
                        for k,v in attr_dom.items():
                            if k == key_1 or k == key_2:
                                return x[1](v)
                #TODO: distinctive attributes (for 2 schema fact-level case)
                else: pass
            else:
                self._rep_set[x_attr] = {}
                self._rep_set[x_attr][(x[0],y[0])]=x[0]
                self._rep_set[x_attr][(y[0],x[0])]=x[0]           
        return x[1](x[0]) 
    
    
    """Xclingo context class."""
    def label(self, text, tup):
        """Given the text of a label and a tuple of symbols, handles the variable instantiation
        and returns the processed text label."""
        if text.type == SymbolType.String:
            text = text.string
        else:
            text = str(text).strip('"')
        for val in tup.arguments:
            text = text.replace("%", val.string if val.type == SymbolType.String else str(val), 1)
        return [String(text)]

              
    def sim(self,X:Symbol,Y:Symbol) -> Number:
        assert type(X) == type(Y)
        self.sim_counter +=1
        # print(self.sim_counter )
        #if not self.ter:
         #   return self.sim_annotated_str(X,Y)
        #else:
        return self.sim_sym(X,Y)
        
    def sim_col(self,X:Symbol,C1:Symbol,Y:Symbol,C2:Symbol) -> Number:
        assert type(X) == type(Y)
        # print("adfsasdfsfsafsd")
        self.sim_counter +=1
        # print(X.number,C1.string,Y.number,C2.string)
        if (X.number,Y.number) in self.sim_pairs:
            return Number(self.sim_pairs[(X.number,Y.number)])
        elif (Y.number,X.number) in self.sim_pairs:
            return Number(self.sim_pairs[(Y.number,X.number)])
        else:
            score = 0
            # TODO:similarity measure with typing
            #x = type_checking(X)
            #y = type_checking(Y)
            #TODO: might need change to customisable one
            short_indicator = 200
            # string simiarities
            # if isinstance(X,String) and isinstance(Y,String):
            if  C1.type == SymbolType.String and C2.type == SymbolType.String:
                if C1.string == DF_EMPTY or C2.string == DF_EMPTY:
                    return Number(int(score)) 
                x_len = len(C1.string)
                y_len = len(C2.string)
                # treating short text-valued entries as sequences of characters
                # measure the editing distance of sequences
                if x_len + y_len <= short_indicator*2:
                    measure = sm.JaroWinkler()
                    score = measure.get_sim_score(C1.string,C2.string)
                # TODO: numeric ids?
                else:
                # treating long text-valued entries as sets of tokens
                # measure the TF-IDF cosine score between token set
                    x_set = self.tokeniser.tokenize(C1.string)
                    y_set = self.tokeniser.tokenize(C2.string)
                    measure = sm.TfIdf()
                    score = measure.get_sim_score(x_set,y_set) 

            # numeric similarities
            # measure the Levenshtein distance for numeric values
            # two numbers is based on the minimum number of operations required to transform one into the other.
            # elif isinstance(X,Number) and isinstance(Y,Number):
            elif  C1.type == SymbolType.Number and C2.type == SymbolType.Number:
                measure = sm.Levenshtein()
                score = measure.get_sim_score(str(C1.number),str(C2.number))
            score = int(score*100)
            self.sim_pairs[(X.number,Y.number)] = score
            return Number(score)
    
    def sim_annotated_str(self,X:Symbol,Y:Symbol): 
        X = X.string.split(':',1)
        X_const= X[1]
        X_aid = X[0]
        Y = Y.string.split(':',1)
        Y_const= Y[1]
        Y_aid = Y[0]
        
        X_const = utils.is_integer(X_const)
        Y_const = utils.is_integer(Y_const)
        if (X_const,Y_const) in self.sim_pairs:
            return Number(self.sim_pairs[(X_const,Y_const)])
        elif ( Y_const,X_const) in self.sim_pairs:
            return  Number(self.sim_pairs[(Y_const,X_const)])
        else:
            score = 0
            short_indicator = 200
            # string simiarities
            # if isinstance(X,String) and isinstance(Y,String):
            if  isinstance(X_const,str) and isinstance(Y_const,str):
                x = X_const.lower().strip()
                y = Y_const.lower().strip()

                if x == DF_EMPTY or y == DF_EMPTY or x== 'ーーー' or y == 'ーーー':
                    return Number(int(score)) 
                x_len = len(x)
                y_len = len(y)
                # treating short text-valued entries as sequences of characters
                # measure the editing distance of sequences
                if x_len + y_len <= short_indicator*2:
                    measure = sm.JaroWinkler()
                    score = measure.get_sim_score(x,y)
                    #score = semantic_sim(x,y)
                # TODO: numeric ids?
                else:
                # treating long text-valued entries as sets of tokens
                # measure the TF-IDF cosine score between token set
                    x_set = self.tokeniser.tokenize(x)
                    y_set = self.tokeniser.tokenize(y)
                    measure = sm.TfIdf()
                    score = measure.get_sim_score(x_set,y_set) 
                    #score = semantic_sim(x,y)

            # numeric similarities
            # measure the Levenshtein distance for numeric values
            # two numbers is based on the minimum number of operations required to transform one into the other.
            # elif isinstance(X,Number) and isinstance(Y,Number):
            elif   isinstance(X_const,int) and isinstance(Y_const,int):
                measure = sm.Levenshtein()
                score = measure.get_sim_score(str(X_const),str(Y_const))
            score = int(score*100)
            self.sim_pairs[(f'{X_aid}:{X_const}',f'{Y_aid}:{Y_const}')] = score
            return Number(score)
    
    def sim_sym(self,X:Symbol,Y:Symbol):
        score = 0
        if X.type == SymbolType.Function: 
            if X.name == 'nan':
                return Number(int(score)) 
        
        if Y.type == SymbolType.Function: 
            if Y.name == 'nan':
                return Number(int(score)) 
        if (X.string,Y.string) in self.sim_pairs:
            return Number(self.sim_pairs[(X.string,Y.string)])
        elif (Y.string,X.string) in self.sim_pairs:
            return  Number(self.sim_pairs[(Y.string,X.string)])
        else:
            #score = 0
            short_indicator = 200
            # string simiarities
            # if isinstance(X,String) and isinstance(Y,String):

            if  X.type == SymbolType.String and Y.type == SymbolType.String:
                x:str = X.string
                x = x.lower().strip()
                y:str = Y.string
                y = y.lower().strip()

                if x == DF_EMPTY or y == DF_EMPTY or x== 'ーーー' or y == 'ーーー':
                    return Number(int(score)) 
                x_len = len(x)
                y_len = len(y)
                # treating short text-valued entries as sequences of characters
                # measure the editing distance of sequences
                if x_len + y_len <= short_indicator*2:
                    measure = sm.JaroWinkler()
                    #measure = sm.Jaccard()
                    #qg3_tok_set = sm.QgramTokenizer(qval=3,return_set=True)
                    #score = measure.get_sim_score(qg3_tok_set.tokenize(x),qg3_tok_set.tokenize(y))
                    #score = semantic_sim(x,y)
                    score = measure.get_sim_score(x,y)
                # TODO: numeric ids?
                else:
                # treating long text-valued entries as sets of tokens
                # measure the TF-IDF cosine score between token set
                    x_set = self.tokeniser.tokenize(x)
                    y_set = self.tokeniser.tokenize(y)
                    measure = sm.TfIdf()
                    score = measure.get_sim_score(x_set,y_set) 
                    #score = semantic_sim(x,y)

            # numeric similarities
            # measure the Levenshtein distance for numeric values
            # two numbers is based on the minimum number of operations required to transform one into the other.
            # elif isinstance(X,Number) and isinstance(Y,Number):
            elif  X.type == SymbolType.Number and Y.type == SymbolType.Number:
                measure = sm.Levenshtein()
                score = measure.get_sim_score(str(X.number),str(Y.number))
            score = int(score*100)
            self.sim_pairs[(X.string,Y.string)] = score
            return Number(score)

    def sim__(self,X,Y) -> Number:
        score = 0
        short_indicator = 200
        # string simiarities
        # if isinstance(X,String) and isinstance(Y,String):
        if  isinstance(X,str) and isinstance(Y,str):
            if X == DF_EMPTY or Y == DF_EMPTY:
                return Number(int(score)) 
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
                x_set = self.tokeniser.tokenize(X)
                y_set = self.tokeniser.tokenize(Y)
                measure = sm.TfIdf()
                score = measure.get_sim_score(x_set,y_set) 

        # numeric similarities
        # measure the Levenshtein distance for numeric values
        # two numbers is based on the minimum number of operations required to transform one into the other.
        # elif isinstance(X,Number) and isinstance(Y,Number):
        elif isinstance(X,int) and isinstance(Y,int):
            measure = sm.Levenshtein()
            score = measure.get_sim_score(str(X),str(Y))
        # print(int(score*100))
        return Number(int(score*100))


if __name__ == "__main__":
    #imdb_test()
    pass
    #print(get_reduced_spec('./experiment/5-uni/music/music.lp')[-1])
    #dl = Dataloader(name = 'cora_tsv')
    #tbl = dl.load_data()
    #schema = Schema('1','cora-non-split',{'pub':(5,tbl)})
    #for a in schema.attrs:
     #   if a.name == 'authors' or a.name == 'editor':
      #      a.is_list = True
    #schema.entity_split(schema.rel_index('cora').id,{'pub':[5,6,9,10,12,13,14],'author':[1],'editor':[4],'venue':[0,2,3,7,8,11,15,16]})
    #context = ERContext(schema=schema)
    
    ##facts = context.generating_facts_(lower_bound=50)
    #pubs = [f for f in facts if f.startswith('pub')]
    #print(len(pubs))
    #s = context.generating_facts_(lower_bound=50)
    #print([ a for a in s if a.startswith('author')])
    #print([ a for a in s if a.startswith(ACTIVE_DOM)])
    #dl = ERContext(name = DBLP_ACM)
    #x = "SENTINEL: An Object-Oriented DBMS With Event-Based Rules"
    #y = "Sentinel: an object-oriented DBMS with event-based rules"
    #print(len(x)+len(y))
    #print(dl.sim(String(x),String(y)))

    
    
    # cnjs_check(None,intersect[0][('title_basics', 3, 'title_basics', 3)][3])
    #ws_pat = re.compile(r"\s*")
    #pat = re.compile(r"\s*%.*\s*")
    #print(pat.findall("            sim(RE1,RE2,100). %  maybe can be grouped by language"))
    #print(line not in ['','\n','\t','\r'] and comment_pat.match(line, re.IGNORECASE)==None)
    #looking at hard rules and soft rules
    # print('1',line,comment_pat.match(line, re.IGNORECASE)==None,'1')
    # print('1.1',bool(line not in ['','\n','\t','\r']) and not bool(comment_pat.match(line, re.IGNORECASE)))
    
    #print(get_sim_attrs("specification.lp"))
