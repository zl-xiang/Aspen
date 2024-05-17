
import utils
from logger import logger
import logger as _log
import pickle
import os


TP_CACHE = './cache/tps' 
FP_CACHE = './cache/fps' 
FN_CACHE = './cache/fns' 
EQ_CACHE = './cache/eqs' 
TRACE_CACHE = './cache/trace'


def eval(sol,truth,by_type=True,ter=False,level = _log.INFO_C):
    eval_log = logger(reg_class='eval',level=level)
    def _eval( sol,truth,name='',):
        if not ter:
            sol = {(t1.split(':',1)[1],t2.split(':',1)[1]) for t1,t2 in sol }
        # [print(s) for s in sol]
        sol = sol.union({(s[1],s[0]) for s in sol})
        truth_size = len(truth)
        truth = truth.union({(x[1],x[0]) for x in truth})
        fn_set = utils.remove_symmetry(truth.difference(sol))
        eval_log.debug(f"[{name}] False Negative pairs : %", [fn_set])
        tp_set = utils.remove_symmetry(truth.intersection(sol))
        fp_set = utils.remove_symmetry(sol.difference(truth))
        

        eval_log.debug(f"[{name}] False Positve pairs : % ", [fp_set])
        
        tp = len(tp_set) # tp
        #tp_path = os.path.join(TP_CACHE,f"tps-{fname}_{name}.pkl")
        #fp_path = os.path.join(FP_CACHE,f"fps-{fname}_{name}.pkl")
        #fn_path = os.path.join(FN_CACHE,f"fns-{fname}_{name}.pkl")
        
        #if not os.path.isfile(tp_path):
           # with open(os.path.join(tp_path), 'wb') as filep:
               # pickle.dump(tp_set, filep)

       # if not os.path.isfile(fn_path):
            #with open(os.path.join(fn_path), 'wb') as filep:
               # pickle.dump(fn_set, filep)
            
        fp = len(fp_set) # fp
        fn = len(fn_set) # fn
        #if not os.path.isfile(fp_path):
          #  with open(os.path.join(fp_path), 'wb') as filep:
                #pickle.dump(fp_set, filep)
        eval_log.info(f"[{name}] ground truth size: {truth_size}")

        eval_log.info(f"[{name}] true positive: {tp}")
        eval_log.info(f"[{name}] false negative: {fn}")
        eval_log.info(f"[{name}] false positve: {fp}")
        eval_log.info(f"============== Results [{name}] | Precision = {utils.precision(tp,fp)} | Recall = {utils.recall(tp,fn)} | F1 = {utils.f1(tp,fp,fn)} ==============")
        return tp, fp, fn
        
    
    eval_log.info(msg="==============Starting Evaluation==============",args=[])
    if isinstance(truth,dict) and not by_type:
        # print(1111)
        all_t = set()
        for k,v in truth.items():
            all_t = all_t.union(set(v)) 
        truth = all_t
        _eval(sol=sol,truth=truth)
    elif isinstance(truth,dict) and by_type:
        sol_dict = {}
        for s in sol:
            if s[0] not in sol_dict:
                sol_dict[s[0]] = set()
            sol_dict[s[0]].add((s[1],s[2]))
        sum_tp = 0
        sum_fp = 0
        sum_fn = 0
        eval_log.debug(sol_dict)
        for k,v in truth.items():
            if k in sol_dict:
                _tp, _fp, _fn = _eval(sol_dict[k],set(v),k)
                sum_tp+= _tp
                sum_fp+=_fp
                sum_fn+=_fn
            else:
                sum_fn+= len( utils.remove_symmetry(set(v)))
                print(f"============== Results for {k} not found!  ==============")
        print(f"============== Results [OVERALL] | Precision = {utils.precision(sum_tp,sum_fp)} | Recall = {utils.recall(sum_tp,sum_fn)} | F1 = {utils.f1(sum_tp,sum_fp,sum_fn)} ==============")
    else:
        _eval(set(sol),set(truth))