import argparse
from typing import Sequence
from pathlib import Path

from ercontrol2 import ERControl
import contextc_
from timeit import default_timer as timer
from program_transformer import program_transformer

from dataloader import Dataloader, save_gts
from er_explainer import ERExplainer
import example_schema as eschema
import utils
import random
from logger import logger
import logger as _log

from metainfo import Schema


import pickle
import os



TP_CACHE = './cache/tps' 
FP_CACHE = './cache/fps' 
FN_CACHE = './cache/fns' 
EQ_CACHE = './cache/eqs' 
TRACE_CACHE = './cache/trace'

SAT = "SAT"
UNSAT = "UNSAT"
UNKNOWN = "UNKNOWN"
SOLVE_STAT = [SAT] 


BASE = "base"
"""base program, grounded as factual knowledge
"""
MAXSOL = "maxsol"

LP_SUFFIX = ".lp"

CACHE_DIR = './cache' 

DBLP = "dblp"
DBPL_SPLIT = "dblp-split"
CORA = "cora"
CORA_SPLIT = "cora-split"
IMDB = "imdb"
MUSIC = "music"
POKEMON = "pokemon"


def get_schema(args,data_dir='')->tuple[Schema,Dataloader]:
    split = args.data
    files = args.lps
    # uniq = args.uniq
    #print(uniq,'=====================================')
    if args.schema == DBLP:
        return eschema.dblp_non_split_schema(files)
    elif args.schema == CORA:
        return eschema.cora_non_split_schema(files)
    elif args.schema == IMDB:
        return eschema.imdb_schema(files=files)
    elif args.schema == MUSIC:
        return eschema.music_schema(split=split,files=files,data_dir=data_dir)
    elif args.schema == POKEMON:
        return eschema.pokemon_schema('50',files)
    else:
        return eschema.other_schema(split=split,files=files)

# ============explaination============

    
def eval(sol,truth,fname='',by_type=True,ter=False,level = _log.INFO_C):
    
    eval_log = logger(reg_class='eval',level=level)
    def _eval( sol,truth,name='',):
        if not ter:
            sol = {(t1.split(':',1)[1],t2.split(':',1)[1]) for t1,t2 in sol }
        # [print(s) for s in sol]
        sol = sol.union({(s[1],s[0]) for s in sol})
        truth_size = len(truth)
        truth = truth.union({(x[1],x[0]) for x in truth})
        fn_set = utils.remove_symmetry(truth.difference(sol))
        #eval_log.debug(f"[{name}] False Negative pairs : %", [fn_set])
        tp_set = utils.remove_symmetry(truth.intersection(sol))
        eval_log.debug(f"[{name}] True Positve pairs : % ", [tp_set])
        fp_set = utils.remove_symmetry(sol.difference(truth))

        eval_log.debug(f"[{name}] False Positve pairs : % ", [fp_set])
        
        tp = len(tp_set) # tp
        tp_path = os.path.join(TP_CACHE,f"tps-{fname}_{name}.pkl")
        fp_path = os.path.join(FP_CACHE,f"fps-{fname}_{name}.pkl")
        fn_path = os.path.join(FN_CACHE,f"fns-{fname}_{name}.pkl")
        
        if not os.path.isfile(tp_path):
            with open(os.path.join(tp_path), 'wb') as filep:
                pickle.dump(tp_set, filep)

        if not os.path.isfile(fn_path):
            with open(os.path.join(fn_path), 'wb') as filep:
                pickle.dump(fn_set, filep)
            
        fp = len(fp_set) # fp
        fn = len(fn_set) # fn
        if not os.path.isfile(fp_path):
            with open(os.path.join(fp_path), 'wb') as filep:
                pickle.dump(fp_set, filep)
        eval_log.info(f"[{name}] ground truth size: {truth_size}")

        eval_log.info(f"[{name}] true positive: {tp}")
        eval_log.info(f"[{name}] false negative: {fn}")
        eval_log.info(f"[{name}] false positve: {fp}")
        eval_log.info(f"============== Results [{name}] | Precision = {utils.precision(tp,fp)} | Recall = {utils.recall(tp,fn)} | F1 = {utils.f1(tp,fp,fn)} ==============")
        return tp, fp, fn
        
    
    eval_log.info(msg="==============Starting Evaluation==============",args=[])
    if isinstance(truth,dict) and not by_type:
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
         # print("---------------truth",truth)
    
        # print(f"============== Results | Precision = {utils.precision(tp,fp)} | Recall = {utils.recall(tp,fn)} | F1 = {utils.f1(tp,fp,fn)} ==============")

def __init_crtl(args,log):
    spec = get_schema(args)
    ternary = 'ter' if args.ternary else ''
        
    # spec = eschema.dblp_non_split_schema()
    log.info('* Schema Name : %',[spec[0].name])
    log.info('* Schema stats : %',[spec[0].stats()])
        # print(spec[0].stats())
    files = args.lps
    file = files[0]
    prg_trans = program_transformer(schema=spec[0])
    ctx = contextc_.ERContext(ter=args.ternary)
    enum = args.enumerate
        # requires number of explainations are the same as number of enumerated models
    er_ctrl = ERControl(ctx,transformer=prg_trans,dataloader=spec[1],args=args,n_enum=enum,n_explanations=str(enum))
    fname=files[0].replace('.lp','').split('/')[-1]
    if args.rec_track: fname = fname.replace('-rec','')
    version = '' if utils.is_empty(args.data) else args.data
    er_ctrl.sim_cache_dir = os.path.join(CACHE_DIR,f'sim-{fname}{version}{ternary}.pkl')
    #print(er_ctrl.sim_cache_dir)
    return er_ctrl

def main(**kwargs):
    args = argparse.Namespace(**kwargs)
    files = args.lps
    debug = args.debug
    print(debug)
    log_level = _log.INFO_C if not debug else _log.DEBUG_C
    if args.asprin: mode = 'maxsol'
    else:mode = 'normal'
    fname=files[0].replace('.lp','').split('/')[-1]
    presimed = args.presimed
    ter = args.ternary
    ter_str = '' if not ter else '-ter'
    ub_eqs_dir = os.path.join(CACHE_DIR,f'{fname}{ter_str}-{args.data}-ub.pkl') if not utils.is_empty(args.data) else os.path.join(CACHE_DIR,f'{fname}{ter_str}-ub.pkl')
    if args.main:
        main_log = logger('main',level=log_level)

        maxsol = args.maxsol
        upperbound = args.ub
        lowerbound = args.lb
        asprin = args.asprin
        # number of models to be enumerated
        er_ctrl = __init_crtl(args=args,log=main_log)
        ctx = er_ctrl.context
        cached_results_dir = os.path.join(CACHE_DIR,f'{fname}-{mode}.pkl')

        sim_facts = set()
        if presimed:
            sim_facts=utils.load_cache(er_ctrl.sim_cache_dir)
        if upperbound:
            spec_ver = program_transformer.UPERBOUND
        elif lowerbound:
            spec_ver = program_transformer.LOWERBOUND
        else:
            spec_ver = program_transformer.ORIGIN  
        # program_transformer.ORIGIN if not upperbound else program_transformer.UPERBOUND
        ub_eqs = None
        if args.ub_guarded:
            ub_eqs = utils.load_cache(ub_eqs_dir)
            #[print(a) for a in ub_eqs]
            sol, all_models = er_ctrl.run(maxsol,files,sim_facts = sim_facts,spec_ver=spec_ver,asprin=asprin,ub_eqs=ub_eqs)
        else:
            sol, all_models = er_ctrl.run(maxsol,files,sim_facts = sim_facts,spec_ver=spec_ver,asprin=asprin,ub_eqs=ub_eqs)
        #with open(cached_results_dir, 'wb') as fp:
           # pickle.dump(sol, fp)
            
        all_models = [set(map(utils.atom2str,model)) for model in all_models]
        model = all_models[0]
            
        main_log.info('* Similarity execution times : %',[ctx.sim_counter])
        main_log.info('* Number of stored sim pairs : %',[len(ctx.sim_pairs)])
        schema_name = er_ctrl.prg_transformer.schema.name
        constraints = False
        if args.trace:
            main_log.info('* number of models in the original program %', [len(all_models)])
            explainer = ERExplainer(er_ctl=er_ctrl,constraints=constraints)
            explainer.expand_trace(args,all_models,er_ctrl,constraints)
        ground_truth = er_ctrl.dataloader.load_ground_truth()
        # main_log.debug('* number of ground truth: % ',[len(ground_truth)])
        by_type = args.typed_eval
        if args.enumerate==1:
            #main_log.debug('cora records with weird behaviour: %', [[s for s in set(sol) ]])
            eval(set(sol),ground_truth,fname,by_type,ter,level=log_level)
        else:
            for i, s in enumerate(sol):
                eval(set(s),ground_truth,fname+str(i),by_type,ter,level=log_level)
    
    elif args.pos_merge:
        by_type = args.typed_eval
        pos_log = logger('pos-merge')
        er_ctrl = __init_crtl(args=args,log=pos_log)
        ctx = er_ctrl.context
        # of the shape c_1,c_2 c_3,c_4 ... c_n,c_m
        merge = ['all'] if 'all' in args.pos_merge else utils.pairs2tups(args.pos_merge)
        attrs = None if 'all' in args.pos_merge else utils.pairs2tups(args.attr)
        if args.ub_guarded:
            ub_eqs = utils.load_cache(ub_eqs_dir)
        sim_facts=utils.load_cache(er_ctrl.sim_cache_dir)
        if 'all' in args.pos_merge:
            # merge = ['all']
            sol, all_models = er_ctrl.pos_merge(sim_facts=sim_facts,ter=ter,ub_eqs=ub_eqs)
            sol, all_models = utils.load_result(sol=sol,a_models=all_models,triple=by_type)
            ground_truth = er_ctrl.dataloader.load_ground_truth()
            eval(set(sol), ground_truth,by_type,ter=ter,level=log_level)
        else:
            merges = utils.pairs2tups(args.pos_merge)

            if args.trace:
                explainer = ERExplainer(er_ctl=er_ctrl)
                explainer.trace_merge(merge = merges,sim_facts=sim_facts,attrs=attrs)
            else:    
                sol, all_models = er_ctrl.pos_merge(merge=merges,sim_facts=sim_facts,ter=ter,attrs=attrs)
                sol, all_models = utils.load_result(sol=sol,a_models=all_models,triple=by_type)

    elif args.cert_merge:
        cert_log = logger('cert-merge',level=log_level)
        er_ctrl = __init_crtl(args=args,log=cert_log)
        ctx = er_ctrl.context
        sim_facts=utils.load_cache(er_ctrl.sim_cache_dir)
        er_ctrl.cm_approx(sim_facts=sim_facts)
        
    elif args.naive_sim:
        sim_log = logger('naive-sim',level=log_level)
        er_ctrl = __init_crtl(args=args,log=sim_log)
        ctx = er_ctrl.context
        # sim_facts=utils.load_cache(er_ctrl.sim_cache_dir)
        er_ctrl.get_naive_sim()
        
    elif args.examine:
        cached_results_dir = os.path.join(CACHE_DIR,f'{fname}-{args.data}-{mode}.pkl') if not utils.is_empty(args.data) else os.path.join(CACHE_DIR,f'{fname}-{mode}.pkl')
        print(cached_results_dir)
        exam_log = logger('examine',level=log_level)
        maxsol = args.maxsol
        by_type = args.typed_eval
        # number of models to be enumerated
        er_ctrl = __init_crtl(args=args,log=exam_log)
        ctx = er_ctrl.context
        ground_truth = er_ctrl.dataloader.load_ground_truth()
        if args.cached and os.path.isfile(cached_results_dir):
            sol, all_models = utils.load_result(cached_results_dir,triple=by_type)
            eval(set(sol),ground_truth,fname,by_type,ter=ter)

    elif args.rec_track:
        rec_log = logger('rec-track',level=log_level)
        er_ctrl = __init_crtl(args=args,log=rec_log)
        ctx = er_ctrl.context
        maxsol = args.maxsol
        sim_facts = set()
        if presimed:
            sim_facts=utils.load_cache(er_ctrl.sim_cache_dir)
            #[print(a) for a in sim_facts]
        sol_iter_dict =  er_ctrl.rec_track_ub(sim_facts=sim_facts) if args.ub else er_ctrl.rec_track_max(sim_facts=sim_facts, files = files,maxsol_dir=maxsol) 
        ground_truth = er_ctrl.dataloader.load_ground_truth()
        by_type = args.typed_eval
        iter_num = len(sol_iter_dict.keys())
        spec_ver = 'UB' if args.ub else 'MAXSOL'
        rec_log.info(f'Evaluate multi-level recursion on {spec_ver} ...')
        if sol_iter_dict is not None and len(sol_iter_dict.keys())>0:
            for iter in range(iter_num):
                rec_log.info(f'Evaluating the {iter+1} iteration ...')
                eval(set(sol_iter_dict[iter+1]),ground_truth,fname,by_type,ter=ter,level=log_level)
                
        
    elif args.getsim:
        start = timer()
        print("* Precomputing upperbound of constants similarities")
        print(f"* Start time: {start} s")
        sim_log = logger('getsim')
        er_ctrl = __init_crtl(args,sim_log)
        # data = '' if utils.is_empty(args.data) else args.data
        print('* Querying facts to be compared on similarity ...')
        tobe_simed, sol, model = er_ctrl.get_simed_tuples()
        print('* Pre-computed sim size :', len(tobe_simed))
        print('* Saving to-be-simed results ...')
        end = timer()
        print(f"* To be simed computation ends in : {end - start} s")
        utils.cache(er_ctrl.sim_cache_dir,tobe_simed)
        cached_results_dir = os.path.join(CACHE_DIR,f'{fname}-{args.data}-{mode}.pkl') if not utils.is_empty(args.data) else os.path.join(CACHE_DIR,f'{fname}-{mode}.pkl')
        all_models = [set(map(utils.atom2str,model))]
        utils.cache(cached_results_dir,(sol,all_models))
        # ub Eq cache
        ub_eqs_dir = os.path.join(CACHE_DIR,f'{fname}{ter_str}-{args.data}-ub.pkl') if not utils.is_empty(args.data) else os.path.join(CACHE_DIR,f'{fname}{ter_str}-ub.pkl')
        ub_eqs =  set([utils.get_atom('up_eq',[str(a) for a in e.arguments])+'.' for e in model if e.name == 'eq' and e.arguments[0]!=e.arguments[1]])
        #[print(a) for a in ub_eqs]
        #ub_eqs = utils.replace_pred('eq','up_eq',ub_eqs)
        utils.cache(ub_eqs_dir,ub_eqs)

    elif args.stats:
        splits = [DBLP,CORA,IMDB,MUSIC,POKEMON]
        for s in splits:
            schema = get_schema(args)

            
        

if __name__ == "__main__":
    # arg parser for python
    parser = argparse.ArgumentParser(description='Main app running options.')
    parser.add_argument(
        "-a",
        "--asprin",
        action="store_true",
        default=False,
        help="Optimisation with asprin",
    )
    parser.add_argument(
        "-c",
        "--cached",
        action="store_true",
        default=False,
        help="Loading from preprocessed ground facts.",

    )
    parser.add_argument(
        "-e",
        "--enumerate",
        type=int,
        default=1,
        help="Enumerating n optimal models.",

    )
    parser.add_argument(
        "--examine",
        action="store_true",
        default=False,
        help="Comparing resulted merges of two specifications.",

    )
    parser.add_argument(
        "--stats",
        action="store_true",
        default=False,
        help="statistics of dataset.",
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=["10","30","50","50-corr","m10+1","m10+2","m10+3","m10+4"],
        default="",
        help="""Level of duplications of dataset.""",
    ) 
    
    parser.add_argument(
        "--schema",
        type=str,
        choices=["dblp","dblp-split","cora","cora-split", "imdb", "music", "pokemon", "other"],
        default="other",
        help="""schema to be used""",
    ) 


    parser.add_argument(
        "--main",
        action="store_true",
        default=False,
        help="Default pipeline ...",

    )
    
    parser.add_argument(
        "--rec-track",
        action="store_true",
        default=False,
        help="Monitoring multi-level recursion ...",

    )
    
    parser.add_argument(
        "--detail",
        action="store_true",
        default=False,
        help="Logging label results ...",
    )
    
    parser.add_argument(
        "--getsim",
        action="store_true",
        default=False,
        help="Query tuples to be measured similarities.",
    )
    
    parser.add_argument(
        "--semi",
        action="store_true",
        default=False,
        help="Semi-column-based representation",
    )

    parser.add_argument(
        "--presimed",
        action="store_true",
        default=False,
        help="Using precomputed similarities",

    )
    
    
    parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Get explainations",
    )
    
    
    parser.add_argument('-m','--maxsol', type=str, help='Directory of preference statement', required=False)
    parser.add_argument('-l','--lps', nargs='+', help='List of logic programs', required=False)
    parser.add_argument('-v','--version', type=int, help='version of relation splits', required=False, default=0)
    
    # xclingo options

    optional_group = parser.add_mutually_exclusive_group()
    optional_group.add_argument('--only-translate', action='store_true',
                        help="Prints the internal translation and exits.")
    optional_group.add_argument('--only-translate-annotations', action='store_true',
                        help="Prints the internal translation and exits.")
    optional_group.add_argument('--only-explanation-atoms', action='store_true',
                        help="Prints the atoms used by the explainer to build the explanations.")
    parser.add_argument('--auto-tracing', type=str, choices=["none", "facts", "all"], default="none",
                        help="Automatically creates traces for the rules of the program. Default: none.")
    
    parser.add_argument('--trace-output', type=str, choices=[ "ascii-trees",
            "translation",
            "graph-models",
            "render-graphs"], 
            default="none",
            help="""Determines the format of the output. "translation" will output the translation 
        together with the xclingo logic program. "graph-models" will output the explanation 
        graphs following clingraph format.""")
    # parser.add_argument('infiles', nargs='+', type=FileType('r'), default=sys.stdin, help="ASP program")
    parser.add_argument(
        "--constraint-explaining",
        type=str,
        choices=["minimize", "all"],
        default="minimize",
        help="""Explains traced constraints of the program. Default: unsat.
        - 'unsat' will only explain constraints if original program is UNSAT. In such a case
        the ocurrence of violated constraints will be minimized when solving the original program.
        - 'minimize' acts as 'unsat' but the first UNSAT check is skipped. Directly minimizes the
        constraints when solving the original program. Useful when the original program is known
        UNSAT but it takes long to check it.
        - 'all' will explain all constraints and will not minimize them before explaining.""",
    )    
    
    parser.add_argument(
        "--pickletrace",
        action='store_true',
        default=False,
    )
    
    parser.add_argument(
        "--typed_eval",
        action='store_true',
        default=False,
    )
    
    parser.add_argument(
        "--ternary",
        action='store_true',
        default=False,
    )
    
    parser.add_argument(
        "--ub",
        help="Upperbound solution",
        action='store_true',
        default=False,
    )
    
    parser.add_argument(
        "--lb",
        help="Lowerbound solution",
        action='store_true',
        default=False,
    )
    
    
    parser.add_argument(
        "--no_show",
        help="Do not generate show directives",
        action='store_false',
        default=True,
    )
    
    parser.add_argument(
        "--debug",
        help="Run with debugging mode",
        action='store_true',
        default=False,
    )
    
    parser.add_argument(
        "--output",
        type=str,
        choices=[
            "ascii-trees",
            "translation",
            "graph-models",
            "render-graphs",
        ],
        default="ascii-trees",
        help="""Determines the format of the output. "translation" will output the translation 
        together with the xclingo logic program. "graph-models" will output the explanation 
        graphs following clingraph format.""",
    )
    
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Only takes effect when used together with --output=render-graphs.",
    )
    
    parser.add_argument('--pos-merge', nargs='+', help='Check whether given pair of merge is possible or not ...', required=False)
    parser.add_argument("--cert-merge",
        action="store_true",
        default=False,
        help="Using precomputed similarities")

    parser.add_argument('--trace-merge', nargs='+', help='Check whether given pair of merge is possible or not and explain the underlying causes', required=False)

    parser.add_argument('--attr', nargs='+', help='What attribute the merge is of', required=False)
    
    
    parser.add_argument('--ub-guarded',  
                        action='store_true',
                        default=False, 
                        help='Reuse computed UB merges to guard the Eq heads', required=False)
    
    parser.add_argument('--naive-sim',  
                        action='store_true',
                        default=False, 
                        help='Evaluating performances of naive sim ... ', required=False)

      
    main(**vars(parser.parse_args()))
    

        
