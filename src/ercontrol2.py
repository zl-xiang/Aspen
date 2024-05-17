
from typing import  Sequence, Tuple
from dataloader import Dataloader
import contextc_
from program_transformer import get_sim_atom, EQ_PRED, ACTIVE_PRED, TO_SIM, EXTERNAL_DIRC

from program_transformer import program_transformer
import program_transformer as pt
import logger as log

import clingo
#from trans_utils import get_reduced_spec
from clingo.control import Control
from clingo import  Symbol
from clingo.symbol import Number,String, Function
from clingo import Model
from clingo.ast import (
    AST,
    ASTType,
)

from asprin.src.main.main import Asprin, AsprinArgumentParser


from _xclingo2.preprocessor import PreprocessorPipeline
from _xclingo2.preprocessor import DefaultExplainingPipeline
from _xclingo2.explainer import Explainer
from _xclingo2.preprocessor.xclingo_ast._ast_shortcuts import is_constraint, literal

from _xclingo2.explainer._logger import XclingoLogger
from _xclingo2.explainer.error import ExplanationControlGroundingError, ExplanationControlParsingError
from _xclingo2.explanation import ExplanationGraphModel

import argparse

import sys
import os
import pickle
import utils

from logger import logger

from timeit import default_timer as timer

TP_CACHE = './cache/tps' 
FP_CACHE = './cache/fps' 
FN_CACHE = './cache/fns' 
EQ_CACHE = './cache/eqs' 

SAT = "SAT"
UNSAT = "UNSAT"
UNKNOWN = "UNKNOWN"
SOLVE_STAT = [SAT] 

GROUNDING = 'Grounding'
SOLVING = 'Solving'


BASE = "base"
"""base program, grounded as factual knowledge
"""
HARD = "hard"
"""program contains hard rule
"""
SOFT = "soft"
"""program contains soft rule
"""
CLOSURE = "closure"
"""program for deriving EqRel closure
"""
RULES = "rules"

LOCAL = "local"

DENIAL = "constraints"
"""program contains denial constraints
"""
MAXSOL = "maxsol"

ITER_PRED = "iter"

LP_SUFFIX = ".lp"

CACHE_DIR = './cache' 

META_PREFIX = '_'

SPEC = 'spec'


class ERControl:
    def __init__(self,context:contextc_.ERContext,
                 transformer:program_transformer,
                 dataloader:Dataloader,
                 args=None, 
                 n_enum=1, 
                 n_explanations='1', 
                 auto_trace='none',        
                 solving_preprocessor_pipeline: PreprocessorPipeline = None,
                 explaining_preprocessor_pipeline: PreprocessorPipeline = None,) -> None:
        
        self.context = context
        self.prg_transformer:program_transformer = transformer
        self.ctl = None
        self.optimiser = Asprin()

        self.args = args
        level = log.INFO_C if not self.args.debug else log.DEBUG_C
        self.ctrl_log = logger('control',level=level)
        
        self.ternary = args.ternary
        self.it_model = set()
        self.shown = set()
        self.n_enum = n_enum
        self.dataloader:Dataloader = dataloader
        self.pre_solving_pipeline = (
            PreprocessorPipeline()
            if solving_preprocessor_pipeline is None
            else solving_preprocessor_pipeline
        )
        # Explainer control
        self.pre_explaining_pipeline = (
            DefaultExplainingPipeline()
            if explaining_preprocessor_pipeline is None
            else explaining_preprocessor_pipeline
        )
        self.n_explanations = n_explanations
        # xclingo explainer
        self.explainer = Explainer([n_explanations], logger=self.ctrl_log.logger)
        self._explainer_context = None
        self.solver_result = None
        self.trace = self.args.trace
        self.sim_cache_dir = os.path.join(CACHE_DIR,f"facts_tobesimed-{self.prg_transformer.schema.name}.pkl")
        #self.data_ver = self.args.data if not utils.is_empty(self.args.data) else ""
    
    def _init_options(self,spec_dir:str,optim_dir:str, asprin=False):
        if self.n_enum == 0: opt_args=["0"]
        elif self.n_enum >1: opt_args=[f"-n={self.n_enum}"]
        else: opt_args =[]
        if asprin:
            aap = AsprinArgumentParser()
            # TODO: shall be passed through command line arguments
            opt_args += ["--debug", spec_dir, optim_dir,"-q=2","--opt-mode=ignore","--approximation=heuristic"]
            try:
                return aap.run(opt_args)
            except Exception as e:
                raise argparse.ArgumentError(None, e.message)
        else:
            # opt_args+= ['--stats']
            return opt_args
            
    def __init_control(self,program:str,spec_dir:str = None,optim_dir:str=None,asprin=False,atom_base:str=None,brave_consq=False):
        if asprin:
            self.optimiser.options, clingo_options, u, prologue, warnings = self._init_options(spec_dir,optim_dir,asprin=asprin)
            print(clingo_options)
            self.ctrl_log.debug('Initialising optimiser with options % ',[self.optimiser.options])
        else:
            #if spec_dir != None:
            clingo_options = self._init_options(spec_dir,None,asprin=asprin)
            self.ctrl_log.debug('Initialising clingo control with options % ',[clingo_options])

        self.ctl =  Control(clingo_options)
        #TODO: to be changed to setting up only before solving rather than initialisation
        if brave_consq : self.ctl.configuration.solve.enum_mode = 'brave' 
        self.add(BASE,[],program)
        self.add(BASE,[],atom_base)
        
        if asprin:
            self.optimiser.control = self.ctl
            return u, prologue, warnings
        
    def _run_optim(self,u, prologue, warnings, base_ctx):
        return self.optimiser.run_wild(u=u, prologue= prologue, 
                                     warnings=warnings,base_ctx = base_ctx, trace = self.trace)
        
    
    def add(self, name: str, parameters: Sequence[str], program: str,trace=False,cache=False) -> None:
        # if no pre-solving processor (options), translate return the program itself
        # if self.args.trace:
        # self.ctrl_log.debug('self trace or not %',[self.trace])
        if trace:
            pstart = self.ctrl_log.timer(proc_name='Pre-solving translation',state=log.START)
            pre_sol_trans = self.pre_solving_pipeline.translate(BASE,program)
            self.ctrl_log.timer(proc_name='Pre-solving translation',state=log.END,start=pstart)
            self.ctl.add(name,parameters,pre_sol_trans)
            try:
                #TODO avoid repetitive translation of EDB
                postart = self.ctrl_log.timer(proc_name='Post solving translation',state=log.START)
                # check if there already exists the translation of the part if base
                if cache:
                    ter_str = '-ter' if self.ternary else '' 
                    cache_name = f'{self.prg_transformer.schema.name}{ter_str}-{name}-trans'
                    pre_exp_trans = utils.get_cache(source_func=self.pre_explaining_pipeline.translate,cache_dir=CACHE_DIR,fname=cache_name,name=BASE, program=program)
                else:
                    pre_exp_trans = self.pre_explaining_pipeline.translate(BASE, program)
                self.ctrl_log.timer(proc_name='Pre-solving translation',state=log.END,start=postart)
                # self.ctrl_log.debug('Pre-explaining translation: % ',[pre_exp_trans])
                self.explainer.add(
                    BASE, parameters, pre_exp_trans
                )
                #return pre_exp_trans
            except RuntimeError as e:
                raise ExplanationControlParsingError(e)
        else:
             self.ctl.add(name,parameters,program)
        
    def extend_explainer(self, name: str, parameters: Sequence[str], program: str) -> None:
        self.explainer.add(name, parameters, program)
        
    def add_show_trace(self, atom: Symbol, conditions: Sequence[Tuple[bool, Symbol]] = []):
        self.explainer.add("base", [], f"_xclingo_show_trace({str(atom)}) :- .")
        
    def _on_model(self, model: Model):
        # terms Select all terms displayed with #show statements in the model.
        self.it_model = model.symbols(atoms=True)
        self.shown = set([str(a) for a in model.symbols(terms=True)])
        
    def _on_unsat(self, model:Model):
        self.it_model = model.symbols(atoms=True)
        
    def _on_sim_model(self, model: Model):
        # terms Select all terms displayed with #show statements in the model.
        for a in model.symbols(atoms=True):
            a = str(a)
            if a.startswith('up_eq('):
                self.it_model.add(a)
        # print(len(self.it_model))
        
    def ground(self,parts:Sequence[tuple[str,Sequence[Symbol]]]=[(BASE,[])],context=None,proc_name = ''):
        proc_name = proc_name if not utils.is_empty(proc_name) else ','.join([f'#{name}' for name,_ in parts])
        vstart = self.ctrl_log.timer(f'{GROUNDING} {proc_name}',log.START)
        self.ctl.ground(parts,context=context)
        self.ctrl_log.timer(f'{GROUNDING} {proc_name}',log.END,vstart)
        

    def get_simed_tuples(self):
        db_facts = self.prg_transformer.get_atombase(ter=self.ternary)
        print('* Atombase size:',len(db_facts))
        db_facts_str = ''.join(db_facts)
        self.ctl = clingo.Control()
        show = self.args.no_show
        sim_spec = self.prg_transformer.get_reduced_spec(ter=self.ternary,show=show)
        sim_rules = sim_spec[0]
        sim_pairs = self.prg_transformer.sim_pairs
        self.ctrl_log.info("Sim pairs in given specification: %", [sim_pairs])
        self.ctrl_log.info("#Sim attributes : %",[len(sim_pairs)])
        sum_stats = self.prg_transformer.get_sim_cat_sum()
        self.ctrl_log.info("#Cat sum of Sim : %, #Sum of sim constants : %",[sum_stats[0],sum_stats[1]])

        mini_thresh = min([v for _,v in sim_pairs.items()])
        sim_rules = '\n'.join(sim_rules)
        self.ctrl_log.info(f"Running similarity filtering program step 1: \n {sim_rules}")
        self.ctl.add(BASE,[],sim_rules)
        self.ctl.add(BASE, [],db_facts_str)
        self.ground([(BASE, [])],context=self.context, proc_name='Similarity Filtering')
        result = self.ctl.solve(on_model = self._on_model)
        
        sim_facts = set()
        sim_facts_lower = set()
        if result.satisfiable:
            self.ctrl_log.info("Sim execution times : %",[self.context.sim_counter])
            for pair,score in self.context.sim_pairs.items():
                if score>=mini_thresh:
                    #sim_facts.add(f'{SIM_PRED}("{pair[0]}","{pair[1]}",{score}).')
                    sim_facts.add(get_sim_atom(pair[0],pair[1],score))
                    sim_facts.add(get_sim_atom(pair[1],pair[0],score))
                else:
                    sim_facts_lower.add(get_sim_atom(pair[0],pair[1],score))
                    sim_facts_lower.add(get_sim_atom(pair[1],pair[0],score))
            # [print(a) for a in sim_facts]
            # [2023-11-18] update: take all sim fact from step 1) and ground the program from step 2)
            self.ctrl_log.info("Sim algorithm step 2, finding complement to sim computed",[])
            ub_model =  set(map(utils.atom2str,self.it_model))
            ub_model.update(sim_facts)
            ub_model.update(sim_facts_lower)
            #[print(a) for a in ub_model if not a.startswith(SIM_PRED)]
            ub_model = ''.join(ub_model)
            atom_base_str = ub_model #+ db_facts_str
            del ub_model
            
            step_2_spec = '\n'.join(sim_spec[-1])
            self.ctrl_log.debug("Sim algorithm step 2 spec: % ",[step_2_spec])
            
            # re-initialise clingo object
            self.ctl = clingo.Control()
            self.ctl.add(BASE,[],step_2_spec)
            self.ctl.add(BASE, [],atom_base_str)
            
            self.ctrl_log.debug("Grounding step 2 starts ...",[])
            self.ground([(BASE, [])], context=self.context)
            sim_compl = set()
            self.ctrl_log.debug("Solving step 2 starts ...",[])
            with  self.ctl.solve(yield_=True) as solution_iterator:
                for model in solution_iterator:
                    [sim_compl.add(a) for a in model.symbols(atoms=True) if a.name == TO_SIM]
            if len(sim_compl) >0:
                compl_inc = 0
                for a in sim_compl:
                    X = str(a.arguments[0]).replace('"','')
                    Y = str(a.arguments[1]).replace('"','')
                    S = utils.sim(X,Y)
                    self.context.sim_counter+=1
                    if not ((X,Y) in self.context.sim_pairs or (Y,X) in self.context.sim_pairs):
                        self.context.sim_pairs[(X,Y)] = S
                    if S >= mini_thresh:
                        compl_inc+=1
                        new_sim = get_sim_atom(X,Y,S)
                        # )f'{SIM_PRED}({X},{Y},{S}).'
                        sim_facts.add(new_sim)
                        
                self.ctrl_log.debug("Step 2 complement increment : %",[compl_inc])
            self.ctrl_log.info("Sim cached #pairs : %",[len(self.context.sim_pairs.keys())])
            # [2023-11-18] update: take all sim fact from step 1) and ground the program from step 2)
            self.ctrl_log.info("Sim facts size: %", [len(sim_facts)])
        return set(sim_facts), self.shown, self.it_model
        #return to_be_simed,sim_pairs
        

    def get_parts (self,db_facts:set,token=None,file=None):
        db_facts_str = ''.join(db_facts)
        self.ctl = Control()
        file = f'{file}{token}.lp'
        self.ctl.load(file)
        self.ctl.add(BASE, [],db_facts_str)
        self.ctl.ground([(BASE, [])], context=self.context)
        self.ctl.solve(on_model=self._on_model)
        model = set([str(a) for a in self.it_model ])
        return model
    # *** explainable facility functions ***
    def explain_model(self, model:set[Symbol]) -> Sequence[ExplanationGraphModel]:
        return self.explainer._compute_graphs_(model,context=self.context)
    
    def run(self,maxsol:str,files:Sequence[str], sim_facts = set(),spec_ver = program_transformer.ORIGIN, asprin = False):
        if not files:
            files = ["-"]
        atom_base= self.prg_transformer.get_atombase(ter=self.ternary)
        # [print(a) for a in atom_base]
        # print(atom_base)
        show = self.args.no_show
        # print(show,'---------------------------------')
        is_ol_sim = sim_facts == None or len(sim_facts)<1
        if not is_ol_sim: atom_base = atom_base.union(sim_facts)
        spec = self.prg_transformer.get_spec(ter=self.ternary,spec_ver=spec_ver,trace=self.trace,show=show,is_ol_sim=is_ol_sim)
        [print(r) for r in spec]
        spec = '\n'.join(spec)

        self.ctrl_log.info("* Atombase size: %",[len(atom_base)])         
        atom_base = ''.join(atom_base)
        if asprin:
            u, prologue, warnings = self.__init_control(program=spec,spec_dir=files[0],optim_dir=maxsol,atom_base=atom_base, asprin=asprin)  
            fname=files[0].replace('.lp','').split('/')[-1] # TODO
            size = int(sys.getsizeof(self.context.sim_pairs)/(1024*1024))
            _solver = self._run_optim(u=u, prologue=prologue, 
                                      warnings=warnings, base_ctx=self.context) 
            size = int(sys.getsizeof(self.context.sim_pairs)/(1024*1024))
            self.ctrl_log.info("* Sim cache size : % MB",[size])
            if self.n_enum==1:
                sol = _solver.get_sol()
                all_models = [_solver.saturated_model] 
                #atoms = [a for a in all_models[0] if not a.name.startswith(META_PREFIX)]
                atoms = all_models[0]
                eqs = [a for a in atoms if a.name.startswith(f'{EQ_PRED}')]
                actives = [a for a in atoms if a.name.startswith(f'{ACTIVE_PRED}')]
                self.ctrl_log.debug("* Sol pairs: % ",[[a for a in sol]])
                self.ctrl_log.info("* Atom size: % ",[len(atoms)])
                self.ctrl_log.info("* Eq size: % ",[len(eqs)])
                self.ctrl_log.info("* Active size: % ",[len(actives)])
                del atoms
                sol = set(map(utils.sym2tup,sol))
                return sol, all_models
            else:
                sol = _solver.get_optim_enum()
                # [print(s) for s in sol]
                all_models = _solver.all_models_enum
                self.ctrl_log.info(f"#Models: {str(len(all_models))}")
                sols = []
                self.ctrl_log.info("* sol numbers (%), model numbers (%)",[len(sol),len(all_models)])
                for i, m in enumerate(all_models):
                    sol[i] = [s for s in sol[i] if utils.is_empty(s.name)]
                    self.ctrl_log.info("* solution for model (%)  ...",[i])
                    atoms = [a for a in m if not a.name.startswith(META_PREFIX)]
                    self.ctrl_log.info("* Atom size (%):  %",[i,len(atoms)])
                    eqs = [a for a in atoms if a.name.startswith(f'{EQ_PRED}')]
                    self.ctrl_log.info(" * Active size (%):  %",[i, len([a for a in atoms if a.name.startswith(f'{ACTIVE_PRED}')])]) 
                    self.ctrl_log.info("*  Eq size (%):  %",[i,len(eqs)])
                    del atoms
                    eqs = set(map(utils.atom2str,eqs))
                    eq_cache = os.path.join(EQ_CACHE,f"eq-{fname}.pkl")
                    if not os.path.isfile(eq_cache):
                        with open(eq_cache, 'wb') as filep:
                            pickle.dump(eqs, filep)
                    sols.append(set(map(utils.sym2tup,sol[i])))
                return sols, all_models
        else:
            self.ctrl_log.info("* Solving specification without optimisation ...",[])
            self.__init_control(program=spec,optim_dir=None,atom_base=atom_base)
            start =  timer()
            self.ground([(BASE,[])],context=self.context)
            sol = []
            models = []
    
            self.ctrl_log.info("* Solving starts....",[])
            start =  timer()
            with  self.ctl.solve(yield_=True) as solution_iterator:
                for model in solution_iterator:
                    sol=set(map(utils.sym2tup, model.symbols(shown=True)))
                    models.append(model.symbols(atoms=True))
                    print(len(sol),len(models))
                end = timer()
                self.ctrl_log.info("* Solving ended in % seconds ....",[end - start])
                self.ctrl_log.info("* Eq size: % ",[len([a for a in models[0] if a.name.startswith(EQ_PRED)])])
                self.ctrl_log.info("* Sol size: % ",[len([a for a in sol])/2])
                self.ctrl_log.debug("* Sol pairs: % ",[[a for a in sol]])
                return sol, models
    
    
    def __rec_track (self, atom_base:set[str], rec_spec:Sequence[str] ) -> dict:
        atom_base = ''.join(atom_base)
        rec_spec = '\n'.join(rec_spec)
        self.ctrl_log.info(f'Recursion level tracking program: \n {rec_spec}')
        self.__init_control(program=rec_spec,atom_base= atom_base)
        sols = set()
        self.ground()
        sol_iter_dict = {}
        i = 1
        # last_model = set()
        while True:
            # self.ctrl_log.info("* Grounding the % iteration....",[str(i)])
            self.ground([(BASE,[]),(SPEC,[Number(i)])],proc_name=f'iteration {str(i)}',context = self.context)
            sols_len = len(sols)
            prev_sols = sols
            #models = []
            self.ctrl_log.info(f"* Solving the {str(i)} iteration....")
            with  self.ctl.solve(yield_=True) as solution_iterator:
                for model in solution_iterator:
                    sol=set(map(utils.sym2tup, model.symbols(shown=True)))
                    # print('---------',len(sol))
                    self.ctrl_log.info(f"* Solving the {str(i)} iteration....")
                    # last_model ={ e for e in model.symbols(atoms=True) if e.name == EQ_PRED}
                    sols = sols.union(sol)
                sols_len_after = len(sols)
                #for m in models:
                    #[print(str(a)) for a in m]
                if sols_len == sols_len_after:
                    self.ctrl_log.info(f"* Converged at the #{str(i-1)} iteration")
                    break
                else: 
                    sol_iter_dict[i] = sols
                    self.ctrl_log.info(f"* #Increments of the {str(i)} iteration: {str((sols_len_after-sols_len)/2)}")
                    self.ctrl_log.debug("* Increments of the % iteration: %",[str(i),list(sols.difference(prev_sols))])
                    i+=1
        return sol_iter_dict
            
    def rec_track_ub(self, sim_facts = set()):
        atom_base = self.prg_transformer.get_atombase(multi_lvl=True,ter=self.ternary)
        #[print(a) for a in atom_base]
        atom_base = atom_base.union(sim_facts)
        show = self.args.no_show
        rec_spec = self.prg_transformer.get_rec_spec(ter=self.ternary,show=show)
        return self.__rec_track(atom_base=atom_base,rec_spec=rec_spec)

    
    def rec_track_max(self,sim_facts:set, files = [], maxsol_dir:str = ''):
        show = self.args.no_show
        max_sol_pairs , max_model= self.run(files=files,maxsol=maxsol_dir,sim_facts=sim_facts,asprin=True)
        max_eqs = [a for a in  max_model[0] if a.name == EQ_PRED]      
        max_eqs = set(map(utils.atom2str, max_eqs)) 
        readoff_atom_base = self.prg_transformer.get_atombase(ter=self.ternary,multi_lvl=True).union(max_eqs).union(sim_facts)
        readoff_program = self.prg_transformer.get_rec_spec(self.ternary,show=show, max=True)
        return self.__rec_track(atom_base=readoff_atom_base, rec_spec=readoff_program)

 

    def pos_merge(self,merge:Sequence[tuple]=None,sim_facts=set(),ter=False,attrs = None):
        atom_base = self.prg_transformer.get_atombase(ter=ter)
        
        is_ol_sim = sim_facts == None or len(sim_facts)<1 
        if not is_ol_sim: atom_base = atom_base.union(sim_facts)
        # [print(a) for a in atom_base]
        atom_base = ''.join(atom_base)
        #print(merge)
        single_pair = merge is not None and (len(merge) == 1 and 'all' not in merge)
        show = self.args.no_show
        program = self.prg_transformer.get_spec(ter=ter,trace=self.trace,show=show,is_ol_sim=is_ol_sim)
        [print(r) for r in program]
        if single_pair: 
            return self.pos_merge_single(program=program,merge=merge,atom_base=atom_base,ter=ter,attrs=attrs)
        else:
            return self.pos_merge_all(program=program,atom_base=atom_base)
        
        
    def pos_merge_single(self,program:list[str], merge:Sequence[tuple]=None,atom_base:str = None ,ter=False,attrs = None):
        #if single_pair: 
        merge = merge[0]
        single_merge = list(merge)
        if ter:
            attr = attrs[0]
            rel = self.prg_transformer.schema.rel_index(attr[0])
            attr_idx = 0
            for i,a in enumerate(rel.attrs):
                if a.name == attr[1]:
                    attr_idx = a.id
                    break
            single_merge.append(attr_idx) 
        self.ctrl_log.info("*  checking possible pair: %",[single_merge])
        pos_pair_dc = self.prg_transformer.get_merge_constraint(merge=single_merge,ter=ter,neg=True)
        program+=pos_pair_dc
        program = '\n'.join(program)
        # print(program)
        self.__init_control(program=program,atom_base= atom_base)
        start =  timer()
        self.ctrl_log.info("* Grounding starts....",[])
        self.ctl.ground([(BASE,[])],context = self.context )
        end = timer()
        self.ctrl_log.info("* Grounding ended in % seconds ....",[end - start])
        start =  timer()
        self.ctrl_log.info("* Solving starts....",[])
        result = self.ctl.solve(on_model=self._on_model)
        end = timer()
        self.ctrl_log.info("* Solving ended in % seconds ....",[end - start])
        if result.unsatisfiable:
            self.ctrl_log.info("* % \is not a possbile merge.",merge)
        elif result.satisfiable:
            # pos_merges = [str(a) for a in self.it_model if a.name.startswith(EQ_PRED)]
            self.ctrl_log.info("*  (%,%) on %.% is a possbile merge.",[merge[0],merge[1],attr[0],attr[1]],)
         #if self.trace or merge[0]=='all':
        return self.shown, [self.it_model], 
    
    
    def pos_merge_all(self,program:list[str],atom_base:str=None):
        program = '\n'.join(program)
        self.ctrl_log.info("*  compute all possible merges ... ")
        self.__init_control(program=program, atom_base= atom_base,brave_consq=True)
        start =  timer()
        self.ctrl_log.info("* Grounding starts....",[])
        self.ctl.ground([(BASE,[])],context = self.context )
        end = timer()
        self.ctrl_log.info("* Grounding ended in % seconds ....",[end - start])
        start =  timer()
        self.ctrl_log.info("* Solving starts....",[])
        result = self.ctl.solve(on_model=self._on_model)
        end = timer()
        self.ctrl_log.info("* Solving ended in % seconds ....",[end - start])
        if result.unsatisfiable:
            self.ctrl_log.info("*  no possible merges found .")
        elif result.satisfiable:
            pos_merges = [str(a) for a in self.it_model if a.name.startswith(EQ_PRED)]
            # print(pos_merges)
            self.ctrl_log.debug("* possible merges found: %",[pos_merges])
        return self.shown, [self.it_model], 
    
            
    def cm_approx(self,sim_facts:set[str])->set[str]:
        self.ctrl_log.info('Compute the set of lower approximation of certain merges ...',[])
        atom_base = self.prg_transformer.get_atombase(ter=self.ternary)
        atom_base = atom_base.union(sim_facts)

        rules = self.prg_transformer.get_spec(ter=self.ternary)
        [print(r) for r in rules]
        hard_rules = [r for r in rules if r.startswith(EQ_PRED)]
        soft_rules = [r for r in rules if r.startswith(ACTIVE_PRED)]
        active_choice = pt.ACTIVE_CHOICE if not self.ternary else pt.ACTIVE_CHOICE_TER
        denials = [r for r in rules if r.startswith(pt.IMPLY)]
        
        atom_base_str = ''.join(atom_base)
        eq_axiom = pt.EQ_AXIOMS_TER_TRACE if self.ternary else pt.EQ_AXIOMS_TRACE
        base = [pt.EMPTY_TGRS,eq_axiom,atom_base_str]
        
        # TODO: to add __init_control an option to initialise without adding program
        self.ctl = Control()
        self.add(BASE,[],'\n'.join(base))
        self.add(HARD,[],'\n'.join(hard_rules))
        self.add(SOFT,[],'\n'.join(soft_rules)+active_choice)
        self.add(DENIAL,[],'\n'.join(denials))
        # ground and solve hard part
        # self.ctl.ground([(BASE,[]),(HARD,[])])
        self.ground([(BASE,[]),(HARD,[])],context=None,)
        self.ctl.solve(on_model=self._on_model)
        hard_merges = set([str(a) for a in self.it_model if a.name.startswith(EQ_PRED)])
        # print(len(hard_merges))
        self.ctrl_log.info(f'# lower bound {str(len(hard_merges))} ...')
        # add and ground soft rules with denials
        self.ground([(BASE,[]),(HARD,[]),(SOFT,[]),(DENIAL,[])],context=None,)
        # enable brave reasoning
        self.ctl.configuration.solve.enum_mode = 'brave' 
        # solving the specification with brave reasoning
        self.ctl.solve(on_model=self._on_model)
        pm = set([str(a) for a in self.it_model if a.name.startswith(EQ_PRED)])
        # exclude hard merges from possible merges
        to_be_tested = pm.difference(hard_merges)
        # to_be_tested = hard_merges
        # to_be_tested = list(to_be_tested)
        # print(to_be_tested)
        self.ctrl_log.info(f'# to be test {str(len(to_be_tested))} ...')
        lower_cm = set()
        checked = set()
        merge_consts = ['i']
        iter_fact = f'{ITER_PRED}({merge_consts[0]})' 
        iter_external = f'{EXTERNAL_DIRC} {iter_fact}.'
        cm_check_name = 'cm_check'
        self.ctl.configuration.solve.enum_mode = 'bt' 
        # for each possible merge
        iter_num = 0
        self.add(cm_check_name,merge_consts,iter_external)
        for p in to_be_tested:
            self.ctrl_log.debug(p)
            p_pred, p_tup = utils.get_atom_tup(p)
            if not self.ternary:
                #merge_consts = ['x','y','i'] 
                #iter_fact = f'iter({merge_consts[2]}).'
                cm_check_ic = f'{pt.IMPLY} {EQ_PRED}({p_tup[0]},{p_tup[1]}),{iter_fact}.'
                #cm_check_ic = f'{pt.IMPLY} {EQ_PRED}({p_tup[0]},{p_tup[1]}).'
            else:
                #merge_consts = ['x','y','n','i']
                #iter_fact = f'iter({merge_consts[3]}).'
                # TODO thinking about here?
                cm_check_ic = f'{pt.IMPLY} {EQ_PRED}({p_tup[0]},{p_tup[1]},{p_tup[2]}),{iter_fact}.'
                #cm_check_ic = f'{pt.IMPLY} {EQ_PRED}({p_tup[0]},{p_tup[1]},{p_tup[2]}).'
            # self.prg_transformer.get_merge_constraint([merge_consts],ter=self.ternary)
            print(cm_check_ic)
            self.add(cm_check_name,merge_consts,cm_check_ic)
            p_tup_sym = p_tup[:]
            p_tup_sym[0] = p_tup[1]
            p_tup_sym[1] = p_tup[0]
            p_tup_sym = tuple(p_tup_sym)
            if p_tup_sym in lower_cm:
                lower_cm.add(tuple(p_tup))
            elif p_tup_sym in checked:
                checked.add(tuple(p_tup))
            else:
                consts = [Number(iter_num)]
                self.ctrl_log.debug(consts)
                self.ground(([(cm_check_name,consts)]),context=None)
                # self.ground(([(cm_check_name,[])]),context=None)
                if not iter_num == 0:
                    self.ctl.assign_external(Function(f"{ITER_PRED}",[Number(iter_num-1)]),False)
                self.ctl.assign_external(Function(f"{ITER_PRED}",[Number(iter_num)]),True)
                result = self.ctl.solve()
                iter_num+=1
                if result.unsatisfiable:
                    self.ctrl_log.info('found one certain merge % ...',[p_tup])
                    lower_cm.add(tuple(p_tup))
                else:
                    checked.add(tuple(p_tup))
        # verify if its indeed a certain merge
            ## create sub program with 2 or 3 parameters depending on ter
            ## for each parameter, ground the sub and solve the whole program again
        # adding to lower_cm if unsatisfiable
        # otherwise continue the iteration
        self.ctrl_log.info('# % new certain merges found', [len(lower_cm)])
        lower_cm = {utils.get_atom_(EQ_PRED,t)[:-1] for t in lower_cm}
        hard_merges.update(lower_cm)
        # self.ctrl_log.debug('Difference between PM and LowerCM %', [pm.difference(hard_merges)])
        return hard_merges