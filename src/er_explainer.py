from typing import  Sequence, Tuple
from clingo.control import Control
from _xclingo2.preprocessor import (
    DefaultExplainingPipeline,
    ConstraintRelaxerPipeline,
    ConstraintExplainingPipeline,
)
from _xclingo2.extensions import load_xclingo_extension
from _xclingo2._utils import print_header, print_version
import program_transformer as pt
import logger as lg
from logger import logger
from ercontrol2 import ERControl,BASE,HARD,SOFT,DENIAL, EQ_PRED, ACTIVE_PRED
import utils
from clingraph.orm import Factbase
from clingraph import compute_graphs as compute_clingraphs
from clingraph.graphviz import render

import pickle
import os

BASE = "base"
TRACE_CACHE = './cache/trace'
GROUNDING = 'Grounding'
SOLVING = 'Solving'

def read_files(files):# file objects are required
    return "\n".join([file.read() for file in files])

class ERExplainer:
    
    def __init__(self,er_ctl:ERControl,constraints=False) -> None:
        self.er_ctl = er_ctl
        self.args = self.er_ctl.args
        self.constraints = constraints
        self.ctrl_log = logger('ER_explainer')
                
        if self.args.auto_tracing != "none":
            self.er_ctl.extend_explainer(
                BASE, [], load_xclingo_extension(f"autotrace_{self.args.auto_tracing}.lp")
            )

        if self.args.trace_output == "render-graphs":
            self.er_ctl.extend_explainer(BASE, [], load_xclingo_extension("graph_locals.lp"))
            self.er_ctl.extend_explainer(BASE, [], load_xclingo_extension("graph_styles.lp"))

        if constraints:
            
            self.er_ctl.pre_solving_pipeline = ConstraintRelaxerPipeline()
            self.er_ctl.pre_explaining_pipeline = ConstraintExplainingPipeline()
            
            self.er_ctl.extend_explainer(
                BASE, [], load_xclingo_extension("violated_constraints_show_trace.lp")
            )
            if self.args.constraint_explaining == "minimize":
                self.er_ctl.extend_explainer(
                    BASE, [], load_xclingo_extension("violated_constraints_minimize.lp")
                )


    def print_explainer_program(self):
        from _xclingo2.xclingo_lp import FIRED_LP, GRAPH_LP, SHOW_LP

        print_version()
        print(FIRED_LP)
        print(GRAPH_LP)
        print(SHOW_LP)

        if self.args.auto_tracing != "none":
            print(load_xclingo_extension(f"autotrace_{self.args.auto_tracing}.lp"))

        if self.args.output == "render-graphs":
            print(load_xclingo_extension("graph_locals.lp"))
            print(load_xclingo_extension("graph_styles.lp"))

        if self.args.debug_output == "explainer-program":
            pipe = DefaultExplainingPipeline()
        elif self.args.debug_output == "unsat-explainer-program":
            pipe = ConstraintExplainingPipeline()
        elif self.args.debug_output == "unsat-solver-program":
            pipe = ConstraintRelaxerPipeline()

        if len(self.args.infiles) > 0:
            print(pipe.translate("translation", read_files(self.args.infiles)))


    def render_graphs(self, optim_models,):
        nmodel = 0
        for x_model in optim_models:
            nmodel += 1
            nexpl = 0
            for graph_model in self.er_ctl.explain_model(x_model):
                nexpl += 1
                render(
                    compute_clingraphs(
                        Factbase.from_model(
                            graph_model, prefix="_xclingo_", default_graph="explanation"
                        ),
                        graphviz_type="digraph",
                    ),
                    name_format=f"explanation{nmodel}-{nexpl}" + "_{graph_name}",
                    format="png",
                    directory=self.args.outdir if self.args.outdir else "out/",
                )

        if nmodel > 0:
            print("Images saved in ./out/")

        return nmodel == 0

    def solve_explain(self, optim_models):
        nmodel = 0
        # xclingo control.solve yield a sequence of XClingoModel
        # xclingo Model is a child class of clingo model
        # where original model accompanied with a explainer object
        # explain_model function of the explainer object derives explainations
        for x_model in optim_models:
            nmodel += 1
            print(f"Answer: {nmodel}")
            #if args.print_models:
            # print(x_model)
            nexpl = 0
            for graph_model in self.er_ctl.explain_model(x_model):
                nexpl += 1
                print(f"##Explanation: {nmodel}.{nexpl}")
                if self.args.output == "graph-models":
                    print(graph_model)
                else:
                    for sym in graph_model.show_trace:
                        e = graph_model.explain(sym)
                        if e is not None:
                            print(e)
            print(f"##Total Explanations:\t{nexpl}")
        if nmodel > 0:
            print(f"Models:\t{nmodel}")
            return False
        else:
            return True
        
    def into_pickle(self, models,  save_on_unsat=False, f_name='dump',):
        pickle_log = logger('main.into_pickle')
        buf = []
        for x_model in models:
            for graph_model in self.er_ctl.explain_model(x_model):
                buf.append(frozenset(str(s) for s in graph_model.symbols(shown=True)))

        if len(buf) == 0 and not save_on_unsat:
            return True

        trace_path = os.path.join(TRACE_CACHE,f"trace-{f_name}.pkl")
        if not os.path.isfile(trace_path):
            with open(trace_path, 'wb') as picklefile:
                pickle.dump(frozenset(buf), picklefile)
                pickle_log.info(f"Results saved as frozen sets at %",[trace_path])

        return False

    def expand_trace(self, models):
        # control object that hold the context of grounded base program
        if self.args.pickletrace:  # default value: ""
            self.into_pickle( models,save_on_unsat=self.constraints)
        elif self.args.output == "render-graphs":
            self.render_graphs(models)
        else:
            # solve base program and explaining
            self.solve_explain(models)
            
    def trace_merge(self,merge:Sequence[tuple]=None,sim_facts:set[str] = None,attrs = None):
        ter = self.er_ctl.ternary
        atom_base = self.er_ctl.prg_transformer.get_atombase(ter=ter)
        atom_base.update(sim_facts)
        merge = merge[0]
        single_merge = list(merge)
        single_merge[0] = f'"{single_merge[0]}"'
        single_merge[1] = f'"{single_merge[1]}"'
        if ter:
            attr = attrs[0]
            rel = self.er_ctl.prg_transformer.schema.rel_index(attr[0])
            attr_idx = 0
            for i,a in enumerate(rel.attrs):
                if a.name == attr[1]:
                    attr_idx = a.id
                    break
            single_merge.append(attr_idx) 
        self.ctrl_log.info("*  checking possible pair: %",[single_merge])
        pos_pair_dc = self.er_ctl.prg_transformer.get_merge_constraint(merge=single_merge,ter=ter,neg=True)
        
        rules = self.er_ctl.prg_transformer.get_spec(ter=ter)
        # 1 check whether its undefined or violated
        hard_rules = [r for r in rules if r.startswith(EQ_PRED)]
        soft_rules = [r for r in rules if r.startswith(ACTIVE_PRED)]
        # print(soft_rules) 
        denials = [r for r in rules if r.startswith(pt.IMPLY)]
        trace_labels = self.er_ctl.prg_transformer.annotations
        # print(trace_labels)
        
        if len(trace_labels) >0:
            for r,l in trace_labels:
                # print(r)
                if r < len(hard_rules):
                    hard_rules[r] = [l,hard_rules[r]]
                elif r<len(hard_rules)+len(soft_rules):
                    r = r - len(hard_rules)
                    soft_rules[r] = [l,soft_rules[r]]
                    #soft_rules.insert(r,l)
        
        hard_rules = utils.flatten_element(hard_rules)
        soft_rules = utils.flatten_element(soft_rules)
        #[print(r) for r in hard_rules]
        #[print(r) for r in soft_rules]
        eq_axiom = pt.EQ_AXIOMS_TER_TRACE if ter else pt.EQ_AXIOMS_TRACE
        active_choice = pt.ACTIVE_CHOICE_TER if ter else pt.ACTIVE_CHOICE
        atom_base_str = ''.join(atom_base)
        trace_pair = pt.TRACE_PAIR if not ter else pt.TRACE_PAIR_TER
        trace_pair = utils.format_string(trace_pair,single_merge,placeholder='^')
        base = [pt.EMPTY_TGRS,eq_axiom,trace_pair]
        # print(soft_rules,'--------------------')
        pos_ic_name = 'pm_ic'
        # TODO: add labels to rules
        #print(base,hard_rules,soft_rules,active_choice,denials,pos_pair_dc)
        self.er_ctl.ctl = Control()
        self.er_ctl.add(BASE,[],'\n'.join(base),trace=True)
        self.er_ctl.add(BASE,[], atom_base_str,trace=True,cache=True)
        self.er_ctl.add(HARD,[],'\n'.join(hard_rules),trace=True)
        self.er_ctl.add(SOFT,[],'\n'.join(soft_rules)+active_choice,trace=True)
        self.er_ctl.add(DENIAL,[],'\n'.join(denials),trace=True)
        self.er_ctl.add(pos_ic_name,[],'\n'.join(pos_pair_dc),trace=True)
        # print('\n'.join(['\n'.join(base), '\n'.join(hard_rules), '\n'.join(soft_rules)+active_choice ]))
        ## ground spec without denial with atom base and obtain M_ub'
        sol_ub = set()
        self.er_ctl.ground([(BASE,[]),(HARD,[]),(SOFT,[])],context=None)
        sstart = self.ctrl_log.timer(f'{SOLVING} #{BASE}, #{HARD}, #{SOFT}',lg.START)
        self.er_ctl.ctl.configuration.solve.enum_mode = 'brave' 
        self.er_ctl.ctl.solve(on_model=self.er_ctl._on_model)
        #with  self.er_ctl.ctl.solve(yield_=True) as solution_iterator:
               # for model in solution_iterator:
                    #[sol_ub.add(str(a)) for a in model.symbols(atoms=True) if a.name == EQ_PRED]
                # sol_ub = {f'{str(e)}.' for e in sol_ub}
        sol_ub = {str(a) for a in self.er_ctl.it_model if a.name.startswith(EQ_PRED)}
        #{print(a) for a in sol_ub}
        self.ctrl_log.timer(f'{SOLVING} #{BASE}, #{HARD}, #{SOFT}',lg.END,sstart)
        
        # [print(e) for e in sol_ub]
        ## check whether c \in M_ub'
        merge_fact = utils.get_atom_(EQ_PRED,single_merge)
        print(merge_fact,'##############')
        if merge_fact[:-1] not in sol_ub:
            ### no: return undefined 
            self.ctrl_log.info("*  Merge (%,%,%) is undefined .",[a for a in single_merge])
            return None
        else:    
            
            ### yes: checking if possible
            #### ground denials
            #### ground extra IC that checks is possible
            gstart = self.ctrl_log.timer(f'{GROUNDING} #{DENIAL}',lg.START)
            self.er_ctl.ctl.ground([(DENIAL,[]),(pos_ic_name,[])])
            self.ctrl_log.timer(f'{GROUNDING} #{DENIAL}',lg.END, gstart)
            self.er_ctl.ctl.configuration.solve.enum_mode = 'bt'
            result = self.er_ctl.ctl.solve(on_model = self.er_ctl._on_model)
            #### if satisfiable: take the stable model to ground the translated trace program to obtain an explaination of the merge
            if result.satisfiable:
                self.ctrl_log.info("*  Merge (%,%,%) is a possible merge .",[a for a in single_merge])
                self.ctrl_log.info("*  Finding explaination ...",[])
                estart = self.ctrl_log.timer(f'Trace merge',lg.START)
                # [print(a) for a in self.er_ctl.it_model]
                self.expand_trace([self.er_ctl.it_model])
                self.ctrl_log.timer(f'Trace merge',lg.END,estart)
            #### otherwise: 
            else: 
                ##### extend UB with mock denials as UB_trace
                hardened_soft = self.er_ctl.prg_transformer.get_ub_spec(soft_rules)
                atoms = self.er_ctl.prg_transformer.get_merge_conditions(single_merge,ter)
                weak_denials =  self.er_ctl.prg_transformer.get_weak_denials(atoms=atoms,ter=ter)
                #trace_eq_options = self.er_ctl.prg_transformer.set_show_trace_condition_w_atom(atoms,ter)
                sol_ub = {u+'.' for u in sol_ub}
                sol_ub_str = ''.join(sol_ub)
                # ''.join(atom_base) + ''.join(sol_ub)
                program = '\n'.join(hard_rules)+'\n'+'\n'.join(hardened_soft)+'\n'+eq_axiom+'\n'+pt.EMPTY_TGRS+'\n'+'\n'.join(weak_denials)
                #+'\n'+trace_eq_options
                print(program)
                self.ctrl_log.info(' * Merge % is not possible due to constraint violation.', [single_merge])
                self.ctrl_log.info(' * Finding explaination of constraint violation, program: % ... ', [program])
                #print(self.er_ctl.ternary)
                self.er_ctl.ctl = Control()
                self.er_ctl.explainer.reset_program()
                # TODO: to update the translation while keeping translation of atom base the same (adding an extra argument to add function)
                self.er_ctl.add(BASE,[],program,trace=True)
                self.er_ctl.add(BASE,[],atom_base_str,trace=True,cache=True)
                self.er_ctl.add(BASE,[],sol_ub_str,trace=True)
                
                # self.er_ctl.__init_control(program=program,atom_base=atom_base)
                ##### taking all eq \in M_ub together with A to ground UB_trace
                vstart = self.ctrl_log.timer(f'{GROUNDING} for violation check',lg.START)
                self.er_ctl.ctl.ground([(BASE,[])])
                self.ctrl_log.timer(f'{GROUNDING} for violation check',lg.END,vstart)
                
                vstart = self.ctrl_log.timer(f'{SOLVING} for violation check',lg.START)
                result = self.er_ctl.ctl.solve(on_model = self.er_ctl._on_model)
                self.ctrl_log.timer(f'{SOLVING} for violation check',lg.END,vstart)
                if result.satisfiable:
                    estart = self.ctrl_log.timer(f'Trace merge denial violation',lg.START)
                    self.expand_trace([self.er_ctl.it_model])
                    self.ctrl_log.timer(f'Trace merge denial violation',lg.END, estart)
                ##### add subprogram with original soft rules with choice head and hardened soft rules
        #1. determine it is a possible merge or not
            ## yes: take the stable model to ground the translated trace program to obtain an explaination of the merge
            ## no:
                ### check whether its undefined or violated
                ### 