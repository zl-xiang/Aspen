import utils
from clingo.core import MessageCode
from timeit import default_timer as timer

DEBUG = 'DEBUG'
ERROR = 'ERROR'
INFO = 'INFO'
DEBUG_C = 0
INFO_C = 1
ERROR_C = 2
LEVEL_MAP = {DEBUG_C:DEBUG, INFO_C:INFO, ERROR_C:ERROR}
START = 1
END = 0

class logger:

    # level [class] message
    TEMPLATE = '* % [%] %'
    
    def __init__(self,reg_class, level = INFO_C) -> None:
        self.reg_class = reg_class.__name__ if isinstance(reg_class,type) else reg_class
        # xclingologger attributes
        self._no_labels = None
        self._no_show_trace = None
        self.level = level
    
    def log(self,level:int,msg:str,args:list=None)-> None:
        if level == None:
            level = INFO_C
        #print(level, self.level)
        if level>=self.level:
            msg = utils.format_string(msg,args)
            level = LEVEL_MAP[level]
            msg = utils.format_string(logger.TEMPLATE,[level, self.reg_class,msg])
            print(msg)
            
    def info(self,msg:str,args:list=None) -> None:
        self.log(INFO_C,msg,args)
        
    def debug(self,msg:str,args:list=None) -> None:
        # print('debug is called !!')
        self.log(level=DEBUG_C,msg=msg,args=args)
    
    def error(self,msg:str,args:list=None) -> None:
         self.log(ERROR_C,msg,args)
    
    def timer(self,proc_name:str,state:int,start:float=None):
        if state == START:
            self.info(' % started ...', [proc_name])
            start = timer()
            return start
        else:
            self.info(' % ended ...', [proc_name])
            end = timer()
            self.info(" % ended in % seconds ....",[proc_name, end - start])

    # xclingologger callbacks
    def logger(self, _code, msg):
        """Logger TODO: more detail."""
        if _code == MessageCode.AtomUndefined:
            if "xclingo_muted" in msg:
                return
            if "_xclingo_label" in msg:
                self._no_labels = True
                return
            if "_xclingo_show_trace" in msg:
                self._no_show_trace = True
        print(msg)

    def print_messages(self):
        """Prints messages from the logger."""
        if self._no_labels:
            print("xclingo info: any atom or rule has been labelled.")
        if self._no_show_trace:
            print("xclingo info: any atom has been affected by a %!show_trace annotation.")
