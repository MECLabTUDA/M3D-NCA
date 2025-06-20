from typing import Dict, Any, List, Tuple, Callable
from src.utils.util_enumy import JobType
import uuid
from qtpy.QtCore import QThread, QMutex, Signal, QObject
from multiprocess import Process, Pipe
from functools import partial
import psutil
import time

import sys
import pdb
"""
This file contains multiprocessing utility used in the Dataflow handler. 
We mostly use this because we were not able to get process pools working reliably under windows. 
Classes contained here should only be accessed indirectly through the dataflowHandler. Should not be used explicitly by any widgets.

"""


class GuiMPReceiver(QThread, QObject):
    """Class that works on a seperate Thread and is intended to Handle Communication to and Synchronization 
    with External Computation Processes.
    One Receiver <-> Many processes. There is no reason for more than one instance of this class to be active.
    This class is very threadsave-> future proofing for eventual removal of the GIL in later python versions.
    """
    sending_processes: Dict[str, Pipe]={}
    processes_mutex: QMutex = QMutex()
    # contains mapping from process ids to pipes that can receive data from those processes
    # process has to send TUple of job id and return Value.

    pending_ids: Dict[Any, Tuple[str, List[str]]] = {}
    pending_mutex: QMutex = QMutex()
    # Dict that maps job id to Tuple consisting of Group ID and list of participating processes.
    
    
    process_returns: Dict[Any, Dict[str, Any]] = {}
    return_mutex: QMutex = QMutex()
    # Contains all returns assoicated with dispatched jobs. 
    # maps job uuid to a dict that matches process id to return.
    # values should be deleted after all processes that compute a result for 
    # a job have returned.


    process_groups: Dict[str, List[str]] = {"default_group": []}
    pg_mutex: QMutex = QMutex()
    # dictionary for groups. Groups can be used to seperate sets of processes that belong to a group. 
    # Each group has its own Signal. 
    # default group is used by default
    # (process_ids, bool) -> Second part sets whether a Group is temporary.
    #Signals are now dynamically created attributes.
    jfs_mutex: QMutex = QMutex()
    # Signal that is triggered when a job completes:
    # One Signal per group.
    
    return_mutex: QMutex = QMutex()

    running: bool = True
    running_mutex: QMutex = QMutex()

    semi_orphaned_threads: Dict[Any, Any] = {}
    sot_mutex: QMutex = QMutex()
    # Dictionary mapping uuid to MultiThreadingFunctionWrapper objects. References 
    # to those objects must be kept, otherwise the running thread gets deleted which leads to errors. 

    def __init__(self):
        super().__init__()
        self.sending_processes = {}
        self.pending_ids = {}
        self.process_returns = {}
        self._add_signal("default_group", Tuple[Any, Dict[str, Any]])

    def _add_semi_orphaned_thread(self, uuid: Any, multi_threading_function_wrapper: Any):
        """
        Should only be used by Dataflowhandler. 
        Adds multi_threading_function_wrapper to watchlist. 
        Threads need to be stored & discarded in the future to prevent deletion by garbage collection.
        """
        self.sot_mutex.lock()
        self.semi_orphaned_threads[uuid] = multi_threading_function_wrapper
        self.sot_mutex.unlock()

    def _kill_semi_orphaned_thread(self, uuid: Any):
        """
        Removes Thread object from local storage. 
        """
        self.sot_mutex.lock()
        if uuid in self.semi_orphaned_threads:
            self.semi_orphaned_threads.pop(uuid)
        self.sot_mutex.unlock()

    def set_running(self, running: bool):
        """Threadsave method that sets running. 
        If running is set to false the process loop is terminated at the beginning of its next iteration.
        Kills its own thread. 
        """
        self.running_mutex.lock()
        self.running = running
        self.running_mutex.unlock()
        
    def _add_communicating_process(self, process_id: str, process_result_pipe: Pipe, group_name: str = "default_group"):
        """
        Adds communicating Process to the internal list of processes. 
        Data can now be received from this process. The given process_id 
        can be used to adress the process. Processes are added to the "default" group if not specified otherwise. 
        If processes should be added to other groups the group has to be explicitly created beforehand.
        args:
            process_id: Process id used to reference the process in the future
            process_pipe: Pipe used to receive results from the process.
        """
        valid_group = False
        try:
            self.pg_mutex.lock()
            if group_name in self.process_groups:
                valid_group = True
        finally:
            self.pg_mutex.unlock()
        if not valid_group:
            raise Exception("Tried to add process to invalid Group")

        self.processes_mutex.lock()
        self.sending_processes[process_id] = process_result_pipe
        self.processes_mutex.unlock()

        self.pg_mutex.lock()
        self.process_groups[group_name].append(process_id)
        self.pg_mutex.unlock()

    def _add_signal(self, name, args):
        """Adds signal attribute to self.
        This needs to be done because Signals seem to 
        be class attributes. More elegant solutions do not work.
        Adding signals with names like __equal___ or __init__ is going to break things.
        """
        cls = self.__class__
        new_cls = type(
            cls.__name__, cls.__bases__,
            {**cls.__dict__, name: Signal(object)},
        )
        self.__class__ = new_cls

    def _has_signal(self, name: str):
        """Not ideal. Probably should also do type checking here. Unlikely to cause problems in practice since UUIDs should be used for signal names."""
        return hasattr(self, name)

    def _remove_communicating_process(self, process_id: str, group: str = "default_group"):
        """
        Removes a process from the list of associated process. 
        Does not remove pending jobs which include the removed proces. 
        Depending on size of process returns this might take up significant memory. 
     
        """
        self.processes_mutex.lock()
        if process_id in self.sending_processes:
            self.sending_processes[process_id] = None
        self.processes_mutex.unlock()

        self.pg_mutex.lock()
        self.process_groups[group].remove(process_id)
        self.pg_mutex.unlock()


    def create_group(self, group_name: str):
        """
        Creates a group with the given name. 
        Processes can be added to this group and handlers registered.
        """
        g_names = []
        self.pg_mutex.lock()
        g_names = list(self.process_groups.keys())
        self.pg_mutex.unlock()
        if group_name in g_names:
            raise Exception("Can't create group where name already exists.")


        self.pg_mutex.lock()
        self.process_groups[group_name] = []
        self.pg_mutex.unlock()


        self.jfs_mutex.lock()
        self._add_signal(group_name, Tuple[Any, Dict[str, Any]])
        self.jfs_mutex.unlock()



    def has_group(self, group_name: str):
        """Returns whether the Group already exists.
        
        """
        g_names = []
        self.pg_mutex.lock()
        g_names = list(self.process_groups.keys())
        self.pg_mutex.unlock()
        return group_name in g_names
    

    def call_group_signal(self, group_name: str, data: Any, dummy_process_name: str, job_id: Any):
        """Makes a call to the Signal associated with the given group name. 

        dummy_process_name used for consistency.
        used for internal purposes, do not use directly
        """
        
        g_names = []
        self.pg_mutex.lock()
        g_names = list(self.process_groups.keys())
        self.pg_mutex.unlock()
        if group_name not in g_names:
            raise Exception("Can't make call to nonexistent group.")
        group_return = (job_id, {dummy_process_name: data})
        
        self.jfs_mutex.lock()
        
        try:
            sig = getattr(self, group_name)
            sig.emit(group_return)
        finally:
            self.jfs_mutex.unlock()
        
    def _register_job(self, job_id: Any, processes: List[str], group: str = "default_group"):
        """
        Registers a new job. Multiple processes can be signed up for a job and the job will only return 
        if all processes associated with a job have returned. All processes that are registered for a job need to be from the same group.
        Originally this was intended to synchronize the return of several computational GUI processes. Would have used process pool, but this does not work 
        reliably under windows.
        ONLY FOR INTERNAL USE.
        """

        raise_err = False
        self.pending_mutex.lock()
        for p_id in processes:
            if p_id not in self.sending_processes:
                raise_err = True
                break
        if not raise_err:
            self.pending_mutex.unlock()
            self.pg_mutex.lock()
            if not set(processes) <= set(self.process_groups[group]):
                
                raise Exception("Cannot dispatch a single job to processes from multiple groups")
            self.pg_mutex.unlock()
            self.pending_mutex.lock()
            self.pending_ids[job_id] = (group, processes)
            self.pending_mutex.unlock()
            
            self.return_mutex.lock()
            self.process_returns[job_id] = {}
            self.return_mutex.unlock()
        else:    
            self.pending_mutex.unlock()
            raise Exception("All requested process ids have to be registered processes.")
    

    def connect_to_finished(self, handle_done: Callable[[Any, Dict[str, Any]], None], group: str = "default_group"):
        """Connects Function to the event that gets triggered when a job is finished. 
        This function is called for any finished job from the corresponding group. 
        It has to take the job uuid 
        as well as a dictionary that maps process id to return type as an argument.
        
        """
        self.jfs_mutex.lock()
        try:
            sig = getattr(self, group)
            sig.connect(handle_done)
        finally:
            self.jfs_mutex.unlock()

    def disconnect_handle(self, handle: Callable[[Any, Dict[str, Any]], None], group: str = "default_group"):
        """
        Attempts to remove callback function from given group.
        """
        self.jfs_mutex.lock()
        try:
            sig = getattr(self, group)
            sig.disconnect(handle)
            print(f"Successfully removed handle: {handle.__qualname__}")
        finally:
            self.jfs_mutex.unlock()

    def run(self):
        """
        Manages jobs: return Data for incomplete jobs is cached until every process associated with the job has returned. 
        Then all listeners will be informed. 
        Also handles management of stashed threads.
        """
        while self.running:
            time.sleep(0.5)
            self.processes_mutex.lock()
            try:
                active_processes = list(self.sending_processes.keys()).copy()
            finally:
                self.processes_mutex.unlock()
                self.sot_mutex.lock()
                try:
                    sots = list(self.semi_orphaned_threads.keys()).copy()
                finally:
                    self.sot_mutex.unlock()

                for sot in sots:
                    dispose = False
                    self.sot_mutex.lock()
                    if self.semi_orphaned_threads[sot].isFinished():
                        
                        self.semi_orphaned_threads[sot].exit()
                        dispose = True
                    self.sot_mutex.unlock()
                    if dispose:
                        self._kill_semi_orphaned_thread(sot)
            for proc_id in active_processes:
                if self.sending_processes[proc_id].poll():
                    job_id, group, ret = self.sending_processes[proc_id].recv()
                    valid_job_id = False
                    self.pending_mutex.lock()
                    try:
                        if job_id in list(self.pending_ids.keys()):
                            valid_job_id = True
                    finally:
                        self.pending_mutex.unlock()
                    if valid_job_id:
                        self.return_mutex.lock()
                        try:
                            self.process_returns[job_id][proc_id] = ret
                        finally:
                            self.return_mutex.unlock()
                        job_workers = []
                        self.pending_mutex.lock()
                        try:
                            job_workers = self.pending_ids[job_id][1].copy()
                        finally:

                            self.pending_mutex.unlock()

                        self.return_mutex.lock()
                        do_return = False
                        try:
                            if all(job_worker in list(self.process_returns[job_id].keys()) for job_worker in job_workers):
                                do_return = True
                        finally: 
                            self.return_mutex.unlock()

                        if do_return:
                            self.pending_mutex.lock()
                            group = "default_group"
                            try:
                                group, _ = self.pending_ids[job_id]
                                self.pending_ids.pop(job_id)
                            finally:
                                self.pending_mutex.unlock()

                            self.return_mutex.lock()
                            try:
                                result_dict = self.process_returns[job_id]
                                self.process_returns.pop(job_id)
                                
                                sig = getattr(self, group)
                                
                                sig.emit((job_id, result_dict))
                            finally:
                                self.return_mutex.unlock()



def dumy_func(args):
    return None
def dummy_imports():
    return None
def guiAgentSkeletonFunction(argument_pipe: Pipe, return_pipe: Pipe, exec_function: Callable, import_function: Callable, core: int,  recursive_argument:bool = False, initial_recursive_value: Any = None, group: str = "default_group"):
    """Skeleton function wrapped around a computation function for multiprocessing that handles communication infrastructure with the main thread. 
    should NEVER be used manually. Is used in the dataflowhandler.

    Args:
    argument_pipe: Pipe through which Keyword Arguments are received: 
    Arguments are received as a tuple containing the job_id as first place:
    Tuple[Any,     Dict[str, Any]]
          job_id   Keyword Args
    retur_pipe: Pipe through which the returns of the function are sent
    recursive_argument: Sets whether a recursive argument should be used. 
    initial_recursive_agument: Initial Value for the recursive Argument.
    If a recursive argument is used, the exec_fuction is expected to return a recursive_argument as 
    well as the actual return in a 2-Tuple. recursive_arguments are used to keep state between individual executions 
    of the exec_function. looping and communication is handled outside the exec_function in order to hide details 
    about inter-process communication from the programmer.
    """
    
    importFunction: Callable = import_function
    centralFunction: Callable = exec_function
    importFunction()
    p=psutil.Process()
    import multiprocessing
    from src.utils.util_enumy import JobType
    p.cpu_affinity(set(range(multiprocessing.cpu_count())))
    rec_arg = initial_recursive_value
    ret_val = None
    job_id = None
    while True:
        if argument_pipe.poll(timeout=0.001):

            job_id, process_args, job_type,  new_rec_arg = argument_pipe.recv()
            if job_type == JobType.UPDATE_RECURSIVE_ARGUMENTS:
                rec_arg = new_rec_arg
            elif job_type == JobType.DISPATCH_JOB:
                if recursive_argument:
                    rec_arg, ret_val = centralFunction(rec_arg, **process_args)
                else:
                    ret_val = centralFunction(**process_args)
                
                return_pipe.send((job_id, group, ret_val))                
                        

        


class GuiJobDispatcher():
    """
    Class that is used to dispatch GUI Jobs. This class is intended to work 
    with a GuiMPReceiver. 
    Should only be used by the dataflowhandler and never manually. 
    Uses the guiAgentskeletonFunction to handle communication to main thread for wrapped function to be executed in another process. 
    
    """
    _arg_send_pipe: Pipe = None
    _result_get_pipe: Pipe = None
    _proc: Process
    def __init__(self, do_function: Callable, import_func: Callable, core: int, use_recursive_arguments: bool = False, initial_recursive_argument = None, group: str = "default_group"):
        the_doing = partial(guiAgentSkeletonFunction, exec_function=do_function, import_function=import_func, group=group, core=core)
        kwargs = {}
        arg_recv, arg_send = Pipe()
        res_recv, res_send = Pipe()
        self._arg_send_pipe = arg_send
        self._result_get_pipe = res_recv
        if not initial_recursive_argument is None:
            kwargs = {"argument_pipe": arg_recv, "return_pipe": res_send, "recursive_argument": use_recursive_arguments, "initial_recursive_value": initial_recursive_argument}
        else:
            kwargs = {"argument_pipe": arg_recv, "return_pipe": res_send, "recursive_argument": use_recursive_arguments}
        self._proc = Process(target=the_doing, kwargs=kwargs)
        self._proc.start()

    @property
    def argument_pipe(self) -> Pipe:
        return self._arg_send_pipe
    
    @property
    def result_pipe(self) -> Pipe:
        return self._result_get_pipe
    def dispatch_job(self, kwargs: Dict[str, Any], job_id: Any, new_recursive_argument: Any = None, job_type: JobType = JobType.DISPATCH_JOB):
        """
        Dispatches a job to the associated process.
        kwargs: arguments with which the associated function is called.
        job_id: uuid of the job
        """
        self._arg_send_pipe.send((job_id, kwargs, job_type, new_recursive_argument))


    def kill(self):
        self._proc.terminate()
        self._proc.join()
        self._arg_send_pipe = None
        self._result_get_pipe = None



