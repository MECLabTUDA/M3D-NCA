from timeit import default_timer as timer
import sys
import threading
from src.utils.DatasetClassCreator import Extended_Loadable_Dataset_Annotations
from src.utils.ModelCreatorVis import Extended_Visualization_Model_Annotations
from src.utils.AgentCreatorVis import Extended_Visualization_Agent_Annotations
from typing import List, Tuple, Any, Dict, Type
from src.utils.Experiment_vis import Experiment_vis
from contextlib import contextmanager
import torch
import numpy as np
from multiprocess import Process, Pipe
from enum import Enum, auto
import uuid
from src.utils.util_enumy import *
from src.utils.GuiMultiprocessing import *

class AgentProcArgType(Enum):
    """
    Job Type for network agent. Only used internally.
    """
    RunAgent = 1,
    ReconfigureAgent = 2


class AgentProcessArgs():
    """
    Argument class for network agent. Only used for network agent and only used internally.
    """
    type: AgentProcArgType
    content: Any # for now either config object or args to run agent

    def __init__(self, content: Any, type: AgentProcArgType):
        self.type = type
        self.content = content







class ResultGetters():
    """This object allows querying for results from the Agent Process.

    Basically just wrapper around a pipe. Only used internally for convenience.
    

    """
    pipe: Any
    def __init__(self, pipe):
        """Initializes the Getter Object

        Args:
            pipe (_type_): Pipe from which this object gets it's Data
        """
        self.pipe = pipe
        
    def get_data(self)-> Tuple[bool, Dict[int, np.ndarray], str]:
        """Queries the underlying pipe for data.
        

        Returns:
            Tuple[bool, np.ndarray]: If data is available: true, net_output, filename
                                    else: false, None, None
        """
        if self.pipe.poll():
            return (True,) + self.pipe.recv()
        else:
            return (False, None, None)
        
def _handle_imports_agent():
    """
    import function for Network agent
    """
    import torch

def _handle_agent_recursive_one_shot(
    rec_arg: Tuple[
                Type[Extended_Visualization_Agent_Annotations], Dict[str, Any], 
                Type[Extended_Visualization_Agent_Annotations], 
                Dict[str, Any]
                ]
            | Tuple[Extended_Visualization_Agent_Annotations, Extended_Loadable_Dataset_Annotations],
    process_args: AgentProcessArgs):
    """Behavior for network agent. 
    One Shot --> The function does not have to loop itself and query for new job arguments, this is handled externally. 
    Handles Inference on Data as well as reconfiguration of agent.
    
    Types and args instead of actual agent objects are used for historic reasons, no longer relevant.

    """
    agent = None
    dataset = None
    if len(rec_arg) == 7:
        # Instantiation of agent, model and dataset classes
        agent_type, agent_args, dataset_type, dataset_args, model_type, model_kwargs, device_type = rec_arg
            
        device = torch.device(device_type)
        
        if isinstance(model_type, List):
            backbone_model = []
            for i in range(len(model_type)):
                model_kwargs[i]["device"] = device
                tmp = model_type[i](**model_kwargs[i])
                
                tmp.eval()
                backbone_model.append(tmp)
        else:
            model_kwargs["device"]=device
            backbone_model = model_type(**model_kwargs)

            backbone_model.eval()
        agent_args["model"]=backbone_model
        dataset = dataset_type(**dataset_args)
        agent_args["dataset"] = dataset
        agent = agent_type(**agent_args)
        
    else:
        agent, dataset = rec_arg

    if process_args.type == AgentProcArgType.RunAgent:
        args = process_args.content
        fname = args['image']
        do_variance = args['do_variance']
        num_runs = args['num_runs']
        args.pop('do_variance')
        args.pop('num_runs')
        print(f"Pushing datapoint {fname} into agent...")
        start = timer()
        if do_variance:
            s_1: Dict[int, np.ndarray] = None
            s_2: Dict[int, np.ndarray] = None
            for i in range(num_runs):

                temp = agent.get_output_for_image_monitored(**args)
                if s_1 is None:
                    s_1 = {}
                    s_2 = {}
                    a = list(temp.keys())
                    for i in a:
                        s_1[i] = np.zeros(shape=temp[i].shape)
                        s_2[i] = np.zeros(shape=temp[i].shape)
                    for i in a:
                        s_1[i] += temp[i]
                        s_2[i] += np.power(temp[i], 2)
                else:
                    for i in a:
                        s_1[i] += temp[i]
                        s_2[i] += np.power(temp[i], 2)
            results = {}
            for i in list(s_1.keys()):

                results[i] = s_2[i]*(1/num_runs) - np.power(s_1[i]*(1/num_runs), 2)

        else:    
            results = agent.get_output_for_image_monitored(**args)
        end = timer()
        print(f"Agent finished after: {end - start}")
        return ((agent, dataset), (AgentProcArgType.RunAgent, results, fname))
    elif process_args.type == AgentProcArgType.ReconfigureAgent:
        agent_args = process_args.content
        agent_args["dataset"] = dataset
        agent = agent_type(**agent_args)
        print(f"reconfigured agent", flush=True)
        sys.stdout.flush()
        return ((agent, dataset), (AgentProcArgType.ReconfigureAgent,))
            
    


                


class MultiThreadingFunctionWrapper(QThread):
    """
    Utility class used to run one-shot function in group. 
    Handles appropriate signalling for termination of operation in Group.
    """
    exec_function: Callable[[], Any]
    group: str
    recv: GuiMPReceiver
    cores: List[int] = None
    uuid: Any
    pain = None
    kwargs: Dict[str, Any] = {}
    def __init__(self, exec_function: Callable[[], Any]|Callable[[List[int]], Any], group: str, receiver: GuiMPReceiver,  cores: List[int] = None, kwargs: Dict[str, Any]={}):
        super().__init__()
        self.exec_function = exec_function
        self.group = group
        self.recv = receiver
        self.cores = cores
        self.uuid = uuid.uuid4()
        self.kwargs = kwargs
        self.recv._add_semi_orphaned_thread(uuid=self.uuid, multi_threading_function_wrapper=self)
        # this is necessary
        # otherwise thread will be collected by garbage collection during computation
    def run(self):
        data = self.exec_function(**self.kwargs)
        self.recv.call_group_signal(group_name=self.group, data=data, dummy_process_name=self.group, job_id=uuid)




class DataFlowHandler():
    """
    Handles Data flow for the whole application. 
    Handles: 
    - Dispatching jobs to the NCA Agent
    - Dispatching Multiprocessing Jobs and notifying associated listeners
    - Dataset Interaction for associated widgets.

    We have decided to handle everything centrally through the Dataflow handler. This was the better decision design wise. 
    Communication between individual widgets should also be handled over the DataFlowHandler through dedicated groups.
    
    """
    dataset: Extended_Loadable_Dataset_Annotations
    agent: Extended_Visualization_Agent_Annotations
    dtype: Type[Extended_Loadable_Dataset_Annotations]
    a_type: Type[Extended_Visualization_Agent_Annotations]
    d_cons: Dict[str, Any]
    a_cons: Dict[str, Any]
    arg_send: Any
    gui_running_processes: Dict[str, GuiJobDispatcher] = {}
    gui_receive_handler: GuiMPReceiver = None
    network_group_id: str = None
    current_data: Dict[int, np.ndarray] = None
    # holds currently displayed working data
    stashed_data: Dict[int, np.ndarray] = None
    # can optionally hold data stashed for later use
    network_process_id: str = None
    model_type: Type[Extended_Visualization_Model_Annotations]|List[Type[Extended_Visualization_Model_Annotations]]
    model_kwargs: Dict[str, Any]|List[Dict[str, Any]]
    device_type: str

    

    data_flow_data_changed_group: str = "data_changed_group"
    def __init__(self, dataset_constructor_arguments: Dict[str, Any], agent_constructor_arguments: Dict[str, Any], 
                 dataset_type: Type[Extended_Loadable_Dataset_Annotations], vis_agent_type: Type[Extended_Visualization_Agent_Annotations], 
                 vis_model_type: Type[Extended_Visualization_Model_Annotations]|List[Type[Extended_Visualization_Model_Annotations]], vis_model_kwargs: Dict[str, Any]|List[Dict[str, Any]], device_type: str):
        """Initializes the Dataflow Handler. Here types and args are used instead of actual objects for historical reasons.

        Args:
            dataset_constructor_arguments (Dict[str, Any]): Agruments for the Dataset Constructor
            agent_constructor_arguments (Dict[str, Any]): Arguments for the constructor for the Agent
                do not include the dataset in this set of arguments
            dataset_type (Type[Extended_Loadable_Dataset_Annotations]): Type of the Dataset
            vis_agent_type (Type[Extended_Visualization_Agent_Annotations]): Type of the Agent
        """
        self.dtype = dataset_type
        self.d_cons = dataset_constructor_arguments
        self.a_cons = agent_constructor_arguments
        self.a_type = vis_agent_type
        self.model_type = vis_model_type
        self.model_kwargs = vis_model_kwargs
        self.device_type = device_type
        self.dataset = self.dtype(**self.d_cons)

        self.create_group(self.data_flow_data_changed_group)
    
    def get_current_data(self) -> Dict[int, np.ndarray]:
        """
        Returns data currently visualized by all widgets that subscribe to the network output.
        """
        return self.current_data
    
    def _handle_network_data_generated(self, data: Tuple[Any, Dict[str, Tuple[AgentProcArgType]|Tuple[AgentProcArgType, Dict[int, np.ndarray], str]]]):    
            """
            Helper class, due to later decision: Originally, Network group was used to listen to only Network output. Since then, other means 
            of changing the globally displayed data were introduced. These are handled through the network group for compatibility with existing code. 
            This necessitated the introduction of an intermediate layer between network group and actual network output. 
            """    
            _, data = data
            keys = list(data.keys())
            my_data = data[keys[0]]
            if len(my_data)==1:
                return
            _, data, fname = my_data
            self.current_data = data
            self.stashed_data = None
            self.call_group_signal(self.data_flow_data_changed_group, data=(FlowDataChangeType.NETWORK, my_data), dummy_process_name="data_stash_process", job_id=uuid.uuid4())

    @contextmanager
    def with_network(self, terminate_handler_on_leave: bool = False) -> Tuple[Extended_Loadable_Dataset_Annotations, List[ResultGetters]]:
        """Contextmanager that allows interaction with an Agent in a seperate process. 
        This uses the newer multiprocessing backend. Handler functions can be registered over a seperate method.
        args:
            terminate_on_leave: Sets whether the contextmanager terminates the handler that manages the 
                process returns upon leaving this context. Multiple terminations of the handler do not cause error.
        """

        if self.network_group_id is None:
            self.network_group_id = str(uuid.uuid4())
            self.network_process_id = str(uuid.uuid4())

        process_handler = GuiJobDispatcher(do_function=_handle_agent_recursive_one_shot, import_func=_handle_imports_agent, 
                                                                  use_recursive_arguments=True, initial_recursive_argument=(self.a_type, self.a_cons, self.dtype, self.d_cons, self.model_type, self.model_kwargs, self.device_type), core=0)

        self.gui_running_processes[self.network_process_id] = process_handler
        if self.gui_receive_handler is None:
            self.gui_receive_handler = GuiMPReceiver()
            self.gui_receive_handler.start()
        if not self.gui_receive_handler.has_group(self.network_group_id):
            self.gui_receive_handler.create_group(self.network_group_id)
        self.gui_receive_handler._add_communicating_process(process_id=self.network_process_id, process_result_pipe=process_handler.result_pipe, group_name=self.network_group_id)
        self.add_handler_function(self._handle_network_data_generated, group=self.network_group_id)
        try:
            yield
        finally:
            process_handler.kill()
            if terminate_handler_on_leave:
                self.terminate_mp_handler()

    def cleanup(self):
        if not self.gui_receive_handler is None:
            pass #TODO: implement graceful termination of MP thread

        
    def add_network_handler(self, func:  Callable[[Tuple[Any, Dict[str, Tuple[AgentProcArgType]|Tuple[AgentProcArgType, Dict[int, np.ndarray], str]]]], None]):
        """Connects function to the signal that signals the completion of a network job. 
        Signal is triggered after a computation as well as after a network reconfiguration. 
        The Handeler function is passed the normal data structure for a group job, 
        however necessarily in this context there only exists one process per group. 
        The process return is either the appropiate AgentProcAgrType for a network reset 
        or a tuple consisting of AgentProcTypeArg, network result and filename.
        ------------
        There are also other events that result in new display data which are anounced through this group. In this case, the resulting data has a slightly different format. 
        Examples of how this is handled can be found in the existing codebase.
        """
        if self.gui_receive_handler is None:
            raise Exception("Cannot add handler function before any processes were created.")
        if self.network_group_id is None:
            raise Exception("Handler function for network can only be attached if a network has previously been initialized.")
        self.gui_receive_handler.connect_to_finished(func, group=self.data_flow_data_changed_group)

    def remove_network_handler(self, func:  Callable[[Tuple[Any, Dict[str, Tuple[AgentProcArgType]|Tuple[AgentProcArgType, Dict[int, np.ndarray], str]]]], None]):
        if self.gui_receive_handler is None:
            raise Exception("Cannot remove handler function before any processes were created.")
        if self.network_group_id is None:
            raise Exception("Handler function for network can only be removed if a network has previously been initialized.")
        self.gui_receive_handler.disconnect_handle(func, group=self.data_flow_data_changed_group)

            
    def reload_dataset(self, dataset_constructor_arguments: Dict[str, Any], dtype: Type[Extended_Loadable_Dataset_Annotations] = None):
        """
        Method used to load in a new dataset. Notation with args + Type for consistency with the Dataset format required for the network process.
        Does not reload the dataset in the network process.
        """
        if not dtype is None:
            self.dtype = dtype
        self.d_cons = dataset_constructor_arguments
        self.dataset = self.dtype(**self.d_cons)

    def slices_per_datapoint(self, filename: str) -> int:
        """Returns number of slices per datapoint. Uses the implementation 
        provided in the dataset base classes. Should be able to handle flexible slice number.
        Returns:
            int: number of slices per datapoint
        """

        return len(self.dataset.get_dataset_index_information()[filename])
        

    def get_image_for_filename(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns image for given Filename. Always returns the whole Image, even when the config specifies a slice axis.

        Args:
            filename (str): Name of file

        Returns:
            Tuple[np.ndarray, np.ndarray]: input Data used by the network, target label
        """
        return self.dataset.get_source_image_for_filename(filename)
    

    def does_network_receive_3d(self) -> bool:
        """Returns whether the network receives 3D data, so whether slicing is just for optical purposes.
        """
        return self.dataset.does_network_receive_3d()

    
    def _set_current_data(self, data: Dict[int, np.ndarray], event_data: FlowDataChangeType = FlowDataChangeType.ABSDIFF):
        """Internal method. Used to set the current working dataset. 
        Stashed data is deleted.
        
        """
        self.current_data = data
        self.stashed_data = None
        self.call_group_signal(self.data_flow_data_changed_group, data=(event_data, self.get_current_data()), dummy_process_name="data_stash_process", job_id=uuid.uuid4())


    def _add_stashed_data(self, data: Dict[int, np.ndarray]):
        self.stashed_data = data

    def switch_current_stash(self):
        """Switches current data to stashed Data. 
        
        Generates a Signal in the NEtwork group. For all subscribing processes this looks like new data was generated by the network. 
        """
        temp = self.current_data
        self.current_data = self.stashed_data
        self.stashed_data = temp
        self.call_group_signal(self.data_flow_data_changed_group, data=(FlowDataChangeType.ABSDIFF, self.get_current_data()), dummy_process_name="data_stash_process", job_id=uuid.uuid4())

    def has_stashed(self) -> bool:
        return self.stashed_data is not None

    def get_filenames_in_dataset(self) -> Tuple[List[Tuple[str, int, int]], List[str]]:
        """Returns Internal list of images which the dataset consists


        Returns:
            _type_: List[Tuple[a, b, c]]: a-> str, name of file
                                    b-> int, patientID
                                    c->int = n --> this is the n-th datapoint for the patient
                                    For 3D Data this is always 0.
                                    The proper ID that is used by the Dataset can be calculated as 
                                    (b+c). Otherwise, the index in the list is also fine
                    List[str]: just the list of the file names
        """
        image_names = list(set([x[0] for x in self.dataset.images_list]))
        return (self.dataset.images_list, image_names)
    
    
    def process_image(self, image: np.ndarray, filename: str, do_variance: bool = False, num_runs: int=5, slice_number: int = None):
        """Processes Image in seperate Agent process. This method can only be called if in the with_network context manager.
        A call is placed to the corresponding Network Agent process. This method doesn't block.

        Args:
            image (np.ndarray): Alternative image over which the Input is to be computed
            filename (str): Filename of the Image for which the output is to be computed
            slice_number: if the network processes individual slices of the data file, this option specifies which slice.
        """

        if self.network_group_id is None or self.network_process_id is None:
            raise Exception("Cannot dispatch image to nonexistent network process.")
        args_tuple = {"image": filename, "slice_number": slice_number, "output_path": "", "altered_input": image, "save_images": False, "do_variance": do_variance, "num_runs": num_runs}
        agent_proc_args = AgentProcessArgs(args_tuple, AgentProcArgType.RunAgent)
        job_arguments= {"process_args": agent_proc_args}
        job_params = {self.network_process_id: job_arguments}
        self.dispatch_job(job_parameters=job_params, group=self.network_group_id)

    def reconfigure_agent(self, agent_args):
        """
        Dispatches a job to the agent process which prompts it to reconfigure itself. Mainly used in the configuration editor. Should be used sparingly elsewhere.
        """
        if self.network_group_id is None or self.network_process_id is None:
            raise Exception("Cannot dispatch image to nonexistent network process.")
        agent_proc_args = AgentProcessArgs(agent_args, AgentProcArgType.ReconfigureAgent)
        job_arguments= {"process_args": agent_proc_args}
        job_params = {self.network_process_id: job_arguments}
        self.dispatch_job(job_parameters=job_params, group=self.network_group_id)

    @contextmanager
    def with_process(self, import_function: Callable, action_function: Callable, process_id: str, core: int = 1,
                     use_recursive_argument: bool = False, initial_recursive_argumet: Any = None, group: str ="default_group", terminate_handler_on_leave: bool = False):
        """import_function and action_function have to be a top level Function, requirement by the pickler used.
        Imports have to be entirely handled by the importFunction. If this is not done, 
        multiprocessing might be reduced to two cores wich significantly hinders the performance of the software.
        action_function should not loop on its own, but rather only do one single computation. Passing of aguments 
        and results should not be handled by this function. 
        if initial_recursive_argument is set to true, the first element returned to the function will 
        be passed back into the fucntion on the next call as the first parameter. 
        action_function should ideally be viewed as a pure function. Any continuous state should be handled 
        through the recursive argument.

    Args:
    import_function: Handles the imports
    action_function: handles the actual computation
    process_id: str: String under which the process can be adressed in the future.
    core: sets cpu affinity, currently ineffective
    recursive_argument: Sets whether a recursive argument should be used. 
    initial_recursive_agument: Initial Value for the recursive Argument.
    group: Group to which the process is associated. Each group is associated with its own event.
    terminate_on_leave: Sets whether the contextmanager terminates the handler that manages the 
        process returns upon leaving this context. Multiple terminations of the handler do not cause error.
    """
        process_handler = GuiJobDispatcher(do_function=action_function, import_func=import_function, 
                                                                  use_recursive_arguments=use_recursive_argument, initial_recursive_argument=initial_recursive_argumet, core=core)

        self.gui_running_processes[process_id] = process_handler
        if self.gui_receive_handler is None:
            self.gui_receive_handler = GuiMPReceiver()
            self.gui_receive_handler.start()
        if not self.gui_receive_handler.has_group(group):
            self.gui_receive_handler.create_group(group)
        self.gui_receive_handler._add_communicating_process(process_id=process_id, process_result_pipe=process_handler.result_pipe, group_name=group)
        try:
            yield
        finally:
            process_handler.kill()
            if terminate_handler_on_leave:
                self.terminate_mp_handler()


    def _get_uuid(self):
        return uuid.uuid4()
    

    def dispatch_job(self, job_parameters: Dict[str, Dict[str, Any]], group: str = "default_group", foreign_job_id: Any = None) -> Any:
        """Dispatches a set of individual jobs to multiple processes. 
        The job as such only returns when all individual jobs inside this job return. 
        Each individual job is computed on its own process.

        job_parameters: Dict mapping process_id (name given by the user) to 
        Keyword arguments for central function inside the process.
        returns the job_id
        group: group to which the job is dispatched
        """
        if foreign_job_id is not None:
            job_id = foreign_job_id
        else:
            job_id = self._get_uuid()
        jobs = list(job_parameters.keys())
        for p_id in jobs:
            if p_id not in self.gui_running_processes:
                raise Exception("Tried to dispatch job to nonexistent process.")
    
        self.gui_receive_handler._register_job(job_id, jobs, group)
        for job in jobs:
            dispatcher = self.gui_running_processes[job]
            dispatcher.dispatch_job(job_parameters[job], job_id)
        return job_id
    
    def update_recursive_parameters(self, process_id: str, new_recursive_argument: Any, group: str = "default_group") -> Any:
        """updates the current recursive Argument for the given process. 
        Recursive parameters are used to hold the state of a given process. An exmple of how this can be implemented is shown in the 
        Network handler function at the beginning of this file.

        job_parameters: Dict mapping process_id (given by user) to 
        Keyword arguments for central function inside the process.
        return the job_id
        group: group to which the job is dispatched
        """
        
        job_id = self._get_uuid()
        if process_id not in self.gui_running_processes:
            raise Exception("Tried to dispatch job to nonexistent process.")

        dispatcher = self.gui_running_processes[process_id]
        dispatcher.dispatch_job({}, job_id=job_id, new_recursive_argument=new_recursive_argument, 
                                job_type=JobType.UPDATE_RECURSIVE_ARGUMENTS)


    def create_group(self, group: str):
        """
        Creates a communication group of the given name. 
        Listeners an be subscribed to a group.
        """
        if self.gui_receive_handler is None:
            self.gui_receive_handler = GuiMPReceiver()
            self.gui_receive_handler.start()
        self.gui_receive_handler.create_group(group_name=group)


    def exist_group(self, group: str) -> bool:
        if self.gui_receive_handler is None:
            self.gui_receive_handler = GuiMPReceiver()
            self.gui_receive_handler.start()
            return False
        return self.gui_receive_handler.has_group(group_name=group)
    
    def call_group_signal(self, group: str, data: Any, dummy_process_name: str, job_id: Any) -> bool:
        """
        Function used to explicitly call a Signal in a given group. 
        Since groups are primarily used to handle communication for multiprocessing, 
        dummy name for process is required. job_id should be a UUID.
        """
        if self.gui_receive_handler is None:
            self.gui_receive_handler = GuiMPReceiver()
            self.gui_receive_handler.run()
        if not self.gui_receive_handler.has_group(group):
            self.gui_receive_handler.create_group(group)
        self.gui_receive_handler.call_group_signal(group_name=group, data=data, dummy_process_name=dummy_process_name, job_id=job_id)

    def terminate_mp_handler(self) -> bool:
        """This method terminates the multiprocessing handler. 
        This method should only be called to terminate all running processes. 
        Afterwards process returns will not be able to be handled anymore.        
        """
        if self.gui_receive_handler is None:
            return True
        self.gui_receive_handler.set_running(False)
        self.gui_receive_handler.quit()
        self.gui_receive_handler.wait()
        self.gui_receive_handler = None
        return True
        


    def add_handler_function(self, func: Callable[[Any, Dict[str, Any]], None], group: str = "default_group"):
        """Connects function to the internal Multiprocessing Handler that then receives result of the returning 
        jobs.
        args: func: This function is called any time a job returns. It receives a tuple consisting of the 
        job id and a dict mapping process_id to returned result. This function runs in the main Thread, 
        so it can be used to interact with GUI elements.
        """
        if self.gui_receive_handler is None:
            self.gui_receive_handler = GuiMPReceiver()
            self.gui_receive_handler.run()
        if not self.gui_receive_handler.has_group(group):
            self.gui_receive_handler.create_group(group)
        self.gui_receive_handler.connect_to_finished(func, group=group)

    def run_function_in_group(self, func: Callable[[List[int]], Any]|Callable[[], Any], group: str = "default_group", cores: List[int] = None, kwargs={}):
        """Executes the given function in a seperate Thread and then distributes the result using 
        the signal corresponding to the given group. If a group with the given name does not exist, 
        a corresponding group is created. 
        The contents of the signal are Tuple[random_uuid, {group_name: Return of the Function}]
        (this is to ensure compatibility as group Signals are intended to be used for Returning the results 
        of multiprocessing jobs that utilize multiple processes.)

     
        """
        if self.gui_receive_handler is None:
            self.gui_receive_handler = GuiMPReceiver()
            self.gui_receive_handler.run()
        if not self.gui_receive_handler.has_group(group):
            self.gui_receive_handler.create_group(group)
        w = MultiThreadingFunctionWrapper(exec_function=func, group=group, receiver=self.gui_receive_handler, cores=cores, kwargs=kwargs)
        w.start()



