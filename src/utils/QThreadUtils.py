from qtpy.QtCore import QRunnable, QThreadPool, QObject, Signal, Slot
from typing import Any, Callable
from multiprocess.pool import Pool
class Signals(QObject):
    started = Signal(int)
    completed = Signal(object)

"""
Deprecated. Still sometimes referenced in some classes, but should not be used

"""
class WorkerThread(QRunnable):
    exec: Callable[[], Any] 
    signals: Signals
    def __init__(self, exec: Callable[[], Any]):
        super().__init__()
        self.exec = exec
        self.signals = Signals()

    @Slot()
    def run(self):
        self.signals.started.emit(0)
        res = self.exec()
        print("COMPLETED", flush=True)
        self.signals.completed.emit(res)

    
class GuiComputationDispatcher():
    mp_pool = None

    @classmethod
    def dispatch_gui_computation(cls, computation: Callable[[], Any], doOnCOmplete: Callable[[Any], None] = None, doOnStart: Callable[[], None] = None):
        """Dispatches Computation to Thread Pool. This is not intended for very heavy computation as the underlying implementation 
        uses threads which prohibits efficient parallelization due to the global interpretor lock.
        Args:
            computation: Computation to be executed. returns some result
            doOnCOmplete: Function that is triggered by completed Computation. Is Called with the Result of the Computation 
            Function as Parameter. Typically used to update GUI with Computation results.
            doOnStart: Function that is Called when the Thread is dispatched
        """
        pool = QThreadPool.globalInstance()
        worker = WorkerThread(computation)
        if not doOnCOmplete is None:
            worker.signals.completed.connect(doOnCOmplete)
        if not doOnStart is None:
            worker.signals.started.connect(doOnStart)
        pool.start(worker)


    @classmethod
    def setup_mp(cls, num_proc: int = None):
        if cls.mp_pool is None:
            cls.mp_pool = Pool(processes=num_proc)
    @classmethod
    def dispatch_comp_mp(cls, computation: Callable[[], Any], doOnComplete: Callable[[Any], None]=None):
        """
        Dispatches Computation to process.
        DoOnComplete is called with the result
        """
        cls.setup_mp()
        cls.mp_pool.apply_async(func=computation, callback=doOnComplete)
    @classmethod
    def dispatch_comp_mp_args(cls, computation: Callable[[Any], Any], args=(), kwargs={}, doOnComplete: Callable[[Any], None]=None):
        """
        Dispatches Computation to process.
        DoOnComplete is called with the result
        """
        def exc_catch(ex):
            print(ex)
        cls.setup_mp()
        cls.mp_pool.apply_async(func=computation, args=args, kwds=kwargs, callback=doOnComplete, error_callback=exc_catch)