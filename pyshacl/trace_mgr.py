from .shape import Trace

class TraceList:
    def __init__(self, trace):
        self._traces:list[Trace] = [trace]
    def add_trace(self, trace:Trace):
        self._traces.append(trace)
    def print(self):
        for trace in self._traces:
            trace.print()
class TraceMgr:
    _instance = None
    _traces:dict[str,TraceList] = {}
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TraceMgr, cls).__new__(cls)
        return cls._instance
    def add_trace(cls, shape_name, trace:Trace):
        if shape_name in cls._traces:
            cls._traces[shape_name].add_trace(trace)
        else:
            cls._traces[shape_name] = TraceList(trace)
    def print(cls):
        for shape_name, trace_list in cls._traces.items():
            print(f"Shape: {shape_name}")
            trace_list.print()
