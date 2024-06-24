from .shape import Trace

class ShapeContainer:
    def __init__(self, shape_name:str, shacl_syntax:str):
        self._shape_name = shape_name
        self._traces:list[Trace] = []
        self.shacl_syntax = shacl_syntax
    def add_trace(self, trace:Trace):
        self._traces.append(trace)
    def print(self):
        for trace in self._traces:
            trace.print()
        print(self.shacl_syntax)
class TraceMgr:
    _instance = None
    _shapes:dict[str, ShapeContainer] = {}
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TraceMgr, cls).__new__(cls)
        return cls._instance
    def add_shape_container(cls, shape_name:str, sc:ShapeContainer):
        cls._shapes[shape_name] = sc 
    def print(cls):
        for shape_name, sc in cls._shapes.items():
            print(f"Shape: {shape_name}")
            sc.print()
