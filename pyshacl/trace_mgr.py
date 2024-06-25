from .shape import Trace
from rdflib import Graph, URIRef
class ShapeContainer:
    def __init__(self, shape_name:str, shacl_syntax:str):
        self._shape_name = shape_name
        self._traces:list[Trace] = []
        self.shacl_syntax = shacl_syntax
        self._shape_uri_name = None
        self._children = []
    def add_children(self, children):
        self._children = children
    def set_shape_uri_name(self, shape_uri_name:str):
        self._shape_uri_name = shape_uri_name
    def add_trace(self, trace:Trace):
        self._traces.append(trace)
    def print(self):
        print("shacl syntax:")
        print(self.shacl_syntax)
        print("children:")
        print(self._children)
        for trace in self._traces:
            trace.print()
    def get_focus_neighbors(self, graph:Graph):
        for trace in self._traces:
            trace.get_focus_neighbors(graph)
class TraceMgr:
    _instance = None
    _shapes:dict[str, ShapeContainer] = {}
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TraceMgr, cls).__new__(cls)
        return cls._instance
    def add_shape_container(cls, shape_name:str, sc:ShapeContainer):
        # for some reason this shape_uri_name is hard to get from shape.py
        # it looks something like nc89b5fc355c0437d99ebb586ed03bbdbb1
        # or urn:my_site_constraints/return-air-temperature2
        shape_uri_name = shape_name.split('>')[0].strip('<>').split("Shape")[1].strip()
        sc.set_shape_uri_name(shape_uri_name)
        cls._shapes[shape_uri_name] = sc 
    def print(cls):
        for shape_name, sc in cls._shapes.items():
            print("="*25,f"Shape: {shape_name}", "="*25)
            sc.print()
            print("="*100)
    def get_focus_neighbors(cls, graph:Graph):
        for shape_name, sc in cls._shapes.items():
            sc.get_focus_neighbors(graph)
    def get_shape(cls, shape_uri_name:str):
        assert shape_uri_name in cls._shapes, f"Shape {shape_uri_name} not found"
        return cls._shapes[shape_uri_name]