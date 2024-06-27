from .shape import Trace, Shape
from rdflib import Graph, URIRef
from pyshacl.constraints import ConstraintComponent
from copy import deepcopy
class ShapeContainer:
    def __init__(self, shape:Shape, copy:bool=False):
        self._shape = shape
        self._shape_name = shape._my_name
        self.shacl_syntax = shape.get_shacl_syntax() 
        self._children:list[str] = shape.get_children()
        self._traces:list[Trace] = []
        for focus_signature, trace in shape._traces.items():
            if copy:
                new_trace = Trace(trace.focus_ls)
                new_trace.focus_neighbors = deepcopy(trace.focus_neighbors)
                self._traces.append(new_trace)
            else:
                self._traces.append(trace)

    @property
    def shape_uri_name(self):
        # for some reason this shape_uri_name is hard to get from shape.py
        # it looks something like nc89b5fc355c0437d99ebb586ed03bbdbb1
        # or urn:my_site_constraints/return-air-temperature2
        shape_uri_name = self._shape_name.split('>')[0].strip('<>').split("Shape")[1].strip()
        return shape_uri_name

    def print(self):
        print(f"shape name: {self.shape_uri_name}")
        print("shacl syntax:")
        print(self.shacl_syntax)
        print("children:")
        print(self._children)
        for trace in self._traces:
            trace.print()
    def get_focus_neighbors(self, graph:Graph):
        for trace in self._traces:
            trace.get_focus_neighbors(graph)

    def get_prompt_string(self):
        s = f"{self.shacl_syntax}\n"
        for trace in self._traces:
            s += trace.get_prompt_string()
        return s 

    #### only be called to add an additional shape container
    #### because pyshacl has a weird way of handling logical constraints when they are the violation source
    def modify_for_logical_source_shape(self, source_cc:ConstraintComponent):
        assert len(self._traces) == 1, f"Expected exactly one trace, got {len(self._traces)}"
        self._traces[0].set_components({source_cc:False})
        self._children = [str(child) for child in source_cc.get_nodes_from_rdf_list()]
        self._shape_name = f"<PropertyShape {self.shape_uri_name}_{source_cc.constraint_name()}>"
        self.shacl_syntax = self._shape.get_shacl_syntax(for_logical=True)

class TraceMgr:
    _instance = None
    _shapes:dict[str, ShapeContainer] = {}
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TraceMgr, cls).__new__(cls)
        return cls._instance
    def add_shape_container(cls, shape_uri_name:str, sc:ShapeContainer):
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
    def get_shape_descendants(cls, shape_uri_name:str) -> list[str]:
        ret = []
        def recurse(shape_uri_name):
            ret.append(shape_uri_name)
            shape = cls.get_shape(shape_uri_name)
            for child in shape._children:
                recurse(child)

        shape = cls.get_shape(shape_uri_name)
        for child in shape._children:
            recurse(child)
        return ret