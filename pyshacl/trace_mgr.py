from .shape import Trace, Shape
from rdflib import Graph, URIRef
from .consts import SH, RDFS 
from rdflib import BNode, RDF
class ShapeContainer:
    def __init__(self, shape:Shape):
        self._shape = shape
        self._shape_name = shape._my_name
        self._shacl_syntax = None 
        self._children:list[str] = [] 
        self._traces:list[Trace] = []
        for focus_signature, trace in shape._traces.items():
            self._traces.append(trace)
    @property
    def children(self):
        if not self._children:
            if self._shacl_syntax is None:
                self.shacl_syntax ## need to populate skipped_properties
            self._children = self._shape.get_children()
        return self._children
    @property
    def shacl_syntax(self):
        if self._shacl_syntax is None:
            self._shacl_syntax, _ = self._shape.get_shacl_syntax(TraceMgr()._prune_shape)
        return self._shacl_syntax
    @property
    def shape_uri_name(self):
        # for some reason this shape_uri_name is hard to get from shape.py
        # it looks something like nc89b5fc355c0437d99ebb586ed03bbdbb1
        # or urn:my_site_constraints/return-air-temperature2
        shape_uri_name = self._shape_name.split('>')[0].strip('<>').split("Shape")[1].strip()
        return shape_uri_name

    def print(self):
        print(f"shape name: {self.shape_uri_name}")
        print("children:")
        print(self.children)
        for trace in self._traces:
            trace.print()
    
    def collect_value_types(self, shape_str:str, value_types:set):
        if self._shape.get_other_shape(URIRef(shape_str)):
            shape = self._shape.get_other_shape(URIRef(shape_str))
        else:
            shape = self._shape.get_other_shape(BNode(shape_str))
        _, cbd_graph = shape.get_shacl_syntax(TraceMgr()._prune_shape)
        def add_class_and_superclasses(cls):
            o_class_and_superclasses = list(TraceMgr()._target_graph.transitive_objects(cls, RDFS["subClassOf"]))
            for cls in o_class_and_superclasses:
                value_types.add(str(cls))
        def traverse_bnode(bnode):
            for s, p, o in cbd_graph.triples((bnode, None, None)):
                if isinstance(o, BNode):
                    traverse_bnode(o)
                elif o!= RDF.nil:
                    add_class_and_superclasses(o)
        for s, p, o in cbd_graph.triples((None, None, None)):
            if p == SH["class"]:
                if type(o) == BNode: 
                    traverse_bnode(o)
                else:
                    add_class_and_superclasses(o)

    def get_prompt_string(self, nested_shapes):
        value_types = set()
        for shape_str in nested_shapes:
            self.collect_value_types(shape_str, value_types)
        assert len(self._traces) == 1, "Only one trace is supported"
        s = "Focus Node: "
        for foc in self._traces[0].focus_ls:
            s += f"<{foc}> "
        s += f"\n{self.shacl_syntax}\n"
        s += self._traces[0].get_prompt_string(TraceMgr()._data_graph, value_types, TraceMgr()._prune_data)
        return s 


class TraceMgr:
    _instance = None
    _shapes:dict[str, ShapeContainer] = {}
    _prune_shape: bool = False
    _prune_data: bool = False
    _data_graph:Graph = None
    _target_graph:Graph = None
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
    def set_target_graph(cls, target_graph:Graph):
        cls._target_graph = target_graph
    def set_data_graph(cls, data_graph:Graph):
        cls._data_graph = data_graph
    def get_shape(cls, shape_uri_name:str):
        assert shape_uri_name in cls._shapes, f"Shape {shape_uri_name} not found"
        return cls._shapes[shape_uri_name]
    def get_shape_descendants(cls, shape_uri_name:str) -> list[str]:
        ret = []
        def recurse(shape_uri_name):
            ret.append(shape_uri_name)
            shape = cls.get_shape(shape_uri_name)
            for child in shape.children:
                recurse(child)

        shape = cls.get_shape(shape_uri_name)
        for child in shape.children:
            recurse(child)
        return ret
    
    def clear(cls):
        cls._shapes.clear()