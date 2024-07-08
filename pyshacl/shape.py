# -*- coding: utf-8 -*-
#
from buildingmotif import get_building_motif
import logging
import sys
from decimal import Decimal
from time import perf_counter
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Type, Union
from rdflib import BNode, Literal, URIRef
import inspect
from rdflib.namespace import RDF
from .consts import SH
from .consts import (
    RDF_type,
    RDFS_Class,
    RDFS_subClassOf,
    SH_deactivated,
    SH_description,
    SH_Info,
    SH_jsFunctionName,
    SH_JSTarget,
    SH_JSTargetType,
    SH_message,
    SH_name,
    SH_order,
    SH_node,
    SH_property,
    SH_resultSeverity,
    SH_select,
    SH_severity,
    SH_SPARQLTarget,
    SH_SPARQLTargetType,
    SH_target,
    SH_targetClass,
    SH_targetNode,
    SH_targetObjectsOf,
    SH_targetSubjectsOf,
    SH_Violation,
    SH_Warning,
)
from .errors import ConstraintLoadError, ConstraintLoadWarning, ReportableRuntimeError, ShapeLoadError
from .helper import get_query_helper_cls
from .helper.expression_helper import value_nodes_from_path
from .pytypes import GraphLike

if TYPE_CHECKING:
    from pyshacl.constraints import ConstraintComponent
    from pyshacl.shapes_graph import ShapesGraph

module = sys.modules[__name__]


class Shape(object):
    __slots__ = (
        'logger',
        'sg',
        'node',
        '_p',
        '_path',
        '_advanced',
        '_deactivated',
        '_severity',
        '_messages',
        '_names',
        '_descriptions',
        '_traces',
        '_my_name'
    )

    def __init__(
        self,
        sg: 'ShapesGraph',
        node: Union[URIRef, BNode],
        p=False,
        path: Optional[Union[URIRef, BNode]] = None,
        logger=None,
    ):
        """
        Shape
        :type sg: ShapesGraph
        :type node: URIRef | BNode
        :type p: bool
        :type path: URIRef | BNode | None
        :type logger: logging.Logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.sg = sg
        self.node = node
        self._p = p
        self._path = path
        self._advanced = False
        self._traces:dict[str, Trace] = {}
        self._my_name:str = str(self) 
        deactivated_vals = set(self.objects(SH_deactivated))
        if len(deactivated_vals) > 1:
            # TODO:coverage: we don't have any tests for invalid shapes
            raise ShapeLoadError(
                "A SHACL Shape cannot have more than one sh:deactivated predicate.",
                "https://www.w3.org/TR/shacl/#deactivated",
            )
        elif len(deactivated_vals) < 1:
            self._deactivated = False  # type: bool
        else:
            d = next(iter(deactivated_vals))
            if not isinstance(d, Literal):
                # TODO:coverage: we don't have any tests for invalid shapes
                raise ShapeLoadError(
                    "The value of sh:deactivated predicate on a SHACL Shape must be a Literal.",
                    "https://www.w3.org/TR/shacl/#deactivated",
                )
            self._deactivated = bool(d.value)
        severity = set(self.objects(SH_severity))
        if len(severity):
            self._severity = next(iter(severity))  # type: Union[URIRef, BNode, Literal]
        else:
            self._severity = SH_Violation
        messages = set(self.objects(SH_message))
        if len(messages):
            self._messages = messages  # type: Set
        else:
            self._messages = set()
        names = set(self.objects(SH_name))
        if len(names):
            self._names = names  # type: Set
        else:
            self._names = set()
        descriptions = set(self.objects(SH_description))
        if len(descriptions):
            self._descriptions = descriptions  # type: Set
        else:
            self._descriptions = set()

    def set_advanced(self, val):
        self._advanced = bool(val)

    def get_other_shape(self, shape_node):
        try:
            return self.sg.lookup_shape_from_node(shape_node)
        except (KeyError, AttributeError):
            # TODO:coverage: we never hit this during a successful test run
            return None

    @property
    def is_property_shape(self):
        return bool(self._p)

    def property_shapes(self):
        # TODO:coverage: this is never used?
        return self.sg.graph.objects(self.node, SH_property)

    @property
    def deactivated(self):
        return self._deactivated

    @property
    def severity(self):
        return self._severity

    @property
    def message(self):
        if self._messages is None:
            return
        for m in self._messages:
            yield m

    @property
    def name(self):
        if self._names is None:
            return
        for n in self._names:
            yield n

    def find_triples_w_bnode(self):
        triples = []
        if self.is_property_shape:
            for s, p, o in self.sg.graph.triples((None, None, self.node)):
                triples.append((s, p, o))
        else:
            for s, p, o in self.sg.graph.triples((self.node, None, None)):
                triples.append((s, p, o))
        return triples

    def __str__(self):
        try:
            name = next(iter(self.name))
        except Exception:
            name = str(self.node)
        if self.is_property_shape:
            kind = "PropertyShape"
        else:
            kind = "NodeShape"
        return "<{} {}>".format(kind, name)

    def __repr__(self):
        if self.is_property_shape:
            p = "True"
        else:
            p = "False"
        names = list(self.name)
        if len(names):
            return "<Shape {} p={} node={}>".format(",".join(names), p, str(self.node))
        else:
            return "<Shape p={} node={}>".format(p, str(self.node))
        # return super(Shape, self).__repr__()

    @property
    def description(self):
        # TODO:coverage: this is never used?
        if self._descriptions is None:
            return
        for d in self._descriptions:
            yield d

    def objects(self, predicate=None):
        return self.sg.graph.objects(self.node, predicate)

    @property
    def order(self):
        order_nodes = list(self.objects(SH_order))
        if len(order_nodes) < 1:
            return Decimal("0.0")
        if len(order_nodes) > 1:
            raise ShapeLoadError(
                "A SHACL Shape can have only one sh:order property.", "https://www.w3.org/TR/shacl-af/#rules-order"
            )
        order_node = next(iter(order_nodes))
        if not isinstance(order_node, Literal):
            raise ShapeLoadError(
                "A SHACL Shape must be a numeric literal.", "https://www.w3.org/TR/shacl-af/#rules-order"
            )
        if isinstance(order_node.value, Decimal):
            order = order_node.value
        elif isinstance(order_node.value, int):
            order = Decimal(order_node.value)
        elif isinstance(order_node.value, float):
            order = Decimal(str(order_node.value))
        else:
            raise ShapeLoadError(
                "A SHACL Shape must be a numeric literal.", "https://www.w3.org/TR/shacl-af/#rules-order"
            )
        return order

    def target_nodes(self):
        return self.sg.graph.objects(self.node, SH_targetNode)

    def target_classes(self):
        return self.sg.graph.objects(self.node, SH_targetClass)

    def implicit_class_targets(self):
        types = list(self.sg.graph.objects(self.node, RDF_type))
        subclasses = list(self.sg.graph.subjects(RDFS_subClassOf, RDFS_Class))
        subclasses.append(RDFS_Class)
        for t in types:
            if t in subclasses:
                return [self.node]
        return []

    def target_objects_of(self):
        return self.sg.graph.objects(self.node, SH_targetObjectsOf)

    def target_subjects_of(self):
        return self.sg.graph.objects(self.node, SH_targetSubjectsOf)

    def path(self):
        if not self.is_property_shape:
            return None
        if self._path is not None:
            return self._path
        raise RuntimeError("property shape has no _path!")  # pragma: no cover

    def target(self):
        target_nodes = self.target_nodes()
        target_classes = self.target_classes()
        implicit_targets = self.implicit_class_targets()
        target_objects_of = self.target_objects_of()
        target_subjects_of = self.target_subjects_of()
        return (target_nodes, target_classes, implicit_targets, target_objects_of, target_subjects_of)

    def advanced_target(self):
        custom_targets = set(self.sg.objects(self.node, SH_target))
        result_set = dict()
        if self.sg.js_enabled:
            use_JSTarget: Union[bool, Type] = True
        else:
            use_JSTarget = False

        for c in custom_targets:
            ct = dict()
            selects = list(self.sg.objects(c, SH_select))
            has_select = len(selects) > 0
            fn_names = list(self.sg.objects(c, SH_jsFunctionName))
            has_fnname = len(fn_names) > 0
            is_types = set(self.sg.objects(c, RDF_type))
            if has_select or (SH_SPARQLTarget in is_types):
                ct['type'] = SH_SPARQLTarget
                SPARQLQueryHelper = get_query_helper_cls()
                qh = SPARQLQueryHelper(self, c, selects[0], deactivated=self._deactivated)
                qh.collect_prefixes()
                ct['qh'] = qh
            elif has_fnname or (SH_JSTarget in is_types):
                if use_JSTarget:
                    JST = getattr(module, "JSTarget", None)
                    if not JST:
                        # Lazy-import JS-Target to prevent RDFLib import error
                        from pyshacl.extras.js.target import JSTarget as JST

                        setattr(module, "JSTarget", JST)
                    ct['type'] = SH_JSTarget
                    ct['targeter'] = JST(self.sg, c)
                else:
                    #  Found JSTarget, but JS is not enabled in PySHACL. Ignore this target.
                    pass
            else:
                found_tt = None
                for t in is_types:
                    try:
                        found_tt = self.sg.get_shacl_target_type(t)
                        break
                    except LookupError:
                        continue
                if not found_tt:
                    msg = "None of these types match a TargetType: {}".format(" ".join(is_types))
                    raise ShapeLoadError(msg, "https://www.w3.org/TR/shacl-af/#SPARQLTargetType")
                bound_tt = found_tt.bind(self, c)
                ct['type'] = bound_tt.shacl_constraint_class()
                if ct['type'] == SH_SPARQLTargetType:
                    ct['qt'] = bound_tt
                elif ct['type'] == SH_JSTargetType:
                    ct['targeter'] = bound_tt
            result_set[c] = ct
        return result_set

    def focus_nodes(self, data_graph):
        """
        The set of focus nodes for a shape may be identified as follows:

        specified in a shape using target declarations
        specified in any constraint that references a shape in parameters of shape-expecting constraint parameters (e.g. sh:node)
        specified as explicit input to the SHACL processor for validating a specific RDF term against a shape
        :return:
        """
        t1 = perf_counter()
        (target_nodes, target_classes, implicit_classes, target_objects_of, target_subjects_of) = self.target()
        if self._advanced:
            advanced_targets = self.advanced_target()
        else:
            advanced_targets = False
        found_node_targets = set()
        # Just add _all_ target_nodes to the set,
        # they don't need to actually exist in the graph
        found_node_targets.update(iter(target_nodes))
        target_classes = set(target_classes)
        target_classes.update(set(implicit_classes))
        found_target_instances = set()
        for tc in target_classes:
            s = data_graph.subjects(RDF_type, tc)
            found_target_instances.update(s)
            subc = data_graph.transitive_subjects(RDFS_subClassOf, tc)
            for subclass in iter(subc):
                if subclass == tc:
                    continue
                s1 = data_graph.subjects(RDF_type, subclass)
                found_target_instances.update(s1)
        found_node_targets.update(found_target_instances)
        found_target_subject_of = set()
        for s_of in target_subjects_of:
            subs = {s for s, o in data_graph.subject_objects(s_of)}
            found_target_subject_of.update(subs)
        found_node_targets.update(found_target_subject_of)
        found_target_object_of = set()
        for o_of in target_objects_of:
            objs = {o for s, o in data_graph.subject_objects(o_of)}
            found_target_object_of.update(objs)
        found_node_targets.update(found_target_object_of)
        if advanced_targets:
            for at_node, at in advanced_targets.items():
                if at['type'] == SH_SPARQLTarget:
                    qh = at['qh']
                    select = qh.apply_prefixes(qh.select_text)
                    results = data_graph.query(select, initBindings=None)
                    if not results or len(results.bindings) < 1:
                        continue
                    for r in results:
                        t = r['this']
                        found_node_targets.add(t)
                elif at['type'] in (SH_JSTarget, SH_JSTargetType):
                    results = at['targeter'].find_targets(data_graph)
                    for r in results:
                        found_node_targets.add(r)
                else:
                    results = at['qt'].find_targets(data_graph)
                    if not results or len(results.bindings) < 1:
                        continue
                    for r in results:
                        t = r['this']
                        found_node_targets.add(t)
        t2 = perf_counter()
        return found_node_targets

    def value_nodes(self, target_graph, focus):
        """
        For each focus node, you can get a set of value nodes.
        For a Node Shape, each focus node has just one value node,
            which is just the focus_node
        :param target_graph:
        :param focus:
        :return:
        """
        if not isinstance(focus, (tuple, list, set)):
            focus = [focus]
        if not self.is_property_shape:
            return {f: set((f,)) for f in focus}
        path_val = self.path()
        focus_dict = {}
        for f in focus:
            focus_dict[f] = value_nodes_from_path(self.sg, f, path_val, target_graph)
        return focus_dict

    def find_custom_constraints(self):
        applicable_custom_constraints = set()
        for c in self.sg.custom_constraints:
            mandatory = (p for p in c.parameters if not p.optional)
            found_all_mandatory = True
            for mandatory_param in mandatory:
                path = mandatory_param.path()
                assert isinstance(path, URIRef)
                found_vals = set(self.sg.objects(self.node, path))
                # found_vals = value_nodes_from_path(self.node, mandatory_param.path(), self.sg.graph)
                found_all_mandatory = found_all_mandatory and bool(len(found_vals) > 0)
            if found_all_mandatory:
                applicable_custom_constraints.add(c)
        return applicable_custom_constraints
    def get_focus_signature(self, focus):
        assert len(focus) > 0
        ret = ""
        for i, f in enumerate(list(focus)):
            if i != len(focus) - 1:
                ret += f"{str(f)}_"
            else:
                ret += str(f) 
        return ret
    def validate(
        self,
        target_graph: GraphLike,
        focus: Optional[
            Union[
                Tuple[Union[URIRef, BNode]],
                List[Union[URIRef, BNode]],
                Set[Union[URIRef, BNode]],
                Union[URIRef, BNode],
            ]
        ] = None,
        abort_on_first: Optional[bool] = False,
        allow_infos: Optional[bool] = False,
        allow_warnings: Optional[bool] = False,
        _evaluation_path: Optional[List] = None,
        mydebug = False
    ):
        stack = inspect.stack()
        for frame_info in stack:
            if frame_info.function == "shacl_validate":
                mydebug = True
                break  # If we encounter any other function, mydebug = False
        if self.deactivated:
            if self.sg.debug:
                self.logger.debug(f"Skipping shape because it is deactivated: {str(self)}")
            return True, []
        if focus is not None:
            lh_shape = False
            rh_shape = True
            if not isinstance(focus, (tuple, list, set)):
                focus = [focus]
            if len(focus) < 1:
                return True, []
        else:
            lh_shape = True
            rh_shape = False
            focus = self.focus_nodes(target_graph)
            if len(focus) < 1:
                # It's possible for shapes to have _no_ focus nodes
                # (they are called in other ways)
                return True, []

        if mydebug:
            focus_signature = self.get_focus_signature(focus)
            if focus_signature not in self._traces:
                self._traces[focus_signature] = Trace(focus)

            if isinstance(self.node, BNode):
#                print(f"=========================Running evaluation of Shape {str(self)} on focus: {focus}=================================")
                subj, depth = self.find_closest_non_blank_parent()
                self._my_name = f"{str(self)} closest non blank parent {subj} is {depth} levels up" 
#                print(f"closest non blank parent {subj} is {depth} levels up")
            else:
                self._my_name = str(self)
#                print(f"=========================Running evaluation of Shape {str(self)} on focus: {focus}=================================")
        if _evaluation_path is None:
            _evaluation_path = []
        elif len(_evaluation_path) >= 30:
            # 27 is the depth required to successfully do the meta-shacl test on shacl.ttl
            path_str = " -> ".join((str(e) for e in _evaluation_path))
            raise ReportableRuntimeError("Evaluation path too deep!\n{}".format(path_str))
        # Lazy import here to avoid an import loop
        CONSTRAINT_PARAMETERS, PARAMETER_MAP = getattr(module, 'CONSTRAINT_PARAMS', (None, None))
        if not CONSTRAINT_PARAMETERS or not PARAMETER_MAP:
            from .constraints import ALL_CONSTRAINT_PARAMETERS, CONSTRAINT_PARAMETERS_MAP

            setattr(module, 'CONSTRAINT_PARAMS', (ALL_CONSTRAINT_PARAMETERS, CONSTRAINT_PARAMETERS_MAP))
            CONSTRAINT_PARAMETERS = ALL_CONSTRAINT_PARAMETERS
            PARAMETER_MAP = CONSTRAINT_PARAMETERS_MAP
        if self.sg.js_enabled or self._advanced:
            search_parameters = CONSTRAINT_PARAMETERS.copy()
            constraint_map = PARAMETER_MAP.copy()
            if self._advanced:
                from pyshacl.constraints.advanced import ExpressionConstraint, SH_expression

                search_parameters.append(SH_expression)
                constraint_map[SH_expression] = ExpressionConstraint
            if self.sg.js_enabled:
                from pyshacl.extras.js.constraint import JSConstraint, SH_js

                search_parameters.append(SH_js)
                constraint_map[SH_js] = JSConstraint
        else:
            search_parameters = CONSTRAINT_PARAMETERS
            constraint_map = PARAMETER_MAP
        parameters = (p for p, v in self.sg.predicate_objects(self.node) if p in search_parameters)
        reports = []
        focus_value_nodes = self.value_nodes(target_graph, focus)
        filter_reports: bool = False
        allow_conform: bool = False
        allowed_severities: Set[URIRef] = set()
        if allow_infos:
            allowed_severities.add(SH_Info)
        if allow_warnings:
            allowed_severities.add(SH_Info)
            allowed_severities.add(SH_Warning)
        if allow_infos or allow_warnings:
            if self.severity in allowed_severities:
                allow_conform = True
            else:
                filter_reports = True

        non_conformant = False
        done_constraints = set()
        run_count = 0
        _evaluation_path.append(self)
        constraint_components = [constraint_map[p] for p in iter(parameters)]
        constraint_component: Type['ConstraintComponent']
        for constraint_component in constraint_components:
            if constraint_component in done_constraints:
                continue
            try:
                c = constraint_component(self)
            except ConstraintLoadWarning as w:
                self.logger.warning(repr(w))
                continue
            except ConstraintLoadError as e:
                self.logger.error(repr(e))
                raise e
            _e_p_copy = _evaluation_path[:]
            _e_p_copy.append(c)
            _is_conform, _reports = c.evaluate(target_graph, focus_value_nodes, _e_p_copy)

            if _is_conform or allow_conform:
                ...
            elif filter_reports:
                all_allow = True
                for v_str, v_node, v_parts in _reports:
                    severity_bits = list(filter(lambda p: p[0] == v_node and p[1] == SH_resultSeverity, v_parts))
                    if severity_bits:
                        all_allow = all_allow and (severity_bits[0][2] in allowed_severities)
                non_conformant = non_conformant or (not all_allow)
            else:
                non_conformant = non_conformant or (not _is_conform)
            reports.extend(_reports)
            run_count += 1
            done_constraints.add(constraint_component)
            if non_conformant and abort_on_first:
                break
            if mydebug:
                self.record_trace(focus, c, non_conformant)
        applicable_custom_constraints = self.find_custom_constraints()
        for a in applicable_custom_constraints:
            if non_conformant and abort_on_first:
                break
            _e_p_copy2 = _evaluation_path[:]
            validator = a.make_validator_for_shape(self)
            _e_p_copy2.append(validator)
            _is_conform, _r = validator.evaluate(target_graph, focus_value_nodes, _e_p_copy2)
            non_conformant = non_conformant or (not _is_conform)
            reports.extend(_r)
            run_count += 1
            if mydebug:
                self.record_trace(focus, c, non_conformant)
#        if mydebug:
#            print(_evaluation_path[-1], "Passes" if not non_conformant else "Fails")
        return (not non_conformant), reports

    def record_trace(self, focus, c, non_conformant):
        focus_signature = self.get_focus_signature(focus)
        assert focus_signature in self._traces 
        self._traces[focus_signature].add_component(c, not non_conformant)
#        print(f"\t\tFocus:{focus}", c, "Passes" if not non_conformant else "Fails")

    def find_closest_non_blank_parent(self) -> URIRef:
        visited = set()

        # Stack for DFS (Depth First Search)
        stack = [(self.node, None, 0)]  # (current node, parent node)
        while stack:
            current, parent, depth = stack.pop()

            # If current node is a blank node and not visited
            if isinstance(current, BNode) and current not in visited:
                visited.add(current)

                # Find all subjects that have the current blank node as object
                for subj in self.sg.graph.subjects(object=current):
                    # If subject is not a blank node, we found our parent
                    if not isinstance(subj, BNode):
                        return subj, depth + 1
                    # Otherwise, add it to the stack to continue search
                    stack.append((subj, current, depth+1))
        assert False  # We should always find a non-blank parent
        return None  # If no non-blank parent is found

    def get_children_from_rdf_list(self, list_node):
        list_children = []
        while list_node and list_node != RDF.nil:
            first = self.sg.graph.value(list_node, RDF.first)
            if first:
                list_children.append(first)
            list_node = self.sg.graph.value(list_node, RDF.rest)
        return list_children
    def get_shacl_syntax(self, for_logical:bool=False):
        ns_mgr = get_building_motif().template_ns_mgr
        g = self.sg.graph
        shape_syntax = []
        logical_constraints = [SH["not"], SH["and"], SH["or"], SH.xone]
        def format_node(node):
            """Format the RDF node for display"""
            if isinstance(node, URIRef):
                return f"{str(ns_mgr.normalizeUri(node))}"
            elif isinstance(node, BNode):
                assert False # I don't think this should be reached
                return f"_:bnode"
            elif isinstance(node, Literal):
                return f'{str(node)}'
            else:
                return str(node)
        def add_properties(node, indent=0):
            for p, o in g.predicate_objects(node):
                indent_str = " " * indent
                if p in [SH["and"], SH["or"], SH.xone]:
                    syntax_str = f"{indent_str}{format_node(p)} [ "
                    list_children = self.get_children_from_rdf_list(o)
                    syntax_str += ", ".join([format_node(c) for c in list_children])
                    syntax_str += " ] ;"
                    shape_syntax.append(syntax_str)
                elif isinstance(o, BNode):
                    shape_syntax.append(f"{indent_str}{format_node(p)} [")
                    add_properties(o, indent + 2)
                    shape_syntax.append(f"{indent_str}] ;")
                elif p != RDF_type:
                    shape_syntax.append(f"{indent_str}{format_node(p)} {format_node(o)} ;")
        # Add the shape type
        rdf_types = list(g.objects(self.node, RDF_type))
        if rdf_types:
            rdf_types = [format_node(t) for t in rdf_types]
            shape_syntax.append(f"{format_node(self.node)} a {', '.join(rdf_types)} ;")

        # Add properties and constraints
        if for_logical:
            logical_ps = [(p,o) for p, o in g.predicate_objects(self.node) if p in logical_constraints]
            indent = 2
            indent_str = " " * indent
            for p, o in logical_ps:
                if p in [SH["and"], SH["or"], SH.xone]:
                    syntax_str = f"{indent_str}{format_node(p)} [ "
                    list_children = self.get_children_from_rdf_list(o)
                    syntax_str += ", ".join([format_node(c) for c in list_children])
                    syntax_str += " ] ;"
                    shape_syntax.append(syntax_str)
                elif isinstance(o, BNode):
                    shape_syntax.append(f"{indent_str}{format_node(p)} [")
                    add_properties(o, indent + 2)
                    shape_syntax.append(f"{indent_str}] ;")
        else:
            add_properties(self.node, 2)

        return "\n".join(shape_syntax)
    def get_children(self):
        child_shapes = set()
        SH_not = SH["not"]
        SH_and = SH["and"]
        SH_or = SH["or"]
        SH_xone = SH.xone
        SH_qualifiedValueShape = SH.qualifiedValueShape

        def add_nested_property_children():
            nested_properties = [SH_property, SH_node, SH_not]
            for prop in nested_properties:
                for obj in self.objects(prop):
                    if isinstance(obj, (BNode, URIRef)):
                        child_shapes.add(obj)

        def add_logical_property_children():
            logical_properties = [SH_and, SH_or, SH_xone]
            for prop in logical_properties:
                for obj in self.objects(prop):
                    if isinstance(obj, (BNode, URIRef)):
                        child_shapes.update(c for c in self.get_children_from_rdf_list(obj))
#                        children = self.get_children_from_rdf_list(obj)
#                        for c in children:
#                            child_shapes.add(c)
#                        add_children_from_rdf_list(obj)

        def add_qualified_value_shape_children():
            for obj in self.objects(SH_qualifiedValueShape):
                if isinstance(obj, (BNode, URIRef)):
                    first_shape = get_first_shape_from_qualifiedvalueshapes(obj)
                    if first_shape:
                        child_shapes.add(first_shape)

#        def add_children_from_rdf_list(list_node):
#            while list_node and list_node != RDF.nil:
#                first = self.sg.graph.value(list_node, RDF.first)
#                if first:
#                    child_shapes.add(first)
#                list_node = self.sg.graph.value(list_node, RDF.rest)

        def get_first_shape_from_qualifiedvalueshapes(qnode):
            shapes = {str(shape).strip("<>").split('Shape')[1].strip() for shape in self.sg.shapes}# if shape._my_name}
            visited = set()
            stack = [qnode]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                if str(current) in shapes:
                    return current
                stack.extend(self.get_nested_objects(current, SH_property, SH_node, SH_qualifiedValueShape))
                stack.extend(self.get_logical_objects(current, SH_and, SH_or, SH_not, SH_xone))
            return None
        
        add_nested_property_children()
        add_logical_property_children()
        add_qualified_value_shape_children()

        return [str(child) for child in list(child_shapes)]
    def get_nested_objects(self, node, *properties):
        objects = []
        for prop in properties:
            for obj in self.sg.graph.objects(node, prop):
                if isinstance(obj, (BNode, URIRef)):
                    objects.append(obj)
        return objects

    def get_logical_objects(self, node, *properties):
        objects = []
        for prop in properties:
            for list_node in self.sg.graph.objects(node, prop):
                if isinstance(list_node, (BNode, URIRef)):
                    while list_node and list_node != RDF.nil:
                        first = self.sg.graph.value(list_node, RDF.first)
                        if first:
                            objects.append(first)
                        list_node = self.sg.graph.value(list_node, RDF.rest)
        return objects
    def isSAT(self):
        for t in self._traces.values():
            if not t.isSAT:
                return False
        return True

class Trace():
    def __init__(self, focus):
        self.focus_ls = focus
        self.components = {}
        self.focus_neighbors = {}
    @property
    def isSAT(self):
        for c, sat in self.components.items():
            if not sat:
                return False
        return True 
    def add_component(self, component, sat):
        self.components[component] = sat

    def set_components(self, components):
        self.components = components

    def get_focus_neighbors(self, graph:GraphLike):
        assert self.focus_neighbors == {}, f"focus_neighbors {self.focus_neighbors}"
        for f in self.focus_ls:
            self._get_neighbors(graph, f)

    def _get_neighbors(self, graph: GraphLike, f: URIRef):
        visited = set()
        all_triples = set()

        def recurse(node):
            if node in visited:
                return
            visited.add(node)
            
            # Find all triples where node is the subject
            for p, o in graph.predicate_objects(subject=node):
                triple = (node, p, o)
                if triple not in all_triples:
                    all_triples.add(triple)
                    if isinstance(o, (URIRef, BNode)):
                        recurse(o)

        recurse(f)
        self.focus_neighbors[f] = all_triples 

    def pretty_print_triples(self, triples):
        def format_node(node):
            if isinstance(node, URIRef):
                return get_building_motif().template_ns_mgr.normalizeUri(node)
            elif isinstance(node, BNode):
                return f"_:bnode"
            else:
                return f'"{str(node)}"^^<{str(node.datatype)}>' if node.datatype else f'"{str(node)}"'
        # Organize triples by subject and predicate
        organized_triples = {}
        subject_type = {}
        for s, p, o in triples:
            s_str = format_node(s)
            p_str = format_node(p)
            o_str = format_node(o)

            if p_str == "rdf:type":
                if s_str not in subject_type:
                    subject_type[s_str] = []
                subject_type[s_str].append(o_str)
                continue

            if s_str not in organized_triples:
                organized_triples[s_str] = {}
            if p_str not in organized_triples[s_str]:
                organized_triples[s_str][p_str] = []
            organized_triples[s_str][p_str].append(o_str)
        ret_str = ""
        for s_str, types in subject_type.items():
            ret_str += f"{s_str} a {', '.join(types)} "
            if s_str in organized_triples:
                ret_str += (";\n")
                num_predicates = len(organized_triples[s_str])
                for i, (p_str, objects) in enumerate(organized_triples[s_str].items()):
                    suffix = ";\n" if i < num_predicates - 1 else ""
                    ret_str += f"  {p_str} {', '.join(objects)} {suffix}"
            ret_str += ".\n"
        return ret_str

    def print(self):
        print(f"Focus: {self.focus_ls}")
        print(f"Is SAT: {self.isSAT}")
        for c, sat in self.components.items():
            print(f"\t{c} is {sat}")
        for f, connections in self.focus_neighbors.items():
            triples = self.focus_neighbors[f]
            print(self.pretty_print_triples(triples))

    def get_prompt_string(self):
        s = "Focus:\n"
        s += ", ".join(self.focus_ls)
        s += "\nReasons:\n"
        for c, sat in self.components.items():
            c_str = str(c).strip("<>").split(" on ")[0].strip()
            s += f"\t{c_str} is violated\n"
        s += "RDF data graph:\n"
        for f, connections in self.focus_neighbors.items():
            triples = self.focus_neighbors[f]
            s += self.pretty_print_triples(triples)
        return s

class ConstraintComponent():
    def __init__(self, component, sat:bool):
        self.component = component
        self.isSAT = sat