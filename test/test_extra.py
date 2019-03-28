# -*- coding: utf-8 -*-
#
# Extra tests which are not part of the SHT or DASH test suites,
# nor the discrete issues tests or the cmdline_test file.
# The need for these tests are discovered by doing coverage checks and these
# are added as required.
import os
from pyshacl import validate
from pyshacl.errors import ReportableRuntimeError

ontology_file_text = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix exOnt: <http://example.com/exOnt#> .

<http://example.com/exOnt> a owl:Ontology ;
    rdfs:label "An example extra-ontology file."@en .

exOnt:Animal a rdfs:Class ;
    rdfs:comment "The parent class for Humans and Pets"@en ;
    rdfs:subClassOf owl:Thing .

exOnt:Human a rdfs:Class ;
    rdfs:comment "A Human being"@en ;
    rdfs:subClassOf exOnt:Animal .

exOnt:Pet a rdfs:Class ;
    rdfs:comment "An animal owned by a human"@en ;
    rdfs:subClassOf exOnt:Animal .

exOnt:hasPet a rdf:Property ;
    rdfs:domain exOnt:Human ;
    rdfs:range exOnt:Pet .

exOnt:nlegs a rdf:Property ;
    rdfs:domain exOnt:Animal ;
    rdfs:range exOnt:integer .

exOnt:Lizard a rdfs:Class ;
    rdfs:subClassOf exOnt:Pet .

"""

shacl_file_text = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix exShape: <http://example.com/exShape#> .
@prefix exOnt: <http://example.com/exOnt#> .

<http://example.com/exShape> a owl:Ontology ;
    rdfs:label "Example Shapes File"@en .

exShape:HumanShape a sh:NodeShape ;
    sh:property [
        sh:class exOnt:Pet ;
        sh:path exOnt:hasPet ;
    ] ;
    sh:property [
        sh:datatype xsd:integer ;
        sh:path exOnt:nLegs ;
        sh:maxInclusive 2 ;
        sh:minInclusive 2 ;
    ] ;
    sh:targetClass exOnt:Human .

exShape:AnimalShape a sh:NodeShape ;
    sh:property [
        sh:datatype xsd:integer ;
        sh:path exOnt:nLegs ;
        sh:maxInclusive 4 ;
        sh:minInclusive 1 ;
    ] ;
    sh:targetClass exOnt:Animal .
"""

data_file_text = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix exOnt: <http://example.com/exOnt#> .
@prefix ex: <http://example.com/ex#> .

ex:Human1 rdf:type exOnt:Human ;
    rdf:label "Amy" ;
    exOnt:nLegs "2"^^xsd:integer ;
    exOnt:hasPet ex:Pet1 .

ex:Pet1 rdf:type exOnt:Lizard ;
    rdf:label "Sebastian" ;
    exOnt:nLegs "4"^^xsd:integer .
"""

data_file_text_bad = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix exOnt: <http://example.com/exOnt#> .
@prefix ex: <http://example.com/ex#> .

ex:Human1 rdf:type exOnt:Human ;
    rdf:label "Amy" ;
    exOnt:nLegs "2"^^xsd:integer ;
    exOnt:hasPet "Sebastian"^^xsd:string .

ex:Pet1 rdf:type exOnt:Lizard ;
    rdf:label "Sebastian" ;
    exOnt:nLegs "g"^^xsd:string .
"""


def test_metashacl_pass():
    res = validate(data_file_text, shacl_graph=shacl_file_text,
                   meta_shacl=True, data_graph_format='turtle',
                   shacl_graph_format='turtle', ont_graph=ontology_file_text,
                   ont_graph_format="turtle", inference='both', debug=True)
    conforms, graph, string = res
    assert conforms


def test_metashacl_fail():
    bad_shacl_text = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <http://example.com/ex#> .

ex:HumanShape a sh:NodeShape ;
    sh:property [
        sh:class ex:Pet ;
        sh:path "2"^^xsd:integer ;
    ] ;
    sh:property [
        sh:datatype xsd:integer ;
        sh:path ex:nLegs ;
        sh:maxInclusive 2 ;
        sh:minInclusive 2 ;
    ] ;
    sh:targetClass ex:Human .

ex:AnimalShape a sh:NodeShape ;
    sh:property [
        sh:datatype xsd:integer ;
        sh:path ex:nLegs ;
        sh:maxInclusive 4 ;
        sh:minInclusive 1 ;
    ] ;
    sh:targetClass ex:Animal .  
"""
    did_error = False
    try:
        res = validate(data_file_text, shacl_graph=bad_shacl_text,
                       meta_shacl=True, data_graph_format='turtle',
                       shacl_graph_format='turtle', ont_graph=ontology_file_text,
                       ont_graph_format="turtle", inference='both', debug=True)
        conforms, graph, string = res
        assert not conforms
    except ReportableRuntimeError as r:
        assert "Shacl Shapes Shacl file" in r.message
        did_error = True
    assert did_error


def test_serialize_report_graph():
    res = validate(data_file_text, shacl_graph=shacl_file_text,
                   data_graph_format='turtle', serialize_report_graph=True,
                   shacl_graph_format='turtle', ont_graph=ontology_file_text,
                   ont_graph_format="turtle", inference='both', debug=True)
    conforms, graph, string = res
    assert isinstance(graph, (str, bytes))


def test_web_retrieve():
    DEB_BUILD_ARCH = os.environ.get('DEB_BUILD_ARCH', None)
    DEB_HOST_ARCH = os.environ.get('DEB_HOST_ARCH', None)
    if DEB_BUILD_ARCH is not None or DEB_HOST_ARCH is not None:
        print("Cannot run web requests in debhelper tests.")
        assert True
        return True
    shacl_file = "https://raw.githubusercontent.com/RDFLib/pySHACL/master/test/resources/cmdline_tests/s1.ttl"
    ont_file = "https://raw.githubusercontent.com/RDFLib/pySHACL/master/test/resources/cmdline_tests/o1.ttl"
    res = validate(data_file_text, shacl_graph=shacl_file, data_graph_format='turtle',
                   shacl_graph_format='turtle', ont_graph=ont_file,
                   ont_graph_format="turtle", inference='both', debug=True)
    conforms, graph, string = res
    assert conforms


def test_web_retrieve_fail():
    DEB_BUILD_ARCH = os.environ.get('DEB_BUILD_ARCH', None)
    DEB_HOST_ARCH = os.environ.get('DEB_HOST_ARCH', None)
    if DEB_BUILD_ARCH is not None or DEB_HOST_ARCH is not None:
        print("Cannot run web requests in debhelper tests.")
        assert True
        return True
    shacl_file = "https://raw.githubusercontent.com/RDFLib/pySHACL/master/test/resources/cmdline_tests/s1.ttl"
    ont_file = "https://raw.githubusercontent.com/RDFLib/pySHACL/master/test/resources/cmdline_tests/o1.ttl"
    res = validate(data_file_text_bad, shacl_graph=shacl_file, data_graph_format='turtle',
                   shacl_graph_format='turtle', ont_graph=ont_file,
                   ont_graph_format="turtle", inference='both', debug=True)
    conforms, graph, string = res
    assert not conforms


my_partial_shapes_text = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <http://example.com/ex1#> .

<http://example.com/ex1> a owl:Ontology ;
    owl:imports <https://raw.githubusercontent.com/RDFLib/pySHACL/master/test/resources/cmdline_tests/s1.ttl> .
"""

my_partial_ont_text = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <http://example.com/ex2#> .

<http://example.com/ex2> a owl:Ontology ;
    owl:imports <https://raw.githubusercontent.com/RDFLib/pySHACL/master/test/resources/cmdline_tests/o1.ttl> .
"""


def test_owl_imports():
    DEB_BUILD_ARCH = os.environ.get('DEB_BUILD_ARCH', None)
    DEB_HOST_ARCH = os.environ.get('DEB_HOST_ARCH', None)
    if DEB_BUILD_ARCH is not None or DEB_HOST_ARCH is not None:
        print("Cannot run owl:imports in debhelper tests.")
        assert True
        return True
    res = validate(data_file_text, shacl_graph=my_partial_shapes_text, data_graph_format='turtle',
                   shacl_graph_format='turtle', ont_graph=my_partial_ont_text,
                   ont_graph_format="turtle", inference='both', debug=True, do_owl_imports=True)
    conforms, graph, string = res
    print(string)
    assert conforms


def test_owl_imports_fail():
    DEB_BUILD_ARCH = os.environ.get('DEB_BUILD_ARCH', None)
    DEB_HOST_ARCH = os.environ.get('DEB_HOST_ARCH', None)
    if DEB_BUILD_ARCH is not None or DEB_HOST_ARCH is not None:
        print("Cannot run owl:imports in debhelper tests.")
        assert True
        return True

    res = validate(data_file_text_bad, shacl_graph=my_partial_shapes_text, data_graph_format='turtle',
                   shacl_graph_format='turtle', ont_graph=my_partial_ont_text,
                   ont_graph_format="turtle", inference='both', debug=True, do_owl_imports=True)
    conforms, graph, string = res
    print(string)
    assert not conforms


if __name__ == "__main__":
    test_metashacl_pass()
    test_metashacl_fail()
    test_web_retrieve()
    test_serialize_report_graph()
    test_owl_imports()