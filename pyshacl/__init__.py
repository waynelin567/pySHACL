# -*- coding: latin-1 -*-
#
from .shape import Shape
from .shapes_graph import ShapesGraph
from .validate import Validator, validate, validate_with_trace
from .trace_mgr import TraceMgr, ShapeContainer

# version compliant with https://www.python.org/dev/peps/pep-0440/
__version__ = '0.25.0'
# Don't forget to change the version number in pyproject.toml, Dockerfile, and CITATION.cff along with this one

__all__ = ['TraceMgr', 'ShapeContainer','validate', 'validate_with_trace', 'Validator', '__version__', 'Shape', 'ShapesGraph']
