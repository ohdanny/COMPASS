"""COMPASS: cell-type-aware spatial isoform analysis."""

from .design import build_design
from .indexing import ParamIndex
from .likelihood import nll, dm_loglik, softmax_ref, spot_totals
from .model import fit_model
from .inference import compass_fit_gene, compass_fit_many
from .splines import build_bspline_basis, build_basis_to_rank
from .tests import omnibus_score_test, wald_tests, wald_tests_by_category
from .tests_classes import (
    FitResult,
    OmnibusResult,
    WaldOneResult,
    WaldResults,
    WaldOneByCategoryResult,
    WaldByCategoryResults,
    CompassResult,
    CompassManyResult,
)

from .utils import setup_logger
__version__ = "0.0.0"

__all__ = [
    "build_design",
    "ParamIndex",
    "nll",
    "dm_loglik",
    "softmax_ref",
    "spot_totals",
    "fit_model",
    "compass_fit_gene",
    "compass_fit_many",
    "build_bspline_basis",
    "build_basis_to_rank",
    "omnibus_score_test",
    "wald_tests",
    "wald_tests_by_category",
    "FitResult",
    "OmnibusResult",
    "WaldOneResult",
    "WaldResults",
    "WaldOneByCategoryResult",
    "WaldByCategoryResults",
    "CompassResult",
    "CompassManyResult",
    "__version__",
    "setup_logger",
]
