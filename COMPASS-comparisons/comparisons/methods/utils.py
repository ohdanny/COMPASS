import jax.numpy as jnp
from dataclasses import is_dataclass, fields
from typing import Any
import numpy as np


import rpy2.robjects as ro
from rpy2.robjects.vectors import (
    ListVector,
    FloatVector,
    IntVector,
    StrVector,
    BoolVector,
    FactorVector,
)
from rpy2.rlike.container import NamedList


def jax_to_numpy(obj: Any) -> Any:
    """Recursively convert JAX arrays to numpy in a nested structure."""
    if isinstance(obj, jnp.ndarray):
        return np.asarray(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return type(obj)(
            **{f.name: jax_to_numpy(getattr(obj, f.name)) for f in fields(obj)}
        )
    if isinstance(obj, dict):
        return {k: jax_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jax_to_numpy(x) for x in obj]
    if isinstance(obj, tuple):
        if hasattr(obj, "_fields"):  # NamedTuple (e.g. OptimizeResults)
            return type(obj)(*(jax_to_numpy(x) for x in obj))
        return tuple(jax_to_numpy(x) for x in obj)
    return obj



def r_to_python(obj):
    """Recursively convert an rpy2 R object into nested Python primitives."""
    if isinstance(obj, NamedList):
        names = obj.names() if callable(obj.names) else obj.names
        return (
            {n: r_to_python(v) for n, v in zip(names, obj)}
            if names
            else [r_to_python(x) for x in obj]
        )

    if isinstance(obj, ListVector):
        names = obj.names
        if names is ro.NULL or names is None:
            return [r_to_python(obj[i]) for i in range(len(obj))]
        names = list(names)
        out = {}
        for i, name in enumerate(names):
            key = name if name not in out else f"{name}__{i}"
            out[key] = r_to_python(obj[i])
        return out
    if isinstance(obj, (FloatVector, IntVector, BoolVector)):
        arr = np.asarray(obj)
        return arr if arr.size > 1 else arr.item()
    if isinstance(obj, StrVector):
        lst = list(obj)
        return lst if len(lst) > 1 else lst[0]
    if isinstance(obj, FactorVector):
        return list(obj)
    if obj is ro.NULL:
        return None
    try:
        return np.asarray(obj)
    except Exception:
        return str(obj)