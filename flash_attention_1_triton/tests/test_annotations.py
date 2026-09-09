"""Require jaxtyping dtype and shape declarations for torch.Tensor signatures."""

import importlib
import inspect
from pathlib import Path
from typing import TypeAliasType, get_args, get_type_hints

import pytest
import torch
from jaxtyping import Shaped


def assert_tensor_annotation(annotation: object) -> None:
    """Reject bare torch.Tensor types, including nested containers and type aliases.

    Args:
        annotation (object): Evaluated Python type annotation to inspect recursively.

    Returns:
        None: Complete if every Tensor annotation declares its dtype and shape.

    Raises:
        AssertionError: If a Tensor annotation lacks dtype or shape metadata.
    """
    if isinstance(annotation, TypeAliasType):
        assert_tensor_annotation(annotation.__value__)
        return
    assert annotation is not torch.Tensor, (
        "Use a jaxtyping dtype and shape instead of bare torch.Tensor"
    )
    if getattr(annotation, "array_type", None) is torch.Tensor:
        assert isinstance(getattr(annotation, "dtypes", None), tuple), (
            "Declare the torch.Tensor dtype"
        )
        assert isinstance(getattr(annotation, "dim_str", None), str), (
            "Declare the torch.Tensor shape"
        )
    for argument in get_args(annotation):
        assert_tensor_annotation(argument)


def test_tensor_signatures() -> None:
    """Check signatures in production code and tests without requiring CUDA.

    Returns:
        None: Complete if all discovered function signatures satisfy the Tensor annotation rule.
    """
    root = Path(__file__).resolve().parents[1]
    for directory in (root / "src", root / "tests"):
        for path in sorted(directory.rglob("*.py")):
            name = ".".join(path.relative_to(directory).with_suffix("").parts)
            name = name.removesuffix(".__init__")
            module = importlib.import_module(name)
            for member in vars(module).values():
                function = inspect.unwrap(member)
                if not inspect.isfunction(function) or function.__module__ != module.__name__:
                    continue
                for parameter, annotation in get_type_hints(function).items():
                    try:
                        assert_tensor_annotation(annotation)
                    except AssertionError as error:
                        raise AssertionError(
                            f"{name}.{function.__name__}: {parameter}: {error}"
                        ) from error


@pytest.mark.parametrize(
    "annotation", [torch.Tensor, tuple[torch.Tensor, ...], Shaped[torch.Tensor, "rows cols"]]
)
def test_tensor_rule_rejects_missing_metadata(annotation: object) -> None:
    """Prove the rule rejects bare, nested, and dtype-unspecified torch.Tensor types.

    Args:
        annotation (object): Deliberately invalid bare, nested, or dtype-unspecified Tensor type.

    Returns:
        None: Complete only if the annotation checker rejects the invalid type.
    """
    with pytest.raises(AssertionError):
        assert_tensor_annotation(annotation)
