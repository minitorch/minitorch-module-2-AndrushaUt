from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from collections import defaultdict

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_pos = list(vals)
    vals_neg = list(vals)

    vals_pos[arg] += epsilon
    vals_neg[arg] -= epsilon

    derivative = (f(*vals_pos) - f(*vals_neg)) / (2 * epsilon)

    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    result = []

    def recurse(current_node: Variable) -> None:
        if current_node.unique_id in visited or current_node.is_constant():
            return
        
        visited.add(current_node.unique_id)

        for neig in current_node.parents:
            recurse(neig)

        result.append(current_node)

    recurse(variable)
    return result[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    nodes_in_order = topological_sort(variable)
    derivatives_map = defaultdict(float)
    derivatives_map[variable.unique_id] = deriv

    for current_node in nodes_in_order:
        node_derivative = derivatives_map[current_node.unique_id]

        if current_node.is_leaf():
            current_node.accumulate_derivative(node_derivative)
            continue

        chain_results = current_node.chain_rule(node_derivative)
        for parent_node, parent_derivative in chain_results:
            if not parent_node.is_constant():
                derivatives_map[parent_node.unique_id] += parent_derivative


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values