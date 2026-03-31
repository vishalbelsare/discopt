"""MILP formulation for tree ensemble models (Misic 2020).

For each tree with L leaves, the encoding introduces L binary variables
(exactly one active) and big-M constraints linking input features to the
selected leaf via split decisions along the root-to-leaf path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import discopt.modeling as dm
from discopt.nn.scaling import OffsetScaling
from discopt.nn.tree import TreeEnsembleDefinition

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable


class TreeEnsembleFormulation:
    """Embed a tree ensemble as MILP constraints.

    Parameters
    ----------
    model : discopt.Model
        Optimization model to add variables and constraints to.
    ensemble : TreeEnsembleDefinition
        The trained tree ensemble.
    prefix : str
        Name prefix for all created variables and constraints.
    scaling : OffsetScaling, optional
        Input/output affine scaling.
    split_eps : float
        Epsilon for encoding strict inequalities ``x[j] > threshold``.
    """

    def __init__(
        self,
        model: Model,
        ensemble: TreeEnsembleDefinition,
        prefix: str,
        scaling: OffsetScaling | None = None,
        split_eps: float = 1e-6,
    ):
        if ensemble.input_bounds is None:
            raise ValueError("TreeEnsembleDefinition.input_bounds is required for MILP formulation")
        self._model = model
        self._ensemble = ensemble
        self._prefix = prefix
        self._scaling = scaling
        self._split_eps = split_eps

    def build(self) -> tuple[Variable, Variable]:
        """Create all variables and constraints.

        Returns
        -------
        inputs : Variable
            Input feature variables, shape ``(n_features,)``.
        outputs : Variable
            Ensemble output variable, shape ``(1,)``.
        """
        m = self._model
        ens = self._ensemble
        pfx = self._prefix
        assert ens.input_bounds is not None  # validated in __init__
        lb, ub = ens.input_bounds

        inputs = m.continuous(
            f"{pfx}_input",
            shape=(ens.n_features,),
            lb=lb,
            ub=ub,
        )

        # Per-feature big-M values
        feat_range = np.asarray(ub, dtype=np.float64) - np.asarray(lb, dtype=np.float64)

        tree_output_exprs = []
        for t, tree in enumerate(ens.trees):
            leaves = tree.leaves
            n_leaves = len(leaves)

            # Binary: exactly one leaf selected
            z = m.binary(f"{pfx}_t{t}_leaf", shape=(n_leaves,))
            m.subject_to(
                dm.sum(lambda k: z[k], over=range(n_leaves)) == 1,
                name=f"{pfx}_t{t}_one_leaf",
            )

            # Split constraints for each leaf's root-to-leaf path
            for l_idx, leaf in enumerate(leaves):
                for node, direction in tree.leaf_ancestors(leaf):
                    j = int(tree.feature[node])
                    thr = float(tree.threshold[node])
                    M_j = float(feat_range[j])

                    if direction == "left":
                        # x[j] <= threshold when this leaf is selected
                        m.subject_to(
                            inputs[j] <= thr + M_j * (1 - z[l_idx]),
                            name=f"{pfx}_t{t}_sL_{node}_{l_idx}",
                        )
                    else:
                        # x[j] > threshold when this leaf is selected
                        m.subject_to(
                            inputs[j] >= thr + self._split_eps - M_j * (1 - z[l_idx]),
                            name=f"{pfx}_t{t}_sR_{node}_{l_idx}",
                        )

            # Tree output: sum of z[l] * leaf_value[l]
            leaf_vals = np.array(
                [float(tree.value[leaf]) for leaf in leaves],
                dtype=np.float64,
            )
            y_t = dm.sum(
                lambda k, _v=leaf_vals: z[k] * float(_v[k]),
                over=range(n_leaves),
            )
            tree_output_exprs.append(y_t)

        # Ensemble output: sum of trees + base_score
        outputs = m.continuous(f"{pfx}_output", shape=(1,))
        total = tree_output_exprs[0]
        for expr in tree_output_exprs[1:]:
            total = total + expr
        if ens.base_score != 0.0:
            total = total + ens.base_score
        m.subject_to(outputs[0] == total, name=f"{pfx}_ensemble_out")

        return inputs, outputs
