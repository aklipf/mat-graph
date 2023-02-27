import torch
import torch.nn.functional as F

from torch_scatter import scatter_add

from typing import Tuple, Union, Optional

from ..utils import shape, assert_tensor_match, build_shapes
from ..data import Edges, Triplets
from ..meshgrid import sparse_meshgrid


class TripletsBuilder:
    def __init__(
        self,
        num_atoms: torch.LongTensor,
        edges: Edges,
        check_tensor: bool = True,
    ):
        if check_tensor:
            self.shapes = assert_tensor_match(
                (num_atoms, shape("b", dtype=torch.long)),
                (edges.src, shape("n", dtype=torch.long)),
                (edges.dst, shape("n", dtype=torch.long)),
                (edges.cell, shape("n", 3, dtype=torch.long)),
            )

            assert edges.src.max() < num_atoms.sum()
            assert edges.dst.max() < num_atoms.sum()
            assert (edges.src[:-1] <= edges.src[1:]).all()
        else:
            self.shapes = build_shapes(
                {
                    "device": num_atoms.device,
                    "b": num_atoms.shape[0],
                    "n": edges.src.shape[0],
                }
            )

        self.edges = edges

        self.num_atoms = num_atoms
        self.struct_idx = torch.arange(
            self.shapes.b, device=self.shapes.device)
        self.batch = self.struct_idx.repeat_interleave(num_atoms)

        self.triplets = None

    def build(self):
        n_atoms = self.num_atoms.sum()

        num_edges = scatter_add(
            torch.ones_like(self.edges.src),
            self.edges.src,
            dim=0,
            dim_size=n_atoms,
        )

        i_triplets, j_triplets = sparse_meshgrid(num_edges)

        mask = i_triplets != j_triplets
        i_triplets = i_triplets[mask]
        j_triplets = j_triplets[mask]

        self.triplets = Triplets(
            src=self.edges.src[i_triplets],
            dst_i=self.edges.dst[i_triplets],
            cell_i=self.edges.cell[i_triplets],
            dst_j=self.edges.dst[j_triplets],
            cell_j=self.edges.cell[j_triplets],
        )
