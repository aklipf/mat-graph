import torch

from .data import Edges, Triplets
from .builder import NeighboursGraphBuilder, TripletsBuilder


@torch.no_grad()
def make_graph(
    cell: torch.FloatTensor,
    pos: torch.FloatTensor,
    num_atoms: torch.LongTensor,
    knn: int = 0,
    cutoff: float = 0.0,
    src_mask: torch.BoolTensor = None,
    dst_mask: torch.BoolTensor = None,
    symetric: bool = False,
    compute_reverse_idx: bool = False,
    check_tensor: bool = True
) -> Edges:

    ngb = NeighboursGraphBuilder(
        cell=cell,
        pos=pos,
        num_atoms=num_atoms,
        knn=knn,
        cutoff=cutoff,
        src_mask=src_mask,
        dst_mask=dst_mask,
        symetric=symetric,
        compute_reverse_idx=compute_reverse_idx,
        check_tensor=check_tensor
    )

    ngb.build()

    return Edges(src=ngb.edges[:, 0], dst=ngb.edges[:, 1], cell=ngb.edges[:, 2:], reverse_idx=ngb.reverse_idx)


@torch.no_grad()
def make_triplets(
    num_atoms: torch.LongTensor,
    edges: Edges,
    check_tensor: bool = True,
) -> Triplets:

    tb = TripletsBuilder(
        num_atoms=num_atoms,
        edges=edges,
        check_tensor=check_tensor,
    )

    tb.build()

    return tb.triplets
