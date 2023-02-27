import torch
import torch.nn.functional as F


def sparse_meshgrid(num_idx: torch.LongTensor) -> torch.LongTensor:
    num_triplets = num_idx.pow(2)
    cumsum_triplets = F.pad(num_triplets.cumsum(0), (1, 0))
    cumsum_edges = F.pad(num_idx.cumsum(0), (1, 0))

    batch_triplets = torch.arange(
        num_idx.shape[0], dtype=torch.long, device=cumsum_triplets.device
    ).repeat_interleave(num_triplets)

    idx = torch.arange(
        cumsum_triplets[-1], dtype=torch.long, device=cumsum_triplets.device
    )
    idx -= cumsum_triplets[batch_triplets]

    i_triplets = (idx % num_idx[batch_triplets]) + cumsum_edges[batch_triplets]
    j_triplets = (
        torch.div(idx, num_idx[batch_triplets], rounding_mode="floor")
        + cumsum_edges[batch_triplets]
    )

    return i_triplets, j_triplets
