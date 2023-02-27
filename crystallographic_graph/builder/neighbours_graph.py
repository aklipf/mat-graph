import torch
import torch.nn.functional as F

from torch_scatter import scatter_add, scatter_max

from typing import Tuple, Union, Optional

from ..utils import shape, assert_tensor_match, build_shapes


class NeighboursGraphBuilder:
    def __init__(
        self,
        cell: torch.FloatTensor,
        pos: torch.FloatTensor,
        num_atoms: torch.LongTensor,
        knn: int,
        cutoff: float,
        src_mask: torch.BoolTensor = None,
        dst_mask: torch.BoolTensor = None,
        symetric: bool = False,
        compute_reverse_idx: bool = False,
        check_tensor: bool = True
    ):
        # check inputs
        assert (compute_reverse_idx and symetric) or (not compute_reverse_idx)

        if src_mask is None:
            src_mask = torch.ones_like(pos[:, 0], dtype=torch.bool)

        if dst_mask is None:
            dst_mask = torch.ones_like(pos[:, 0], dtype=torch.bool)

        if check_tensor:
            self.shapes = assert_tensor_match(
                (cell, shape("b", 3, 3, dtype=torch.float32)),
                (pos, shape("n", 3, dtype=torch.float32)),
                (num_atoms, shape("b", dtype=torch.long)),
                (src_mask, shape("n", dtype=torch.bool)),
                (dst_mask, shape("n", dtype=torch.bool)),
            )

            assert num_atoms.sum() == pos.shape[0]
        else:
            self.shapes = build_shapes(
                {"device": cell.device, "b": cell.shape[0], "n": pos.shape[0]}
            )

        # setup graph properties
        self.symetric = symetric
        self.compute_reverse_idx = compute_reverse_idx

        # define limits
        self.knn = knn
        self.cutoff = cutoff

        # setup batch
        self.cell = cell
        self.pos = pos
        self.num_atoms = num_atoms
        self.src_mask = src_mask
        self.dst_mask = dst_mask

        self.struct_idx = torch.arange(
            self.shapes.b, device=self.shapes.device)
        self.batch = self.struct_idx.repeat_interleave(num_atoms)

        self.num_src = scatter_add(
            src_mask.int(), self.batch, dim=0, dim_size=self.shapes.b
        )
        self.num_dst = scatter_add(
            dst_mask.int(), self.batch, dim=0, dim_size=self.shapes.b
        )

        # setup src to dst edges index
        batch_struct, src, dst = self.build_coords_(self.num_src, self.num_dst)

        offset_src = F.pad(torch.cumsum(self.num_src, 0), (1, 0))
        offset_dst = F.pad(torch.cumsum(self.num_dst, 0), (1, 0))
        src += offset_src[batch_struct]
        dst += offset_dst[batch_struct]

        idx = torch.arange(self.shapes.n, dtype=torch.long,
                           device=self.shapes.device)

        self.src_idx = idx[self.src_mask][src]
        self.dst_idx = idx[self.dst_mask][dst]

        self.unique_src = torch.unique_consecutive(self.src_idx)

        self.src_idx_inv = torch.zeros_like(self.src_mask, dtype=torch.long)
        self.src_idx_inv[self.unique_src] = torch.arange(
            self.unique_src.shape[0], dtype=torch.long, device=self.shapes.device
        )

        # setup graph buffer
        self.distances = torch.empty(
            (0,), dtype=torch.float32, device=self.shapes.device
        )
        self.edges = torch.empty(
            (0, 5), dtype=torch.long, device=self.shapes.device)
        self.reverse_idx = None

    def subdivide_(
        self,
        batch: torch.LongTensor,
        idx: torch.LongTensor,
        block_size: torch.LongTensor,
    ):
        inner_batch = torch.div(idx, block_size[batch], rounding_mode="trunc")
        inner_idx = idx % block_size[batch]

        return inner_batch, inner_idx

    def build_coords_(self, *count):
        coords = []

        count = torch.stack(count, dim=0)

        device = count.device

        count_prod = count.prod(dim=0)
        total_count = count_prod.sum()

        idx = torch.arange(total_count, dtype=torch.long, device=device)
        batch_struct = torch.arange(
            count.shape[1], dtype=torch.long, device=device
        ).repeat_interleave(count_prod)

        cumsum_struct = F.pad(torch.cumsum(count_prod, 0), (1, 0))

        idx = idx - cumsum_struct[batch_struct]

        coords.append(batch_struct)

        for i in range(1, count.shape[0]):
            batch_current, idx = self.subdivide_(
                batch_struct,
                idx,
                count[i:].prod(dim=0),
            )
            coords.append(batch_current)

        coords.append(idx)

        return torch.stack(coords, dim=0)

    def blend_with_edges_(
        self, coords_cell: torch.IntTensor, num_cell: torch.IntTensor
    ):
        num_edges = self.num_src * self.num_dst
        num = num_edges * num_cell

        batch_blended = self.struct_idx.repeat_interleave(num)
        idx = torch.arange(
            batch_blended.shape[0], dtype=torch.long, device=self.shapes.device
        )
        ptr = F.pad(num.cumsum(dim=0), (1, 0))
        idx -= ptr[batch_blended]

        idx_edges = idx % num_edges[batch_blended]
        idx_cell = torch.div(
            idx, num_edges[batch_blended], rounding_mode="trunc")

        ptr_edges = F.pad(num_edges.cumsum(dim=0), (1, 0))
        idx_edges += ptr_edges[batch_blended]

        ptr_cell = F.pad(num_cell.cumsum(dim=0), (1, 0))
        idx_cell += ptr_cell[batch_blended]

        edges = torch.cat(
            (
                self.src_idx[idx_edges].unsqueeze(1),
                self.dst_idx[idx_edges].unsqueeze(1),
                coords_cell[idx_cell].long(),
            ),
            dim=1,
        )

        return edges

    def gen_edges_range_(
        self, start: torch.IntTensor, limit: torch.IntTensor
    ) -> torch.LongTensor:
        count = 2 * limit - 1

        coords = self.build_coords_(*(count.t()))

        struct_idx = coords[0]
        cell_coords = coords.t()[:, 1:] - limit[struct_idx] + 1

        mask = (
            (cell_coords <= (-start[struct_idx])
             ) | (start[struct_idx] <= cell_coords)
        ).any(dim=1)

        cell_coords = cell_coords[mask]

        num_cell = (2 * limit - 1).prod(dim=1) - (2 * start - 1).clamp(min=0).prod(
            dim=1
        )

        return self.blend_with_edges_(cell_coords, num_cell)

    def push(self, start: Optional[torch.IntTensor], limit: torch.IntTensor = None):
        # check inputs
        if limit is None:
            start, limit = torch.zeros_like(start), start

        # calculate coordinate to evaluate
        edges = self.gen_edges_range_(start, limit)

        # calculate the new distances
        e_ij = self.pos[edges[:, 1]] + \
            edges[:, 2:].float() - self.pos[edges[:, 0]]
        v_ij = torch.bmm(
            self.cell[self.batch[edges[:, 0]]],
            e_ij.unsqueeze(2),
        ).squeeze(2)
        r_ij = v_ij.norm(dim=1)

        # concatenate and reorder inside a single batch sorted by distance
        self.distances = torch.cat((self.distances, r_ij), dim=0)
        self.edges = torch.cat((self.edges, edges), dim=0)

        self.distances, idx = torch.sort(self.distances)

        self.edges = self.edges[idx]

        idx = torch.sort(self.edges[:, 0], stable=True).indices

        self.distances = self.distances[idx]
        self.edges = self.edges[idx]

        # filter the results
        mask = self.distances > 1e-3
        self.distances = self.distances[mask]
        self.edges = self.edges[mask]

        k_distances = self.get_k_distance(self.knn)
        threshold = k_distances + self.cutoff + 1e-6

        mask = self.distances <= threshold[self.src_idx_inv[self.edges[:, 0]]]

        self.distances = self.distances[mask]
        self.edges = self.edges[mask]

        return k_distances

    def get_k_distance(self, k: int):
        if k == 0:
            return torch.zeros_like(self.unique_src, dtype=torch.float32)

        if self.distances.shape[0] == 0:
            return torch.full(
                (self.shapes.n,), fill_value=float("inf"), device=self.shapes.device
            )

        edge_i = self.edges[:, 0].contiguous()

        idx_left = torch.bucketize(self.unique_src, edge_i, right=False)
        idx_right = torch.bucketize(self.unique_src, edge_i, right=True)

        kth = idx_left + k - 1

        k_distances = self.distances[kth.clamp(max=self.edges.shape[0] - 1)]

        k_distances[idx_right <= kth] = float("inf")

        return k_distances

    def get_height(self) -> torch.FloatTensor:
        normal = torch.cross(
            self.cell[:, :, [1, 2, 0]], self.cell[:, :, [2, 0, 1]], dim=1
        )

        normal = F.normalize(normal, dim=1)

        height = (normal * self.cell).sum(dim=1)

        return height

    def closest_border(self, limit: torch.IntTensor) -> torch.FloatTensor:
        normal = torch.cross(
            self.cell[:, :, [1, 2, 0]], self.cell[:, :, [2, 0, 1]], dim=1
        )

        pos = self.pos[self.unique_src]
        batch = self.batch[self.unique_src]

        normal = F.normalize(normal, dim=1)

        height = (normal * self.cell).sum(dim=1)

        cart_pos = torch.bmm(self.cell[batch], pos.unsqueeze(2)).squeeze(2)
        proj = (normal[batch] * cart_pos[:, :, None]).sum(dim=1)

        left = (1 - limit) * height
        right = limit * height

        dist = torch.stack((proj - left[batch], right[batch] - proj), dim=2)

        closest_dist, closest_dim = dist.min(dim=2).values.min(dim=1)

        return closest_dist, closest_dim

    def invert_edges(self, edges: torch.LongTensor) -> torch.LongTensor:
        inverted = torch.empty_like(edges)
        inverted[:, [0, 1]] = edges[:, [1, 0]]
        inverted[:, [2, 3, 4]] = -edges[:, [2, 3, 4]]
        return inverted

    def tokenize(self, edges: torch.LongTensor) -> torch.LongTensor:
        max_idx = edges[:, :2].max()+1
        min_coords, max_coords = edges[:, 2:].min(), edges[:, 2:].max()
        diff_coords = max_coords-min_coords+1

        return ((((edges[:, 0]*max_idx+edges[:, 1])*diff_coords+edges[:, 2]-min_coords) *
                diff_coords+edges[:, 3]-min_coords)*diff_coords+edges[:, 4]-min_coords)

    def get_symetric(self, edges: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        inverted = self.invert_edges(edges)
        full_edges = torch.cat((edges, inverted), dim=0)

        tokens = self.tokenize(full_edges)

        values, index = tokens.sort()
        mask = F.pad(values[:-1] != values[1:], (1, 0), value=True)

        index = index[mask]

        filtered_edges = full_edges[index]
        filtered_tokens = tokens[index]

        return filtered_edges, filtered_tokens

    def get_reverse_idx(self, edges: torch.LongTensor, tokens: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        reverse_tokens = self.tokenize(self.invert_edges(edges))

        reverse_idx = torch.bucketize(reverse_tokens, tokens)

        return reverse_idx

    def build(self):
        start = torch.full_like(
            self.cell[:, :, 0], fill_value=0, dtype=torch.int32)
        limit = torch.full_like(
            self.cell[:, :, 0], fill_value=1, dtype=torch.int32)

        search = True

        batch = self.batch[self.unique_src]

        while search:

            k_distances = self.push(start, limit)

            search_area = k_distances + self.cutoff

            closest, closest_dim = self.closest_border(limit)

            # check if all search area are inside of the evaluated area
            should_continue = closest <= search_area

            mask = should_continue[:, None].int() & F.one_hot(
                closest_dim, num_classes=3
            )
            mask_iter = scatter_max(mask, batch, dim=0, dim_size=self.shapes.b)[
                0
            ].bool()

            # iterate if needed
            start = limit.clone()
            limit[mask_iter] += 1

            search = should_continue.any()

        if self.symetric:
            self.edges, tokens = self.get_symetric(self.edges)

            if self.compute_reverse_idx:
                self.reverse_idx = self.get_reverse_idx(self.edges, tokens)
