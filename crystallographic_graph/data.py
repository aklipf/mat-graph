from dataclasses import dataclass

import torch


class Tensors:
    def to(self, device: torch.device):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.to(device)


@dataclass
class Edges(Tensors):
    src: torch.LongTensor
    dst: torch.LongTensor
    cell: torch.LongTensor
    reverse_idx: torch.LongTensor


@dataclass
class Triplets(Tensors):
    src: torch.LongTensor
    dst_i: torch.LongTensor
    cell_i: torch.LongTensor
    dst_j: torch.LongTensor
    cell_j: torch.LongTensor
