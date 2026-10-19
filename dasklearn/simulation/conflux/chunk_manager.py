import copy
from math import ceil
from random import Random
from typing import Dict, List

import torch
from torch import nn


class ChunkManager:

    @staticmethod
    def chunk_model(model: nn.Module, num_chunks: int, weights=None) -> List[torch.Tensor]:
        # Chunk
        flat_params = ChunkManager.get_flat_params(model)
        total_elements = flat_params.numel()
        if not 1 <= num_chunks <= total_elements:
            raise ValueError("Chunk count must be between one and the model's state size")
        if weights is not None:
            if len(weights) != num_chunks:
                raise ValueError("One partition weight is required per chunk")
            return list(torch.split(flat_params, ChunkManager.partition_sizes(total_elements, weights)))
        chunk_size = total_elements // num_chunks
        chunks: List[torch.Tensor] = [flat_params[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]

        # Handle any remaining elements
        if total_elements % num_chunks != 0:
            remaining = flat_params[num_chunks * chunk_size:]
            chunks[-1] = torch.cat([chunks[-1], remaining])

        return chunks

    @staticmethod
    def partition_sizes(total, weights):
        """Deterministic integer apportionment, conserving every tensor/byte."""
        import math
        if not weights or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Partition weights must be finite, nonnegative and have positive sum")
        raw = [total * w / sum(weights) for w in weights]
        sizes = [int(x) for x in raw]
        for index in sorted(range(len(weights)), key=lambda i: (-(raw[i] - sizes[i]), i))[:total - sum(sizes)]:
            sizes[index] += 1
        return sizes

    @staticmethod
    def get_flat_params(model):
        param_tensors = [param.data.view(-1) for param in model.state_dict().values()]
        flat_params = torch.cat(param_tensors)
        return flat_params

    @staticmethod
    def reconstruct_model(chunks: List[List[torch.Tensor]], model: nn.Module) -> nn.Module:
        for idx in range(len(chunks)):
            assert chunks[idx], "No chunks received at index %d!" % idx

        # Aggregate the chunks at every index
        averaged_chunks = [torch.mean(torch.stack(chunks_at_idx), dim=0) for chunks_at_idx in chunks]

        # Reconstruct the flat tensor
        flat_params = torch.cat(averaged_chunks)

        # Copy the flat tensor into the model
        pointer = 0
        for param in model.state_dict().values():
            numel = param.data.numel()
            param_shape = param.data.shape
            param.data.copy_(flat_params[pointer:pointer + numel].view(param_shape))
            pointer += numel

        return model

    @staticmethod
    def weighted_reconstruct_model(chunks: List[torch.Tensor], model: nn.Module) -> nn.Module:
        flat_params = torch.cat(chunks)

        # Copy the flat tensor into the model
        pointer = 0
        for param in model.state_dict().values():
            numel = param.data.numel()
            param_shape = param.data.shape
            param.data.copy_(flat_params[pointer:pointer + numel].view(param_shape))
            pointer += numel

        return model
