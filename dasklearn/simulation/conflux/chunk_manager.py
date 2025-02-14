import copy
from math import ceil
from random import Random
from typing import Dict, List

import torch
from torch import nn


class ChunkManager:

    @staticmethod
    def chunk_model(model: nn.Module, num_chunks: int) -> List[torch.Tensor]:
        # Chunk
        flat_params = ChunkManager.get_flat_params(model)
        total_elements = flat_params.numel()
        chunk_size = total_elements // num_chunks
        chunks: List[torch.Tensor] = [flat_params[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]

        # Handle any remaining elements
        if total_elements % num_chunks != 0:
            remaining = flat_params[num_chunks * chunk_size:]
            chunks[-1] = torch.cat([chunks[-1], remaining])

        return chunks

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
        for chunk_idx, chunks_at_idx in enumerate(chunks):
            chunks[chunk_idx] = torch.mean(torch.stack(chunks_at_idx), dim=0)

        # Reconstruct the flat tensor
        flat_params = torch.cat(chunks)

        # Copy the flat tensor into the model
        pointer = 0
        for param in model.state_dict().values():
            numel = param.data.numel()
            param_shape = param.data.shape
            param.data.copy_(flat_params[pointer:pointer + numel].view(param_shape))
            pointer += numel

        return model

    def aggregate_received_chunks(self):
        
        self.received_chunks = None
