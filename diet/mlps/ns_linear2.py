import torch
from torch import nn
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


class NeuroselectiveLinear(nn.Module):
    """
    A simplified Linear layer that operates with reduced dimensions.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            in_indices: Optional[torch.Tensor] = None,
            out_indices: Optional[torch.Tensor] = None,
            bias: bool = True,
            device: Optional[Union[str, torch.device]] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Store original dimensions for reference
        self.original_in_features = in_features
        self.original_out_features = out_features

        # Get actual dimensions based on indices
        self.active_in_features = len(in_indices) if in_indices is not None else in_features
        self.active_out_features = len(out_indices) if out_indices is not None else out_features

        # Store indices
        self.register_buffer('in_indices', in_indices)
        self.register_buffer('out_indices', out_indices)

        # Create the reduced linear layer
        self.linear = nn.Linear(self.active_in_features, self.active_out_features, bias=bias, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve original shape information
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        # Flatten batch dimensions
        if len(batch_dims) > 0:
            x_flat = x.reshape(-1, self.original_in_features)
        else:
            x_flat = x.unsqueeze(0)  # Add batch dimension if none

        # Select input features if needed
        if self.in_indices is not None:
            x_active = torch.index_select(x_flat, 1, self.in_indices)
        else:
            x_active = x_flat

        # Apply reduced linear layer
        output_active = self.linear(x_active)

        # Expand output to original size if needed
        if self.out_indices is not None:
            output_flat = torch.zeros(x_flat.shape[0], self.original_out_features,
                                      device=output_active.device, dtype=output_active.dtype)
            output_flat[:, self.out_indices] = output_active
        else:
            output_flat = output_active

        # Restore original batch dimensions
        if len(batch_dims) > 0:
            output = output_flat.reshape(batch_dims + (self.original_out_features,))
        else:
            output = output_flat.squeeze(0)  # Remove batch dimension if added

        return output

    @classmethod
    def from_linear(
            cls,
            original_module: nn.Linear,
            in_indices: Optional[torch.Tensor] = None,
            out_indices: Optional[torch.Tensor] = None,
            **kwargs
    ):
        """Creates a NeuroselectiveLinear layer from a standard nn.Linear layer."""
        if not isinstance(original_module, nn.Linear):
            raise TypeError("original_module must be an instance of nn.Linear")

        orig_out_features, orig_in_features = original_module.weight.shape
        has_bias = original_module.bias is not None
        device = original_module.weight.device
        dtype = original_module.weight.dtype

        instance = cls(
            in_features=orig_in_features,
            out_features=orig_out_features,
            in_indices=in_indices,
            out_indices=out_indices,
            bias=has_bias,
            device=device,
            dtype=dtype
        )

        # Copy weights and biases
        with torch.no_grad():
            if in_indices is not None and out_indices is not None:
                # Both input and output dimensions are reduced
                reduced_weight = original_module.weight[out_indices][:, in_indices]
                instance.linear.weight.copy_(reduced_weight)

                if has_bias and instance.linear.bias is not None:
                    reduced_bias = original_module.bias[out_indices]
                    instance.linear.bias.copy_(reduced_bias)

            elif in_indices is not None:
                # Only input dimension is reduced
                reduced_weight = original_module.weight[:, in_indices]
                instance.linear.weight.copy_(reduced_weight)

                if has_bias and instance.linear.bias is not None:
                    instance.linear.bias.copy_(original_module.bias)

            elif out_indices is not None:
                # Only output dimension is reduced
                reduced_weight = original_module.weight[out_indices]
                instance.linear.weight.copy_(reduced_weight)

                if has_bias and instance.linear.bias is not None:
                    reduced_bias = original_module.bias[out_indices]
                    instance.linear.bias.copy_(reduced_bias)

            else:
                # No reduction, just copy
                instance.linear.weight.copy_(original_module.weight)

                if has_bias and instance.linear.bias is not None:
                    instance.linear.bias.copy_(original_module.bias)

        return instance

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"original_in={self.original_in_features}, original_out={self.original_out_features}, "
                f"active_in={self.active_in_features}, active_out={self.active_out_features}, "
                f"bias={self.linear.bias is not None})")