import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple, Dict
from transformers import DetrConfig
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.detr.modeling_detr import (
    BaseModelOutput, DetrDecoderOutput, DetrModelOutput,
    DetrObjectDetectionOutput, DetrSegmentationOutput
)
from transformers import DetrPreTrainedModel
from transformers.image_transforms import center_to_corners_format
from scipy.optimize import linear_sum_assignment
from timm import create_model
from torch import Tensor


class DetrConvEncoder(nn.Module):
    """
    Convolutional Encoder for DETR with optional support for a timm-based backbone.
    """

    def __init__(self, config: DetrConfig) -> None:
        """
        Initializes the convolutional encoder.

        Args:
            config (DetrConfig): Configuration object containing model parameters.
        """
        super().__init__()

        self.config = config

        if config.use_timm_backbone:
            kwargs = getattr(config, "backbone_kwargs", {})
            out_indices = kwargs.pop("out_indices", (1, 2, 3, 4))
            num_channels = kwargs.pop("in_chans", config.num_channels)

            if config.dilation:
                kwargs["output_stride"] = kwargs.get("output_stride", 16)

            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=out_indices,
                in_chans=num_channels,
                **kwargs
            )

        self.model = backbone
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        backbone_model_type = config.backbone or (config.backbone_config and config.backbone_config.model_type)
        if backbone_model_type is None:
            raise ValueError("Either `backbone` or `backbone_config` should be provided in the config.")

        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if not any(layer in name for layer in ["layer2", "layer3", "layer4"]):
                        parameter.requires_grad_(False)
                else:
                    if not any(stage in name for stage in ["stage.1", "stage.2", "stage.3"]):
                        parameter.requires_grad_(False)

    def forward(self, pixel_values: Tensor, pixel_mask: Tensor) -> List[tuple]:
        """
        Forward pass for the convolutional encoder.

        Args:
            pixel_values (Tensor): Input image tensor of shape (batch_size, channels, height, width).
            pixel_mask (Tensor): Binary mask indicating valid pixels.

        Returns:
            List[tuple]: A list of feature maps and their corresponding masks.
        """
        features = self.model(pixel_values)

        output = []
        for feature_map in features:
            # Downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            output.append((feature_map, mask))

        return output


class DetrSinePositionEmbedding(nn.Module):
    """
    Computes sine-based position embeddings for DETR.
    """

    def __init__(self, embedding_dim: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None) -> None:
        """
        Initializes the sine position embedding module.

        Args:
            embedding_dim (int): Dimension of the embedding.
            temperature (int): Scaling factor for position encoding.
            normalize (bool): Whether to normalize position embeddings.
            scale (Optional[float]): Scaling factor applied to normalized embeddings.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, pixel_values: Tensor, pixel_mask: Tensor) -> Tensor:
        """
        Forward pass for sine position embedding.

        Args:
            pixel_values (Tensor): Input feature map tensor of shape (batch_size, channels, height, width).
            pixel_mask (Tensor): Binary mask indicating valid pixels.

        Returns:
            Tensor: Position embeddings of shape (batch_size, embedding_dim, height, width).
        """
        if pixel_mask is None:
            raise ValueError("No pixel mask provided.")

        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=-1).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=-1).flatten(3)

        pos = torch.cat((pos_x, pos_y), dim=3).permute(0, 3, 1, 2)
        return pos


class DetrLearnedPositionEmbedding(nn.Module):
    """
    Learned 2D position embeddings for DETR.
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        """
        Initializes the learned position embedding module.

        Args:
            embedding_dim (int): Dimension of the embeddings.
        """
        super().__init__()
        self.row_embedding = nn.Embedding(50, embedding_dim)
        self.column_embedding = nn.Embedding(50, embedding_dim)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes learned position embeddings.

        Args:
            pixel_values (torch.Tensor): Input feature map tensor of shape (batch_size, channels, height, width).
            pixel_mask (Optional[torch.Tensor]): Binary mask indicating valid pixels.

        Returns:
            torch.Tensor: Learned position embeddings of shape (batch_size, embedding_dim, height, width).
        """
        height, width = pixel_values.shape[-2:]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)

        x_emb = self.column_embedding(width_values)
        y_emb = self.row_embedding(height_values)

        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(pixel_values.shape[0], 1, 1, 1)

        return pos


def build_position_encoding(config) -> nn.Module:
    """
    Builds position encoding based on configuration.

    Args:
        config: Configuration object containing position embedding type.

    Returns:
        nn.Module: Position embedding module.
    """
    n_steps = config.d_model // 2
    if config.position_embedding_type == "sine":
        return DetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        return DetrLearnedPositionEmbedding(n_steps)
    else:
        raise ValueError(f"Unsupported position embedding type: {config.position_embedding_type}")


class DetrAttention(nn.Module):
    """
    Multi-head self-attention mechanism for DETR.
    """

    def __init__(self, embed_dim: int, num_heads: int, drop_out: float = 0.0, bias: bool = True) -> None:
        """
        Initializes the multi-head attention module.

        Args:
            embed_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            drop_out (float): Dropout probability.
            bias (bool): Whether to use bias in projection layers.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = drop_out
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** 0.5
        self.bias = bias

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        """
        Reshapes the input tensor for multi-head attention.

        Args:
            tensor (torch.Tensor): Input tensor.
            seq_len (int): Sequence length.
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Reshaped tensor.
        """
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Adds positional embeddings if provided.

        Args:
            tensor (torch.Tensor): Input tensor.
            object_queries (Optional[torch.Tensor]): Positional embeddings.

        Returns:
            torch.Tensor: Tensor with positional embeddings added.
        """
        return tensor if object_queries is None else tensor + object_queries

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        spatial_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> tuple:
        """
        Computes multi-head self-attention.

        Args:
            hidden_states (torch.Tensor): Input hidden states of shape (batch_size, seq_len, embed_dim).
            attention_mask (Optional[torch.Tensor]): Attention mask.
            object_queries (Optional[torch.Tensor]): Positional queries.
            key_value_states (Optional[torch.Tensor]): Key-value states for cross-attention.
            spatial_position_embeddings (Optional[torch.Tensor]): Spatial embeddings.
            output_attentions (bool): Whether to output attention weights.

        Returns:
            tuple: (attn_output, attn_weights)
                - attn_output (torch.Tensor): Attention output of shape (batch_size, seq_len, embed_dim).
                - attn_weights (Optional[torch.Tensor]): Attention weights if `output_attentions` is True.
        """
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.size()

        if object_queries is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, object_queries)

        if spatial_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, spatial_position_embeddings)

        query_states = self.q_proj(hidden_states) * self.scaling

        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, "
                f"but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, target_len, source_len)}, "
                    f"but is {attention_mask.size()}"
                )

            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class DetrEncoderLayer(nn.Module):
    """
    Transformer encoder layer for DETR.
    """

    def __init__(self, config: DetrConfig) -> None:
        """
        Initializes an encoder layer.

        Args:
            config (DetrConfig): Configuration object containing model parameters.
        """
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            drop_out=config.attention_dropout
        )

        self.self_attn_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        object_queries: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> tuple:
        """
        Forward pass of the encoder layer.

        Args:
            hidden_states (torch.Tensor): Input hidden states of shape (batch_size, seq_len, embed_dim).
            attention_mask (torch.Tensor): Attention mask.
            object_queries (Optional[torch.Tensor]): Optional object queries for positional encoding.
            output_attentions (bool): Whether to output attention weights.

        Returns:
            tuple: (hidden_states, attn_weights) if `output_attentions=True`, else just (hidden_states,).
        """
        residual = hidden_states

        # Self-attention
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            output_attentions=output_attentions
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_norm(hidden_states)

        # Feed-forward network
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # Prevent NaN or Inf values
        if self.training and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class DetrDecoderLayer(nn.Module):
    """
    Transformer decoder layer for DETR.
    """

    def __init__(self, config: DetrConfig) -> None:
        """
        Initializes a decoder layer.

        Args:
            config (DetrConfig): Configuration object containing model parameters.
        """
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            drop_out=config.attention_dropout
        )

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.encoder_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            drop_out=config.attention_dropout
        )

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple:
        """
        Forward pass of the decoder layer.

        Args:
            hidden_states (torch.Tensor): Input hidden states of shape (batch_size, seq_len, embed_dim).
            attention_mask (Optional[torch.Tensor]): Self-attention mask.
            object_queries (Optional[torch.Tensor]): Positional queries.
            query_position_embeddings (Optional[torch.Tensor]): Query embeddings for positional encoding.
            encoder_hidden_states (Optional[torch.Tensor]): Encoder hidden states.
            encoder_attention_mask (Optional[torch.Tensor]): Mask for encoder-decoder attention.
            output_attentions (Optional[bool]): Whether to return attention weights.

        Returns:
            tuple: (hidden_states, self_attn_weights, cross_attn_weights) if `output_attentions=True`,
                   else just (hidden_states,).
        """
        residual = hidden_states

        # Self-attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            object_queries=query_position_embeddings,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-attention
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                object_queries=query_position_embeddings,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                spatial_position_embeddings=object_queries,
                output_attentions=output_attentions,
            )

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Feed-forward network
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class DetrEncoder(nn.Module):
    """
    Transformer encoder consisting of multiple `DetrEncoderLayer` layers.
    """

    def __init__(self, config: DetrConfig) -> None:
        """
        Initializes the transformer encoder.

        Args:
            config (DetrConfig): Configuration object containing model parameters.
        """
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        self.layers = nn.ModuleList([DetrEncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> BaseModelOutput:
        """
        Forward pass for the encoder.

        Args:
            input_embeds (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask (Optional[torch.Tensor]): Attention mask.
            object_queries (Optional[torch.Tensor]): Positional queries.
            output_attentions (Optional[bool]): Whether to return attention weights.
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary.

        Returns:
            BaseModelOutput: Object containing last hidden state, hidden states, and attention weights.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, input_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states += (hidden_states,)

            if self.training and torch.rand(1) < self.layerdrop:  # Apply LayerDrop
                continue

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                object_queries=object_queries,
                output_attentions=output_attentions
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            encoder_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions
        )


class DetrDecoder(nn.Module):
    """
    Transformer decoder consisting of multiple `DetrDecoderLayer` layers.
    """

    def __init__(self, config: DetrConfig) -> None:
        """
        Initializes the transformer decoder.

        Args:
            config (DetrConfig): Configuration object containing model parameters.
        """
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        self.layers = nn.ModuleList([DetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> DetrDecoderOutput:
        """
        Forward pass for the decoder.

        Args:
            inputs_embeds (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask (Optional[torch.Tensor]): Self-attention mask.
            encoder_hidden_states (Optional[torch.Tensor]): Encoder hidden states.
            encoder_attention_mask (Optional[torch.Tensor]): Encoder-decoder attention mask.
            object_queries (Optional[torch.Tensor]): Positional queries.
            query_position_embeddings (Optional[torch.Tensor]): Query embeddings.
            output_attentions (Optional[bool]): Whether to return attention weights.
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary.

        Returns:
            DetrDecoderOutput: Object containing last hidden state, hidden states, and attention weights.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeds
        input_shape = hidden_states.shape[:-1]

        combined_attention_mask = None

        if attention_mask is not None:
            combined_attention_mask = _prepare_4d_attention_mask(
                attention_mask, hidden_states.dtype, tgt_len=input_shape[-1]
            )

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, hidden_states.dtype, tgt_len=input_shape[-1]
            )

        intermediate = () if self.config.auxiliary_loss else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training and torch.rand(1) < self.dropout:  # Apply LayerDrop
                continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=combined_attention_mask,
                object_queries=object_queries,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions
            )

            hidden_states = layer_outputs[0]

            if self.config.auxiliary_loss:
                hidden_states = self.layernorm(hidden_states)
                intermediate += (hidden_states,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.config.auxiliary_loss:
            intermediate = torch.stack(intermediate)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions, intermediate] if v is not None)

        return DetrDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            intermediate_hidden_states=intermediate
        )


class DetrConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder: nn.Module, position_embedding: nn.Module) -> None:
        """
        Initializes the convolutional model with positional embeddings.

        Args:
            conv_encoder (nn.Module): Convolutional encoder model.
            position_embedding (nn.Module): Position embedding module.
        """
        super().__init__()
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def forward(
        self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor]]:
        """
        Forward pass for convolutional model with position embeddings.

        Args:
            pixel_values (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            pixel_mask (torch.Tensor): Binary mask indicating valid pixels.

        Returns:
            Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor]]:
                - List of (feature_map, pixel_mask) tuples from the encoder.
                - List of corresponding position embeddings.
        """
        # Get feature maps and masks from convolutional encoder
        out = self.conv_encoder(pixel_values, pixel_mask)

        # Compute position embeddings for each feature map
        pos = [self.position_embedding(feature_map, mask).to(feature_map.dtype) for feature_map, mask in out]

        return out, pos


class DetrModel(DetrPreTrainedModel):
    """
    The core DETR model that combines convolutional and transformer-based processing.
    """

    def __init__(self, config: DetrConfig) -> None:
        """
        Initializes the DETR model.

        Args:
            config (DetrConfig): Configuration object containing model parameters.
        """
        super().__init__(config)

        # Initialize convolutional encoder with position embeddings
        backbone = DetrConvEncoder(config)
        object_queries = build_position_encoding(config)
        self.backbone = DetrConvModel(backbone, object_queries)

        # Projection layer to match feature map dimensions with transformer input
        self.input_projection = nn.Conv2d(
            in_channels=backbone.intermediate_channel_sizes[-1],
            out_channels=config.d_model,
            kernel_size=1
        )

        # Query position embeddings for transformer decoder
        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        # Transformer-based encoder and decoder
        self.encoder = DetrEncoder(config)
        self.decoder = DetrDecoder(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ):
        """
        Forward pass for the DETR model.

        Args:
            pixel_values (torch.FloatTensor): Input images of shape (batch_size, channels, height, width).
            pixel_mask (Optional[torch.LongTensor]): Mask indicating valid pixels.
            decoder_attention_mask (Optional[torch.FloatTensor]): Attention mask for the decoder.
            encoder_outputs (Optional[torch.FloatTensor]): Precomputed encoder outputs.
            inputs_embeds (Optional[torch.FloatTensor]): Input embeddings for the model.
            decoder_inputs_embeds (Optional[torch.FloatTensor]): Input embeddings for the decoder.
            output_attentions (bool): Whether to return attention weights.
            output_hidden_states (bool): Whether to return hidden states.
            return_dict (bool): Whether to return outputs as a dictionary.

        Returns:
            DetrModelOutput: Model output containing hidden states, attentions, and other relevant tensors.
        """
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        # If no pixel mask is provided, assume all pixels are valid
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)

        # Extract feature maps and position embeddings
        features, object_queries_list = self.backbone(pixel_values, pixel_mask)
        feature_map, mask = features[-1]

        # Project feature map to match transformer input dimension
        projected_feature_map = self.input_projection(feature_map)

        # Flatten feature maps for transformer processing
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)
        flattened_mask = mask.flatten(1)

        # Process feature maps through the transformer encoder
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_embeds=flattened_features,
                attention_mask=flattened_mask,
                object_queries=object_queries,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        # Generate query position embeddings for transformer decoder
        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)

        # Process queries through the transformer decoder
        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            object_queries=object_queries,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return DetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
        )


class DetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _upcast(t: Tensor) -> Tensor:
    """
    Protects from numerical overflows in multiplications by upcasting to the equivalent higher type.

    Args:
        t (Tensor): Input tensor.

    Returns:
        Tensor: Upcasted tensor.
    """
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of bounding boxes in (x1, y1, x2, y2) format.

    Args:
        boxes (Tensor): Bounding boxes of shape `(N, 4)`, expected to be in (x1, y1, x2, y2) format.

    Returns:
        Tensor: Area of each bounding box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1 (Tensor): First set of bounding boxes.
        boxes2 (Tensor): Second set of bounding boxes.

    Returns:
        Tuple[Tensor, Tensor]: IoU matrix and union matrix.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (right_bottom - left_top).clamp(min=0)
    inter = width_height[:, :, 0] * width_height[:, :, 1]
    union = area1[:, None] + area2 - inter

    return inter / union, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Computes the Generalized IoU (GIoU) between two sets of bounding boxes.

    Args:
        boxes1 (Tensor): First set of bounding boxes.
        boxes2 (Tensor): Second set of bounding boxes.

    Returns:
        Tensor: Generalized IoU matrix.
    """
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] format, but got {boxes2}")

    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


class DetrHungarianMatcher(nn.Module):
    """
    Computes an optimal assignment between targets and predictions using the Hungarian algorithm.

    Args:
        class_cost (float): Weight of classification error in matching cost.
        bbox_cost (float): Weight of L1 error of bounding box coordinates.
        giou_cost (float): Weight of GIoU loss in matching cost.
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1) -> None:
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs cannot be zero.")

    @torch.no_grad()
    def forward(self, outputs: dict, targets: List[dict]) -> List[Tuple[Tensor, Tensor]]:
        """
        Matches predictions to targets using the Hungarian algorithm.

        Args:
            outputs (dict): Model predictions containing `logits` and `pred_boxes`.
            targets (List[dict]): Ground truth targets containing `class_labels` and `boxes`.

        Returns:
            List[Tuple[Tensor, Tensor]]: Indices mapping predictions to ground truth.
        """
        batch_size, num_queries = outputs["logits"].shape[:2]
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])

        class_cost = -out_prob[:, target_ids]
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def sigmoid_focal_loss(inputs: Tensor, targets: Tensor, num_boxes: int, alpha: float = 0.25, gamma: float = 2) -> Tensor:
    """
    Computes sigmoid focal loss.

    Args:
        inputs (Tensor): Predicted logits.
        targets (Tensor): Ground truth labels.
        num_boxes (int): Number of bounding boxes.
        alpha (float): Weighting factor for balancing positive vs. negative examples.
        gamma (int): Exponent for modulating easy vs. hard examples.

    Returns:
        Tensor: Computed loss.
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def dice_loss(inputs: Tensor, targets: Tensor, num_boxes: int) -> Tensor:
    """
    Computes DICE loss for mask prediction.

    Args:
        inputs (Tensor): Predicted masks.
        targets (Tensor): Ground truth masks.
        num_boxes (int): Number of masks.

    Returns:
        Tensor: Computed loss.
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).sum() / num_boxes


class NestedTensor:
    """
    Nested tensor structure that holds both tensor values and masks.
    """

    def __init__(self, tensors: Tensor, mask: Optional[Tensor]) -> None:
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device) -> "NestedTensor":
        return NestedTensor(self.tensors.to(device), self.mask.to(device) if self.mask is not None else None)

    def decompose(self) -> Tuple[Tensor, Optional[Tensor]]:
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)


# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
class DetrLoss(nn.Module):
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box).

    A note on the `num_classes` argument (copied from original repo in detr.py): "the naming of the `num_classes`
    parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where `max_obj_id` is
    the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass `num_classes` to
    be 91. As another example, for a dataset that has a single class with `id` 1, you should pass `num_classes` to be 2
    (`max_obj_id` + 1). For more details on this, check the following discussion
    https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"


    Args:
        matcher (`DetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, matcher, num_classes, eos_coef, losses):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        # upsample predictions to the target size
        source_masks = nn.functional.interpolate(
            source_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        source_masks = source_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
    

class DetrForObjectDetection(DetrPreTrainedModel):
    """
    DETR model for object detection.

    This model consists of a convolutional backbone, a transformer-based encoder-decoder, 
    and classification and bounding box regression heads.
    """

    def __init__(self, config: DetrConfig) -> None:
        """
        Initializes the DETR model for object detection.

        Args:
            config (DetrConfig): Configuration object containing model parameters.
        """
        super().__init__(config)
        self.model = DetrModel(config)

        # Classification head
        self.class_labels_classifier = nn.Linear(config.d_model, config.num_labels + 1)

        # Bounding box prediction head
        self.bbox_predictor = DetrMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[Dict[str, torch.Tensor]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> DetrSegmentationOutput:
        """
        Forward pass for object detection.

        Args:
            pixel_values (torch.FloatTensor): Input images of shape (batch_size, channels, height, width).
            pixel_mask (Optional[torch.LongTensor]): Mask indicating valid pixels.
            decoder_attention_mask (Optional[torch.FloatTensor]): Attention mask for the decoder.
            encoder_outputs (Optional[torch.FloatTensor]): Precomputed encoder outputs.
            inputs_embeds (Optional[torch.FloatTensor]): Input embeddings for the model.
            decoder_inputs_embeds (Optional[torch.FloatTensor]): Input embeddings for the decoder.
            labels (Optional[List[Dict[str, torch.Tensor]]]): List of dictionaries with ground truth labels and boxes.
            output_attentions (Optional[bool]): Whether to return attention weights.
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            return_dict (Optional[bool]): Whether to return outputs as a dictionary.

        Returns:
            DetrSegmentationOutput: Model output containing classification logits, bounding box predictions,
            hidden states, and optionally, computed losses.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get transformer model outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.class_labels_classifier(outputs[0])  # Classification logits
        pred_boxes = self.bbox_predictor(outputs[0]).sigmoid()  # Bounding box predictions

        loss, loss_dict, auxiliary_outputs = None, None, None

        if labels is not None:
            # Compute Hungarian matching
            matcher = DetrHungarianMatcher(
                class_cost=self.config.class_cost,
                bbox_cost=self.config.bbox_cost,
                giou_cost=self.config.giou_cost
            )

            # Define losses
            losses = ["labels", "boxes", "cardinality", "masks"]
            criterion = DetrLoss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                eos_coef=self.config.eos_coefficient,
                losses=losses
            )
            criterion.to(self.device)

            # Compute loss
            outputs_loss = {
                "logits": logits,
                "pred_boxes": pred_boxes
            }

            if hasattr(self.config, "auxiliary_loss") and self.config.auxiliary_loss:
                intermediate = outputs.intermediate_hidden_states if return_dict else outputs[-1]
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs

            loss_dict = criterion(outputs_loss, labels)

            # Compute total loss as a weighted sum of individual losses
            weight_dict = {
                "loss_ce": 1,
                "loss_bbox": self.config.bbox_loss_coefficient,
                "loss_giou": self.config.giou_loss_coefficient,
                "loss_mask": self.config.mask_loss_coefficient,
                "loss_dice": self.config.dice_loss_coefficient
            }

            if hasattr(self.config, "auxiliary_loss") and self.config.auxiliary_loss:
                aux_weight_dict = {f"{k}_{i}": v for i in range(self.config.decoder_layers - 1) for k, v in weight_dict.items()}
                weight_dict.update(aux_weight_dict)

            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not return_dict:
            output = (logits, pred_boxes)
            if auxiliary_outputs is not None:
                output += auxiliary_outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return DetrSegmentationOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def _set_aux_loss(self, outputs_class: torch.Tensor, outputs_coord: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Sets auxiliary losses for intermediate decoder layers.

        Args:
            outputs_class (torch.Tensor): Classification logits for intermediate layers.
            outputs_coord (torch.Tensor): Bounding box predictions for intermediate layers.

        Returns:
            List[Dict[str, torch.Tensor]]: List of dictionaries containing auxiliary outputs.
        """
        return [{"logits": c, "pred_boxes": b} for c, b in zip(outputs_class[:-1], outputs_coord[:-1])]