import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """
    2-layer Graph Attention Network encoder.

    Architecture:
        Layer 1: in=obs_dim, out=hidden_dim, heads=num_heads -> hidden_dim * num_heads
        Layer 2: in=hidden_dim*num_heads, out=embed_dim, heads=1 -> embed_dim (h'_i)

    Uses shared weights across all agents for scalability.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Layer 1: multi-head attention
        self.gat1 = GATConv(
            in_channels=obs_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,  # output: hidden_dim * num_heads
        )

        # Layer 2: single-head attention for final embedding
        self.gat2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=embed_dim,
            heads=1,
            dropout=dropout,
            concat=False,  # output: embed_dim
        )

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, obs_dim]
            edge_index: Graph connectivity [2, num_edges] (COO format)
            return_attention: If True, also return attention weights

        Returns:
            h: Spatially-enriched embeddings [num_nodes, embed_dim]
            attn_weights: (optional) attention weights from layer 2
        """
        # Layer 1
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        if return_attention:
            h, (edge_idx, attn_w) = self.gat2(
                h, edge_index, return_attention_weights=True
            )
            return h, attn_w
        else:
            h = self.gat2(h, edge_index)
            return h
