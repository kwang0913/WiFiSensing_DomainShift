"""Real-valued Tx/Rx node-edge encoders for the configured MATLAB layout.

Rows are Tx-major, then Rx (amplitude) or pair (phase/product), then
subcarrier. Pair order and endpoint roles are 1->2, 1->3, 2->3; raw
relative phase is angle(H_i * conj(H_j)). Magnitude STFT does not retain
this phase sign, although its pair identity remains available. STFT width
is MATLAB frequency-first: frame * bins + frequency.
"""
import torch
from torch import nn
from .common import PairedModel, Residual, conv_block


class TimeLinkEncoder(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.layers = nn.Sequential(
            conv_block(inputs, 32, (3, 7), (1, 4), (1, 3)),
            conv_block(32, 64, (3, 5), (2, 4), (1, 2)),
            Residual(64, dimensions=2),
            nn.AdaptiveAvgPool2d((1, 8))
            )

    def forward(self, x):
        return self.layers(x).squeeze(-2)


class STFTLinkEncoder(nn.Module):
    def __init__(self, inputs, bins):
        super().__init__()
        self.bins = bins
        # Shared over subcarriers and links; small channels control activation memory.
        self.spectral = nn.Sequential(
            conv_block(inputs, 8, 3, 2, 1),
            conv_block(8, 16, 3, 2, 1),
            nn.AdaptiveAvgPool2d((1, 8)))
        self.subcarriers = nn.Sequential(
            conv_block(16, 64, (3, 1), (2, 1), (1, 0)),
            Residual(64, dimensions=2),
            nn.AdaptiveAvgPool2d((1, 8)))

    def forward(self, x):
        n, c, k, width = x.shape
        # Flattened STFT is frequency-first, not frame-first.
        x = x.reshape(n, c, k, width // self.bins, self.bins)
        x = x.permute(0, 2, 1, 4, 3).reshape(n * k, c, self.bins, -1)
        x = self.spectral(x).squeeze(-2)
        x = x.reshape(n, k, 16, 8).permute(0, 2, 1, 3)
        return self.subcarriers(x).squeeze(-2)


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, outputs),
                         nn.GELU(),
                         nn.Linear(outputs, outputs))


class RxMessagePassing(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_update = mlp(192, 64)
        self.to_source = mlp(192, 64)
        self.to_target = mlp(192, 64)
        self.node_update = mlp(128, 64)
        self.edge_norm = nn.LayerNorm(64)
        self.node_norm = nn.LayerNorm(64)

    def forward(self, nodes, edges):
        # [..., 3 nodes/edges, 64]; ordered endpoint roles retain the MATLAB pair convention.
        pairs = ((0, 1), (0, 2), (1, 2))
        updated, incoming = [], [[], [], []]
        for e, (i, j) in enumerate(pairs):
            context = torch.cat((nodes[..., i, :], nodes[..., j, :], edges[..., e, :]), -1)
            edge = self.edge_norm(edges[..., e, :] + self.edge_update(context))
            updated.append(edge)
            context = torch.cat((nodes[..., i, :], nodes[..., j, :], edge), -1)
            incoming[i].append(self.to_source(context))
            incoming[j].append(self.to_target(context))
        new_nodes = []
        for i in range(3):
            message = torch.stack(incoming[i], -2).mean(-2)
            update = self.node_update(torch.cat((nodes[..., i, :], message), -1))
            new_nodes.append(self.node_norm(nodes[..., i, :] + update))
        return torch.stack(new_nodes, -2), torch.stack(updated, -2)


class LinkBranch(nn.Module):
    def __init__(self, bins=None):
        super().__init__()
        self.node_encoder = TimeLinkEncoder(1) if bins is None else STFTLinkEncoder(1, bins)
        self.edge_encoder = TimeLinkEncoder(2) if bins is None else STFTLinkEncoder(2, bins)
        self.graph = RxMessagePassing()
        self.rx_fusion = mlp(384, 128)
        self.tx_fusion = mlp(384, 128)
        self.temporal = nn.Sequential(Residual(128), nn.AdaptiveAvgPool1d(1), nn.Flatten())

    @staticmethod
    def split_links(x):
        b, _, _, width = x.shape
        nodes = x[:, 0].reshape(b * 9, 1, 30, width)
        edges = x[:, 1:].reshape(b, 2, 3, 3, 30, width)
        edges = edges.permute(0, 2, 3, 1, 4, 5).reshape(b * 9, 2, 30, width)
        return nodes, edges

    def forward(self, x):
        b = x.shape[0]
        nodes, edges = self.split_links(x)
        # [batch, time position, Tx, Rx/pair, channel]
        nodes = self.node_encoder(nodes).reshape(b, 3, 3, 64, 8).permute(0, 4, 1, 2, 3)
        edges = self.edge_encoder(edges).reshape(b, 3, 3, 64, 8).permute(0, 4, 1, 2, 3)
        nodes, edges = self.graph(nodes, edges)
        tx = self.rx_fusion(torch.cat((nodes, edges), -2).flatten(-2))
        fused = self.tx_fusion(tx.flatten(-2))
        return self.temporal(fused.transpose(1, 2))


class DualLinkGraph(PairedModel):
    def __init__(self, classes, embedding_dim=32, stft_frequency_bins=32):
        super().__init__(classes, embedding_dim, stft_frequency_bins)
        self.time_encoder = LinkBranch()
        self.stft_encoder = LinkBranch(stft_frequency_bins)
        self.fusion = nn.Sequential(nn.Linear(256, 128), nn.GELU())
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)
