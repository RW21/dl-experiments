import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask_mode=False):
        super().__init__()
        assert k % heads == 0
        self.k, self.heads = k, heads

        self.tokeys = nn.Linear(k, k, bias=False)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tovalues = nn.Linear(k, k, bias=False)

        # self.unifyheads = nn.Linear(k, k)

        self.mask_mode = mask_mode

    # Does multihead attention
    def forward(self, x):
        b, t, k = x.shape
        h = self.heads

        queries = self.toqueries(x)
        keys = self.tokeys(x)
        values = self.tovalues(x)

        s = k // h

        keys = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values = values.view(b, t, h, s)

        # keys = keys.transpose(1, 2).contigious().view(b*h, t, s)
        # queries = queries.transpose(1,2).contigious().view(b*h, t, s)
        # values = values.transpose(1,2).contiguous().view(b*h, t, s)
        keys = keys.transpose(1, 2).reshape(b*h, t, s)
        queries = queries.transpose(1,2).reshape(b*h, t, s)
        values = values.transpose(1,2).reshape(b*h, t, s)

        w = torch.bmm(queries, keys.transpose(1,2),)

        if self.mask_mode:
            indices = torch.triu_indices(t, t, offset=1)
            w[:, indices[0], indices[1]] = float('-inf')
        else:
            # Do we need this for when masked?
            w = w / k**(.5)

        w = F.softmax(w, dim=2)

        out = torch.bmm(w, values).view(b, h, t, s)
        # out = out.transpose(1,2).contigious().view(b, t, s*h)
        out = out.transpose(1,2).reshape(b, t, s*h)
        return out 

class TransformerBlock(nn.Module):
    def __init__(self, k, heads, mask_mode=False):
        super().__init__()

        self.attention = SelfAttention(k, heads = heads, mask_mode=mask_mode)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended+x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)


class ClassificationTransformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()

        self.num_token = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        t_blocks = []
        for _ in range(depth):
            t_blocks.append(TransformerBlock(k=k, heads=heads))
        self.t_blocks = nn.Sequential(*t_blocks)

        self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x):
        tokens = self.token_emb(x)
        b, t, k = tokens.shape

        if x.get_device() >= 0:
            device = 'cuda'
        else:
            device = 'cpu'

        positions = torch.arange(t).to(device)
        positions = self.pos_emb(positions).unsqueeze(0).expand(b, t, k)

        x = tokens + positions
        x = self.t_blocks(x)

        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)


class GenerationTransformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens):
        super().__init__()

        self.num_token = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        t_blocks = []
        for _ in range(depth):
            t_blocks.append(TransformerBlock(k=k, heads=heads, mask_mode=True))
        self.t_blocks = nn.Sequential(*t_blocks)

        self.toprobs = nn.Linear(k, num_tokens)

    def forward(self, x):
        tokens = self.token_emb(x)
        b, t, k = tokens.shape

        if x.get_device() >= 0:
            device = 'cuda'
        else:
            device = 'cpu'

        positions = torch.arange(t).to(device)
        positions = self.pos_emb(positions).unsqueeze(0).expand(b, t, k)

        x = tokens + positions
        x = self.t_blocks(x)

        x = self.toprobs(x.view(b*t, k)).view(b, t, self.num_token)

        return F.log_softmax(x, dim=2)

