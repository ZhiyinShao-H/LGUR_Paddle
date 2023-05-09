# import torch
# from torch import nn, einsum
# import torch.nn.functional as F
# from torch.nn import init
import paddle
from paddle import nn
from paddlenlp.ops import einsum
# from einops import rearrange, repeat


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 1)
        init.constant(m.bias.data, 0)

class Residual(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Layer):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.GELU(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class PreNorm_3(nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, y, **kwargs):
        return self.fn(self.norm(x), self.norm(y),**kwargs)

class Residual_3(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, y,**kwargs):
        return self.fn(x, y, **kwargs) + x

class Residual_3_y(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, y,**kwargs):
        return self.fn(x, y, **kwargs) + y





class Attention_mydecoder(nn.Layer):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.dim_head = dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias_attr = False)
        self.to_k = nn.Linear(dim, inner_dim, bias_attr=False)
        self.to_v = nn.Linear(dim, inner_dim, bias_attr=False)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,txt, image ,kv_mask = None,q_mask = None):
        b1, n1, _, h = *image.shape, self.heads
        b2, n2, _, h = *txt.shape, self.heads

        q = self.to_q(txt)
        # q = rearrange(q,'b n (h d) -> b h n d', h=h)
        q = q.reshape(shape=[0, 0, h, self.dim_head])
        q = q.transpose(perm=[0, 2, 1, 3])


        k = self.to_k(image)
        v = self.to_v(image)
        k = k.reshape(shape=[0, 0, h, self.dim_head])
        k = k.transpose(perm=[0, 2, 1, 3])
        v = v.reshape(shape=[0, 0, h, self.dim_head])
        v = v.transpose(perm=[0, 2, 1, 3])
        # k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), kv)
        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = paddle.matmul(x=q,
                                y=k,
                                transpose_y=True) * self.scale

        # mask_value = -paddle.finfo(dots.dtype).max
        mask_value = paddle.to_tensor('-inf')
        if kv_mask is not None:
            assert kv_mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            q_mask = q_mask.unsqueeze(2).unsqueeze(1)
            kv_mask = kv_mask.unsqueeze(1).unsqueeze(1)
            mask = q_mask * kv_mask
            # mask = rearrange(q_mask, 'b i -> b () i ()') * rearrange(kv_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        # out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = paddle.matmul(attn,v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = out.transpose(perm=[0, 2, 1, 3])
        out = out.reshape(shape=[0, 0, out.shape[2] * out.shape[3]])
        out =  self.to_out(out)
        return out


class Transformer_mydecoder(nn.Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                # Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual_3(PreNorm_3(dim, Attention_mydecoder(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                # PreNorm_3(dim, Attention_DECODER(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, y,kv_mask = None,q_mask = None):
        # for attn, attn_decode, ff in self.layers:
        #     # x = attn(x, mask = q_mask)
        #     x = attn_decode(x, y , kv_mask=kv_mask,q_mask = q_mask)
        #     x = ff(x)
        for attn_decode, ff in self.layers:
            # x = attn(x, mask = q_mask)
            x = attn_decode(x, y , kv_mask=kv_mask,q_mask = q_mask)
            x = ff(x)
        return x

class conv(nn.Layer):

    def __init__(self, input_dim, output_dim, relu=False, BN=False):
        super(conv, self).__init__()

        block = []
        block += [nn.Conv2D(input_dim, output_dim, kernel_size=1, bias_attr=False)]

        if BN:
            block += [nn.BatchNorm2D(output_dim)]
        if relu:
            block += [nn.ReLU()]

        self.block = nn.Sequential(*block)
        self.block.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(3).squeeze(2)
        return x

class mydecoder(nn.Layer):
    def __init__(self, *, opt,dim, depth, heads, mlp_dim, pool = 'cls', patch_dim = 2048, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        # self.to_DIM_txt= nn.Linear(patch_dim, dim)
        # self.to_DIM_img = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer_mydecoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, txt,img, kv_mask = None,q_mask = None): # img [64,48,2048] txt [64,Length,2048]
        # img = self.to_DIM_img(img)
        # txt = self.to_DIM_txt(txt)
        b_img, n_img, _ = img.shape
        b_txt, n_txt, _ = txt.shape
        # x += self.pos_embedding[:, :(n + 1)]
        # img = self.dropout(img)
        x = self.transformer(txt, img, kv_mask,q_mask)
        # x = self.to_latent(x)
        # return self.mlp_head(x)
        return x

