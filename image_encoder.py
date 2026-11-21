"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # TODO-02 约定：把 [B,N,D] 拆成 [B,H,N,D/H]。
        # 本图像模型：D=768、H=12、D/H=64。
        #
        # 存储的 qkv 权重把 D=768 维特征映射到 D=768 维输出，即所有 H 个头的
        # query/key/value 行只是沿特征轴简单拼接。要还原出"按头"的视图，先把
        # D 轴 reshape 成 (H, D/H)，再把 head 轴移到 token 轴前面：
        #   [B, N, D] -> [B, N, H, D/H] -> [B, H, N, D/H]
        B, N, D = x.shape
        H = self.heads
        head_dim = D // H
        x = x.reshape(B, N, H, head_dim)   # 把最后一维拆成 H 组
        x = x.transpose(0, 2, 1, 3)        # 把 head 轴放到 token 轴之前
        return x

    def __call__(self, x, causal=False):
        # 给定：存储的线性层权重采用 [out,in] 布局，参数保持不变。
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key 点积，再除以 sqrt(head_dim) 缩放。
        # q,k,v: [B,H,N,64]，scores 必须是 [B,H,N,N]。
        # scores 让每个 query token 与每个 key token 比较。k 的 shape 是
        # [B,H,N,64]，所以 k.transpose(0,1,3,2) 得到 [B,H,64,N]，相乘后得到
        # [B,H,N,N] 的分数矩阵。缩放系数对应讲义里的 1/sqrt(head_dim)。
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(np.float32(self.head_dim))

        # 给定：为后续文本编码器兼容保留。图像使用 causal=False。
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # 提供的数值辅助函数

        # TODO-03b: 用系数对 value 行做加权求和，再合并各头输出。
        # context: [B,H,N,64]，merged: [B,N,D]。
        # 每个 query token 汇聚出对 value token 的系数加权和。
        context = coefficients @ v                  # [B,H,N,64]
        # 合并头：先转成 [B,N,H,64]，再把 H*head_dim=D 展平。
        merged = context.transpose(0, 2, 1, 3).reshape(context.shape[0],
                                                       context.shape[2], -1)

        # TODO-03c: 输出投影，使用 out_weight 和 out_bias。
        # 返回 [B,N,D]。注意存储的线性层权重的方向。
        # out_weight 存成 [out,in] = [D,D]，所以相乘前要先转置。
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # 给定：先对每个 token 归一化，再计算其注意力更新量。
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: 第一个残差连接，结果命名为 residual。
        # 残差路径保留的是"原始的 x"（Norm 1 之前的），与讲义一致。
        residual = x + update

        # 给定：第二个 Norm，以及精确的 checkpoint 激活辅助函数。
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2。
        # hidden: [B,N,3072]，mlp_update: [B,N,768]（图像塔）。
        # 返回 residual + mlp_update。两条残差路径都保持 [B,N,D]。
        # fc1_weight 是 [3072,768] = [out,in]；fc2_weight 是 [768,3072] = [out,in]。
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01: 分离 patch 网格轴；重排；展平。
        # 输入 [B,3,224,224]，返回 [B,49,3072]。
        #
        # 每张图是 224x224 = 7x7 个 32x32 的 patch。把空间轴 reshape 成 7x7 的
        # 32x32 小块网格，再重排，使 patch 索引按"从左到右、从上到下"排列，
        # 每个 patch 按 C,H,W 的顺序展平。
        B, C, H, W = images.shape
        P = self.patch_weight.shape[1] // C   # 3072 // 3 = 1024 = 32*32
        patch_size = int(np.sqrt(P))           # 32
        grid = H // patch_size                 # 7
        # [B, C, grid, patch, grid, patch] -> [B, grid, grid, C, patch, patch]
        patches = images.reshape(B, C, grid, patch_size, grid, patch_size)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        # 展平每个 patch（C*patch*patch）和每张图的 49 行。
        return patches.reshape(B, grid * grid, C * patch_size * patch_size)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: 把 patches 投影成 [B,49,768]。这里没有 patch 偏置。
        # patch_weight 是 [768,3072] = [out,in]，所以相乘前要先转置。
        projected = patches @ self.patch_weight.T      # [B,49,768]

        # 给定：每张图一个提供的 CLS 行。
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: 在 token 轴上把 CLS 拼到最前面，再加上学习到的位置编码。
        # 返回 [B,50,768]。self.position 沿 B 维广播。
        x = np.concatenate([cls, projected], axis=1)   # [B,50,768]
        return x + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # 选中 CLS、应用 post_norm、再用 self.projection 投影。
        # [B,50,768] -> [B,768] -> [B,512]。visual.proj 已经是 [in,out]。
        # 把投影后的数组命名为 output。最后的单位向量辅助函数已给定。
        cls_token = tokens[:, 0, :]                    # [B,768] CLS 行
        normalized = self.post_norm(cls_token)         # [B,768] post-Norm
        output = normalized @ self.projection          # [B,512]，proj 是 [in,out]

        return unit_vector(output)