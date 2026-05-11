
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import os
import numpy as np
import torch
import torch.nn as nn
from torch.export import Dim

import torch_mlir
from torch_mlir import fx, ir
from torch._decomp import get_decompositions
from torch_mlir.extras.fx_decomp_util import get_decomposition_table

import tiktoken
from torch.utils.data import Dataset, DataLoader

# ========== User parameters ==========
PRETRAINED_NAME = "gpt2"    # the 355M model
NEW_NUM_LAYERS = 1                 # change this to the number of transformer blocks you want
EXPORT_Prefill_PATH = "linalg2_gpt2_small_reduced_torch.mlir"
EXPORT_Decode_PATH = "Decode_linalg2_gpt2_small.mlir"
OPSET = 19
DEVICE = "cpu"                     # or "cuda" if you want to use GPU (and have CUDA & torch)
MAX_SEQUENCE_LENGTH = 64           # sequence length to export / test with
DUMMY_TEXT = "Hello world"

import time
import tiktoken
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False
        )

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):

        if x.dim() == 2:
            num_tokens, d_in = x.shape
            b = 1  
        else:
            b, num_tokens, d_in = x.shape


        keys_new = self.W_key(x)  
        values_new = self.W_value(x)
        queries = self.W_query(x)

        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        if use_cache:
            mask_bool = self.mask.bool()[
                self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
            ]
            self.ptr_current_pos += num_tokens_Q
        else:
            mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        context_vec = context_vec.view(-1, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.current_pos = 0        
        self.final_norm = LayerNorm(cfg["emb_dim"])
    

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = x.view(-1, x.size(-1))
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        x = self.final_norm(x)
        x = x.view(batch_size, seq_len, -1)
        return x

    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()





def get_config_from_hf(name,layer):
    """从 Hugging Face 获取官方配置并转换为自定义格式"""
    config = GPT2Config.from_pretrained(name)
    
    custom_cfg = {
        "vocab_size": config.vocab_size,
        "context_length": config.n_ctx,
        "emb_dim": config.n_embd,
        "n_heads": config.n_head,
        "n_layers": layer,
        "drop_rate": 0.0,      # 推理模式通常设为 0
        "qkv_bias": True       # GPT2 官方带有 Bias
    }
    return custom_cfg
def load_weights_from_hf(custom_model, name):
    # 加载官方权重
    hf_model = GPT2LMHeadModel.from_pretrained(name)
    hf_state_dict = hf_model.state_dict()
    custom_state_dict = custom_model.state_dict()

    # 映射字典示例 (重点：GPT2 官方线性层是 Conv1D，需要转置)
    mapping = {
        "tok_emb.weight": "transformer.wte.weight",
        "pos_emb.weight": "transformer.wpe.weight",
        "final_norm.scale": "transformer.ln_f.weight",
        "final_norm.shift": "transformer.ln_f.bias",
        "out_head.weight": "lm_head.weight",
    }

    # 循环遍历层，映射 TransformerBlock
    for i in range(len(custom_model.trf_blocks)):
        custom_base = f"trf_blocks.{i}"
        hf_base = f"transformer.h.{i}"
        
        # 注意：你朋友的代码将 Q, K, V 分开了，但 GPT2 官方是将它们合并在 c_attn 中的
        # 这里需要拆分官方权重 (QKV 是并排存储的)
        qkv_weight = hf_state_dict[f"{hf_base}.attn.c_attn.weight"].t() # 转置
        emb_dim = qkv_weight.shape[1]
        
        custom_state_dict[f"{custom_base}.att.W_query.weight"] = qkv_weight[:emb_dim, :]
        custom_state_dict[f"{custom_base}.att.W_key.weight"] = qkv_weight[emb_dim:2*emb_dim, :]
        custom_state_dict[f"{custom_base}.att.W_value.weight"] = qkv_weight[2*emb_dim:, :]
        
        # 映射投影层和 FF 层
        custom_state_dict[f"{custom_base}.att.out_proj.weight"] = hf_state_dict[f"{hf_base}.attn.c_proj.weight"].t()
        custom_state_dict[f"{custom_base}.att.out_proj.bias"] = hf_state_dict[f"{hf_base}.attn.c_proj.bias"]
        
        custom_state_dict[f"{custom_base}.ff.layers.0.weight"] = hf_state_dict[f"{hf_base}.mlp.c_fc.weight"].t()
        custom_state_dict[f"{custom_base}.ff.layers.0.bias"] = hf_state_dict[f"{hf_base}.mlp.c_fc.bias"]
        custom_state_dict[f"{custom_base}.ff.layers.2.weight"] = hf_state_dict[f"{hf_base}.mlp.c_proj.weight"].t()
        custom_state_dict[f"{custom_base}.ff.layers.2.bias"] = hf_state_dict[f"{hf_base}.mlp.c_proj.bias"]
        
        # LayerNorm
        custom_state_dict[f"{custom_base}.norm1.scale"] = hf_state_dict[f"{hf_base}.ln_1.weight"]
        custom_state_dict[f"{custom_base}.norm1.shift"] = hf_state_dict[f"{hf_base}.ln_1.bias"]
        custom_state_dict[f"{custom_base}.norm2.scale"] = hf_state_dict[f"{hf_base}.ln_2.weight"]
        custom_state_dict[f"{custom_base}.norm2.shift"] = hf_state_dict[f"{hf_base}.ln_2.bias"]

    # 应用基础映射
    for k, v in mapping.items():
        if k in custom_state_dict:
            custom_state_dict[k] = hf_state_dict[v]

    custom_model.load_state_dict(custom_state_dict)
    return custom_model


def build_reduced_model(name, layer):
    """一键构建函数"""
    cfg_dict = get_config_from_hf(name,layer)
    model = GPTModel(cfg_dict)
    model = load_weights_from_hf(model, name)
    model.eval()
    
    # 获取对应的分词器 (Tiktoken 用于 GPT2)
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer, model

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits, _ = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def get(name,layer):
    
    # GPT_CONFIG_124M = {
    #     "vocab_size": 50257,     # Vocabulary size
    #     "context_length": 1024,  # Context length
    #     "emb_dim": 768,          # Embedding dimension
    #     "n_heads": 12,           # Number of attention heads
    #     "n_layers": 1,          # Number of layers
    #     "drop_rate": 0.1,        # Dropout rate
    #     "qkv_bias": False        # Query-Key-Value bias
    # }

    torch.manual_seed(123)
    tokenizer, model = build_reduced_model(name,layer)
    model.eval()  # disable dropout

    #start_context = "Hello, I am"

    encoded = tokenizer.encode(DUMMY_TEXT)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", DUMMY_TEXT)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=1024#get_config_from_hf(PRETRAINED_NAME)["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)

    return model, tokenizer



# def export_to_pt(model, tokenizer, export_path, device="cpu"):
#     model = model.to(device).eval()
#     # shows only gemm
#     #dummy_text = "aster aster"
#     #dummy_text = "Hello"
#     inputs = tokenizer(DUMMY_TEXT, return_tensors="pt", padding=True, truncation=True)
#     # print(inputs)
#     input_ids = inputs["input_ids"].to(device)
#     # print(input_ids)

#     # 将输入改为单个词LINALG_ON_TENSORS
#     # input_ids = torch.tensor([[1234]], dtype=torch.long).to(device) # 形状为 (1, 1)
#     #input_ids = torch.ones((1, MAX_SEQUENCE_LENGTH), dtype=torch.long).to(device)
    
    
#     # 1. dynamic shapes

#     #batch = Dim("batch", min=1, max=16)  # batch will = 1 during inference
#     seq = Dim("seq", min=1, max=1023)
#     #dynamic_shapes = {"input_ids": {0: batch, 1: seq}}
#     dynamic_shapes = {"input_ids": {1: seq}}
    
#     # input_ids = torch.ones((1, 8), dtype=torch.int64).to(device)

#     #generate_text(model, tokenizer)

#     class WrapperModel(torch.nn.Module):
#         def __init__(self, m):
#             super().__init__()
#             self.m = m
#         def forward(self, input_ids):
#             # outputs = self.m(input_ids=input_ids)
#             # return outputs.logits

#             # explicit pass the mask
#             mask = torch.ones_like(input_ids).to(torch.float32)
#             outputs = self.m(input_ids=input_ids, attention_mask=mask)
#             return outputs.logits

#     wrapper = WrapperModel(model)
#     decomp_table = get_decomposition_table()
#     # TorchScript trace
#     # mlir_module = torch_mlir.torchscript.compile(wrapper, (input_ids,), output_type="linalg-on-tensors", use_tracing=True)
#     mlir_module = fx.export_and_import(
#         wrapper, 
#         input_ids, 
#         #dynamic_shapes=dynamic_shapes, # static
#         output_type="torch", # torch,LINALG_ON_TENSORS
#         strict=False,
#         decomposition_table=decomp_table,
#     )

 

#     with open(export_path, "w") as f:
#         f.write(str(mlir_module))
#     print(f"Success!")


def export_decoding_mlir(model, tokenizer, export_path, device="cpu"):
    # 确保模型在正确的设备和模式
    model = model.to(device).eval()
    
    # 关键点 1：定义符合原模型 forward 签名的 Wrapper
    class DecodingWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        
        def forward(self, input_ids):
            # 严格匹配原模型的参数：in_idx 和 use_cache
            # 我们强制 use_cache=True 来导出 Decoding 逻辑
            logits = self.m(input_ids, use_cache=True)
            return logits

    wrapper = DecodingWrapper(model)

    # 关键点 2：构造符合 Decoding 阶段形状的输入 (1, 1)
    # 假设我们只推理当前这一个 token
    decoding_input = torch.zeros((1, 1), dtype=torch.long, device=device) #torch.tensor([[50256]], dtype=torch.long).to(device) 

    # 关键点 3：预热 (Warm-up)
    # 这一步非常重要！因为原代码中 cache 开始是 None。
    # 如果不运行这一步，FX 追踪到的 cache 分支可能是无效的。

    # make kv cache has 1023 
    prefill_len = model.pos_emb.num_embeddings -1 # 1023
    print("kv cache size:", prefill_len)
    prefill_input = torch.zeros((1, prefill_len), dtype=torch.long, device=device)
    
    with torch.no_grad():
        model.reset_kv_cache()
        # _ = wrapper(decoding_input)
        # model.reset_kv_cache()

        # 先跑 1023 token，让 cache 进入“满上下文前一位”的状态
        _ = wrapper(prefill_input)

    # 关键点 4：导出配置
    decomp_table = get_decomposition_table()

    print("Starting MLIR export for decoding step...")
    try:
        # 使用 fx.export_and_import
        mlir_module = fx.export_and_import(
            wrapper, 
            decoding_input, 
            # 如果你想支持动态长度，这里可以加 dynamic_shapes
            output_type="linalg-on-tensors", 
            strict=False,
            decomposition_table=decomp_table,
        )

        with open(export_path, "w") as f:
            f.write(str(mlir_module))
        print(f"Success! Decoding step MLIR saved to {export_path}")
        
    except Exception as e:
        print(f"Export failed! Error: {e}")
        print("\n提示：如果报错涉及 'torch.cat'，是因为 linalg 对动态增长的 Tensor 支持较难。")
        print("建议先尝试将 output_type 改为 'torch' 看看能否成功导出 Torch 方言的 MLIR。")
    
if __name__ == "__main__":
    tokenizer, reduced_model = build_reduced_model(PRETRAINED_NAME, NEW_NUM_LAYERS)
    
    #export_to_pt(reduced_model, tokenizer, EXPORT_Prefill_PATH, device=DEVICE)
    export_decoding_mlir(reduced_model, tokenizer, EXPORT_Decode_PATH, device=DEVICE)
