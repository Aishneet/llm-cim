import time
import torch
import torch.nn as nn
import tiktoken

from transformers import GPT2LMHeadModel, GPT2Config
from torch.export import Dim

import torch_mlir
from torch_mlir import fx
from torch_mlir.extras.fx_decomp_util import get_decomposition_table


# =========================
# User parameters
# =========================
PRETRAINED_NAME = "gpt2-medium"
NEW_NUM_LAYERS = 1                 # set to 24 if you want full GPT-2 medium
EXPORT_PREFILL_PATH = "Prefill_linalg2_gpt2_medium_2048.mlir"
EXPORT_DECODE_PATH = "Decode_linalg2_gpt2_medium_4096.mlir"
DEVICE = "cpu"                     # "cuda" if available
DUMMY_TEXT = "Hello world"
CONTEXT_LENGTH = 4096
# =========================


#####################################
# Model components
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # remove GEMV
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):
        # This code is written for the reduced-export path where batch=1.
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

        attn_scores = queries @ keys.transpose(2, 3)

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

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # keep it 2D for the no-GEMV export path
        context_vec = context_vec.view(-1, self.d_out)
        context_vec = self.out_proj(context_vec)

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
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.current_pos = 0
        self.final_norm = LayerNorm(cfg["emb_dim"])
        #self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        if use_cache:
            pos_ids = torch.arange(
                self.current_pos,
                self.current_pos + seq_len,
                device=in_idx.device,
                dtype=torch.long,
            )
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)

        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # flatten to 2D for the no-GEMV path
        x = x.view(-1, x.size(-1))

        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)

        x = self.final_norm(x)
        x = x.view(batch_size, seq_len, -1)
        #logits = self.out_head(x)
        return x

    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0


#####################################
# Generation helpers
#####################################
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_text_simple_cached(model, idx, max_new_tokens, context_size=None, use_cache=True):
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


#####################################
# HF config / weight loading
#####################################
def get_config_from_hf(name, layer, context_length=4096):
    config = GPT2Config.from_pretrained(name)

    custom_cfg = {
        "vocab_size": config.vocab_size,
        "context_length": context_length,
        "emb_dim": config.n_embd,
        "n_heads": config.n_head,
        "n_layers": layer,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    return custom_cfg


def load_weights_from_hf(custom_model, name):
    hf_model = GPT2LMHeadModel.from_pretrained(name)
    hf_state_dict = hf_model.state_dict()
    custom_state_dict = custom_model.state_dict()

    mapping = {
        "tok_emb.weight": "transformer.wte.weight",
        "final_norm.scale": "transformer.ln_f.weight",
        "final_norm.shift": "transformer.ln_f.bias",
        "out_head.weight": "lm_head.weight",
    }

    # Extend positional embeddings from HF's 1024 to our target context length
    hf_pos = hf_state_dict["transformer.wpe.weight"]          # [1024, emb_dim]
    custom_pos = custom_state_dict["pos_emb.weight"]          # [context_length, emb_dim]
    custom_pos[:hf_pos.shape[0]] = hf_pos
    if custom_pos.shape[0] > hf_pos.shape[0]:
        custom_pos[hf_pos.shape[0]:] = hf_pos[-1].unsqueeze(0).repeat(
            custom_pos.shape[0] - hf_pos.shape[0], 1
        )
    custom_state_dict["pos_emb.weight"] = custom_pos

    for i in range(len(custom_model.trf_blocks)):
        custom_base = f"trf_blocks.{i}"
        hf_base = f"transformer.h.{i}"

        qkv_weight = hf_state_dict[f"{hf_base}.attn.c_attn.weight"].t()
        emb_dim = qkv_weight.shape[1]

        custom_state_dict[f"{custom_base}.att.W_query.weight"] = qkv_weight[:emb_dim, :]
        custom_state_dict[f"{custom_base}.att.W_key.weight"] = qkv_weight[emb_dim:2 * emb_dim, :]
        custom_state_dict[f"{custom_base}.att.W_value.weight"] = qkv_weight[2 * emb_dim:, :]
        custom_state_dict[f"{custom_base}.att.W_query.bias"] = hf_state_dict[f"{hf_base}.attn.c_attn.bias"][:emb_dim]
        custom_state_dict[f"{custom_base}.att.W_key.bias"] = hf_state_dict[f"{hf_base}.attn.c_attn.bias"][emb_dim:2 * emb_dim]
        custom_state_dict[f"{custom_base}.att.W_value.bias"] = hf_state_dict[f"{hf_base}.attn.c_attn.bias"][2 * emb_dim:]

        custom_state_dict[f"{custom_base}.att.out_proj.weight"] = hf_state_dict[f"{hf_base}.attn.c_proj.weight"].t()
        custom_state_dict[f"{custom_base}.att.out_proj.bias"] = hf_state_dict[f"{hf_base}.attn.c_proj.bias"]

        custom_state_dict[f"{custom_base}.ff.layers.0.weight"] = hf_state_dict[f"{hf_base}.mlp.c_fc.weight"].t()
        custom_state_dict[f"{custom_base}.ff.layers.0.bias"] = hf_state_dict[f"{hf_base}.mlp.c_fc.bias"]
        custom_state_dict[f"{custom_base}.ff.layers.2.weight"] = hf_state_dict[f"{hf_base}.mlp.c_proj.weight"].t()
        custom_state_dict[f"{custom_base}.ff.layers.2.bias"] = hf_state_dict[f"{hf_base}.mlp.c_proj.bias"]

        custom_state_dict[f"{custom_base}.norm1.scale"] = hf_state_dict[f"{hf_base}.ln_1.weight"]
        custom_state_dict[f"{custom_base}.norm1.shift"] = hf_state_dict[f"{hf_base}.ln_1.bias"]
        custom_state_dict[f"{custom_base}.norm2.scale"] = hf_state_dict[f"{hf_base}.ln_2.weight"]
        custom_state_dict[f"{custom_base}.norm2.shift"] = hf_state_dict[f"{hf_base}.ln_2.bias"]

    for k, v in mapping.items():
        if k in custom_state_dict and v in hf_state_dict:
            custom_state_dict[k] = hf_state_dict[v]

    custom_model.load_state_dict(custom_state_dict, strict=True)
    return custom_model


def build_reduced_model(name, layer, context_length=4096):
    cfg_dict = get_config_from_hf(name, layer, context_length=context_length)
    model = GPTModel(cfg_dict)
    model = load_weights_from_hf(model, name)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer, model


#####################################
# Export helpers
#####################################
def export_to_pt(model, tokenizer, export_path, device="cpu"):
    model = model.to(device).eval()

    inputs = tokenizer(DUMMY_TEXT, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)

    seq = Dim("seq", min=1, max=CONTEXT_LENGTH)
    dynamic_shapes = {"input_ids": {1: seq}}

    class WrapperModel(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids):
            outputs = self.m(input_ids, use_cache=False)
            return outputs

    wrapper = WrapperModel(model)
    decomp_table = get_decomposition_table()

    mlir_module = fx.export_and_import(
        wrapper,
        input_ids,
        # dynamic_shapes=dynamic_shapes,
        output_type="torch",
        strict=False,
        decomposition_table=decomp_table,
    )

    with open(export_path, "w") as f:
        f.write(str(mlir_module))
    print(f"Saved prefill MLIR to {export_path}")


def export_decoding_mlir(model, tokenizer, export_path, device="cpu"):
    model = model.to(device).eval()

    class DecodingWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids):
            logits = self.m(input_ids, use_cache=True)
            return logits

    wrapper = DecodingWrapper(model)

    decoding_input = torch.zeros((1, 1), dtype=torch.long, device=device)

    prefill_len = model.pos_emb.num_embeddings - 1
    print("kv cache size:", prefill_len)

    prefill_input = torch.zeros((1, prefill_len), dtype=torch.long, device=device)

    with torch.no_grad():
        model.reset_kv_cache()
        _ = wrapper(prefill_input)

    decomp_table = get_decomposition_table()

    print("Starting MLIR export for decoding step...")
    try:
        mlir_module = fx.export_and_import(
            wrapper,
            decoding_input,
            output_type="linalg-on-tensors",
            strict=False,
            decomposition_table=decomp_table,
        )

        with open(export_path, "w") as f:
            f.write(str(mlir_module))
        print(f"Success! Decoding step MLIR saved to {export_path}")

    except Exception as e:
        print(f"Export failed! Error: {e}")
        print("If torch.cat causes issues, try exporting with output_type='torch' first.")


#####################################
# Main
#####################################
def main():
    torch.manual_seed(123)
    tokenizer, reduced_model = build_reduced_model(
        PRETRAINED_NAME,
        NEW_NUM_LAYERS,
        context_length=CONTEXT_LENGTH,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reduced_model.to(device).eval()

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=reduced_model,
        idx=encoded_tensor,
        max_new_tokens=20,
        context_size=CONTEXT_LENGTH,
        use_cache=True,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50 * '='}\n{22 * ' '}OUT\n{50 * '='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0]) / total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")

    # Optional exports
    # export_to_pt(reduced_model, tokenizer, EXPORT_PREFILL_PATH, device=DEVICE)
    export_decoding_mlir(reduced_model, tokenizer, EXPORT_DECODE_PATH, device=DEVICE)


if __name__ == "__main__":
    main()