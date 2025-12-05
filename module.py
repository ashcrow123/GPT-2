import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from dataclasses import dataclass
import os
import inspect
import math
@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1
    block_size: int = 1024
# Module
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        assert (
            config.n_embd %config.n_head==0
        ), "Embedding size needs to be divisible by n_head"
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn=nn.Linear(self.n_embd,3*self.head_dim*self.n_head)
        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd)
        # self.scale=torch.tensor(self.head_dim ** -0.5)
    def forward(self, x):
        qkv=self.c_attn(x) #(B,seq_len,3*head_dim*n_head)
        q, k, v = qkv.split(self.n_head*self.head_dim, dim=2) #(B,seq_len,head_dim*n_head)
        B,seq_len,_=q.shape
        q=q.view(B,seq_len,self.n_head,self.head_dim).transpose(1,2) #(B,n_head,seq_len,head_dim)
        k=k.view(B,seq_len,self.n_head,self.head_dim).transpose(1,2) #(B,n_head,seq_len,head_dim)
        # score=q @ k.transpose(-2,-1)*self.scale #(B,n_head,seq_len,seq_len)
        # masked_mat=torch.tril(torch.ones(seq_len,seq_len,devide=x.device))
        # masked_score=score.masked_fill(masked_mat==0,float('-inf'))
        # weights=F.softmax(masked_score,dim=-1) #(B,n_head,seq_len,seq_len)
        v=v.view(B,seq_len,self.n_head,self.head_dim).transpose(1,2) #(B,n_head,seq_len,head_dim)
        # out=weights @ v #(B,n_head,seq_len,head_dim)
        out=F.scaled_dot_product_attention(q,k,v,is_causal=True) #(B,n_head,seq_len,head_dim)
        out=out.transpose(1,2).contiguous().view(B,seq_len,self.n_head*self.head_dim) #(B,seq_len,n_embd)
        out=self.c_proj(out)   
        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)
        self.activation=nn.GELU(approximate='tanh')
    def forward(self,x):
        x=self.c_fc(x)
        x=self.activation(x)
        x=self.c_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attn=MultiHeadAttention(config=config)
        self.mlp=MLP(config=config)
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.ln_2=nn.LayerNorm(config.n_embd)
    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x
    
    
class GPT2(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size,config.n_embd),
            "wpe": nn.Embedding(config.block_size,config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd)        
            })
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.lm_head.weight=self.transformer["wte"].weight
        self.apply(self._init_weights)
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self,idx,targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_embeddings=self.transformer["wpe"](pos)
        token_embeddings=self.transformer["wte"](idx)
        x=pos_embeddings+token_embeddings
        for block in self.transformer["h"]:
            x=block(x)
        x=self.transformer["ln_f"](x)
        logits=self.lm_head(x)
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    @classmethod
    def from_pretrained(cls,model_type):
        """Loads pretrained model weights from huggingface"""
        assert (model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl','./gpt2'}), "model_type must be one of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'"
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        config_args = {
            './gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 
        config_args['block_size'] = 1024 
        config=GPT2Config(**config_args)
        model=GPT2(config=config)
        sd=model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
        model_hf=GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf=model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} in hf vs {len(sd_keys)} in local"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        global master_process
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    