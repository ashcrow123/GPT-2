from module import *
from dataloader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
#-----------------------------distributed-------------------------
from torch.distributed import init_process_group,destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available(),"cuda for ddp is ready."
    init_process_group(backend="nccl")
    ddp_rank=os.environ.get("RANK")
    ddp_local_rank=os.environ.get("LOCAL_RANK")
    ddp_world_size=os.environ.get("WORLD_SIZE")
    device=f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device=device)
    master_process=(ddp_rank==0)
else:
    ddp_rank=0
    ddp_local_rank=0
    ddp_world_size=1
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device:{device}")
    master_process=True
device_type="cuda" if device.startswith("cuda") else "cpu"

#-----------------------------training-------------------------
torch.set_float32_matmul_precision("high")

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
B=64
T=1024
total_batch_size=524288 #about 0.5M tokens per batch
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps=total_batch_size//(B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
train_loader=DataLoaderLite(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size,split='train')
val_loader=DataLoaderLite(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size,split='val')

config=GPT2Config(vocab_size=50304)
model=GPT2(config=config)
model=model.to(device)
model=torch.compile(model)
if ddp:
    model=DDP(model,device_ids=[ddp_local_rank])
raw_model=model.module if ddp else model
max_lr=6e-4
min_lr=0.1*max_lr
warmup_steps=1000
max_steps=20000
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)
optimizer=raw_model.configure_optimizers(weight_decay=1e-1,learning_rate=max_lr,device_type=device_type)
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
if master_process:
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

import time
for step in range(max_steps):
    t0=time.time()
    last_step=(step==max_steps-1)
    if last_step or step % 250==0:
        model.eval()
        with torch.no_grad():
            val_loader.reset()
            val_loss_accum=0
            val_loss_steps=20
            for _ in range(val_loss_steps):
                x,y=val_loader.next_batch()
                x,y=x.to(device),y.to(device)
                with torch.autocast(device_type=device_type,dtype=torch.bfloat16):
                    logits,loss=model(x,targets=y)
                loss=loss/val_loss_steps
                val_loss_accum+=loss.detach()
        if ddp:
            dist.reduce(val_loss_accum,op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)         
    model.train()
    optimizer.zero_grad()
    loss_accum=0
    for micro_step in range(grad_accum_steps):
        x,y=train_loader.next_batch()
        x,y=x.to(device),y.to(device)
        with torch.autocast(device_type=device_type,dtype=torch.bfloat16):
            logits,loss=model(x,y)
        loss/=grad_accum_steps
        loss_accum+=loss.detach()
        loss.backward()
    if ddp:
        dist.reduce(loss_accum,op=dist.ReduceOp.AVG)
    norm=nn.utils.clip_grad_norm_(model.parameters(),1.0)
    lr=get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    optimizer.step()
    if device_type=="cuda":
        torch.cuda.synchronize()
    t1=time.time()
    dt=t1-t0
    tokens_accum_count=train_loader.B*train_loader.T*grad_accum_steps*ddp_world_size
    tokens_per_sec=tokens_accum_count/dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()


        

                
  
