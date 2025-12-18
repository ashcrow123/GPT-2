import torch
from module import *
def load_model_from_checkpoint(checkpoint_path):
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path,weights_only=False)
    
    # 从checkpoint获取config
    config = checkpoint['config']
    
    # 创建模型实例
    model = GPT2(config)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model=torch.compile(model)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model'])
    
    # 设置为评估模式
    model.eval()
    
    # 其他信息
    step = checkpoint['step']
    val_loss = checkpoint['val_loss']
    
    print(f"成功加载模型: step={step}, val_loss={val_loss:.4f}")
    return model, config, step, val_loss
checkpoint_path="./log/model_20000.pt"
model,_,_,_=load_model_from_checkpoint(checkpoint_path=checkpoint_path)
device="cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoTokenizer
import time
import torch.nn.functional as F
enc=AutoTokenizer.from_pretrained("Yi-1.5-6B")
prompt="大语言模型是"
tokens=enc.encode(prompt)
tokens=torch.tensor(tokens,dtype=torch.long,device=device).unsqueeze(0).repeat(5,1)
t1=time.time()
while tokens.shape[1]<150:
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits,_=model(tokens)
            probs=F.softmax(logits[:,-1,:],dim=-1)
            next_token=torch.multinomial(probs,1)
    tokens=torch.cat([tokens,next_token],dim=-1)
t2=time.time()
print(f"Generation time: {t2-t1} seconds")
tokens=tokens.to("cpu").numpy()
tokens=tokens.tolist()
for i in tokens:
    text=enc.decode(i)
    print(text)
    print("\n------------------------------------------------------------------------------------------------------------------------\n")
        