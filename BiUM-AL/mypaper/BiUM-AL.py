import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from gmlp import gMLP, MemoryBuffer
from moe import IncrementalMoE_loc
from ResNet import ResNet
from dipole_physics import generate_dipole_data_torch, load_tensor  # 假设你已经把函数整理到 gMLP_BNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------- 配置 -----------------
result_dir = './pos_withss_result_pre140_resnet'
os.makedirs(result_dir, exist_ok=True)

model1_ckpt = os.path.join(result_dir, "finl0516_7_cal.pkl")
model2_ckpt = os.path.join(result_dir, "finl0516_7_pos_val.pkl")
buffer_path = os.path.join(result_dir, "0516_7replay_buffer.pt")

data_ss = '../data/ss_CA0.txt'
data_cc = '../data/cc_CA0.txt'
val_ss = '../data/ss_VAL0.txt'
val_cc = '../data/cc_VAL0.txt'

buffer_capacity = 100000
update_batch_size = 80
density_max = 0.6 * 3822 - 104
density_max /= 3718

# ----------------- 加载数据 -----------------
ss_all = load_tensor(data_ss).to(device)
cc_all = load_tensor(data_cc).to(device)[:, :5]
ss_test = load_tensor(val_ss).to(device)
cc_test = load_tensor(val_cc).to(device)[:, :5]

# ----------------- 初始化 buffer -----------------
buffer = MemoryBuffer(buffer_capacity)
if os.path.exists(buffer_path):
    buffer.buffer = torch.load(buffer_path)
gate_buffer = MemoryBuffer(buffer_capacity)

# ----------------- 模型初始化 -----------------
model1 = gMLP(5, 48, 256, 3, scalar_sigma=0.001).to(device)
model2 = ResNet().to(device)
model1.load_state_dict(torch.load(model1_ckpt, map_location=device))
model2.load_state_dict(torch.load(model2_ckpt, map_location=device))
model1.eval()
model2.eval()

moe2 = IncrementalMoE_loc(48, 5, 256, 1, gating_hidden_dims=[256, 1],
                           pretrained_expert_state=model2.state_dict()).to(device)
moe2.eval()

# ----------------- 按 z 区域划分 -----------------
def get_group_from_z(z):
    z = float(z)
    if z in [48, 68, 88, 108]: return 0
    if z in [58, 78, 98, 118]: return 1
    return -1

z_vals = cc_all[:, 2].cpu().numpy()
seq_groups = [np.nonzero(np.isin(z_vals, [48, 68, 88, 108]))[0].tolist(),
              np.nonzero(np.isin(z_vals, [58, 78, 98, 118]))[0].tolist()]

# ----------------- 主动增量学习 Loop -----------------
max_rounds = 20
used_groups = set()
region_sequence = []

for round_id in range(max_rounds):
    # 选择未采样的区域
    for gid, idxs in enumerate(seq_groups):
        if gid not in used_groups:
            best_gid = gid
            break
    else:
        print("所有区域已采样完毕")
        break

    region_sequence.append(best_gid)
    used_groups.add(best_gid)
    print(f"\n=== 第 {round_id+1} 轮，区域 {best_gid} ===")

    # 当前区域样本
    region_indices = torch.tensor(seq_groups[best_gid], dtype=torch.long, device=device)
    cc_region = cc_all[region_indices]
    ss_region = ss_all[region_indices]
    R = cc_region.size(0)

    # ----------------- 计算不确定度 -----------------
    with torch.no_grad():
        mu1, sigma1 = model1(cc_region)
        ss_cal = generate_dipole_data_torch(cc_region) + mu1
        _, sigma2 = model2(ss_cal)

    var1 = sigma1.clamp(1e-4).pow(2)
    var2 = sigma2[:, :3].clamp(1e-4).pow(2)
    trace1 = var1.prod(dim=1).pow(1 / 48)
    trace2 = var2.prod(dim=1).pow(1 / 3)
    trace1_norm = (trace1 - trace1.min()) / (trace1.max() - trace1.min() + 1e-8)
    trace2_norm = (trace2 - trace2.min()) / (trace2.max() - trace2.min() + 1e-8)
    loc_var = trace1_norm + trace2_norm

    # ----------------- 按概率抽样 -----------------
    k = max(1, int(R * density_max))
    probs = F.softmax(loc_var, dim=0)
    sel_rel = np.random.choice(R, size=k, replace=False, p=probs.cpu().numpy())
    sel_idxs = region_indices[sel_rel]
    new_x, new_y = cc_all[sel_idxs], ss_all[sel_idxs]

    buffer.add(new_x.cpu(), new_y.cpu())
    print(f"新增样本: {len(sel_idxs)} 条")

    # ----------------- gate_buffer -----------------
    center = cc_region[:, :3].mean(dim=0, keepdim=True)
    dist = torch.norm(cc_region[:, :3] - center, dim=1)
    g_idx = region_indices[torch.argsort(dist)[:10]]
    gate_buffer.add(cc_all[g_idx].cpu(), ss_all[g_idx].cpu())

# ----------------- 训练 gating -----------------
x_gate, y_gate = buffer.sample(len(gate_buffer.buffer))
x_gate, y_gate = x_gate.to(device), y_gate.to(device)
region_to_expert = {gid: idx for idx, gid in enumerate(region_sequence)}
y_region_all = torch.tensor([region_to_expert[get_group_from_z(z)] for z in x_gate[:, 2].cpu().numpy()],
                            dtype=torch.long, device=device)

all_train_mus, all_train_sigmas = moe2.forward_experts(y_gate)
train_ds = TensorDataset(x_gate, y_gate, all_train_mus, all_train_sigmas, y_region_all)
train_loader = DataLoader(train_ds, batch_size=update_batch_size, shuffle=False)

for exp in moe2.experts:
    for p in exp.parameters():
        p.requires_grad = False
for p in moe2.gating.parameters():
    p.requires_grad = True

optimizer_gate = torch.optim.Adam(moe2.gating.parameters(), lr=1e-3)
update_epochs = 500
eps = 1e-8

for ep in tqdm(range(update_epochs), desc='Training gating'):
    losses_epoch = []
    for x_b, y_b, mus, sigmas, y_region in train_loader:
        weights = moe2.gating(y_b)
        p_target = weights.gather(dim=1, index=y_region.unsqueeze(1)).squeeze(1)
        loss = -torch.log(p_target.clamp(min=eps)).mean()
        optimizer_gate.zero_grad()
        loss.backward()
        optimizer_gate.step()
        losses_epoch.append(loss.item())
    if ep % 50 == 0:
        print(f"Epoch {ep}, Loss: {np.mean(losses_epoch):.6f}")

# ----------------- 联合训练 -----------------
all_x, all_y = buffer.sample(len(buffer.buffer))
all_x, all_y = all_x.to(device), all_y.to(device)

# 生成每个样本的专家标签
y_region_all = torch.tensor(
    [region_to_expert[get_group_from_z(z)] for z in all_x[:, 2].cpu().numpy()],
    dtype=torch.long, device=device
)

train_ds = TensorDataset(all_x, all_y, y_region_all)
train_loader = DataLoader(train_ds, batch_size=update_batch_size, shuffle=True)

# 开启 gating + 专家模型训练
for exp in moe2.experts:
    for p in exp.parameters():
        p.requires_grad = True
for p in moe2.gating.parameters():
    p.requires_grad = True

optimizer_joint = torch.optim.Adam(
    list(moe2.gating.parameters()) + list(moe2.experts.parameters()), lr=1e-3
)

epochs_joint = 300
eps = 1e-8

for ep in tqdm(range(epochs_joint), desc='Joint training'):
    losses_epoch = []
    for x_b, y_b, y_region in train_loader:
        mus, sigmas = moe2.forward_experts(y_b)  # 所有专家输出
        weights = moe2.gating(y_b)               # gating 输出
        p_target = weights.gather(dim=1, index=y_region.unsqueeze(1)).squeeze(1)
        loss = -torch.log(p_target.clamp(min=eps)).mean()
        optimizer_joint.zero_grad()
        loss.backward()
        optimizer_joint.step()
        losses_epoch.append(loss.item())
    if ep % 50 == 0:
        print(f"Epoch {ep}, Loss: {np.mean(losses_epoch):.6f}")

# 保存训练结果
torch.save(moe2.state_dict(), os.path.join(result_dir, 'moe_joint_trained.pt'))
print("联合训练完成，模型已保存。")
