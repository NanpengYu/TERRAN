import os
import pickle
import torch

@torch.jit.script
def compute_returns(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, gae_lambda: float):
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)

    for t in torch.arange(rewards.shape[0] - 1, -1, -1, device=rewards.device):
        if t == rewards.shape[0] - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t]
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

    returns = advantages + values
    return returns, advantages

def count_avg_route_len(data):
    non_zero_mask = (data != 0).float()

    # 确保每一行的末尾有个0，避免 route 末端没有终点
    padded_data = torch.cat([data, torch.zeros((data.size(0), 1))], dim=1)
    padded_mask = (padded_data != 0).float()

    # 找到每个非零段的起点（0 -> 非零）和终点（非零 -> 0）
    start_points = (padded_mask[:, :-1] == 0) & (padded_mask[:, 1:] == 1)
    end_points = (padded_mask[:, :-1] == 1) & (padded_mask[:, 1:] == 0)

    # 获取起点和终点的索引
    start_indices = start_points.nonzero(as_tuple=False)  # (batch, idx)
    end_indices = end_points.nonzero(as_tuple=False)  # (batch, idx)

    # 确保起点和终点匹配
    if start_indices.size(0) != end_indices.size(0):
        return -1  # 可能数据有异常情况

    # 计算每个 route 的长度
    route_lengths = end_indices[:, 1] - start_indices[:, 1]

    # 计算总 route 数量和平均长度
    total_routes = route_lengths.numel()
    average_route_length = route_lengths.float().mean().item() if total_routes > 0 else 0

    return average_route_length
    
def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)