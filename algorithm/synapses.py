import torch


def forward_synapse(alpha, syn, h1, t, out, v, noise_strength=0.0):
    noise = torch.randn(syn.size()) * noise_strength
    new_syn = alpha * syn + h1[:, t] + noise
    return new_syn


def recurrent_synapse(alpha, syn, h1, t, out, v, noise_strength=0.0):
    noise = torch.randn(syn.size()).to(syn) * noise_strength
    #names = ["alpha", "syn", "h1", "out", "v", "noise"]
    #for i, obj in enumerate([alpha, syn, h1, out, v, noise]):
    #    print(f"{names[i]}, {obj.shape}")
    new_syn = alpha * syn + h1[:, t] + torch.mm(out, v) + noise
    return new_syn
