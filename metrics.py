import torch

def exp_effective_rank_torch(A, eps=1e-12):
    """
    Calcule l'effective rank exponentiel d'une matrice
    """
    S = torch.linalg.svdvals(A)
    S = S[S > eps]
    
    p = S / S.sum()
    entropy = -torch.sum(p * torch.log(p))
    effective_rank = torch.exp(entropy)
    return effective_rank.cpu().numpy()


def nuclear_over_operator_norm_torch(A):
    """
    Ratio: norme nucléaire / norme opérateur
    """
    s = torch.linalg.svdvals(A)
    nuclear_norm = s.sum()
    operator_norm = s.max()
    return (nuclear_norm / operator_norm).item()


def cosine_separation_loss(H, eps=1e-8):
    """
    Pénalité pour encourager l'orthogonalité des lignes de H
    """
    Hn = H / (torch.norm(H, dim=1, keepdim=True) + eps)
    G = Hn @ Hn.T
    I = torch.eye(G.size(0), device=H.device)
    return - torch.norm(G - I, p='fro')**2
