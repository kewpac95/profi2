import torch
import numpy as np
from profis.utils import smiles2sparse_KRFP, smiles2sparse_ECFP
from rdkit import Chem

def fingerprint_similarity(fp_true, fp_pred, method="tanimoto", fallback_reward=0.05):
    fp_true = fp_true.round().to(torch.uint8)
    fp_pred = fp_pred.round().to(torch.uint8)
    if method == "tanimoto":
        intersection = (fp_true & fp_pred).float().sum(dim=1)
        union = ((fp_true | fp_pred).float().sum(dim=1)).clamp(min=1.0)
        sim = intersection / union
        sim[fp_pred.sum(dim=1) == 0] = fallback_reward
        return sim
    else:
        raise NotImplementedError

def smiles_batch_to_fps(smiles_list, fp_type="KRFP", fp_len=2048):
    fps = []
    valid_mask = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("invalid SMILES")
            valid_mask.append(True)
            if fp_type == "KRFP":
                arr = smiles2sparse_KRFP(smi, fp_len=fp_len)
            elif fp_type == "ECFP":
                arr = smiles2sparse_ECFP(smi, n_bits=fp_len)
            else:
                raise ValueError("Unknown fp_type")
            arr = np.asarray(arr)
            if arr.size != fp_len:
                arr2 = np.zeros(fp_len, dtype=np.float32)
                arr2[:min(arr.size, fp_len)] = arr[:min(arr.size, fp_len)]
                arr = arr2
        except Exception:
            arr = np.zeros(fp_len, dtype=np.float32)
            valid_mask.append(False)
        fps.append(torch.tensor(arr, dtype=torch.float32))
    return torch.stack(fps), valid_mask

def decode_indices_to_smiles(indices, charset):
    smiles_list = []
    for row in indices:
        s = ""
        for idx in row:
            if idx < len(charset):
                c = charset[idx]
                if c == "[end]": break
                if c not in ("[start]", "[nop]"): s += c
        smiles_list.append(s)
    return smiles_list

def log_likelihood_from_logits(logits, target_indices):
    log_probs = torch.log_softmax(logits, dim=-1)
    ll = torch.gather(log_probs, 2, target_indices.unsqueeze(-1)).squeeze(-1)
    return ll.sum(dim=1)

def rl_smiles_fp_dap_training_step(
    agent_encoder,
    agent_decoder,
    prior_encoder,
    prior_decoder,
    fp_batch,
    charset,
    fp_type="KRFP",
    reward_method="tanimoto",
    sigma=60.0,
    max_len=100,
    device="cpu",
):
    agent_encoder.eval()
    agent_decoder.train()
    batch_size = fp_batch.size(0)
    fp_len = fp_batch.size(1)

    with torch.no_grad():
        mu, logvar = agent_encoder(fp_batch.to(device))
        if hasattr(agent_encoder, "sampling"):
            latent = agent_encoder.sampling(mu, logvar)
        else:
            from profis.net import Profis
            if isinstance(agent_encoder, Profis):
                latent = agent_encoder.sampling(mu, logvar)
            else:
                epsilon = torch.randn_like(logvar)
                latent = mu + torch.exp(0.5 * logvar) * epsilon

    agent_logits = agent_decoder(latent, max_len=max_len)
    agent_probs = torch.softmax(agent_logits, dim=-1)
    sampled_indices = torch.multinomial(
        agent_probs.view(-1, agent_probs.size(-1)), 1
    ).view(batch_size, max_len)
    smiles = decode_indices_to_smiles(sampled_indices.cpu().numpy(), charset)
    fp_prime, valid_mask = smiles_batch_to_fps(smiles, fp_type=fp_type, fp_len=fp_len)
    fp_prime = fp_prime.to(device)
    valid_mask = torch.tensor(valid_mask, dtype=torch.float32, device=device)
    rewards = fingerprint_similarity(fp_batch.to(device), fp_prime, method=reward_method)
    rewards = rewards * valid_mask  # 0 reward za niepoprawne SMILES

    # AGENT log-likelihood for sampled sequence
    agent_log_probs = torch.log(torch.gather(agent_probs, 2, sampled_indices.unsqueeze(-1)).squeeze(-1) + 1e-8)
    agent_ll = agent_log_probs.sum(dim=1)

    # PRIOR log-likelihood for the same sequence (detach so no grad)
    with torch.no_grad():
        mu_prior, logvar_prior = prior_encoder(fp_batch.to(device))
        if hasattr(prior_encoder, "sampling"):
            latent_prior = prior_encoder.sampling(mu_prior, logvar_prior)
        else:
            from profis.net import Profis
            if isinstance(prior_encoder, Profis):
                latent_prior = prior_encoder.sampling(mu_prior, logvar_prior)
            else:
                epsilon = torch.randn_like(logvar_prior)
                latent_prior = mu_prior + torch.exp(0.5 * logvar_prior) * epsilon
        prior_logits = prior_decoder(latent_prior, max_len=max_len)
        prior_ll = log_likelihood_from_logits(prior_logits, sampled_indices.to(device))

    augmented_ll = prior_ll + sigma * rewards
    loss = ((augmented_ll - agent_ll) ** 2).mean()

    sampling_validity = valid_mask.mean().item()

    return loss, {
        "loss": loss.item(),
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "smiles": smiles,
        "rewards": rewards.cpu().numpy(),
        "prior_ll_mean": prior_ll.mean().item(),
        "agent_ll_mean": agent_ll.mean().item(),
        "augmented_ll_mean": augmented_ll.mean().item(),
        "sampling_validity": sampling_validity,
    }