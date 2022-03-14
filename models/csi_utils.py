import diffdist.functional as distops
import torch.utils.data

from models.layers import *


# code borrowed from https://github.com/alinlab/CSI


def get_simclr_augmentation(image_size):
    # parameter for resizecrop
    # resize_scale = (P.resize_factor, 1.0) # resize scaling factor
    resize_scale = (0.08, 1.0)  # resize scaling factor

    # if P.resize_fix: # if resize_fix is True, use same scale
    # resize_scale = (P.resize_factor, P.resize_factor)

    # Align augmentation
    color_jitter = ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = RandomColorGrayLayer(p=0.2)
    resize_crop = RandomResizedCropLayer(scale=resize_scale, size=None)

    # Transform define #
    transform = nn.Sequential(
        color_jitter,
        color_gray,
        resize_crop)

    return transform


def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    '''

    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)

    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')

    return sim_matrix


def NT_xent(sim_matrix, temperature=0.5, chunk=2, eps=1e-8):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    loss = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

    return loss


def Supervised_NT_xent(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    if multi_gpu:
        gather_t = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
        labels = torch.cat(distops.all_gather(gather_t, labels))
    labels = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    labels = labels.contiguous().view(-1, 1)
    Mask = torch.eq(labels, labels.t()).float().to(device)
    # Mask = eye * torch.stack([labels == labels[i] for i in range(labels.size(0))]).float().to(device)
    Mask = Mask / (Mask.sum(dim=1, keepdim=True) + eps)

    loss = torch.sum(Mask * sim_matrix) / (2 * B)

    return loss


def Semisupervised_NT_xent(sim_matrix, len_p, len_u, temperature=0.5, chunk=2, eps=1e-8, flag=False, w=1):
    """
    :param w: weight of unlabeled samples
    :param flag: if true other positive examples considered in contrastive loss as negatives, if false only
        unlabeled data considered as negative

    :return: contrastive loss for PU-CSI
    """
    B = sim_matrix.size(0) // (chunk * 2)  # B = B' / chunk / 2

    # Bs = B // 4
    if not flag:
        sim_matrix = sim_matrix[: 2 * len_p, :]
        mask = torch.zeros_like(sim_matrix)
        mask[:, 2 * len_p:] = 1
        mask[torch.arange(len_p), len_p + torch.arange(len_p)] = 1
        mask[B + torch.arange(len_p), torch.arange(len_p)] = 1

        # add self shift as negative examples
        # for i in range(7):
        #     mask[torch.arange(0, 2 * B - (i + 1) * Bs), (i + 1) * Bs + torch.arange(2 * B - (i + 1) * Bs)] = 1
        #     mask[(i + 1) * Bs + torch.arange(2 * B - (i + 1) * Bs), torch.arange(2 * B - (i + 1) * Bs)] = 1

    else:
        sim_matrix = sim_matrix[: 2 * len_p, :]
        mask = torch.ones_like(sim_matrix)
        mask[torch.arange(2 * len_p), torch.arange(2 * len_p)] = 0
        mask[:, 2 * len_p:] = w

    sim_matrix = torch.exp(sim_matrix / temperature) * mask  # only negative samples

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix[:, : 2 * len_p]

    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    loss = torch.sum(sim_matrix[:len_p, len_p:].diag() + sim_matrix[len_p:, :len_p].diag()) / (2 * len_p)

    return loss
