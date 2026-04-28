import torch


def flow_loss_func(flow_preds, flow_gt, valid,
                   max_flow=400,
                   gamma=0.9
                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    if valid.max() < 0.5:
        pass

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        'mag': mag.mean().item()
    }

    return flow_loss, metrics


def sparse_flow_loss_func(
    flow_preds, flow_gt, valid,
    max_flow=400,
    gamma=0.9,
    num_points=8192,
):
    """Loss computed on a random subset of N pixel locations.

    This is the implicit-decoder counterpart of ``flow_loss_func``.
    Instead of evaluating every pixel, we randomly sample ``num_points``
    valid pixels per image and supervise only those.  This keeps memory
    bounded even for very high-resolution outputs and aligns with
    InfiniDepth's random coordinate-depth pair training strategy.

    Args:
        flow_preds: list of [B, 2, H, W] flow predictions (multi-scale).
        flow_gt:    [B, 2, H, W] ground-truth flow.
        valid:      [B, H, W] validity mask.
        max_flow:   Discard pixels with GT magnitude > max_flow.
        gamma:      Exponential weighting for multi-scale supervision.
        num_points: Number of random pixels to sample per image.
    Returns:
        (flow_loss, metrics)  matching the signature of flow_loss_func.
    """
    B, _, H, W = flow_gt.shape
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
    valid = (valid >= 0.5) & (mag < max_flow)     # [B, H, W]

    # Sample random indices per batch element
    sampled_gt_list = []
    sampled_pred_lists = [[] for _ in range(n_predictions)]
    sampled_valid_list = []

    for b in range(B):
        valid_idx = valid[b].nonzero(as_tuple=False)  # [K, 2]  (y, x)
        K = valid_idx.shape[0]
        if K == 0:
            continue
        n = min(num_points, K)
        perm = torch.randperm(K, device=valid_idx.device)[:n]
        sel = valid_idx[perm]  # [n, 2]
        yy, xx = sel[:, 0], sel[:, 1]

        sampled_gt_list.append(flow_gt[b, :, yy, xx])        # [2, n]
        sampled_valid_list.append(torch.ones(n, device=flow_gt.device))
        for i in range(n_predictions):
            sampled_pred_lists[i].append(flow_preds[i][b, :, yy, xx])

    if len(sampled_gt_list) == 0:
        metrics = {'epe': 0.0, 'mag': mag.mean().item()}
        return torch.tensor(0.0, device=flow_gt.device, requires_grad=True), metrics

    gt_cat = torch.cat(sampled_gt_list, dim=1)        # [2, total_n]
    valid_cat = torch.cat(sampled_valid_list, dim=0)   # [total_n]

    for i in range(n_predictions):
        pred_cat = torch.cat(sampled_pred_lists[i], dim=1)  # [2, total_n]
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (pred_cat - gt_cat).abs().mean()
        flow_loss += i_weight * i_loss

    last_pred_cat = torch.cat(sampled_pred_lists[-1], dim=1)
    epe = torch.sum((last_pred_cat - gt_cat) ** 2, dim=0).sqrt()

    metrics = {
        'epe': epe.mean().item(),
        'mag': mag.mean().item(),
    }

    return flow_loss, metrics