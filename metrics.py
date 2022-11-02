import torch


def f_score(pr, gt):

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((2 * tp) / (2 * tp + fn + fp))

    return score


def accuracy(pr, gt):

    tp = torch.sum(gt == pr)
    score = tp / gt.reshape(-1).shape[0]
    return score


def precision(pr, gt):

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp) / (tp + fp)

    return score


def recall(pr, gt):
    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp / (tp + fn))

    return score


def specificity(pr, gt):

    neg_gt = torch.abs(torch.sub(gt, 1))
    neg_pr = torch.abs(torch.sub(pr, 1))

    tn = torch.sum(neg_gt * neg_pr)
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tn) / (tn + fp)

    return score


def batch_metric(pr, gt, metric):

    assert pr.shape == gt.shape

    pr_data, gt_data = pr.cpu().detach(), gt.cpu().detach()

    b, c, _, _ = pr_data.shape

    total = 0

    for i in range(b):

        total += metric(pr_data[i, 1:, ...], gt_data[i, 1:, ...])

    return total/b
