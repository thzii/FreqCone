"""SelEx issue #11: level >= 1 pseudo-labels come back in [predicted known; predicted unknown] order."""
import numpy as np
import torch

from conftest import needs_cuda
from freqcone.hypcd import loader_order


def test_loader_order_inverts_the_known_unknown_concatenation():
    rng = np.random.default_rng(0)
    mask = rng.random(50) < 0.4
    labels_loader = torch.arange(50) * 10                      # label of image i in loader order
    order = np.concatenate([np.nonzero(mask)[0], np.nonzero(~mask)[0]])
    as_returned = labels_loader[torch.as_tensor(order)]        # what fit_mix returns at levels >= 1
    assert not torch.equal(as_returned, labels_loader)          # the upstream readers would misread these
    assert torch.equal(loader_order(as_returned, mask), labels_loader)


def test_loader_order_is_the_identity_when_known_images_come_first():
    """Level 0 splits by the labelled mask, and the loader lists labelled images first: nothing to fix there."""
    mask = np.array([True] * 7 + [False] * 5)
    labels = torch.randint(0, 9, (12,))
    assert torch.equal(loader_order(labels, mask), labels)


@needs_cuda
def test_upstream_fit_mix_returns_known_first_and_the_fix_realigns():
    from methods.clustering.faster_mix_k_means_pytorch import K_Means
    g = torch.Generator().manual_seed(0)
    k, per = 6, 20
    centers = torch.randn(k, 16, generator=g) * 10
    truth = torch.arange(k).repeat_interleave(per)
    feats = (centers[truth] + torch.randn(k * per, 16, generator=g)).cuda()
    mask = np.zeros(k * per, dtype=bool)
    mask[np.random.default_rng(1).permutation(k * per)[: k * per // 2]] = True
    mask &= (truth.numpy() < 3)                                   # "predicted known": classes 0-2 only
    km = K_Means(k=k, tolerance=1e-4, max_iterations=10, init='k-means++', n_init=1, random_state=0,
                 n_jobs=None, pairwise_batch_size=1024, mode=None, protos=None, cluster_size=None)
    km.fit_mix(feats[~torch.as_tensor(mask).cuda()], feats[torch.as_tensor(mask).cuda()],
               truth[torch.as_tensor(mask)].cuda())
    returned = km.labels_.cpu()
    fixed = loader_order(returned, mask)

    def purity(lab):
        return sum(int(torch.bincount(truth[lab == c]).max()) for c in torch.unique(lab)) / truth.numel()
    assert purity(fixed) > 0.99 and purity(returned) < 0.9
