import torch
import pytest
from delta_jepa.loss import h3_warrant_loss, friction_score, mcl_loss


def test_h3_warrant_loss():
    warrants = torch.randn(8, 128)
    loss = h3_warrant_loss(warrants)
    assert loss >= 0
    assert torch.isfinite(loss)


def test_friction_score():
    predictions = [torch.randn(4, 512) for _ in range(10)]
    friction = friction_score(predictions)
    assert 0 <= friction <= 1


def test_mcl_loss():
    pred = torch.tensor([0.9, 0.8, 0.3, 0.2])
    gt = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss = mcl_loss(pred, gt)
    assert loss >= 0