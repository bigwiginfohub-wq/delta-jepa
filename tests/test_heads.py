import torch
import pytest
from delta_jepa.heads import H3Head, BoundaryHead, MCLHead


def test_h3_head():
    head = H3Head(latent_dim=512, warrant_dim=128)
    z = torch.randn(4, 512)
    warrant = head(z)
    assert warrant.shape == (4, 128)
    assert warrant.min() >= -1.0
    assert warrant.max() <= 1.0


def test_boundary_head():
    head = BoundaryHead(latent_dim=512, boundary_dim=64)
    z = torch.randn(4, 512)
    boundary = head(z)
    assert boundary.shape == (4, 64)
    assert (boundary >= 0).all()
    assert (boundary <= 1).all()


def test_mcl_head():
    head = MCLHead(latent_dim=512)
    z_pred = torch.randn(4, 512)
    z_target = torch.randn(4, 512)
    mcl = head(z_pred, z_target)
    assert mcl.shape == (4,)
    assert (mcl >= 0).all()
    assert (mcl <= 1).all()
