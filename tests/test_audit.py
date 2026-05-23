import torch
from delta_jepa.audit import generate_audit_report, AuditReport


def test_generate_audit_report():
    prediction = torch.randn(10, 512)
    warrant = torch.randn(10, 128)
    boundary = torch.randn(10, 64)
    
    report = generate_audit_report(
        prediction=prediction,
        h3_warrant=warrant,
        boundary=boundary,
        friction=0.25,
        mcl=0.75,
        horizon=10
    )
    
    assert isinstance(report, AuditReport)
    assert report.friction_score == 0.25
    assert report.mcl_coefficient == 0.75
    assert report.drift_detected is False  # 0.25 < 0.3 threshold
    assert report.to_dict() is not None