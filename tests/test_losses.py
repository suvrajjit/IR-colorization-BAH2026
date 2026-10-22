import os
import sys
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from losses import Stage1PhysicsStructuralLoss, Stage2ColorizationLoss

def test_losses():
    print("Testing Stage 1 Loss...")
    loss1 = Stage1PhysicsStructuralLoss(lambda_physics=0.1, lambda_ssim=0.5)
    pred_100m = torch.rand(2, 1, 512, 512)
    gt_100m = torch.rand(2, 1, 512, 512)
    input_200m = torch.rand(2, 1, 256, 256)
    batch_constants = {
        "radiance_mult": torch.tensor([0.00038, 0.00038]),
        "radiance_add": torch.tensor([0.1, 0.1])
    }
    loss_val1, metrics1 = loss1(pred_100m, gt_100m, input_200m, batch_constants)
    print("Stage 1 Loss:", loss_val1.item())
    print("Stage 1 Metrics:", metrics1)

    print("Testing Stage 2 Loss...")
    loss2 = Stage2ColorizationLoss(lambda_ssim=0.5)
    pred_rgb = torch.rand(2, 3, 512, 512)
    gt_rgb = torch.rand(2, 3, 512, 512)
    loss_val2, metrics2 = loss2(pred_rgb, gt_rgb)
    print("Stage 2 Loss:", loss_val2.item())
    print("Stage 2 Metrics:", metrics2)
    print("All loss tests passed!")

if __name__ == "__main__":
    test_losses()
