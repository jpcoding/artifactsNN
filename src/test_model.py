import torch 
from utils import normalize_residual, denormalize_residual, normalize_to_minus1_1_, denormalize_from_minus1_1_ 
def test_dncnn3d(config):
    
    model = config.model.to(config.device)
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=config.device))
    model.eval()
    criterion = config.criterion
    test_loader = config.test_loader
    device = config.device
    
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            residuals = targets - inputs
            residual_bound = residuals.abs().view(residuals.size(0), -1).max(dim=1)[0]

            residuals = normalize_residual(residuals, residual_bound)
            input_min, input_max = normalize_to_minus1_1_(inputs)
            _ = normalize_to_minus1_1_(targets, input_min, input_max)

            predicted_residuals = model(inputs)

            loss = criterion(predicted_residuals, residuals)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"[Test] Average Loss: {avg_test_loss:.6e}")
    return avg_test_loss


def test_3d(config):
    model = config.model.to(config.device)
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=config.device))
    model.eval()
    criterion = config.criterion
    test_loader = config.test_loader
    device = config.device
    
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_min, input_max = normalize_to_minus1_1_(inputs)
            normalize_to_minus1_1_(targets, input_min, input_max)
            predicted_residuals = model(inputs)
            loss = criterion(predicted_residuals, targets)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"[Test] Average Loss: {avg_test_loss:.6e}")
    return avg_test_loss