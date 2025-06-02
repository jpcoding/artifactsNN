import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.utils import normalize_to_minus1_1_, normalize_residual
from src.training.train_model  import range_penalty 

def finetune3d(config):
    model = config.model.to(config.device)
    model.train()

    if config.freeze_early:
        for name, param in model.named_parameters():
            if name.startswith('dncnn.0') or name.startswith('dncnn.1'):
                param.requires_grad = False

    criterion = config.criterion
    optimizer = config.optimizer
    dataloader = config.tune_loader  

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        for input_tensor, target_tensor, *_ in dataloader:
            input_tensor = input_tensor.to(config.device)
            target_tensor = target_tensor.to(config.device)
            residuals = target_tensor - input_tensor 
            residual_bound = residuals.abs().view(residuals.size(0), -1).max(dim=1)[0]
            residuals = normalize_residual(residuals, residual_bound)
            normalize_to_minus1_1_(input_tensor)
        
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output, residuals)
            if config.lambda_range > 0:
                loss += config.lambda_range * range_penalty(output, bound=config.range_bound)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Fine-tune] Epoch {epoch+1}/{config.num_epochs} - Loss: {epoch_loss / len(dataloader):.6f}")

    torch.save(model.state_dict(), config.fine_tune_save_path)
    model.eval()
    # clear cache
    torch.cuda.empty_cache()

    return model
