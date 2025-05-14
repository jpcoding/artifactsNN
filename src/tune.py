import torch
import torch.nn as nn
import torch.optim as optim

def finetune_dncnn3d(model, original,quantized, fine_tune_save_path,  device='cuda', num_epochs=5, lr=1e-4,  
                     freeze_early=False):
    model = model.to(device)
    model.train()
    
    # Compute residual ground truth
    residual = original - quantized
    residual_bound = max(abs(residual.min()), abs(residual.max()))

    # Normalize input to [-1, 1]
    quantized_norm = (quantized - quantized.min()) / (quantized.max() - quantized.min()) * 2 - 1

    # Prepare input/target tensors
    input_tensor = torch.from_numpy(quantized_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    target_tensor = torch.from_numpy(residual / residual_bound).unsqueeze(0).unsqueeze(0).float().to(device)

    # Optionally freeze early layers (e.g., conv + relu at start)
    if freeze_early:
        for name, param in model.named_parameters():
            if 'dncnn.0' in name or 'dncnn.1' in name:  # First Conv3D and ReLU
                param.requires_grad = False

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
        print(f"[Fine-tune] Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.6f}")

    # Save fine-tuned model (optional)
    torch.save(model.state_dict(), fine_tune_save_path)

    # Switch back to eval mode
    model.eval()

    return model, residual_bound

