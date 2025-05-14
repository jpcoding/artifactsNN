import os
import torch
import torch.nn as nn
import torch.optim as optim

def range_penalty(predicted, bound=1.0):
    excess = torch.relu(predicted.abs() - bound)
    return (excess ** 2).mean()

def train_dncnn3d(model, train_loader, val_loader, device, args, model_str="dncnn3d", save_model="default", 
                  criterion=nn.MSELoss(), optimizer=None, num_epochs=100, patience=20, lambda_range=0.0):

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            residuals = targets - inputs
            residual_bound = residuals.abs().view(residuals.size(0), -1).max(dim=1)[0]  # [B]

            residuals = normalize(residuals, residual_bound)

            input_min, input_max = normalize_to_minus1_1_(inputs)  # in-place
            _ = normalize_to_minus1_1_(targets.clone(), input_min, input_max)  # targets_norm (not used)

            optimizer.zero_grad()
            predicted_residuals = model(inputs)

            loss = criterion(predicted_residuals, residuals)

            if lambda_range > 0:
                loss += lambda_range * range_penalty(predicted_residuals)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                residuals = targets - inputs
                residual_bound = residuals.abs().view(residuals.size(0), -1).max(dim=1)[0]

                input_min, input_max = normalize_to_minus1_1_(inputs)
                residuals = normalize(residuals, residual_bound)

                predicted_residuals = model(inputs)

                predicted_residuals_denorm = denormalize(predicted_residuals, residual_bound)
                inputs_denorm = denormalize_from_minus1_1_(inputs, input_min, input_max)
                predicted_targets = inputs_denorm + predicted_residuals_denorm

                loss = criterion(predicted_residuals, residuals)

                if lambda_range > 0:
                    loss += lambda_range * range_penalty(predicted_residuals)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4e} | Val Loss: {avg_val_loss:.4e}")

        # Save all models
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, f"{model_str}_epoch_{epoch}.pth"))

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Save best model
    best_model_path = f"best_residual_model_{model_str}_{save_model}.pth"
    torch.save(best_state, best_model_path)
    print(f"Best model saved to {best_model_path}")
