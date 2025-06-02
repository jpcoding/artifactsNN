import os
import torch
import torch.nn as nn
import torch.optim as optim
from ..utils.utils import normalize_residual, denormalize_residual, normalize_to_minus1_1_, denormalize_from_minus1_1_ 
from types import SimpleNamespace


def range_penalty(predicted, bound=1.0, p =2):
    excess = torch.relu(predicted.abs() - bound)
    return (excess ** p).mean()

def sign_penalty(predicted, target, p =2):
    sign_diff = (predicted.sign() != target.sign()).float()
    return (sign_diff ** p).mean()
    


def train_residual_3d(config):
    
    model = config.model.to(config.device)
    optimizer = config.optimizer
    criterion = config.criterion
    train_loader = config.train_loader
    val_loader = config.val_loader
    device = config.device
    model_str = config.model_str
    num_epochs = config.num_epochs
    patience = config.patience
    lambda_range = config.lambda_range
    best_model_path = config.best_model_path ## best model save to this path 
    outputs_dir = config.outputs_dir ## all models are saved in this directory 
    penalty_order =  config.range_penalty_order
    penalty_fn = config.penalty_fn

    

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
            residual_bound = residuals.abs().view(residuals.size(0), -1).max(dim=1)[0]

            residuals = normalize_residual(residuals, residual_bound)
            input_min, input_max = normalize_to_minus1_1_(inputs)
            _ = normalize_to_minus1_1_(targets.clone(), input_min, input_max)
            optimizer.zero_grad()
            predicted_residuals = model(inputs)
            loss = criterion(predicted_residuals, residuals)
            if lambda_range > 0 and penalty_fn == "range_penalty": 
                loss += lambda_range * range_penalty(predicted_residuals, p = penalty_order)
            elif lambda_range > 0 and penalty_fn == "sign_penalty": 
                loss += lambda_range * sign_penalty(predicted_residuals, residuals, p = penalty_order) 

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
                residuals = normalize_residual(residuals, residual_bound)
                predicted_residuals = model(inputs)
                loss = criterion(predicted_residuals, residuals)
                if lambda_range > 0 and penalty_fn == "range_penalty": 
                    loss += lambda_range * range_penalty(predicted_residuals, p = penalty_order)
                elif lambda_range > 0 and penalty_fn == "sign_penalty": 
                    loss += lambda_range * sign_penalty(predicted_residuals, residuals, p = penalty_order) 
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4e} | Val Loss: {avg_val_loss:.4e}")
        torch.save(model.state_dict(), os.path.join(outputs_dir, f"{model_str}_epoch_{epoch}.pth"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    torch.save(best_state, best_model_path)
    print(f"Best model saved to {best_model_path}")
    

# The target is the original data and the input is the decoposed data. 
def train_3d(config): 
    model = config.model.to(config.device)
    optimizer = config.optimizer
    criterion = config.criterion
    train_loader = config.train_loader
    val_loader = config.val_loader
    device = config.device
    model_str = config.model_str
    num_epochs = config.num_epochs
    patience = config.patience
    best_model_path = config.best_model_path ## best model save to this path 
    outputs_dir = config.outputs_dir ## all models are saved in this directory 

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            input_min, input_max = normalize_to_minus1_1_(inputs)
            normalize_to_minus1_1_(targets, input_min, input_max)
            optimizer.zero_grad()
            predicted_targets = model(inputs)
            
            loss = criterion(predicted_targets, targets)

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
                input_min, input_max = normalize_to_minus1_1_(inputs)
                normalize_to_minus1_1_(targets, input_min, input_max)
                predicted_targets = model(inputs)
                loss = criterion(predicted_targets, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4e} | Val Loss: {avg_val_loss:.4e}")
        torch.save(model.state_dict(), os.path.join(outputs_dir, f"{model_str}_epoch_{epoch}.pth"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    torch.save(best_state, best_model_path) 
    print(f"Best model saved to {best_model_path}") 
