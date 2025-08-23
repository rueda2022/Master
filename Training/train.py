import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def split_data(simulation, N):
    inputs = simulation['inputs']
    outputs = simulation['outputs']
    split_idx = int(0.8 * N)
    X_train, X_val = inputs[:split_idx], inputs[split_idx:]
    y_train, y_val = outputs[:split_idx], outputs[split_idx:]

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)

    return X_train, y_train, X_val, y_val


# Función para graficar el historial
def plot_training(history, model_name):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=history['train_loss'], label='Training Loss', color='#1f77b4', linewidth=2.5)
    sns.lineplot(data=history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2.5, linestyle='--')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss (MSE)', fontsize=16)
    plt.title(f'Training and Validation Loss for {model_name}', fontsize=18)
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.show()


# PyTorch model classes
class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        # Estrategia: Mantener la misma estructura pero ajustar hiperparámetros
        self.layers = nn.Sequential(
            nn.Linear(input_size, 48),          # Aumentado a 48 neuronas
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.1),                  # Cambio a LeakyReLU para mejor gradiente
            nn.Dropout(0.15),                   # Reducido ligeramente para permitir mejor aprendizaje
            nn.Linear(48, 32),                  # Capa intermedia más grande
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            nn.Linear(32, 20)
        )
    
    def forward(self, x):
        return self.layers(x)


class CorrelatedModel(nn.Module):
    def __init__(self, input_size):
        super(CorrelatedModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 20)
        )
        
    def forward(self, x):
        return self.layers(x)


class GammaModel(nn.Module):
    def __init__(self, input_size):
        super(GammaModel, self).__init__()
        # Arquitectura más profunda y con más capacidad
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),          # Aumentado a 64 neuronas
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),                    # Reducido para evitar regularización excesiva
            
            nn.Linear(64, 96),                  # Capa más grande
            nn.BatchNorm1d(96),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(96, 64),                  
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            
            nn.Linear(64, 32),                  
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),                    # Dropout más bajo en capas finales
            
            nn.Linear(32, 20)
        )
        
        # Inicialización mejorada de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.layers(x)


class DefaultModel(nn.Module):
    def __init__(self, input_size):
        super(DefaultModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 20)
        )
        
    def forward(self, x):
        return self.layers(x)


# PyTorch training function
def train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, patience=None, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Loss and optimizer - añadimos weight_decay para mejor regularización
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler mejorado
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_weights = None
    early_stop_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        # Update scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping check
        if patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.state_dict().copy()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    model.load_state_dict(best_model_weights)
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
    
    return model, history


# Main training function
def train_model(simulation, dist=('normal', 'normal'), corr=False):
    X_train, y_train, X_val, y_val = split_data(simulation, 10000)
    
    if dist == ('normal', 'normal') and not corr:
        model = SimpleModel(input_size=4)
        epochs = 40                  # Aumentado a 40 épocas para permitir mejor convergencia
        batch_size = 128             # Aumentado a 128 para entrenamiento más estable
        model, history = train_pytorch_model(
            model, X_train, y_train, X_val, y_val, 
            epochs, batch_size, patience=15, lr=0.0005  # Learning rate más bajo para ajuste fino
        )
        return model, history
        
    elif dist == ('normal', 'normal') and corr:
        model = CorrelatedModel(input_size=5)
        epochs = 50
        batch_size = 128
        model, history = train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, patience=15)
        return model, history
        
    elif dist == ('gamma', 'gamma') and not corr:
        model = GammaModel(input_size=4)  # Cambiado a GammaModel (con G mayúscula)
        epochs = 80                        # Aumentado para permitir mejor convergencia
        batch_size = 64                    
        model, history = train_pytorch_model(
            model, X_train, y_train, X_val, y_val, 
            epochs, batch_size, patience=20, lr=0.0005  # Learning rate reducido, paciencia aumentada
        )
        return model, history
    
    else:
        model = DefaultModel(input_size=4)
        epochs = 30
        batch_size = 32
        model, history = train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)
        return model, history


def mse_calculation(y, y_h):
    mse = 0
    n = len(y)
    for i in range(n):
        mse += (y[i]- y_h[i])**2

    return mse/n