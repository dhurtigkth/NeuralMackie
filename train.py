import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
from dataparser import audio_parser
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class EncodeTensor(JSONEncoder,Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(json.NpEncoder, self).default(obj)

class LSTM_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # LSTM expects input of shape (batch_size, sequence_length, input_size)
        self.lstm = nn.LSTM(input_size=6, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 512)  # Map from hidden size to output size (512)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length=512, input_size=6)
        x, _ = self.lstm(x)  # Output shape: (batch_size, 512, hidden_size=50)
        x = x[:, -1, :]       # Take the last time step (batch_size, 50)
        x = self.linear(x)    # Output shape: (batch_size, 512)
        return x

def train(x, y, n_epochs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using " + str(device))
        torch.cuda.empty_cache()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)
        x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        # Set up the model
        model = LSTM_Model().to(device)
        optimizer = optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()
        
        # Load the training data
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=2)
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=2)

        print("training started..")
        for epoch in range(n_epochs):
            model.train()
            for X_batch, y_batch in tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Validation
            if epoch % 100 != 0:
                torch.cuda.empty_cache() # Try this, maybe it helps
                continue
            print("evaluating model..")
            model.eval()
            test_rmse_total = 0.0
            num_batches = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    y_pred = model(X_batch)  # Run inference on smaller batches
                    batch_rmse = torch.sqrt(loss_fn(y_pred, y_batch))  # Compute RMSE for the batch
                    test_rmse_total += batch_rmse.item()
                    num_batches += 1
                    torch.cuda.empty_cache()  # Free up memory
            print("Epoch %d: test RMSE %.4f" % (epoch, test_rmse_total/num_batches))
            return model
    



#data, targets = audio_parser("Training Data/C", "Training Data/D", 512)
#trained_model = train(data, targets, 2000)


#with open('models/gru_torch.json', 'w') as json_file:
#    json.dump(trained_model.state_dict(), json_file,cls=EncodeTensor)
