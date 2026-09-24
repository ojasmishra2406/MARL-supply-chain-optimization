import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

class BaselineForecaster:
    def fit(self, history):
        pass
        
    def predict(self, history, horizon=1):
        raise NotImplementedError

class NaiveForecaster(BaselineForecaster):
    def predict(self, history, horizon=1):
        # Predict the last observed value
        last_val = history[-1] if len(history) > 0 else 0
        return [last_val] * horizon

class MovingAverageForecaster(BaselineForecaster):
    def __init__(self, window=3):
        self.window = window
        
    def predict(self, history, horizon=1):
        if len(history) == 0:
            return [0] * horizon
        val = np.mean(history[-self.window:]) if len(history) >= self.window else np.mean(history)
        return [val] * horizon

class ExponentialSmoothingForecaster(BaselineForecaster):
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        
    def predict(self, history, horizon=1):
        if len(history) == 0:
            return [0] * horizon
        
        # Calculate ES over the history
        s = history[0]
        for val in history[1:]:
            s = self.alpha * val + (1 - self.alpha) * s
            
        return [s] * horizon

class XGBoostForecaster(BaselineForecaster):
    def __init__(self, window=5):
        self.window = window
        self.model = None
        
    def fit(self, history_series):
        # Prepare supervised dataset
        import xgboost as xgb
        X, y = [], []
        for i in range(len(history_series) - self.window):
            X.append(history_series[i:i+self.window])
            y.append(history_series[i+self.window])
            
        if not X:
            return
            
        X = np.array(X)
        y = np.array(y)
        
        self.model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=1)
        self.model.fit(X, y)
        
    def predict(self, history, horizon=1):
        if self.model is None or len(history) < self.window:
            # Fallback to naive if not enough history
            last_val = history[-1] if len(history) > 0 else 0
            return [last_val] * horizon
            
        preds = []
        current_input = list(history[-self.window:])
        
        for _ in range(horizon):
            x = np.array([current_input])
            pred = self.model.predict(x)[0]
            preds.append(max(0, pred))  # Demand can't be negative
            current_input.pop(0)
            current_input.append(pred)
            
        return preds

class LSTMForecaster(BaselineForecaster):
    def __init__(self, window=5, epochs=50):
        self.window = window
        self.epochs = epochs
        self.model = None
        
    def fit(self, history_series):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        class SimpleLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
                self.linear = nn.Linear(16, 1)
                
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.linear(out[:, -1, :])
                return out
                
        self.model = SimpleLSTM()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        X, y = [], []
        for i in range(len(history_series) - self.window):
            X.append([[v] for v in history_series[i:i+self.window]])
            y.append([history_series[i+self.window]])
            
        if not X:
            return
            
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        
        # We run it on CPU to avoid interfering with Phase 6 GPU jobs
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self.model(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()
            
        self.model.eval()

    def predict(self, history, horizon=1):
        import torch
        if self.model is None or len(history) < self.window:
            last_val = history[-1] if len(history) > 0 else 0
            return [last_val] * horizon
            
        preds = []
        current_input = list(history[-self.window:])
        
        with torch.no_grad():
            for _ in range(horizon):
                x_t = torch.tensor([[[v] for v in current_input]], dtype=torch.float32)
                pred = self.model(x_t).item()
                preds.append(max(0, pred))
                current_input.pop(0)
                current_input.append(pred)
                
        return preds

def evaluate_forecaster(forecaster, df, window, horizon=1):
    y_true = []
    y_pred = []
    
    # We evaluate on each episode separately to avoid cross-episode leakage
    for ep in df["episode"].unique():
        ep_data = df[df["episode"] == ep]["demand"].values
        
        for i in range(window, len(ep_data) - horizon + 1):
            history = ep_data[:i]
            actual = ep_data[i:i+horizon]
            
            pred = forecaster.predict(history, horizon)
            
            y_true.extend(actual)
            y_pred.extend(pred)
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # sMAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.mean(np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)) * 100
    
    bias = np.mean(y_pred - y_true)
    
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "Bias": bias}

if __name__ == "__main__":
    train_df = pd.read_csv("data/forecasting/train_demand.csv")
    test_df = pd.read_csv("data/forecasting/test_demand_ood.csv")
    
    window = 5
    
    models = {
        "Naive": NaiveForecaster(),
        "MovingAverage": MovingAverageForecaster(window=window),
        "ExpSmoothing": ExponentialSmoothingForecaster(alpha=0.3),
        "XGBoost": XGBoostForecaster(window=window),
        "LSTM": LSTMForecaster(window=window, epochs=100)
    }
    
    # Train ML models
    train_series = train_df["demand"].values
    models["XGBoost"].fit(train_series)
    models["LSTM"].fit(train_series)
    
    print("=== Evaluation on IN-DISTRIBUTION Data ===")
    for name, model in models.items():
        metrics = evaluate_forecaster(model, train_df, window=window, horizon=1)
        print(f"{name:15} | MAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f} | sMAPE: {metrics['sMAPE']:.2f}% | Bias: {metrics['Bias']:.2f}")
        
    print("\n=== Evaluation on OUT-OF-DISTRIBUTION Data ===")
    for name, model in models.items():
        metrics = evaluate_forecaster(model, test_df, window=window, horizon=1)
        print(f"{name:15} | MAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f} | sMAPE: {metrics['sMAPE']:.2f}% | Bias: {metrics['Bias']:.2f}")
