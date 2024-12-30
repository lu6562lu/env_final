import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 讀取資料
data_incomplete = pd.read_csv(
    "ExampleTrainData(IncompleteAVG)/IncompleteAvgDATA_17.csv"
)
data_complete = pd.read_csv("ExampleTrainData(AVG)/AvgDATA_17.csv")

# 轉換 Serial 為時間戳並去掉秒數
data_incomplete["Timestamp"] = data_incomplete["Serial"].astype(str).str[:-2]
data_incomplete["Timestamp"] = pd.to_datetime(
    data_incomplete["Timestamp"], format="%Y%m%d%H%M"
)
data_incomplete = data_incomplete.set_index("Timestamp")

data_complete["Timestamp"] = data_complete["Serial"].astype(str).str[:-2]
data_complete["Timestamp"] = pd.to_datetime(
    data_complete["Timestamp"], format="%Y%m%d%H%M"
)
data_complete = data_complete.set_index("Timestamp")

# 提取所需欄位
features = [
    "WindSpeed(m/s)",
    "Pressure(hpa)",
    "Temperature(°C)",
    "Humidity(%)",
    "Sunlight(Lux)",
]
target = "Power(mW)"

# 合併資料
combined_data = pd.concat(
    [data_incomplete[features + [target]], data_complete[features + [target]]]
)
combined_data = combined_data.sort_index()


# 每一天單獨處理，僅填補早上 7 點到下午 5 點的時間段
filled_data = combined_data.copy()  # 複製原始資料，這樣不會破壞原始資料
# 確保填補資料框具有欄位
for day in pd.date_range(
    start=combined_data.index.min().date(),
    end=combined_data.index.max().date(),
    freq="D",
):
    # 設定填補範圍
    start_time = day.replace(hour=7, minute=0, second=0)
    end_time = day.replace(hour=16, minute=50, second=0)
    time_range_to_fill = pd.date_range(start=start_time, end=end_time, freq="10T")

    # 跳過 end_time 早於早上 9 點的天數
    if end_time < day.replace(hour=9, minute=0, second=0):
        continue

    # 在填補範圍內進行移動平均值補缺
    for time in time_range_to_fill:
        if time not in filled_data.index:
            # 將該時間點的缺失值填充為滑動窗口的均值
            # 獲取最近的五個時間點進行平均計算
            filled_data.loc[time] = 0
            window = filled_data.loc[:time].tail(4)  # 取出之前五個有效數據點
            if not window.empty:
                filled_data.loc[time] = window.mean()
            else:
                filled_data.loc[time] = 0  # 如果沒有有效數據，用 0 填補

# 最終的 filled_data 是僅針對早上 7 點到下午 5 點補全的資料
combined_data = filled_data.sort_index()


#
# 整理 x_train 和 y_train
x_train_list = []
y_train_list = []
review_day = 1

# 假設 combined_data 已經存在
day_count = 0
for day in pd.date_range(
    start=combined_data.index.min(), end=combined_data.index.max(), freq="D"
):
    # 從第n天開始計算9
    if day_count < review_day:
        day_count += 1
        continue

    # 取得當天的資料
    day_data = combined_data.loc[
        day : day + timedelta(days=1) - timedelta(seconds=1)
    ]  # 取該天的資料

    # 確保取出的時間範圍是有效的
    morning_range = pd.date_range(
        start=day.replace(hour=7, minute=0, second=0),
        end=day.replace(hour=8, minute=50, second=0),
        freq="T",
    )
    evening_range = pd.date_range(
        start=day.replace(hour=9, minute=0, second=0),
        end=day.replace(hour=17, minute=0, second=0),
        freq="T",
    )

    # 提取當天 7點到9點的特徵資料
    x_day = day_data.loc[day_data.index.isin(morning_range)]

    # 9點到下午5點的目標資料
    y_day = day_data.loc[day_data.index.isin(evening_range), "Power(mW)"]

    # 加入前n天的完整資料
    for i in range(1, review_day + 1):
        prev_day = day - timedelta(days=i)
        prev_day_data = combined_data.loc[
            prev_day : prev_day + timedelta(days=1) - timedelta(seconds=1)
        ]

        # 如果前一天資料為空，跳過
        if prev_day_data.empty:
            break

        x_day = pd.concat([prev_day_data[features], x_day[features]]) #### [features]
    x_day = np.array(x_day)

    # 確保資料是非空的，然後儲存
    if x_day.size > 0 and y_day.size > 0:
        x_train_list.append(x_day[:, 1:])  # 儲存每個時間段的特徵值
        y_train_list.append(y_day)  # 儲存對應的目標值

# 將 list 合併並轉換為 Tensor
x_train = np.array(x_train_list, dtype=np.float32)  # 確保資料形狀一致
y_train = np.array(y_train_list, dtype=np.float32)

x_train_tensor = torch.tensor(x_train)
y_train_tensor = torch.tensor(y_train)

print(f"x_train shape: {x_train_tensor.shape}")
print(f"y_train shape: {y_train_tensor.shape}")
# (191, 193, 5)  (191, 48)

# 標準化
scaler_features = StandardScaler()
scaler_target = StandardScaler()

# 對特徵和目標進行標準化
x_train_standardized = scaler_features.fit_transform(
    x_train.reshape(-1, x_train.shape[-1])
).reshape(x_train.shape)
y_train_standardized = scaler_target.fit_transform(y_train.reshape(-1, 1)).reshape(
    y_train.shape
)

# 轉換為 Tensor
x_train_tensor = torch.tensor(x_train_standardized, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_standardized, dtype=torch.float32)

print(f"x_train shape: {x_train_tensor.shape}")
print(f"y_train shape: {y_train_tensor.shape}")


################ test data #################

test_data = pd.read_csv("ExampleTestData/upload.csv")
test_data.columns = ["date/time/id", "answer"]
date_list = []
time_list = []
station_id_list = []

for index in range(len(test_data)):
    date = str(test_data["date/time/id"][index])[:8]
    time = str(test_data["date/time/id"][index])[8:12]
    station_id = str(test_data["date/time/id"][index])[12:]

    date_list.append(date)
    time_list.append(time)
    station_id_list.append(station_id)

test_data["date"] = date_list
test_data["time"] = time_list
test_data["id"] = station_id_list

test_data.drop(["date/time/id"], axis=1, inplace=True)
station_filter = test_data["id"] == "17"
test_data = test_data[station_filter]
test_date = test_data["date"].value_counts().index

x_test_list = []
for day in pd.to_datetime(test_date):
    # 取得當天的資料
    day_data = combined_data.loc[
        day : day + timedelta(days=1) - timedelta(seconds=1)
    ]  # 取該天的資料

    # 確保取出的時間範圍是有效的
    morning_range = pd.date_range(
        start=day.replace(hour=7, minute=0, second=0),
        end=day.replace(hour=8, minute=50, second=0),
        freq="T",
    )
    evening_range = pd.date_range(
        start=day.replace(hour=9, minute=0, second=0),
        end=day.replace(hour=17, minute=0, second=0),
        freq="T",
    )

    # 提取當天 7點到9點的特徵資料
    x_day = day_data.loc[day_data.index.isin(morning_range)]

    # 加入前n天的完整資料
    for i in range(1, review_day + 1):
        prev_day = day - timedelta(days=i)
        prev_day_data = combined_data.loc[
            prev_day : prev_day + timedelta(days=1) - timedelta(seconds=1)
        ]

        # 如果前一天資料為空，跳過
        if prev_day_data.empty:
            break

        x_day = pd.concat([prev_day_data[features], x_day[features]]) ###[features]
    x_day = np.array(x_day)

    # 確保資料是非空的，然後儲存
    if x_day.size > 0:
        x_test_list.append(x_day[:, 1:])  # 儲存每個時間段的特徵值

x_test = np.array(x_test_list, dtype=np.float32)

x_test_tensor = torch.tensor(x_test)
y_test_tensor = torch.tensor(np.array(test_data[station_filter]["answer"]).reshape(12,48))

# 標準化數據
x_test_standardized = scaler_features.fit_transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
x_test_tensor = torch.tensor(x_test_standardized, dtype=torch.float32)

y_test =  np.array(test_data[station_filter]["answer"]).reshape(12,48)
y_test_standardized = scaler_target.fit_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
y_test_tensor = torch.tensor(y_test_standardized, dtype=torch.float32)

#%%
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, 64), 
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 128), 
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        x = x[:, 60:, :]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = self.output_layer(x)
        # print(output.size())
        # print(output[:, 12:60, :].size())
        # output = output[:, 12:60, :]
        output = torch.squeeze(output, dim=2)
        return output


class LSTMForecastingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pred_len):
        super(LSTMForecastingModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, pred_len * output_dim)  # 將隱藏層映射到整個未來序列的輸出
        self.pred_len = pred_len
        self.output_dim = output_dim

    def forward(self, x):
        # LSTM 輸出
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)
        # 只使用最後一個時間步的隱藏狀態進行預測
        lstm_last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        output = self.fc(lstm_last_hidden)  # (batch_size, pred_len * output_dim)
        output = output.view(-1, self.pred_len, self.output_dim)  # (batch_size, pred_len, output_dim)
        output = torch.squeeze(output, dim=2)
        return output



class final_MLP(nn.Module):
    def __init__(self):
        super(final_MLP, self).__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(32, 1)
        )
    def forward(self, x):
        output = self.output_layer(x)
        return output
    
class full_model(nn.Module):
    def __init__(
            self, 
            model_MLP, 
            model_lstm,
            # model_final_MLP,  
    ):
        super(full_model, self).__init__()
        self.model_MLP = model_MLP
        self.model_lstm = model_lstm
        # self.model_final_MLP = model_final_MLP
    
    def forward(self, x):
        mlp_output = self.model_MLP(x)
        # print(mlp_output.size())
        lstm_output = self.model_lstm(mlp_output)

        # final_output = self.model_final_MLP(final_input)
        # final_output = self.model_final_MLP(mlp_output)
        output = torch.squeeze(lstm_output, dim=2)
        # print(final_output.size())
        return output

mlp_input_dim = 4
mlp_hidden_dim = 8
mlp_output_dim = 4

MLP_model = MLP(input_dim=mlp_input_dim, hidden_dim=mlp_hidden_dim, output_dim=mlp_output_dim)
LSTM_model = LSTMForecastingModel(input_dim=4, hidden_dim=128, num_layers=5, output_dim=1, pred_len=48)
# final_MLP_model = final_MLP()

model = LSTMForecastingModel(input_dim=4, hidden_dim=128, num_layers=5, output_dim=1, pred_len=48)
# model = full_model(model_MLP=MLP_model, model_lstm=LSTM_model)
model.train()

batch_size = 16
epochs = 10000
learning_rate = 0.0005
patience = 30

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_val_loss = float("inf")  
early_stop_counter = 0  


# 分割訓練集和驗證集
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_train_tensor, y_train_tensor, test_size=0.2, random_state=42
)

# 創建訓練和驗證 DataLoader
train_dataset = TensorDataset(x_train_split, y_train_split)
val_dataset = TensorDataset(x_val_split, y_val_split)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

for epoch in range(epochs):
    model.train()
    train_epoch_loss = 0

    # 訓練階段
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()

    train_epoch_loss /= len(train_loader)

    # 驗證階段
    model.eval()
    val_epoch_loss = 0
    with torch.no_grad():
        for x_val_batch, y_val_batch in val_loader:
            val_output = model(x_val_batch)
            val_loss = criterion(val_output, y_val_batch)
            val_epoch_loss += val_loss.item()

    val_epoch_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

    # Early Stopping 檢查
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("模型已保存為 'best_model.pth'")
    else:
        early_stop_counter += 1
        print(f"Early Stopping counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print(f"早停條件滿足，在第 {epoch + 1} 個 epoch 停止訓練。最佳驗證損失：{best_val_loss:.4f}")
        break


test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_data_loader = DataLoader(test_dataset, batch_size=12)

model.eval()
model.load_state_dict(torch.load("best_model.pth"))

y_pred_list = []
for batch_idx, (x_batch, y_batch) in enumerate(test_data_loader):
    y_pred = model(x_batch)
    y_pred_list.append(y_pred.detach().numpy())

y_pred_list = np.array(y_pred_list)
y_pred_list = np.squeeze(y_pred_list, axis=0)

# 還原標準化
y_pred = scaler_target.inverse_transform(y_pred_list.reshape(-1, 1)).reshape(y_pred_list.shape)


def calculate_total_score(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    return np.sum(np.abs(y_pred - y_true))

print(calculate_total_score(y_test, y_pred))

fig, ax = plt.subplots(figsize=(10,7), nrows=3, ncols=4)

date = 0
for row in range(3):
    for col in range(4):
        ax[row, col].plot(y_pred[date].flatten(), label="predicted data")
        ax[row, col].plot(y_test[date].flatten(), label="test data")
        
        date += 1
ax[0, 3].legend()



