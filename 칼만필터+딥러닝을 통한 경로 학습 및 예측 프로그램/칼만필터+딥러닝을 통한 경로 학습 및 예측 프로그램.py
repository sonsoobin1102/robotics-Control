import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.widgets import Slider, CheckButtons

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# [한글 폰트 설정]
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 데이터 불러오기
# ==========================================
csv_path = r'C:\Users\1102s\OneDrive\Sonsoobin\현직자수업\수요일 sw개발\4주차\IMU_GPS_sensor_data.csv'

try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")
    sys.exit()

time = data['time'].values
gps_x = data['gps_x'].values
gps_y = data['gps_y'].values
absolute_x = data['absolute_x'].values
absolute_y = data['absolute_y'].values
imu_acceleration_x = data['imu_acceleration_x'].values
imu_acceleration_y = data['imu_acceleration_y'].values

# 변수명 매핑
gt_x = absolute_x
gt_y = absolute_y

# ==========================================
# 2. AI 모델 학습 (결과 고정)
# ==========================================
scaler_imu = MinMaxScaler(); norm_imu = scaler_imu.fit_transform(data[['imu_acceleration_x', 'imu_acceleration_y']].values)
scaler_gps = MinMaxScaler(); norm_gps = scaler_gps.fit_transform(data[['gps_x', 'gps_y']].values)
scaler_gt = MinMaxScaler(); norm_gt = scaler_gt.fit_transform(data[['absolute_x', 'absolute_y']].values)

SEQ_LENGTH = 10
def create_dataset(imu, gps, gt):
    X_i, X_g, Y = [], [], []
    for i in range(len(imu)-SEQ_LENGTH):
        X_i.append(imu[i:i+SEQ_LENGTH])
        X_g.append(gps[i+SEQ_LENGTH])
        Y.append(gt[i+SEQ_LENGTH])
    return torch.FloatTensor(np.array(X_i)), torch.FloatTensor(np.array(X_g)), torch.FloatTensor(np.array(Y))

X_imu_t, X_gps_t, Y_gt_t = create_dataset(norm_imu, norm_gps, norm_gt)
train_size = int(len(X_imu_t)*0.7)

class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 32, batch_first=True)
        self.fc_gps = nn.Linear(2, 32)
        self.fusion = nn.Sequential(nn.Linear(64,64), nn.ReLU(), nn.Linear(64,2))
    def forward(self, i, g):
        o, _ = self.lstm(i)
        return self.fusion(torch.cat((o[:,-1,:], torch.relu(self.fc_gps(g))), dim=1))

model = FusionNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("AI 모델 학습 중... (잠시만 기다려주세요)")
for i in range(150):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_imu_t[:train_size], X_gps_t[:train_size]), Y_gt_t[:train_size])
    loss.backward()
    optimizer.step()
print("학습 완료!")

model.eval()
with torch.no_grad():
    ai_pred_norm = model(X_imu_t, X_gps_t).numpy()
    ai_pred = scaler_gt.inverse_transform(ai_pred_norm)

# ==========================================
# 3. 칼만 필터
# ==========================================
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.x = np.array([[0], [0], [0], [0]]) 
        self.P = np.eye(4)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_variance
        self.R = np.eye(2) * measurement_variance

    def predict(self, dt, imu_acceleration):
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        acceleration = np.array([[0.5 * imu_acceleration[0] * dt**2], 
                                 [0.5 * imu_acceleration[1] * dt**2],
                                 [imu_acceleration[0] * dt],
                                 [imu_acceleration[1] * dt]])
        self.x = self.F @ self.x + acceleration
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, gps_measurement):
        y = gps_measurement - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

def run_kalman_filter(process_variance, measurement_variance):
    kf = KalmanFilter(process_variance, measurement_variance)
    
    # 초기 위치 보정
    kf.x = np.array([[gps_x[0]], [gps_y[0]], [0], [0]]) 
    
    estimates_x = []
    estimates_y = []

    for i in range(len(time)):
        dt = time[i]-time[i-1] if i>0 else 0.1
        imu_acceleration = [imu_acceleration_x[i], imu_acceleration_y[i]]
        gps_measurement = np.array([[gps_x[i]], [gps_y[i]]])
        
        if i > 0: 
            kf.predict(dt, imu_acceleration)
        
        kf.update(gps_measurement)
        estimates_x.append(kf.x[0,0])
        estimates_y.append(kf.x[1,0])
        
    return estimates_x, estimates_y

# ==========================================
# 4. 시각화
# ==========================================
fig, ax = plt.subplots(figsize=(12, 9))
plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.25)

slice_idx = SEQ_LENGTH 

# 1. GPS
p_gps = ax.scatter(gps_x[slice_idx:], gps_y[slice_idx:], s=15, c='green', alpha=0.3, 
                   label='GPS measured') 
# 2. GT
p_gt = ax.scatter(gt_x[slice_idx:], gt_y[slice_idx:], s=8, c='red', alpha=0.5, 
                  label='absolute_path(GT)') 

# 3. KF (초기값 1.0으로 수정)
init_process_variance = 0.001 
init_measurement_variance = 1.0 # [수정] 2.0 -> 1.0
estimates_x, estimates_y = run_kalman_filter(init_process_variance, init_measurement_variance)
p_kf, = ax.plot(estimates_x[slice_idx:], estimates_y[slice_idx:], c='blue', linewidth=2, 
                label='Kalman Filter predicted(GPS+IMU)') 

# 4. AI Train
p_ai_train, = ax.plot(ai_pred[:train_size, 0], ai_pred[:train_size, 1],
                      c='cyan', linewidth=3, alpha=0.9, label='AI (Train)') 
# 5. AI Test
p_ai_test, = ax.plot(ai_pred[train_size:, 0], ai_pred[train_size:, 1],
                      c='orange', linewidth=3, linestyle='--', alpha=1.0, label='AI (Test)') 

ax.set_title("칼만필터 + Ai 딥러닝 시뮬레이션") 
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.grid(True)

# --- 슬라이더 ---
ax_measurement_variance = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_process_variance = plt.axes([0.25, 0.05, 0.65, 0.03])

# [수정] 최대값 10.0 -> 5.0 (기존의 절반)
s_measurement_variance = Slider(ax_measurement_variance, 'Measurement Variance', 0.1, 5.0, valinit=init_measurement_variance)
s_process_variance = Slider(ax_process_variance, 'Process Variance', 0.001, 0.1, valinit=init_process_variance, valfmt='%1.5f')

def update(val):
    process_variance = s_process_variance.val
    measurement_variance = s_measurement_variance.val
    new_est_x, new_est_y = run_kalman_filter(process_variance, measurement_variance)
    p_kf.set_data(new_est_x[slice_idx:], new_est_y[slice_idx:])
    fig.canvas.draw_idle()

s_measurement_variance.on_changed(update)
s_process_variance.on_changed(update)

# --- 체크박스 ---
lines = [p_gps, p_gt, p_kf, p_ai_train, p_ai_test]
labels = [l.get_label() for l in lines]
visibility = [True] * len(lines)

# 범례 위치 유지 (Y축 뚫림 방지: left=0.08)
ax_check = plt.axes([0.08, 0.28, 0.25, 0.22]) 
ax_check.set_facecolor('white')

check = CheckButtons(ax_check, labels, visibility)

def toggle(label):
    idx = labels.index(label)
    lines[idx].set_visible(not lines[idx].get_visible())
    plt.draw()

check.on_clicked(toggle)

plt.show()