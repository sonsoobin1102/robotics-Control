import sys
import numpy as np
import platform
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QComboBox, QFrame, 
                             QGroupBox, QDoubleSpinBox, QPushButton, QAbstractSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 수학 및 기구학 함수 (변경 없음)
# ==========================================

# UR5 DH Parameters
a = np.array([0, -0.425, -0.39225, 0, 0, 0])
d = np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])
alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])

def dh_transform(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),                d],
        [0,              0,                            0,                            1]
    ])

def forward_kinematics(theta_list, a, d, alpha):
    T = np.eye(4)
    positions = [np.array([0, 0, 0])]
    for i in range(len(theta_list)):
        T_i = dh_transform(theta_list[i], d[i], a[i], alpha[i])
        T = T @ T_i
        positions.append(T[:3, 3])
    return T, positions

def rotation_matrix_to_rpy(R):
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def compute_jacobian(thetas, a, d, alpha, tool_len, delta=1e-6):
    J = np.zeros((6, 6))
    T0, _ = forward_kinematics(thetas, a, d, alpha)
    p0 = T0[:3, 3] + T0[:3, 2] * tool_len
    rpy0 = rotation_matrix_to_rpy(T0[:3, :3])
    
    for i in range(6):
        th_new = thetas.copy()
        th_new[i] += delta
        Ti, _ = forward_kinematics(th_new, a, d, alpha)
        pi = Ti[:3, 3] + Ti[:3, 2] * tool_len
        rpyi = rotation_matrix_to_rpy(Ti[:3, :3])
        
        J[:3, i] = (pi - p0) / delta
        J[3:, i] = (rpyi - rpy0) / delta
    return J

def inverse_kinematics(target_pos, target_rpy, initial_guess, a, d, alpha, tool_len, max_iter=50, tol=1e-2):
    if initial_guess is None: thetas = np.zeros(6)
    else: thetas = initial_guess.copy()

    for i in range(max_iter):
        T, _ = forward_kinematics(thetas, a, d, alpha)
        p_curr = T[:3, 3] + T[:3, 2] * tool_len
        rpy_curr = rotation_matrix_to_rpy(T[:3, :3])
        
        pos_error = target_pos - p_curr
        if np.linalg.norm(pos_error) < tol:
            return thetas, True
            
        rpy_error = target_rpy - rpy_curr
        e = np.concatenate([pos_error, rpy_error])
        
        J = compute_jacobian(thetas, a, d, alpha, tool_len)
        lambda_val = 0.01
        dtheta = J.T @ np.linalg.inv(J @ J.T + (lambda_val**2)*np.eye(6)) @ e
        thetas += dtheta
        
    final_err = np.linalg.norm(target_pos - p_curr)
    return thetas, (final_err < tol)

# ==========================================
# 2. 커스텀 위젯
# ==========================================
class CleanSpinBox(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setDecimals(2)
        self.setRange(-360.0, 360.0)
    def textFromValue(self, value):
        text = f"{value:.2f}"
        if '.' in text: text = text.rstrip('0').rstrip('.')
        return text

# ==========================================
# 3. Matplotlib 캔버스 (크기 확장 및 여백 제거)
# ==========================================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100): # [수정] 기본 사이즈 확대
        # 한글 폰트 설정
        if platform.system() == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        elif platform.system() == 'Darwin':
            plt.rc('font', family='AppleGothic')
        plt.rcParams['axes.unicode_minus'] = False

        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#ffffff')
        
        # ★ [핵심] 여백 제거 (이 부분이 없으면 그래프가 작아 보임)
        # top, bottom, left, right 값을 조절하여 캔버스 끝까지 활용
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.05)

        # --------------------------------------------------------
        # 레이아웃 설정 (GridSpec)
        # --------------------------------------------------------
        # width_ratios=[1, 6, 1] -> 가운데(그래프)가 양옆 빈공간보다 6배 큼 (약 75% 너비 사용)
        self.gs = self.fig.add_gridspec(2, 3, 
                                        height_ratios=[4, 1],   # [수정] 로봇 화면 비율 더 늘림
                                        width_ratios=[1, 8, 1]) # [수정] 하단 그래프 너비 확장

        # [상단] 3D 로봇 화면 (모든 열 사용 -> 꽉 차게)
        self.ax = self.fig.add_subplot(self.gs[0, :], projection='3d')
        self.ax.set_facecolor('#ffffff')
        
        # [하단] 그래프 (가운데 열만 사용)
        self.ax_det = self.fig.add_subplot(self.gs[1, 1])
        self.init_det_graph()

        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # 3D 그래프 꽉 차게 보이도록 뷰 조정
        self.ax.set_box_aspect(None, zoom=0.95)

    def init_det_graph(self):
        self.ax_det.set_title("Singularity Determinant (det(J))", fontsize=10)
        self.ax_det.set_ylim(-0.15, 0.15) 
        self.ax_det.set_xlim(-1.0, 1.0)
        self.ax_det.set_yticks(np.arange(-0.1, 0.11, 0.1))
        self.ax_det.set_xticks([]) 
        self.ax_det.axhline(0, color='black', linewidth=1)
        self.ax_det.grid(True, linestyle=':', alpha=0.6)

        self.bar_container = self.ax_det.bar([0], [0], width=1.8, color='green')
        self.text_warning = self.ax_det.text(0, 0, "Singularity!", color='red', 
                                             fontsize=14, fontweight='bold', 
                                             ha='center', va='center', visible=False)

    def plot_robot(self, positions, tool_pos, tool_name, T_final):
        self.ax.clear()
        xs, ys, zs = zip(*positions)
        
        self.ax.plot(xs, ys, zs, 'o-', linewidth=3, markersize=6, color='#2980b9', label='UR5 Link')
        
        joint_names = ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            label = joint_names[i] if i < len(joint_names) else f"F{i}"
            self.ax.text(x, y, z + 0.05, label, fontsize=9, color='black', ha='center')

        current_tip = positions[-1]
        if tool_pos is not None:
            color = '#d35400' if "Welder" in tool_name else '#f39c12'
            self.ax.plot([xs[-1], tool_pos[0]], [ys[-1], tool_pos[1]], [zs[-1], tool_pos[2]], 
                         'o-', linewidth=4, color=color, label=f'Tool: {tool_name}')
            current_tip = tool_pos

        r_deg, p_deg, y_deg = 0, 0, 0
        if T_final is not None:
            rpy = rotation_matrix_to_rpy(T_final[:3, :3])
            r_deg, p_deg, y_deg = np.degrees(rpy)
        
        # 텍스트 박스 위치 및 스타일
        info_text = (f"Pos : ({current_tip[0]:.2f}, {current_tip[1]:.2f}, {current_tip[2]:.2f})\n"
                     f"Ori : ({r_deg:.2f}, {p_deg:.2f}, {y_deg:.2f})")
        
        self.ax.text2D(0.95, 0.95, info_text, transform=self.ax.transAxes, 
                       color='black', fontsize=11, fontweight='bold',
                       ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))

        if T_final is not None:
            length = 0.15
            start = current_tip
            ux = T_final[:3, 0] * length
            uy = T_final[:3, 1] * length
            uz = T_final[:3, 2] * length
            
            self.ax.quiver(start[0], start[1], start[2], ux[0], ux[1], ux[2], color='r', linewidth=1.5)
            self.ax.quiver(start[0], start[1], start[2], uy[0], uy[1], uy[2], color='g', linewidth=1.5)
            self.ax.quiver(start[0], start[1], start[2], uz[0], uz[1], uz[2], color='b', linewidth=1.5)

        limit = 0.9 # [수정] 축 범위 약간 축소하여 로봇이 더 커 보이게 함
        self.ax.set_xlim([-limit, limit]); self.ax.set_ylim([-limit, limit]); self.ax.set_zlim([0, 1.4])
        self.ax.set_xlabel('X-axis'); self.ax.set_ylabel('Y-axis'); self.ax.set_zlabel('Z-axis')
        
        self.ax.legend(loc='upper left', fontsize=9)
        self.ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    def plot_det(self, det_val):
        self.bar_container[0].set_height(det_val)
        threshold = 0.0001
        is_singular = abs(det_val) < threshold

        if is_singular:
            self.bar_container[0].set_color('red')
            text_y = 0.05 if det_val >= 0 else -0.05
            self.text_warning.set_position((0, text_y))
            self.text_warning.set_visible(True)
        else:
            self.bar_container[0].set_color('green')
            self.text_warning.set_visible(False)
        self.draw()

# ==========================================
# 4. 메인 윈도우
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.home_pose_deg = [0, -90, 90, -90, -90, 0]
        self.current_thetas = np.radians(self.home_pose_deg)
        self.is_updating = False
        self.tool_data = {"없음": 0.0, "Gripper (집게)": 0.15, "Welder (용접기)": 0.25}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("2주차 업무지시 프로그램")
        # [복구] 전체 창 크기 1400x800 유지
        self.setGeometry(100, 100, 1400, 800) 
        self.setStyleSheet("""
            QMainWindow { background-color: #ffffff; }
            QLabel { color: #333333; font-size: 14px; }
            QGroupBox { 
                border: 1px solid #cccccc; border-radius: 5px; margin-top: 10px; 
                font-weight: bold; color: #333333; background-color: #ffffff;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
            QSlider::groove:horizontal { height: 8px; background: #e0e0e0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #2980b9; width: 16px; margin: -4px 0; border-radius: 8px; }
            QComboBox { background-color: #f0f0f0; color: #333; padding: 5px; border: 1px solid #ccc; border-radius: 3px; }
            QFrame { background-color: #f9f9f9; border-right: 1px solid #ddd; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # [좌측 패널] (크기 고정 400)
        left_panel = QFrame()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)

        header = QLabel("로봇 제어")
        header.setFont(QFont("Arial", 20, QFont.Bold))
        header.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        header.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(header)

        self.btn_reset = QPushButton("카메라 시점 초기화")
        self.btn_reset.setStyleSheet("""
            QPushButton { background-color: #7f8c8d; color: white; font-weight: bold; padding: 8px; border-radius: 5px; }
            QPushButton:hover { background-color: #95a5a6; }
        """)
        self.btn_reset.clicked.connect(lambda: self.canvas.ax.view_init(elev=30, azim=-60) or self.canvas.draw())
        left_layout.addWidget(self.btn_reset)

        tool_group = QGroupBox("툴 선택")
        tool_layout = QVBoxLayout()
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(self.tool_data.keys())
        self.tool_combo.currentIndexChanged.connect(self.update_robot)
        tool_layout.addWidget(self.tool_combo)
        tool_group.setLayout(tool_layout)
        left_layout.addWidget(tool_group)

        # 관절 제어 슬라이더
        slider_group = QGroupBox("관절 제어")
        slider_layout = QVBoxLayout()
        self.spinboxes = []
        self.sliders = []

        for i in range(5, -1, -1):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Theta{i+1}"))
            
            sl = QSlider(Qt.Horizontal)
            sl.setRange(-1800, 1800)
            init_val = self.home_pose_deg[i]
            sl.setValue(int(init_val*10))
            sl.setFocusPolicy(Qt.NoFocus)
            
            sb = CleanSpinBox()
            sb.setFixedWidth(70)
            sb.setSingleStep(1.0)
            sb.setAlignment(Qt.AlignCenter)
            sb.setStyleSheet("background-color: #ffffff; color: #2980b9; font-weight: bold; border: 1px solid #ccc;")
            sb.setValue(init_val)

            sl.valueChanged.connect(lambda v, s=sb: s.setValue(v/10))
            sb.valueChanged.connect(lambda v, s=sl: s.setValue(int(v*10)))
            sb.valueChanged.connect(self.update_robot)
            
            self.sliders.insert(0, sl); self.spinboxes.insert(0, sb)
            row.addWidget(sl); row.addWidget(sb)
            slider_layout.addLayout(row)
        slider_group.setLayout(slider_layout)
        left_layout.addWidget(slider_group)

        # 목표 좌표 제어
        pos_group = QGroupBox("목표 좌표 제어")
        pos_layout = QVBoxLayout()
        self.pos_spinboxes = []
        self.pos_sliders = []

        for axis in ['X', 'Y', 'Z']:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Target {axis}"))
            sl = QSlider(Qt.Horizontal); sl.setRange(-1000, 1000)
            sb = CleanSpinBox(); sb.setFixedWidth(70); sb.setSingleStep(0.01); sb.setRange(-1, 1)

            sl.valueChanged.connect(lambda v, s=sb: s.setValue(v/1000))
            sb.valueChanged.connect(lambda v, s=sl: s.setValue(int(v*1000)))
            sb.valueChanged.connect(self.update_pos)
            
            self.pos_sliders.append(sl); self.pos_spinboxes.append(sb)
            row.addWidget(sl); row.addWidget(sb)
            pos_layout.addLayout(row)
        pos_group.setLayout(pos_layout)
        left_layout.addWidget(pos_group)

        # 좌표 표시 라벨
        self.pos_info = QLabel("현재 좌표: (0.000, 0.000, 0.000)")
        self.pos_info.setStyleSheet("""
            background-color: #2980b9; color: white; font-weight: bold; 
            font-size: 14px; padding: 10px; border-radius: 5px; margin-top: 10px;
        """)
        self.pos_info.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.pos_info)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # [우측] 캔버스
        self.canvas = MplCanvas(self, width=8, height=8) 
        main_layout.addWidget(self.canvas, stretch=1)

        self.update_robot()

    def update_robot(self):
        if self.is_updating: return
        self.is_updating = True

        # 1. 값 읽기 & FK
        degs = [sb.value() for sb in self.spinboxes]
        self.current_thetas = np.radians(degs[::-1])
        T, pos = forward_kinematics(self.current_thetas, a, d, alpha)
        
        tool_name = self.tool_combo.currentText()
        tool_len = self.tool_data[tool_name]
        
        tool_tip = T[:3, 3] + T[:3, 2] * tool_len
        T_final = T.copy()
        T_final[:3, 3] = tool_tip

        # 2. 좌표 UI 업데이트
        self.pos_info.setText(f"현재 좌표: ({tool_tip[0]:.3f}, {tool_tip[1]:.3f}, {tool_tip[2]:.3f})")
        
        for i, val in enumerate(tool_tip):
            self.pos_spinboxes[i].blockSignals(True)
            self.pos_spinboxes[i].setValue(val)
            self.pos_spinboxes[i].blockSignals(False)
            self.pos_sliders[i].blockSignals(True)
            self.pos_sliders[i].setValue(int(val*1000))
            self.pos_sliders[i].blockSignals(False)

        # 3. 그래프용 Det 계산
        J = compute_jacobian(self.current_thetas, a, d, alpha, tool_len)
        det_val = np.linalg.det(J) 
        
        # 4. 캔버스 그리기
        self.canvas.plot_robot(pos, tool_tip, tool_name, T_final)
        self.canvas.plot_det(det_val)

        self.is_updating = False

    def update_pos(self):
        if self.is_updating: return
        self.is_updating = True
        try:
            target = np.array([sb.value() for sb in self.pos_spinboxes])
            tool_len = self.tool_data[self.tool_combo.currentText()]
            
            new_q, success = inverse_kinematics(target, np.zeros(3), self.current_thetas, a, d, alpha, tool_len)
            
            if not np.isnan(new_q).any():
                self.current_thetas = new_q
                degs = np.degrees(new_q)[::-1]
                for i, v in enumerate(degs):
                    v = (v + 180) % 360 - 180
                    self.spinboxes[i].blockSignals(True)
                    self.spinboxes[i].setValue(v)
                    self.sliders[i].setValue(int(v*10))
                    self.spinboxes[i].blockSignals(False)
                
                self.is_updating = False
                self.update_robot()
        except: pass
        self.is_updating = False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())