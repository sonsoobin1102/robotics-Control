본 프로그램은 **자유도 계산**부터, **6축 로봇 팔의 실시간 제어 및 특이점 감지**, 그리고 **Kalman Filter와 LSTM을 활용한 위치 추정 비교 분석**까지 로봇 소프트웨어의 핵심 기술을 단계별로 구현하고 검증하였습니다.

### **Week 1. 자유도 자동 계산 유틸리티 (DOF Calculator)**
복잡한 기구학적 메커니즘 설계 시 필수적인 자유도(DOF, Degrees of Freedom) 계산을 자동화하여 설계 효율성을 높인 프로그램입니다.

* **주요 기능**:
    * **Kutzbach Formula 기반 계산**: 링크(Link)와 조인트(Joint)의 개수, 각 관절의 파라미터를 입력받아 시스템 전체의 자유도를 산출.
    * 설계 초기 단계에서의 메커니즘 검증 자동화.

### **Week 2~3. 6자유도 로봇 실시간 제어 및 특이점 감지 시뮬레이터**
6-DOF 로봇 팔의 운동학(Kinematics)을 해석하고, 제어 불능 상태를 사전에 감지하는 통합 시뮬레이터입니다.

* **주요 기능**:
    * **정기구학 (Forward Kinematics)**: DH Parameter를 기반으로 각 관절 각도에 따른 로봇 끝단(End-effector)의 좌표 시각화.
    * **역기구학 (Inverse Kinematics)**: 목표 좌표 입력 시, 해당 위치 도달을 위한 각 관절의 회전각 역산 및 모션 시각화.
    * **특이점 (Singularity) 실시간 감지**: 자코비안(Jacobian) 행렬을 실시간 분석하여 행렬식이 0에 수렴할 경우(제어 불능 위험) 경고 메시지 출력.

### **Week 4. GPS/IMU 센서 퓨전 & LSTM 비교 분석 시뮬레이터**
불확실한 센서 데이터를 융합하여 위치 정확도를 높이는 고전적 알고리즘과 최신 딥러닝 기법의 성능을 비교 분석합니다.

* **주요 기능**:
    * **Kalman Filter 센서 퓨전**:
        * *Predict*: IMU 가속도 데이터를 적분하여 물리적 이동 경로 추정.
        * *Update*: GPS 관측 데이터를 반영하여 누적 오차 보정.
        * *Interaction*: 파라미터(Q, R 등) 조절을 통해 필터 민감도 변화 실시간 확인.
    * **LSTM 기반 경로 예측**:
        * 시계열 데이터 처리에 특화된 LSTM 모델 적용.
        * 전체 경로를 학습(Train) 구간과 예측(Test) 구간으로 분리하여 AI 모델의 일반화 성능 시각적 검증.

---

## 🛠 Tech Stack

| Category | Technology |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3.x-blue) |
| **Robotics & Math** | Numpy, DH Parameters, Jacobian Matrix, Kutzbach Formula |
| **Simulation & GUI** | PyQt5, Matplotlib, Slider Widgets |
| **AI & Data** | PyTorch, Scikit-learn, Pandas, LSTM |

---

## 🧮 Key Concepts & Theory

본 프로젝트에서 활용된 핵심 수학적/공학적 개념에 대한 정의입니다.

### **Robotics (Kinematics & Control)**
* **자유도 (DOF, Degrees of Freedom)**: 로봇이나 기구가 3차원 공간 안에서 독립적으로 움직일 수 있는 방향(축)의 개수입니다.
* **Kutzbach Formula**: 기구의 링크(Link) 수와 조인트(Joint) 수를 대입하여 기구 전체의 자유도를 산출하는 공식입니다.
* **DH Parameter (Denavit-Hartenberg)**: 로봇의 인접한 관절과 링크 사이의 좌표 변환을 수학적으로 정의하기 위한 4가지 표준 파라미터($a, \alpha, d, \theta$)입니다.
* **Forward Kinematics (정기구학)**: 각 관절의 회전 각도($\theta$)가 주어졌을 때, 로봇 끝단의 위치와 자세를 계산하는 과정입니다.
* **Inverse Kinematics (역기구학)**: 로봇 끝단이 도달해야 할 목표 위치가 주어졌을 때, 이를 달성하기 위해 필요한 각 관절의 각도를 역으로 계산하는 과정입니다.
* **Jacobian Matrix**: 관절 공간의 속도($\dot{q}$)와 작업 공간(로봇 끝단)의 속도($\dot{x}$) 사이의 관계를 나타내는 미분 행렬입니다.
* **Singularity (특이점)**: 자코비안 행렬의 행렬식이 0이 되어 역행렬이 존재하지 않는 상태입니다. 이 상태에서는 로봇이 특정 방향으로 움직일 수 없거나 제어가 불가능해집니다.

### **State Estimation & AI**
* **Sensor Fusion**: 서로 다른 특성을 가진 센서들(예: 절대 위치를 주는 GPS vs 상대 움직임을 측정하는 IMU)을 결합하여, 단일 센서보다 더 정확하고 신뢰성 높은 데이터를 얻는 기술입니다.
* **Kalman Filter**: 잡음(Noise)이 포함된 측정값(GPS)과 시스템의 예측값(IMU)을 확률적 가중치인 **칼만 이득(Kalman Gain)**으로 결합하여, 실제 값에 가장 가까운 상태를 추정하는 재귀적 필터링 알고리즘입니다.
* **RNN (Recurrent Neural Network)**: 이전 단계의 출력이 다음 단계의 입력으로 다시 들어가는 순환 구조를 가져, 데이터의 순서 정보가 중요한 시계열 처리에 적합한 신경망입니다.
* **LSTM (Long Short-Term Memory)**: 기존 RNN의 장기 의존성(Vanishing Gradient) 문제를 해결한 모델로, 긴 시간의 데이터 패턴을 효과적으로 기억하여 로봇 이동 경로와 같은 시계열 예측에 탁월합니다.
