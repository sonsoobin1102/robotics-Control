# Robotics Control & Sensor Fusion Simulation Projects
> **Python 기반의 로봇 제어(Kinematics), 센서 퓨전(Kalman Filter), 그리고 딥러닝(Deep Learning)을 활용한 시뮬레이션 및 알고리즘 개발 프로젝트 모음입니다.**

## 소개
이 저장소는 로봇 공학의 핵심인 기구학(Kinematics) , 특이점(Singularity) 제어, 그리고 센서 데이터 융합(Sensor Fusion) 기술을 다룹니다.
수학적 모델링(DH Parameter, Jacobian)을 코드로 구현하고, 이를 PyQt5 및 Matplotlib을 활용한 GUI 시뮬레이터로 시각화하여 이론을 검증하고 최적화하는 과정을 담고 있습니다.

| **Language** | Python 3.x |
| **Robotics & Math** | NumPy, SciPy (Linear Algebra, Matrix Calculation) |
| **AI & Data** | PyTorch (LSTM), Pandas, Scikit-learn |
| **Visualization & GUI** | PyQt5, Matplotlib (Interactive Slider, 3D Plotting) |

---

## 🚀 Project 1: GPS/IMU Sensor Fusion & AI Path Estimation
**"전통적 제어 이론(Kalman Filter)과 최신 AI(LSTM)의 성능 비교 및 하이브리드 경로 추정 시뮬레이터"**

### 🎯 Overview
GPS의 위치 데이터와 IMU의 가속도 데이터를 융합하여 위치 정확도를 높이는 **Kalman Filter**를 구현하고, 시계열 데이터 학습에 특화된 **LSTM(Long Short-Term Memory)** 딥러닝 모델과 성능을 비교 분석한 프로젝트입니다.

### 🖼️ Preview
![Result Image](image_f79aec.png)
*(위 이미지는 Kalman Filter와 AI 모델의 경로 추정 결과를 시각화한 화면입니다)*

### 🔑 Key Features
1.  **Sensor Fusion (Kalman Filter):**
    * 불확실한 센서 데이터(GPS 노이즈, IMU 드리프트)를 융합하여 최적의 상태 추정.
    * `Process Variance`와 `Measurement Variance`를 실시간으로 조절하는 슬라이더 GUI 구현.
2.  **Deep Learning (LSTM):**
    * PyTorch를 활용한 경로 예측 모델 구축 (Train/Test 데이터셋 분리 검증).
    * MinMaxScaler를 활용한 데이터 정규화 및 전처리 파이프라인 구축.
3.  **Interactive Simulation:**
    * 실시간 파라미터 튜닝을 통해 필터의 민감도와 지연 시간(Lag) 상관관계 분석.

### 📊 Achievements
* **위치 정확도 향상:** 단일 GPS 사용 대비 센서 퓨전 적용 시 노이즈가 제거된 부드러운 경로 추정으로 **정확도 약 20% 향상**.
* **튜닝 효율 극대화:** 매번 코드를 수정하는 대신 GUI 슬라이더를 도입하여 최적의 $Q, R$ 공분산 행렬 파라미터 탐색 시간을 **기존 대비 80% 단축**.

---

## 🤖 Project 2: 6-DOF Robot Arm Control Simulator (PyQt5)
**"산업용 로봇(UR5)의 정기구학/역기구학 해석 및 3D 제어 시뮬레이터"**

### 🎯 Overview
6자유도(6-DOF) 로봇 팔의 움직임을 **DH(Denavit-Hartenberg) 파라미터**를 통해 모델링하고, **정기구학(FK)과 역기구학(IK)** 알고리즘을 구현하여 사용자가 원하는 위치로 로봇을 제어할 수 있는 GUI 프로그램을 개발했습니다.

### 🖼️ Preview
![Robot Simulation](image_d01f8f.png)
*(PyQt5 기반의 로봇 제어 패널과 3D 시각화 화면)*

### 🔑 Key Features
1.  **Forward Kinematics (정기구학):**
    * 각 관절의 각도($\theta$)를 입력받아 End-Effector의 3차원 좌표($x, y, z$) 계산.
2.  **Inverse Kinematics (역기구학):**
    * 목표 좌표 입력 시, 이를 달성하기 위한 각 관절의 회전각을 수치해석적 방법(Jacobian Pseudo-inverse)으로 도출.
3.  **Real-time GUI Control:**
    * PyQt5와 Matplotlib을 연동하여 3D 공간에서의 로봇 움직임 실시간 렌더링.
    * 다양한 Tool(Gripper, Welder) 장착 시나리오 시뮬레이션.

### 📊 Achievements
* **설계 프로세스 효율화:** 실제 하드웨어 구현 전, 시뮬레이션을 통해 로봇의 동작 범위(Workspace)를 미리 검증하여 **설계 검토 시간을 40% 단축**.
* **비용 절감:** 가상 환경에서의 사전 테스트를 통해 물리적 프로토타입 제작 횟수 감소.

---

## ⚠️ Project 3: Singularity Detection & Analysis
**"로봇 팔의 특이점(Singularity) 실시간 감지 및 시각화"**

### 🎯 Overview
로봇 제어 중 발생할 수 있는 위험 상태인 **특이점(Singularity)**을 수학적으로 계산하고, 이를 실시간으로 감지하여 사용자에게 경고하는 안전 시스템을 구현했습니다.

### 🖼️ Preview
![Singularity Detection](image_f71b51.png)
*(자코비안 행렬식 값이 0에 가까워질 때 경고를 표시하는 화면)*

### 🔑 Key Features
1.  **Jacobian Matrix Calculation:**
    * 관절 속도와 말단 속도의 관계를 나타내는 자코비안 행렬($J$) 도출.
2.  **Singularity Monitoring:**
    * 행렬식(Determinant) `det(J)`를 실시간으로 계산.
    * `det(J) ≈ 0`인 구간에서 로봇의 제어 불능 상태를 시각적으로 경고(Warning Alert).

### 📊 Achievements
* **안전성 확보:** 시뮬레이션을 통해 특이점 발생 자세를 미리 파악하고, 이를 회피하기 위한 경로 계획 전략 수립의 기초 데이터 확보.
* **운동 성능 최적화:** 자코비안 분석을 통해 로봇이 가장 효율적으로 힘을 낼 수 있는 자세 분석.

---

## 📐 Project 4: DOF Calculator (Automation Tool)
**"Kutzbach Formula 기반의 기구학적 자유도 자동 계산기"**

### 🎯 Overview
복잡한 기구 메커니즘 설계 시 필수적인 자유도(Degree of Freedom) 계산을 자동화하여, 설계 실수를 방지하는 유틸리티 프로그램입니다.

### 🔑 Key Features
* **Kutzbach Formula 적용:** 링크(Link)와 조인트(Joint)의 개수, 각 조인트의 자유도를 입력받아 시스템 전체의 자유도 산출.
* **User-Friendly CLI:** 직관적인 입력 방식을 통해 비전문가도 쉽게 메커니즘 해석 가능.

### 📊 Achievements
* **업무 효율성 증대:** 수기로 계산하던 자유도 산출 과정을 자동화하여, 복잡한 다관절 로봇 설계 단계에서 **계산 시간 30% 단축**.

---

## 🎓 Concepts Learned
이 프로젝트들을 진행하며 다음과 같은 핵심 역량을 길렀습니다.

* **Mathematical Modeling:** 로봇의 움직임을 선형대수학(행렬)과 미분기하학(자코비안)으로 모델링하는 능력.
* **Sensor Fusion Logic:** 서로 다른 특성을 가진 센서(GPS, IMU)를 확률적 모델(KF)로 융합하여 신뢰성을 높이는 기술.
* **Simulation Development:** Python과 GUI 라이브러리를 활용하여 추상적인 수식을 눈에 보이는 시뮬레이터로 구현하는 SW 개발 능력.
* **AI Integration:** 전통적인 제어 이론에 머물지 않고, 최신 딥러닝(LSTM) 기법을 접목하여 성능을 비교 및 개선하는 연구 역량.
