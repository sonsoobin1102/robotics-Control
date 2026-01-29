# 🤖 Robotics Control & Sensor Fusion Simulation Project

## 📖 Project Overview
본 프로젝트는 **4주간 진행된 로봇 공학 및 자율주행 알고리즘 개발 프로젝트**입니다.
기구학적 메커니즘 설계의 기초가 되는 **자유도(DOF) 계산**부터, **6축 로봇 팔의 실시간 제어 및 특이점 감지**, 그리고 **Kalman Filter와 LSTM을 활용한 위치 추정 비교 분석**까지 로봇 소프트웨어의 핵심 기술을 단계별로 구현하고 검증하였습니다.

---

## 📅 Project Roadmap

### **Week 1. 자유도 자동 계산 유틸리티 (DOF Calculator)**
복잡한 기구학적 메커니즘 설계 시 필수적인 자유도(DOF) 계산을 자동화하여 설계 효율성을 높인 유틸리티입니다.

* **주요 기능**:
    * **Kutzbach Formula 기반 계산**: 링크(Link)와 조인트(Joint)의 개수, 각 관절의 파라미터를 입력받아 시스템 전체의 자유도를 산출합니다.
    * **설계 검증 자동화**: 수작업 계산 오류를 방지하고 메커니즘 설계 초기 단계의 검증 속도를 향상시켰습니다.

### **Week 2~3. 6자유도 로봇 실시간 제어 및 특이점 감지 시뮬레이터**
6-DOF 로봇 팔의 운동학(Kinematics)을 해석하고, 제어 불능 상태를 사전에 감지하는 통합 시뮬레이터입니다.

* **주요 기능**:
    * **정기구학 (Forward Kinematics)**: DH Parameter를 기반으로 각 관절 각도에 따른 로봇 끝단(End-effector)의 좌표를 시각화합니다.
    * **역기구학 (Inverse Kinematics)**: 목표 좌표 입력 시, 해당 위치로 이동하기 위한 각 관절의 회전각을 역산합니다.
    * **특이점 (Singularity) 감지**: 자코비안(Jacobian) 행렬을 실시간 분석하여 값이 0에 수렴(제어 불능 위험)할 경우 경고를 표시합니다.

### **Week 4. GPS/IMU 센서 퓨전 & LSTM 비교 분석 시뮬레이터**
불확실한 센서 데이터를 융합하여 위치 정확도를 높이는 고전적 알고리즘과 최신 딥러닝 기법의 성능을 비교 분석합니다.

* **주요 기능**:
    * **Kalman Filter 센서 퓨전**:
        * *Predict*: IMU 가속도 데이터를 적분하여 물리적 이동 경로 추정.
        * *Update*: GPS 관측 데이터를 반영하여 누적 오차 보정.
        * *Interaction*: 파라미터 조절을 통해 필터 민감도 변화를 실시간으로 확인 가능.
    * **LSTM 기반 경로 예측**:
        * 시계열 데이터 처리에 특화된 LSTM 모델 적용.
        * 전체 경로를 학습(Train) 구간과 예측(Test) 구간으로 분리하여 AI 모델의 일반화 성능을 시각적으로 검증.

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
* **자유도 (DOF)**: 로봇이나 기구가 공간 안에서 독립적으로 움직일 수 있는 방향의 개수.
* **Kutzbach Formula**: 기구의 링크와 조인트 수로 자유도를 산출하는 공식.
* **DH Parameter**: 로봇의 관절과 링크 사이의 좌표 변환을 수학적으로 정의하기 위한 4가지 표준 파라미터.
* **Forward / Inverse Kinematics**: 관절 각도로 위치를 구하거나(Forward), 원하는 위치를 위해 각도를 역산(Inverse)하는 기법.
* **Jacobian Matrix**: 관절의 회전 속도가 로봇 끝단의 이동 속도에 미치는 영향을 나타내는 미분 행렬.
* **Singularity (특이점)**: 자코비안 행렬식이 0이 되어 로봇이 특정 방향으로 움직일 수 없는 상태.

### **State Estimation & AI**
* **Kalman Filter**: 잡음이 포함된 측정값(GPS)과 예측값(IMU)을 확률적 가중치(Kalman Gain)로 결합하여 참값을 추정하는 알고리즘.
* **Sensor Fusion**: 서로 다른 특성의 센서를 결합하여 단일 센서보다 높은 신뢰성을 얻는 기술.
* **LSTM**: 긴 시간의 데이터 패턴을 기억할 수 있어 시계열 데이터(로봇 경로 등) 예측에 적합한 딥러닝 모델.

---

## 🚀 How to run

### Prerequisites
Python 3.x 환경이 필요하며, 아래 라이브러리를 설치해야 합니다.

```bash
pip install numpy matplotlib PyQt5 torch pandas scikit-learn
