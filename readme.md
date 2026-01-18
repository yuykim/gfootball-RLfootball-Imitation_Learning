# ⚽ RL-Soccer: Imitation Learning (Behavior Cloning)

이 프로젝트는 **Google Research Football (G-Football)** 환경에서
**인간 전문가의 플레이 데이터를 수집**하고, 이를 바탕으로
**이미테이션 러닝(Imitation Learning, Behavior Cloning)** 기반 축구 정책을 학습·비교하는 연구 프로젝트입니다.

본 프로젝트의 핵심 목표는 **성공(득점)과 실패(무득점) 데이터의 학습 효과 차이**를 분석하는 것입니다.

---

## 🎮 실제 플레이 예시

![Gameplay Demo](../doc/gameplay.gif)

* **사람의 전문가 플레이 영상**
  [https://drive.google.com/file/d/1nhdAwGLTyb9hywAcStkV3nlO3njdwcVK/view](https://drive.google.com/file/d/1nhdAwGLTyb9hywAcStkV3nlO3njdwcVK/view)
  *(약 6,000 프레임)*

* **학습된 모델 플레이 영상**
  [https://drive.google.com/file/d/1fYemqlV27fp6omK9S_TlL1f8KAjSiQB-/view](https://drive.google.com/file/d/1fYemqlV27fp6omK9S_TlL1f8KAjSiQB-/view)

---

## 📂 프로젝트 구조 (Project Structure)

```text
Imiation_Learning/
    ├── 01_collect_data.py          # 전문가 플레이 데이터 수집
    ├── 02A_train_goal.py         # 골 성공 데이터만 학습
    ├── 02B_train_nogoal.py       # 골 실패 데이터만 학습
    ├── 02C_train_mix.py      # 골 70% + 노골 30% 혼합 학습
    ├── 03_test.py               # 학습된 모델 성능 테스트
    │
    ├── expert_dataset/
    │   ├── goal/                # 득점 성공 에피소드 데이터
    │   └── no_goal/             # 득점 실패 에피소드 데이터
    │
    ├── il_model_goal_only.pth
    ├── il_model_nogoal_only.pth
    └── il_model_mix70_goal30_nogoal.pth
```

---

## 🔑 주요 특징 (Key Features)

### 1. 하이브리드 입력 시스템 (Dual Input Support)

* **게임패드(Gamepad)** 및 **키보드(Keyboard)** 입력을 모두 지원
* 모든 입력은 G-Football의 `CoreAction` 기반 **이산 액션 인덱스**로 통일
* 입력 장치에 무관한 **일관된 전문가 데이터 수집**

---

### 2. 에피소드 단위 데이터 분리 (Goal / No-Goal Split)

* 각 에피소드를 종료 시점 기준으로 자동 분류

  * **Goal Episode**: 득점 성공
  * **No-Goal Episode**: 무득점 종료
* 데이터는 다음과 같이 분리 저장됨:

  ```text
  expert_dataset/
    ├── goal/
    └── no_goal/
  ```

👉 이를 통해 **성공/실패 데이터의 학습 효과를 정량적으로 비교 가능**

---

### 3. 3가지 Behavior Cloning 학습 전략

| 학습 전략         | 설명                        |
| ------------- | ------------------------- |
| Goal-only     | 득점 성공 데이터만 사용             |
| No-goal-only  | 득점 실패 데이터만 사용             |
| Mixed (70/30) | Goal 70% + No-goal 30% 혼합 |

👉 **“성공 경험만 학습하는 것이 항상 좋은가?”**라는 질문을 실험적으로 검증

---

### 4. 동적 모델 아키텍처 (Dynamic Architecture)

* 데이터셋에서 등장한 **액션 수를 자동 추론**
* 최대 **32개 이산 액션**에 대해 출력 레이어 자동 구성
* Cross Entropy Loss 기반 다중 클래스 분류

---

## 🛠️ 사용 방법 (Usage)

### Step 1. 전문가 데이터 수집

```bash
python 01_play_and_collect.py
```

* 권장 FPS: **18**
* 에피소드 종료 시 자동으로 `goal / no_goal` 폴더에 분류 저장

---

### Step 2. 모델 학습

#### (1) 골 성공 데이터만 학습

```bash
python 02A_train_goal.py
```

#### (2) 골 실패 데이터만 학습

```bash
python 02B_train_nogoal.py
```

#### (3) 골 70% + 노골 30% 혼합 학습

```bash
# 학습 비율은 수정 가능
python 02C_train_mix.py
```

**공통 학습 설정**

* Loss Function: Cross Entropy Loss
* Optimizer: Adam
* Epoch: 500
* Batch Size: 64

---

### Step 3. 모델 테스트

```bash
python 03_test.py
```

* 학습된 모델의 실제 경기 플레이 확인
* Softmax Sampling을 적용하여 **행동 다양성 유지**

---

## 📊 기술 데이터 규격 (Technical Specification)

### Observation Space

* **Representation**: `simple115v2`
* **Dimension**: 115

  * 공 위치·속도
  * 아군/적군 선수 22명의 위치 및 속도

### Action Space

* **Action Set**: `Full`
* **Number of Actions**: 최대 32
* **Type**: Discrete

---

