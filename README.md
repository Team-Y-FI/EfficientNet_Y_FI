## 0) 실습 요약
- **데이터셋**: TFDS `oxford_flowers102` (`train/validation/test`, `as_supervised=True`)
- **입력 크기**: `224×224`
- **모델**: `tf.keras.applications.EfficientNetB0` (ImageNet pretrained, `include_top=False`)
- **태스크**: 이미지 **분류(Classification)** (102 classes)
- **학습**: 2-stage
  - Stage 1: backbone freeze → head만 학습 (Adam lr=1e-3)
  - Stage 2: backbone 일부 fine-tune (BatchNorm freeze, 뒤 30%만 학습, Adam lr=1e-5)
- **추가 분석**:
  - train/val 곡선 + 일반화 갭 확인
  - test Top-1 / Top-5 / Macro-F1
  - confusion matrix 기반 “가장 많이 헷갈린 클래스 쌍” 출력
  - 파라미터 수/가중치 용량 추정, (선택) FLOPs 추정, latency(ms/image) 측정
- **경량 모델 실험(variant 비교)**:
  - EfficientNet **B0/B1/B2**를 같은 파이프라인에서 3 epochs(head-only)로 빠르게 비교
  - val_acc vs params / val_acc vs latency 시각화

---

# 1. 모델 조사

## 1.1 모델 등장 배경
CNN 성능을 올리는 가장 흔한 방법은 모델을 키우는 것(스케일업)이었는데,  
문제는 **depth/width/resolution 중 뭘 어떻게 키워야 “효율적으로” 성능이 오르는지**가 명확하지 않았다는 점이다.

EfficientNet은 이 문제를 **Compound Scaling(Depth/Width/Resolution을 균형 있게 함께 키우기)**로 정리하고,  
그 규칙이 잘 먹히도록 **효율 좋은 baseline(B0)**부터 제시한 모델 패밀리다.

> 노트북에서는 B0를 메인 모델로 학습하고, 마지막에 B0/B1/B2를 “경량 비교”로 추가 실험한다.

---

## 1.2 핵심 구조와 동작 원리

### (A) Backbone 관점: EfficientNet을 “특징 추출기”로 사용
노트북의 기본 사용 방식은 아래 형태다.

- EfficientNet(Backbone) = 이미지에서 특징(feature) 뽑는 몸통
- Head = `GlobalAveragePooling2D → Dropout(0.3) → Dense(102, softmax)`

즉, EfficientNet을 **backbone(특징 추출기)**로 쓰고, 데이터셋(Flowers102)에 맞는 head만 새로 붙여 분류를 수행한다.

### (B) Compound Scaling 개념(핵심 메시지)
EfficientNet이 유명한 이유는 단순히 “분류 모델 하나”라서가 아니라,  
모델을 키울 때 **Depth/Width/Resolution을 같이 키우는 규칙(Compound Scaling)**을 제안했기 때문이다.

- Depth ↑ : 더 깊은 표현(계층적 특징)
- Width ↑ : 더 넓은 채널(풍부한 특징 용량)
- Resolution ↑ : 더 많은 입력 디테일

노트북의 “경량 모델 실험(variant 비교)”는 이 철학을 현실적으로 체감하게 해주는 파트로,  
같은 데이터/파이프라인에서 B0→B1→B2로 갈수록 **정확도·파라미터·속도**가 어떻게 바뀌는지 확인한다.

---

## 1.3 기존 모델 대비 장점과 한계 (실습 기준)
### 장점(실습에서 체감되는 포인트)
- ImageNet 사전학습 기반으로 **전이학습/파인튜닝이 안정적으로 동작**
- 작은 모델(B0)로도 시작하기 좋아서 **실습 환경(Colab)에서 현실적인 선택지**
- variant(B0/B1/B2)를 바꿔가며 **성능-비용 트레이드오프**를 만들기 쉬움

### 한계/주의점(노트북에서도 고려한 부분)
- 데이터가 크지 않으면 fine-tuning을 과하게 풀 때 불안정해질 수 있어  
  → 노트북에서는 **BatchNorm을 고정**하고, backbone의 **뒤 30%만** 풀어 학습한다.
- FLOPs는 이론 지표라 실제 속도는 환경 영향을 받음  
  → 그래서 노트북은 **latency(ms/image) 실측**도 같이 본다.

---

# 2. 데이터셋 조사

## 2.1 선택 데이터셋
- **TFDS**: `oxford_flowers102`
- **형태**: `as_supervised=True` → `(image, label)` 형태로 로드
- **클래스 수**: `ds_info.features["label"].num_classes` (노트북에서 출력)

## 2.2 이미지 크기/전처리(노트북 설정 그대로)
- 입력 크기 통일: `IMG_SIZE = 224`
- EfficientNet 전용 전처리: `tf.keras.applications.efficientnet.preprocess_input`
- 데이터 증강(Train 전용):
  - `RandomFlip("horizontal")`
  - `RandomRotation(0.05)`
  - `RandomZoom(0.1)`

## 2.3 Data pipeline
- `train_ds`: shuffle(2048) → map(train_map_fn) → batch(32) → prefetch
- `val_ds/test_ds`: map(test_map_fn) → batch(32) → prefetch

---

# 3. 실험 설계 및 수행

## 3.1 가능한 태스크 예시(정리)
EfficientNet은 원래 분류 모델이지만, backbone으로도 많이 쓰인다.

- **이미지 분류**: EfficientNet + classification head
- **객체 탐지**: EfficientNet backbone + detection head (별도 구성 필요)
- **이미지 분할**: EfficientNet encoder + decoder (별도 구성 필요)
- **전이학습/파인튜닝**: 위 태스크들에 공통으로 적용 가능한 학습 전략

> 단, `EfficientNet_.ipynb`에서 실제로 수행한 태스크는 **이미지 분류** 1개다.

---

## 3.2 실제 수행한 실험 1: EfficientNet-B0 (Transfer Learning → Fine-tuning)

### 모델 구성
```text
EfficientNetB0 (include_top=False, weights="imagenet")
→ GlobalAveragePooling2D
→ Dropout(0.3)
→ Dense(NUM_CLASSES=102, softmax)
```

### Stage 1: 전이학습(Feature extractor)
- `base.trainable = False`
- optimizer: Adam(lr=1e-3)
- epochs: 최대 10
- callbacks:
  - EarlyStopping(patience=3, restore_best_weights=True)
  - ReduceLROnPlateau(patience=2, factor=0.2)

### Stage 2: 파인튜닝(Fine-tuning)
- `base.trainable = True`
- BatchNorm 레이어는 고정(`layer.trainable=False`)
- backbone의 앞 70%는 고정, 뒤 30%만 학습
- optimizer: Adam(lr=1e-5)
- epochs: 최대 10 (callbacks 동일)

### 평가/출력(노트북에서 수행)
- `model.evaluate(test_ds)`로 test accuracy 출력
- `history1 + history2`를 이어붙여 train/val accuracy/loss 곡선 시각화
- 일반화 갭(train - val) 출력
- train/val/test 각각 evaluate 해서 과적합 경향 체크

---

## 3.3 실제 수행한 실험 2: 상세 성능지표 + 오류 분석

노트북은 test에서 아래를 추가로 계산한다.
- **Top-1 accuracy**
- **Top-5 accuracy** (`TopKCategoricalAccuracy(k=5)`)
- **Macro-F1**
  - sklearn이 있으면 `sklearn.metrics.f1_score(average="macro")`
  - 없으면 numpy fallback으로 직접 계산

또한 confusion matrix를 만든 뒤,
- 102-class 전체 heatmap 대신
- **가장 많이 헷갈린 label pair(true→pred) Top-N**을 출력한다.

---

## 3.4 실제 수행한 실험 3: 파라미터/속도 효율 측정

노트북에서 수행한 효율 측정은 아래다.
- `model.count_params()`로 **총 파라미터 수**
- `base.count_params()`로 **backbone 파라미터 수**
- float32 가정 시 가중치 크기(대략 MB) 추정
- (선택) try/except로 FLOPs 근사 추정
- latency(ms/image):
  - test batch를 대상으로 warm-up 후 반복 실행
  - `ms/image`로 출력

---

## 3.5 경량 모델 실험: EfficientNet variant 비교(B0/B1/B2)
노트북의 “경량 모델 실험”은 다음처럼 진행한다.

### 목적
- 같은 데이터/전처리/학습 설정에서
- EfficientNet variant(B0/B1/B2)에 따라
- **정확도 ↔ 파라미터 ↔ 속도**가 어떻게 달라지는지 빠르게 확인

### 설정(노트북 그대로)
- 입력/파이프라인: `IMG_SIZE=224`, 동일한 `train_ds/val_ds/test_ds`
- 모델: EfficientNet B0/B1/B2 (ImageNet pretrained, backbone freeze)
- 학습: **epochs=3 (head-only)** 빠른 비교
- 측정:
  - `val_acc`, `test_acc`
  - `params`
  - `ms_per_image`(latency)
  - `acc_per_Mparams = val_acc / (params/1e6)`

### 시각화
- **Val Accuracy vs Params**
- **Val Accuracy vs Latency(ms/image)**

---

# 4. 결과 분석

---

## 4.1 B0 2-stage 학습 결과 (Transfer Learning → Fine-tuning)

### (1) Stage 1: backbone freeze + head 학습 (10 epochs)
- 최종(10 epoch) 기준  
  - **Train acc**: 0.9497  
  - **Val acc**: 0.8304  
  - **Val loss**: 0.9902  

> Stage 1만으로도 val acc가 0.83까지 올라가서, pretrained backbone이 “특징 추출기”로 충분히 먹힌다는 걸 확인.

### (2) Stage 2: 부분 파인튜닝 (10 epochs)
노트북 설정:
- backbone(`EfficientNetB0`)을 `trainable=True`로 바꾸고,
- **BatchNorm 레이어는 freeze**(안정성 목적),
- backbone 레이어의 **앞 70%는 다시 freeze**, **뒤 30%만 학습**하도록 설정.

최종(10 epoch) 기준:
- **Final train acc**: 0.9725  
- **Final  val  acc**: 0.8578  
- **Generalization gap (train - val)**: 0.1147  
- **Final train loss**: 0.1753  
- **Final  val  loss**: 0.6131  

---

## 4.2 최종 성능 요약 (Train / Val / Test)

노트북에서 `model.evaluate()`로 동일 모델을 각각 평가한 결과:

- **Train**: acc 0.9931 | loss 0.0930  
- **Val**:   acc 0.8578 | loss 0.6131  
- **Test**:  acc 0.8323 | loss 0.6849  

추가 지표(테스트셋 기준):
- **Top-1 acc**: 0.8323  
- **Top-5 acc**: 0.9478  
- **Macro-F1 (sklearn)**: 0.8227  

해석 포인트:
- Top-5가 0.9478로 높게 나온 건, **꽃(102클래스)처럼 클래스 간 경계가 미세한 문제에서 “상위 후보”는 꽤 잘 잡는 편**이라는 의미로 설명하기 좋다.

---

## 4.3 오류/혼동 분석 (Confusion)

### (1) 가장 많이 헷갈린 라벨 쌍 (true → pred, 상위 12개)
- petunia → hibiscus : 23  
- hibiscus → mallow : 16  
- petunia → pink primrose : 16  
- passion flower → clematis : 15  
- petunia → morning glory : 14  
- geranium → trumpet creeper : 13  
- petunia → balloon flower : 11  
- azalea → clematis : 11  
- lotus → water lily : 11  
- camellia → mallow : 9  
- rose → camellia : 9  
- petunia → tree mallow : 9  

### (2) 클래스별 정확도 하위(Per-class acc, n=표본 수)
- sweet pea : acc=0.250 (n=36)  
- snapdragon : acc=0.299 (n=67)  
- japanese anemone : acc=0.343 (n=35)  
- petunia : acc=0.466 (n=238)  
- siam tulip : acc=0.476 (n=21)  
- canterbury bells : acc=0.500 (n=20)  
- hibiscus : acc=0.523 (n=111)  
- columbine : acc=0.530 (n=66)  
- azalea : acc=0.566 (n=76)  
- desert-rose : acc=0.605 (n=43)  

해석 포인트:
- 혼동이 큰 쌍(예: petunia 계열)은 **꽃잎 형태/색감이 유사하거나 촬영 구도·배경이 비슷한 경우**로 설명하기 좋다.
- per-class acc가 낮은 클래스는 표본 수가 적거나(n이 작은 클래스), 유사 클래스가 많아 fine-grained 난이도가 높을 가능성이 있다.

---

## 4.4 성능 대비 파라미터 효율(경량/효율 지표)

### (1) 파라미터/모델 크기
- **Backbone params**: 4,049,571  
- **Total model params**: 4,180,233  
- **Approx weights size (float32)**: 15.95 MB  

> 분류 head(Dense 102) 때문에 total params가 backbone보다 약 13만 정도만 증가(=헤드는 가벼움).

### (2) FLOPs (선택 측정)
- **Approx FLOPs (batch=1)**: 801,046,907  

> FLOPs는 “이론 연산량”이고, 실제 속도는 하드웨어/라이브러리 최적화 영향을 받음.

### (3) Latency (간단 벤치마크)
- **Batch latency (sec)**: 0.266400  
- **Per-image latency (ms)**: 8.325  (batch_size=32)

---

## 4.5 경량 모델 비교 실험 (B0/B1/B2 빠른 비교)

노트북 후반부에서는 EfficientNet 변형(B0/B1/B2)을 **동일 입력(224×224), 동일 파이프라인, head-only 학습 3 epochs** 조건으로 빠르게 비교했다.  
(이 비교는 “완전 공정한” EfficientNet 공식 설정(각 B별 권장 해상도)과는 다르지만, **파라미터/지연시간 대비 성능 경향**을 빠르게 보기에는 유용하다.)

요약(DataFrame `df_eff` 출력):

| variant | val_acc | test_acc | params | ms_per_image | acc_per_Mparams |
|---|---:|---:|---:|---:|---:|
| B0 | 0.687255 | 0.655228 | 4,180,233 | 9.760399 | 0.164406 |
| B1 | 0.711765 | 0.679297 | 6,705,901 | 12.735525 | 0.106140 |
| B2 | 0.700000 | 0.659619 | 7,912,287 | 11.975868 | 0.088470 |

해석 포인트:
- **절대 정확도(val/test)**는 B1이 가장 높게 나왔지만,
- **acc_per_Mparams(파라미터 100만개당 정확도)** 기준으로 보면 B0가 가장 높게 나와서, “경량/효율” 관점에서는 B0의 장점이 드러난다.
- latency(ms_per_image)는 B0가 가장 빠른 편이라, **서비스/엣지 환경에서는 B0가 현실적 선택**이 될 수 있다.

---

## 4.6 최종 결론
이번 실습에서는 Flowers102에서 EfficientNet-B0를 ImageNet 사전학습 backbone으로 사용했을 때,  
Stage1(헤드만 학습)만으로도 val acc 0.83 수준까지 빠르게 도달했고,  
Stage2에서 backbone의 뒤 30%만 파인튜닝해 val acc 0.8578 / test acc 0.8323까지 개선되었다.  
또한 파라미터 418만(약 15.95MB) 수준에서 Top-5 0.9478을 기록해, fine-grained 분류에서 “효율 대비 준수한 성능”을 보여줬다.

---

## Appendix: 실행 환경 메모
노트북 첫 셀에서 TFDS만 설치한다.
```bash
pip install tensorflow-datasets
```

---

## References
- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*.
- TFDS: `oxford_flowers102`
****
