# CRUSH 프로젝트 - Unsupervised Pseudo-Label 방식으로 전환

## 변경 요약

이전 버전의 핵심 문제 4가지를 해결하기 위해 다음을 수정했습니다.

### 이전 문제점
1. **TF-IDF + seed 1개**로 K-Means → 단어 표면 클러스터링 (의미 카테고리 못 잡음)
2. **DNA + WJ 합쳐서 클러스터링** → 데이터 출처별로 나뉠 위험
3. **EMA centroid + hidden 동시 업데이트** → 피드백 루프로 representation collapse
4. **WJ에는 정답 라벨 없는데 DNA 라벨 강제 적용** → 라벨 미스매치

### 변경 후 구조

```
[학습 시점 1회만]
DNA 프롬프트들
  → base 모델 forward (target_layers 중간 레이어)
  → hidden state (의미 공간)
  → K-Means(k=5)
  → pseudo-labels [0,1,2,3,4] (의미 모름, 임의 인덱스)
  → 각 클러스터 hidden 평균 → centroids (L2 정규화 후 고정)

[학습]
  → DNA: pull/push loss로 같은 pseudo-label끼리 모음 / 다름끼리 밂
  → WJ : KL/Benign loss에는 참여, pull/push에는 미참여
  → centroid는 학습 내내 변하지 않음

[검증]
  → DNA의 risk_area(정답) 라벨로:
    - Silhouette / Davies-Bouldin
    - 새 K-Means 돌려서 NMI / ARI / Hungarian Accuracy
    - t-SNE 시각화 (Before vs After)
```

## 파일별 변경점

### `dataset.py` (대대적 수정)
- K-Means / TF-IDF / seed 코드 **전부 제거**
- `data_unsafe_true_labels`: DNA의 risk_area 정답 라벨 (검증용으로만)
- `data_unsafe_request_prompts`: DNA의 request 부분만 (clustering.py가 사용)
- `data_unsafe_labels`: pseudo-label, 외부에서 `set_pseudo_labels()`로 주입
- WJ의 `wj_unsafe_labels` 완전 제거
- `__getitem__`에서 WJ에 라벨 미부여 (`labels`는 DNA만)

### `crush_loss.py` (대대적 수정)
- `CentroidManager` (EMA 기반) **폐기**
- `FixedCentroidHolder` 신규: 사전 계산된 centroid 보관/이동만 담당, 업데이트 없음
- `calc_crush_loss`: margin 0.2 → 0.5, 기능은 동일
- 1000 스텝 backprop 시뮬레이션에서 centroid 변화량 0 검증 완료

### `clustering.py` (신규)
- `extract_request_hidden`: base 모델로 request 마지막 토큰 hidden 추출
- `run_kmeans_clustering`: K-Means (n_init=10으로 안정성)
- `compute_centroids_from_hidden`: 클러스터 평균 → L2 정규화 → 고정 centroid
- `diagnose_clustering`:
  - 클러스터별 샘플 분포
  - true_labels와 NMI/ARI 사전 계산
  - 클러스터별 대표 프롬프트 5개 출력 (사람이 의미 검증)
- `cluster_unsafe_prompts`: 전체 파이프라인 + hidden 캐싱 지원

### `trainer.py` (정밀 수정)
- `centroid_mgr` → `centroid_holder`
- `update_centroids` 호출 **전부 제거**
- `_compute_loss`에서 DNA 배치 크기(`dna_batch_size`) 명시적 분리
- `_calc_loss`에서 unsafe hidden 슬라이싱: pull/push에는 DNA 부분(앞 절반)만 사용
- `labels`에 -1 또는 누락 시 명시적 에러 raise (silent fallback 제거)

### `train.py` (대대적 수정)
- 학습 시작 전 클러스터링 단계 **신규 추가**:
  1. base 모델로 DNA hidden 추출 (cluster_layer = target_layers 중간)
  2. K-Means → pseudo_labels + centroids
  3. dataset.set_pseudo_labels() 주입
  4. FixedCentroidHolder 생성
- LoRA 적용은 클러스터링 후
- 초기 centroid + pseudo_labels + true_labels + cluster_layer를
  `crush_centroids_initial.pt`에 저장 (evaluate.py가 사용)

### `evaluate.py` (대대적 수정)
- `INIT_CENTROIDS_PATH` 추가 (cluster_layer 자동 동기화)
- 평가 데이터 균등 샘플링: pseudo-label이 아닌 **true_labels(risk_area) 기준**
- `evaluate_separation`: true-label 기준 Silhouette / D-B
- `evaluate_clustering_recovery` (신규):
  - 학습 후 latent에서 새 K-Means 돌려서
  - NMI / ARI / Hungarian Accuracy 계산
  - "라벨 없이 학습했는데 클러스터가 자연 발생함" 검증
- t-SNE: true-label로 색칠 (Before vs After)
- Confusion matrix 출력

## 실행 순서

1. **학습**: `python train.py ...` (기존 인자 그대로)
   - 시작 시점에 클러스터링 진단 출력 (대표 프롬프트, NMI, 분포 등)
   - 이때 NMI가 너무 낮으면(<0.1) cluster_layer 변경 검토
2. **평가**: `python evaluate.py`
   - Before/After Silhouette/D-B 비교
   - **★ Before/After NMI/ARI 비교가 핵심 결과**
   - t-SNE 이미지로 시각 확인

## 핵심 검증 가설

> H1: 학습 후 latent에서 새로 K-Means를 돌리면 true_labels(risk_area)와의
>     NMI/ARI/Hungarian Accuracy가 학습 전보다 개선된다.

이 H1이 참이면: "pseudo-label로 unsupervised 학습했는데 latent에 의미적 카테고리가 자연 발생"이라는 contribution 입증.

만약 H1이 거짓이면: pseudo-label과 true-label이 무관해서 학습이 의미 카테고리를 못 만들었다는 뜻 → cluster_layer / 클러스터링 알고리즘 / hidden 공간 재검토.

## 결정사항 (이전 대화에서 합의됨)

- ❶ DNA만 pull/push, WJ는 KL/Benign loss만
- ❷ 학습 시작 전 1회 클러스터링 (DeepCluster epoch마다 재클러스터링은 future work)
- ❸ Centroid는 사전 계산 후 고정
- ❹ 클러스터링 공간 = target_layers 중간 레이어 hidden state
- ❺ Hidden 추출 위치 = request 마지막 토큰

## 단위 테스트 통과 항목

- ✅ K-Means → centroid 계산 → L2 정규화 (norms ≈ 1.0)
- ✅ FixedCentroidHolder가 1000 스텝 backprop 후 변화량 0
- ✅ NMI/ARI/Hungarian Accuracy = 1.0 (이상적 mock 데이터)
- ✅ DNA/WJ 슬라이싱 후 loss 계산 정상
- ✅ labels = -1 (미주입) 시 ValueError로 명시 차단
- ✅ 모든 파일 syntax + import 일치