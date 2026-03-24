# SUDA Datasets

NIDS(Network Intrusion Detection System) 실험을 위한 데이터셋입니다.

## 데이터셋 목록

| Dataset | Samples | Features | Benign | Attack | Source |
|---------|---------|----------|--------|--------|--------|
| **nslkdd** | 148,517 | 41 | 51.9% | 48.1% | NSL-KDD |
| **unswnb15** | 257,673 | 42 | 36.1% | 63.9% | UNSW-NB15 |
| **cicids2018** | 43,036 | 78 | 3.3% | 96.7% | CIC-IDS2018 |
| **cidds** | 94,066 | 5 | 99.6% | 0.4% | CIDDS |

## 설치

```bash
# 전체 다운로드
uv run python datasets/download.py --all

# 개별 다운로드
uv run python datasets/download.py --dataset nslkdd
uv run python datasets/download.py --dataset unswnb15
uv run python datasets/download.py --dataset cicids2018
uv run python datasets/download.py --dataset cidds
```

## 데이터 형식

각 데이터셋은 `.npy` 파일로 저장됩니다:
- Shape: `(n_samples, n_features + 1)`
- 마지막 열: Binary label (0=benign, 1=attack)
- Dtype: float32 (features), int64 (label)

## 사용법

### 기본 로드

```python
from src.data.nids import load_dataset, get_dataset_info

# 데이터셋 정보 확인
info = get_dataset_info("nslkdd")
print(f"Samples: {info.n_samples}, Features: {info.n_features}")

# 데이터 로드
X, y = load_dataset("nslkdd")
```

### Drift Stream 생성

```python
from src.data.nids import make_sudden_drift_stream, make_label_shift_stream
import numpy as np

rng = np.random.default_rng(42)

# Sudden Drift: benign ratio 변화
X_stream, y_stream = make_sudden_drift_stream(
    "nslkdd",
    rng,
    total_samples=40000,
    drift_point=20000,
    pre_benign_ratio=0.7,   # drift 전: 70% benign
    post_benign_ratio=0.3,  # drift 후: 30% benign
)

# Label Shift: benign → attack 전환
X_stream, y_stream = make_label_shift_stream(
    "nslkdd",
    rng,
    total_samples=40000,
    drift_point=20000,
)
```

### Holdout Test Set

```python
from src.data.nids import sample_holdout

# 자연 비율로 샘플링
X_test, y_test = sample_holdout("nslkdd", rng, n_samples=2000)

# 특정 비율로 샘플링
X_test, y_test = sample_holdout(
    "nslkdd", rng,
    n_samples=2000,
    benign_ratio=0.5
)
```

## 데이터셋 상세

### NSL-KDD

- **원본**: KDD Cup 99 개선 버전
- **특징**: 중복 제거, 균형 잡힌 난이도 분포
- **공격 유형**: DoS, Probe, R2L, U2R
- **출처**: [GitHub](https://github.com/jmnwong/NSL-KDD-Dataset)

### UNSW-NB15

- **원본**: Australian Centre for Cyber Security
- **특징**: 현대적 공격 패턴, 9가지 공격 유형
- **공격 유형**: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms
- **출처**: [HuggingFace](https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15)

### CIC-IDS2018

- **원본**: Canadian Institute for Cybersecurity
- **특징**: 최신 공격 시나리오, 불균형 심함
- **공격 유형**: Brute Force, Heartbleed, Botnet, DoS, DDoS, Web Attacks, Infiltration
- **출처**: [HuggingFace](https://huggingface.co/datasets/Najet-hamdi/CIC-IDS2018)

### CIDDS

- **원본**: Coburg Intrusion Detection Data Sets
- **특징**: 극심한 불균형 (공격 0.4%), 실제 환경 기반
- **출처**: [HuggingFace](https://huggingface.co/datasets/caffeinatedcherrychic/cidds-aggregated)

## 주의사항

1. **cicids2018**: 공격 비율이 96.7%로 극단적으로 불균형합니다
2. **cidds**: 공격 비율이 0.4%로 극단적으로 불균형합니다
3. 모든 데이터셋은 전처리되어 NaN/Inf가 0으로 대체됩니다

## API Reference

```python
# 사용 가능한 함수들
from src.data.nids import (
    list_datasets,          # 데이터셋 목록 반환
    load_dataset,           # X, y 반환
    get_dataset_info,       # DatasetInfo 반환
    get_dataset_path,       # Path 반환
    make_stream,            # 셔플된 스트림 생성
    make_sudden_drift_stream,  # Drift 스트림 생성
    make_label_shift_stream,   # Label shift 스트림 생성
    sample_holdout,         # 홀드아웃 세트 생성
    validate_datasets,      # 모든 데이터셋 존재 여부 확인
)
```
