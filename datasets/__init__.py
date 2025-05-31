"""
Some codes are partially adapted from
https://github.com/AaltoML/generative-inverse-heat-dissipation/blob/main/scripts/datasets.py
"""

# 데이터셋 모듈의 진입점 역할을 하는 파일
# 다양한 데이터셋을 로드하고 전처리하는 유틸리티 함수들을 제공

import numpy as np
import torch

from datasets.quadratic import QuadraticDataset

def data_scaler(data):
    """
    데이터를 [-1, 1] 범위로 정규화하는 함수
    
    역할:
    - 신경망 학습 시 수치적 안정성을 위해 데이터를 표준 범위로 변환
    - 원본 데이터가 [0, 1] 범위일 때 [-1, 1]로 변환
    
    Parameters:
        data: 정규화할 데이터 텐서 (일반적으로 [0, 1] 범위)
    
    Returns:
        [-1, 1] 범위로 정규화된 데이터
    
    의존성:
        - runner/hilbert_runner.py에서 학습 데이터 전처리 시 사용
    """
    return data * 2. - 1.


def data_inverse_scaler(data):
    """
    정규화된 데이터를 원래 범위로 역변환하는 함수
    
    역할:
    - 모델 출력이나 샘플링 결과를 원본 데이터 범위로 복원
    - [-1, 1] 범위의 데이터를 [0, 1] 범위로 변환
    
    Parameters:
        data: 역변환할 데이터 텐서 ([-1, 1] 범위)
    
    Returns:
        [0, 1] 범위로 역변환된 데이터
    
    의존성:
        - runner/hilbert_runner.py에서 샘플링 결과 후처리 시 사용
    """
    return (data + 1.) / 2.

def get_dataset(config):
    """
    설정 파일에 따라 적절한 데이터셋을 생성하고 반환하는 팩토리 함수
    
    역할:
    - 설정에 지정된 데이터셋 타입에 따라 해당 데이터셋 객체 생성
    - 학습용과 테스트용 데이터셋을 별도의 시드로 생성하여 데이터 누출 방지
    
    구현 방식:
    - 데이터셋 이름에 따라 적절한 클래스를 인스턴스화
    - 학습/테스트 데이터셋은 다른 시드(42, 43)를 사용하여 독립성 보장
    
    Parameters:
        config: 데이터셋 설정을 포함한 설정 객체
            - config.data.dataset: 데이터셋 이름 ("Quadratic" 등)
            - config.data.num_data: 데이터 샘플 수
            - config.data.dimension: 데이터 차원 (함수의 점 개수)
    
    Returns:
        dataset: 학습용 데이터셋 객체
        test_dataset: 테스트용 데이터셋 객체
    
    의존성:
        - main.py에서 데이터셋 초기화 시 호출
        - runner/hilbert_runner.py에서 HilbertDiffusion 초기화 시 사용
    
    확장성:
        - 새로운 데이터셋 추가 시 elif 문을 추가하여 쉽게 확장 가능
        - Melbourne, Gridwatch 등 다른 1D 데이터셋도 지원 가능
    """
    if config.data.dataset == "Quadratic":
        # Quadratic 데이터셋 생성
        # seed=42: 학습 데이터의 재현성을 위한 고정 시드
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                num_points=config.data.dimension,
                                seed=42)
        # seed=43: 테스트 데이터는 다른 시드로 생성하여 독립성 보장
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43)
    else:
        raise NotImplementedError(f"Unknown dataset: {config.data.dataset}")
    return dataset, test_dataset


