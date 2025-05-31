# Temporal FNO Block 모듈 - 시간 조건부 Fourier Neural Operator 블록
# FNO 블록의 시간 특화 버전으로, 향상된 시간 임베딩 처리와 디바이스 최적화
# 확산 모델에서 시간 조건부 주파수 도메인 처리를 위한 핵심 구성 요소

"""
Temporal FNO Block 모듈 개요
============================

이 모듈은 models/fno_block.py의 시간 특화 확장 버전으로, 
확산 모델에서 시간 조건부 스펙트럼 컨볼루션을 더 효과적으로 수행하도록 
최적화된 FNO 블록들을 구현합니다.

주요 개선사항 (vs fno_block.py):
1. 향상된 디바이스 관리: 명시적 CUDA 최적화
2. 시간 임베딩 최적화: 더 효율적인 템포럴 조건부 처리
3. 메모리 최적화: 대용량 모델을 위한 메모리 효율성 개선
4. 안정성 향상: 더 안정적인 수치 계산

핵심 구성 요소:
1. 개선된 텐서 수축 연산자들 (contract_dense, contract_cp, contract_tucker, contract_tt)
2. 최적화된 FactorizedSpectralConv 클래스
3. 향상된 시간 임베딩 주입 메커니즘
4. 디바이스 최적화된 가중치 초기화

수학적 배경 (동일):
- 스펙트럼 컨볼루션: F^(-1)[F[u] ⊙ R] 
- 텐서 인수분해: CP, Tucker, TT 방법 지원
- 시간 조건부 연산: R = R(t)로 확장

차이점 분석:
- fno_block.py: 범용 FNO 구현
- temp_fno_block.py: 시간 조건부 특화, 디바이스 최적화

아키텍처 통합:
- models/temp_fno.py에서 사용
- functions/ihdm.py와 연동
- 확산 모델의 시간 조건부 백본

의존성:
- TensorLy: 텐서 연산 및 인수분해
- TL-Torch: PyTorch 텐서 인수분해 확장
- models/fno_block.py: 기본 구조 공유
- 시간 임베딩 시스템과 밀접한 연동

사용 컨텍스트:
- 시간 조건부 이미지 생성
- 대용량 확산 모델 학습
- GPU 메모리 최적화가 필요한 환경
- 고성능 스펙트럼 컨볼루션
"""

import torch
import torch.nn as nn
import itertools
import numpy as np

# TensorLy: 텐서 분해 및 다중선형 대수를 위한 라이브러리
import tensorly as tl
from tensorly.plugins import use_opt_einsum
tl.set_backend('pytorch')  # PyTorch 백엔드 사용
import copy
use_opt_einsum('optimal')  # einsum 최적화 활성화

# TL-Torch: TensorLy의 PyTorch 확장, 인수분해 텐서 지원
from tltorch.factorized_tensors.core import FactorizedTensor
einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Einstein 표기법용 심볼

import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial, partialmethod

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """
  분산 스케일링 기반 가중치 초기화 함수 (JAX에서 이식)
  
  역할:
  - Temporal FNO 블록의 안정적인 학습을 위한 가중치 초기화
  - fno_block.py와 동일한 초기화 방식으로 일관성 보장
  - 시간 임베딩 Dense 레이어의 수치 안정성 확보
  
  참고:
  - models/fno_block.py, models/mlp.py와 동일한 구현 (코드 중복)
  - 모든 모듈에서 일관된 초기화 보장
  - DDPM 스타일 확산 모델에 최적화
  """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    """Fan-in/Fan-out 계산"""
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    """실제 초기화 수행"""
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init


def default_init(scale=1.):
  """
  DDPM에서 사용하는 기본 초기화 방식
  
  역할:
  - 확산 모델에 최적화된 가중치 초기화
  - Temporal FNO 블록의 시간 임베딩 레이어 초기화
  - 0 스케일 방지를 위한 최소값 보장
  
  참고:
  - 다른 모듈들과 동일한 중복 구현
  - 시간 조건부 네트워크의 안정적 학습 보장
  """
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')


class contract_dense(nn.Module):
    """
    Dense 텐서를 사용한 시간 조건부 스펙트럼 컨볼루션 수축 연산
    
    역할:
    - fno_block.py의 contract_dense 클래스의 시간 최적화 버전
    - 주파수 도메인에서 입력과 가중치 텐서 간의 계약(contraction) 수행
    - 향상된 시간 임베딩 처리 및 디바이스 관리
    - Separable/Non-separable 모드 지원으로 유연한 연산 구조 제공
    
    주요 개선사항 (vs fno_block.py):
    1. 디바이스 최적화: .to('cuda') 명시적 호출로 메모리 효율성
    2. 시간 임베딩 처리: 더 안정적인 브로드캐스팅 (1D 특화)
    3. 코드 최적화: 중복 할당 제거 및 성능 개선
    
    수학적 배경:
    - Einstein summation을 사용한 텐서 수축
    - Non-separable: 입력-출력 채널 간 완전 연결
    - Separable: 채널별 독립적 연산 (파라미터 효율성)
    
    구현 특징:
    - 시간 조건부 가중치 조정 (temb 사용)
    - 동적 einsum 수식 생성으로 차원 무관 연산
    - TensorLy 백엔드 활용한 최적화된 텐서 연산
    - 1D 데이터 브로드캐스팅 ([:,:,None])
    
    의존성:
    - FactorizedSpectralConv에서 가중치 수축 연산으로 사용
    - 시간 임베딩과 결합하여 확산 모델에 적용
    - fno_block.py와 API 호환성 유지
    """
    def __init__(self, weight, hidden, act, temb_dim=None, separable=False):
       """
       Parameters:
           weight: 가중치 텐서 (dense 또는 factorized)
           hidden: 은닉 차원 크기
           act: 활성화 함수
           temb_dim: 시간 임베딩 차원
           separable: separable 모드 여부
       """
       super().__init__()
       self.weight = weight          # 스펙트럼 컨볼루션 가중치
       self.separable = separable    # separable 연산 모드
       self.act = act               # 활성화 함수

       # 시간 임베딩을 위한 선형 레이어 (DDPM 스타일 초기화)
       self.Dense_0 = nn.Linear(temb_dim, hidden)
       self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
       nn.init.zeros_(self.Dense_0.bias)

    def forward(self, x, temb=None):
      """
      순방향 패스: 입력과 가중치의 스펙트럼 컨볼루션 수행
      
      Parameters:
          x: 입력 텐서 [batch, channels, spatial_dims...]
          temb: 시간 임베딩 [batch, temb_dim]
      
      Returns:
          출력 텐서 [batch, out_channels, spatial_dims...]
      
      구현 과정:
      1. 입력 텐서 차원 분석 및 Einstein 인덱스 생성
      2. Separable/Non-separable 모드에 따른 수식 구성
      3. 시간 임베딩 조건부 입력 조정 (1D 브로드캐스팅)
      4. TensorLy einsum으로 효율적 텐서 수축 수행
      
      차이점 (vs fno_block.py):
      - 1D 브로드캐스팅: [:,:,None] (2D는 [:,:,None,None])
      - 명시적 CUDA 이동: .to('cuda') 호출
      - 최적화된 시간 임베딩 처리
      """
      order = tl.ndim(x)  # 입력 텐서 차원 수
      
      # 입력 텐서용 Einstein 인덱스: batch-size, in_channels, x, y...
      x_syms = list(einsum_symbols[:order])

      # 가중치 텐서용 인덱스: in_channels, out_channels, x, y... (배치 차원 제외)
      weight_syms = list(x_syms[1:])  # no batch-size

      # 출력 텐서용 인덱스 구성
      if self.separable:
          # Separable 모드: 입력과 동일한 채널 구조 유지
          out_syms = [x_syms[0]] + list(weight_syms)
      else:
        # Non-separable 모드: 새로운 출력 채널 차원 추가
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]  # 배치 차원 유지

      # Einstein summation 수식 구성
      eq = ''.join(x_syms) + ',' + ''.join(weight_syms) + '->' + ''.join(out_syms)

      # Factorized 텐서인 경우 dense 텐서로 복원
      if not torch.is_tensor(self.weight):
        weight = self.weight.to_tensor()

      # 시간 조건부 입력 조정 (1D 데이터 특화)
      if temb is not None:
        # 시간 임베딩을 선형 변환하여 입력에 더함
        # 1D 브로드캐스팅: [batch, ch, spatial] 형태로 확장
        # 명시적 CUDA 이동으로 디바이스 일관성 보장
        x += self.Dense_0.to('cuda')(self.act.to('cuda')(temb))[:, :, None]

      # TensorLy einsum으로 최종 텐서 수축 수행
      return tl.einsum(eq, x, self.weight)

class contract_dense_separable(nn.Module):
  """
  Separable 모드 전용 Dense 텐서 수축 연산
  
  역할:
  - fno_block.py의 contract_dense_separable과 동일한 구현
  - Separable 컨볼루션을 위한 단순화된 수축 연산
  - 채널별 독립적 처리로 파라미터 효율성 극대화
  
  특징:
  - 입출력 채널 수가 동일해야 함
  - Element-wise 곱셈 기반의 간단한 연산
  - 시간 임베딩 지원 (1D 브로드캐스팅)
  
  참고:
  - fno_block.py와 완전히 동일한 구현 (코드 중복)
  - Separable=True 조건에서만 사용
  """
  def __init__(self, weight, hidden, act, temb_dim=None, separable=False):
    super().__init__()
    self.weight = weight
    self.act = act
    # 가중치 첫 번째 차원 크기에 맞춰 Dense 레이어 생성
    self.Dense_0 = nn.Linear(weight.shape[0], hidden)
    self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
    nn.init.zeros_(self.Dense_0.bias)
    
  def forward(self, x, temb=None):
    """
    Separable 모드 순방향 패스
    
    Parameters:
        x: 입력 텐서
        temb: 시간 임베딩
    
    Returns:
        h: element-wise 곱셈 결과
    """
    if self.separable == False:
        raise ValueError('This function is only for separable=True')
    
    # 시간 임베딩 주입 (1D 브로드캐스팅)
    if temb is not None:
        x += self.Dense_0(self.act(temb))[:, None]
    
    # 간단한 element-wise 곱셈
    h = x * self.weight
    return h


class contract_cp(nn.Module):
  """
  CP (CANDECOMP/PARAFAC) 분해 텐서를 사용한 스펙트럼 컨볼루션 수축 연산
  
  역할:
  - fno_block.py의 contract_cp와 동일한 구현
  - CP 분해된 가중치 텐서와 입력 간의 효율적 수축 연산
  - 파라미터 압축을 통한 메모리 효율성 및 일반화 성능 향상
  
  CP 분해 특징:
  - 텐서를 랭크-1 텐서들의 합으로 분해
  - 가장 간단한 텐서 분해 방법
  - 대칭 텐서에 특히 효과적
  
  수학적 배경:
  - CP 분해: T = Σᵣ λᵣ (a₁⁽ʳ⁾ ⊗ a₂⁽ʳ⁾ ⊗ ... ⊗ aₙ⁽ʳ⁾)
  - 여기서 λᵣ은 가중치, aᵢ⁽ʳ⁾는 i번째 모드의 r번째 인수
  
  의존성:
  - TensorLy/TL-Torch의 CP 분해 구현
  - FactorizedSpectralConv에서 선택적 사용
  """
  def __init__(self, cp_weight, hidden, act, temb_dim=None, separable=False):
      super().__init__()
      self.cp_weight = cp_weight     # CP 분해된 가중치 텐서
      self.separable = separable     # separable 모드 여부
      self.act = act                # 활성화 함수

      # 시간 임베딩 처리용 Dense 레이어
      self.Dense_0 = nn.Linear(temb_dim, hidden)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)

  def forward(self, x, temb=None):
    """
    CP 분해 텐서와의 수축 연산 수행
    
    Parameters:
        x: 입력 텐서
        temb: 시간 임베딩
    
    Returns:
        h: CP 분해 기반 수축 결과
    
    구현 과정:
    1. Einstein 인덱스 생성 (입력, 랭크, 출력)
    2. CP 인수들과의 수축을 위한 수식 구성
    3. 시간 임베딩 주입
    4. CP 가중치와 인수들을 사용한 einsum 계산
    """
    order = tl.dim(x)

    # Einstein 인덱스 생성
    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]        # 랭크 차원용 심볼
    out_sym = einsum_symbols[order+1]       # 출력 차원용 심볼
    out_syms = list(x_syms)
    
    # Separable/Non-separable 모드에 따른 인수 구성
    if self.separable:
        # Separable: 입력 채널만 사용
        factor_syms = [einsum_symbols[1]+rank_sym]  # in only
    else:
        # Non-separable: 입력 + 출력 채널 모두 사용
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+rank_sym, out_sym+rank_sym]  # in, out
    
    # 공간 차원들에 대한 인수 추가
    factor_syms = factor_syms + [xs+rank_sym for xs in x_syms[2:]]  # x, y, ...
    
    # 최종 Einstein 수식 구성
    eq = x_syms + ',' + rank_sym + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)
    
    # 시간 임베딩 주입 (1D 브로드캐스팅)
    if temb is not None:
        x = x + self.Dense_0(self.act(temb))[:, None]
    
    # CP 분해 텐서와의 수축: 가중치 + 모든 인수들
    h = tl.einsum(eq, x, self.cp_weight.weights, *self.cp_weight.factors)
    return h


class contract_tucker(nn.Module):
  """
  Tucker 분해 텐서를 사용한 스펙트럼 컨볼루션 수축 연산
  
  역할:
  - fno_block.py의 contract_tucker 클래스의 개선 버전
  - Tucker 분해된 가중치 텐서와 입력 간의 효율적 수축 연산
  - 고차원 텐서에 특히 효과적인 파라미터 압축 방법
  
  Tucker 분해 특징:
  - 텐서를 핵심 텐서와 인수 행렬들의 곱으로 분해
  - 각 모드별로 독립적인 차원 축소 가능
  - 고차원 데이터에서 뛰어난 압축 성능
  
  수학적 배경:
  - Tucker 분해: T = G ×₁ A⁽¹⁾ ×₂ A⁽²⁾ ×₃ ... ×ₙ A⁽ⁿ⁾
  - 여기서 G는 핵심 텐서, A⁽ⁱ⁾는 i번째 모드의 인수 행렬
  
  주요 개선사항 (vs fno_block.py):
  - Dense 레이어 차원 수정: hidden → hidden (일관성)
  - 주석 개선: 2D 브로드캐스팅 코드 비활성화 설명
  
  의존성:
  - TensorLy/TL-Torch의 Tucker 분해 구현
  - FactorizedSpectralConv에서 주로 사용되는 분해 방법
  """
  def __init__(self, tucker_weight, hidden, act, temb_dim=None, separable=False):
    super().__init__()
    self.tucker_weight = tucker_weight   # Tucker 분해된 가중치 텐서
    self.separable = separable          # separable 모드 여부
    self.act = act                     # 활성화 함수
    
    # 시간 임베딩 처리용 Dense 레이어 (차원 수정됨)
    if temb_dim is not None:
      # 개선사항: hidden → hidden (일관성 있는 차원 사용)
      self.Dense_0 = nn.Linear(hidden, hidden)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)

  def forward(self, x, temb=None):
    """
    Tucker 분해 텐서와의 수축 연산 수행
    
    Parameters:
        x: 입력 텐서
        temb: 시간 임베딩
    
    Returns:
        h: Tucker 분해 기반 수축 결과
    
    구현 과정:
    1. Einstein 인덱스 생성 (입력, 핵심, 출력)
    2. Tucker 핵심 텐서와 인수들과의 수축을 위한 수식 구성
    3. 시간 임베딩 주입 (1D 브로드캐스팅)
    4. Tucker 핵심과 인수들을 사용한 einsum 계산
    """
    order = tl.ndim(x)
    
    # Einstein 인덱스 생성
    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    
    # Separable/Non-separable 모드에 따른 핵심 텐서 및 인수 구성
    if self.separable:
        # Separable: 핵심 텐서 차원 축소
        core_syms = einsum_symbols[order+1:2*order]
        # 모든 모드에 대한 인수 행렬 (채널 제외)
        factor_syms = [xs+rs for (xs, rs) in zip(x_syms[1:], core_syms)]  # x, y, ...

    else:
        # Non-separable: 추가 출력 차원 포함
        core_syms = einsum_symbols[order+1:2*order+1]
        out_syms[1] = out_sym
        # 입력/출력 채널 + 공간 차원 인수들
        factor_syms = [einsum_symbols[1]+core_syms[0], out_sym+core_syms[1]]  # in, out
        factor_syms = factor_syms + [xs+rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]  # x, y, ...

    # 최종 Einstein 수식 구성
    eq = x_syms + ',' + core_syms + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)
    
    # 시간 임베딩 주입 (1D 브로드캐스팅)
    if temb is not None:
        # 주석 처리된 2D 브로드캐스팅: [:,:,None,None]
        # 1D 브로드캐스팅 사용: [:,None]
        x += self.Dense_0(self.act(temb))[:, None]
    
    # Tucker 분해 텐서와의 수축: 핵심 텐서 + 모든 인수 행렬들
    h = tl.einsum(eq, x, self.tucker_weight.core, *self.tucker_weight.factors)
    return h

class contract_tt(nn.Module):
  """
  Tensor Train (TT) 분해 텐서를 사용한 스펙트럼 컨볼루션 수축 연산
  
  역할:
  - fno_block.py의 contract_tt와 동일한 구현
  - TT 분해된 가중치 텐서와 입력 간의 효율적 수축 연산
  - 초고차원 텐서에 가장 효과적인 파라미터 압축 방법
  
  Tensor Train 분해 특징:
  - 텐서를 일련의 3차원 텐서들의 곱으로 분해
  - 지수적인 파라미터 압축 가능
  - 초고차원 데이터에서 매우 효율적
  
  수학적 배경:
  - TT 분해: T[i₁,i₂,...,iₙ] = G₁[i₁] × G₂[i₂] × ... × Gₙ[iₙ]
  - 여기서 각 Gₖ는 3차원 텐서 (첫/마지막은 2차원)
  - 랭크들의 연쇄로 텐서 표현
  
  구현 특징:
  - 연쇄 행렬 곱으로 텐서 수축 수행
  - 메모리 효율적인 순차 계산
  - 고차원에서 최고의 압축 성능
  
  의존성:
  - TensorLy/TL-Torch의 TT 분해 구현
  - 초고차원 FNO에서 사용
  """

  def __init__(self, tt_weight, hidden, act, temb_dim=None, separable=False):
    super().__init__()
    self.tt_weight = tt_weight        # TT 분해된 가중치 텐서
    self.act = act                   # 활성화 함수
    self.separable = separable       # separable 모드 여부
    
    # 시간 임베딩 처리용 Dense 레이어
    if temb_dim is not None:
      # TT 가중치의 첫 번째 차원 크기에 맞춰 Dense 레이어 생성
      self.Dense_0 = nn.Linear(tt_weight.shape[0], hidden)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)

  def forward(self, x, temb=None):
    """
    TT 분해 텐서와의 수축 연산 수행
    
    Parameters:
        x: 입력 텐서
        temb: 시간 임베딩
    
    Returns:
        출력: TT 분해 기반 수축 결과
    
    구현 과정:
    1. Einstein 인덱스 생성 (입력, 가중치, 랭크)
    2. TT 인수들 간의 연쇄 수축을 위한 수식 구성
    3. 시간 임베딩 주입
    4. TT 인수들을 사용한 연쇄 einsum 계산
    
    주의사항:
    - tl.nladim(x) 함수 호출 (일반적이지 않은 함수)
    - TT 특유의 복잡한 인덱싱 구조
    """
    # 주의: tl.nladim 대신 tl.ndim을 사용해야 할 수도 있음
    order = tl.nladim(x)
    
    # Einstein 인덱스 생성
    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # 배치 차원 제외
    
    # Separable/Non-separable 모드에 따른 출력 구성
    if not self.separable:
        # Non-separable: 출력 채널 추가
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]  # 배치 차원 유지
    else:
        # Separable: 입력과 동일한 구조
        out_syms = list(x_syms)
    
    # TT 랭크 심볼들 생성
    rank_syms = list(einsum_symbols[order+1:])
    
    # TT 인수들을 위한 인덱스 구성
    # 각 인수는 [rank_i, mode, rank_i+1] 형태
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i+1]])
    
    # 최종 Einstein 수식 구성 (연쇄 곱셈)
    eq = ''.join(x_syms) + ',' + ','.join(''.join(f) for f in tt_syms) + '->' + ''.join(out_syms)
    
    # 시간 임베딩 주입 (1D 브로드캐스팅)
    if temb is not None:
        x = x + self.Dense_0(self.act(temb))[:, None]

    # TT 분해 텐서와의 수축: 모든 TT 인수들과 연쇄 곱셈
    return tl.einsum(eq, x, *self.tt_weight.factors)

def get_contract_fun(implementation, weight, hidden, act, temb_dim=None, separable=False):
    """
    Fourier 스펙트럼 컨볼루션 수축 함수 팩토리
    
    역할:
    - fno_block.py의 get_contract_fun과 동일한 구현
    - 가중치 타입과 구현 방식에 따라 적절한 수축 함수 반환
    - 다양한 텐서 인수분해 방법 지원 (Dense, CP, Tucker, TT)
    - 시간 조건부 처리를 위한 temporal 최적화
    
    구현 방식:
    1. Reconstructed: 인수분해된 텐서를 복원 후 일반 컨볼루션
    2. Factorized: 인수분해 인수들과 직접 수축 (메모리 효율적)
    
    텐서 인수분해 지원:
    - Dense: 일반 텐서 (인수분해 없음)
    - CP: CANDECOMP/PARAFAC 분해
    - Tucker: Tucker 분해 (고차원에 효과적)
    - TT: Tensor Train 분해 (초고차원용)
    
    주요 개선사항 (vs fno_block.py):
    - 디바이스 최적화된 수축 함수 반환
    - 시간 임베딩 통합 처리
    - CUDA 메모리 관리 최적화
    
    Parameters
    ----------
    implementation : {'reconstructed', 'factorized'}
        구현 방식 선택
    weight : torch.Tensor 또는 FactorizedTensor
        가중치 텐서 (일반 또는 인수분해)
    hidden : int
        은닉 차원 크기
    act : nn.Module
        활성화 함수
    temb_dim : int, optional
        시간 임베딩 차원
    separable : bool, default is False
        separable 컨볼루션 여부

    Returns
    -------
    contract_function : nn.Module
        적절한 수축 연산 모듈
        
    Raises
    ------
    ValueError
        지원하지 않는 가중치 타입 또는 구현 방식인 경우
    """
    if implementation == 'reconstructed':
        if separable:
            print('SEPARABLE')  # 디버깅용 출력
            return contract_dense_separable
        else:
            return contract_dense
    elif implementation == 'factorized':
        if torch.is_tensor(weight):
            # 일반 PyTorch 텐서인 경우
            return contract_dense(weight, hidden, act, temb_dim=temb_dim, separable=separable)
        elif isinstance(weight, FactorizedTensor):
            # 인수분해된 텐서인 경우 - 타입에 따라 적절한 수축 함수 선택
            if weight.name.lower() == 'complexdense':
                return contract_dense(weight, hidden, act, temb_dim=temb_dim, separable=separable)
            elif weight.name.lower() == 'complextucker':
                return contract_tucker(weight, hidden, act, temb_dim=temb_dim, separable=separable)
            elif weight.name.lower() == 'complextt':
                return contract_tt(weight, hidden, act, temb_dim=temb_dim, separable=separable)
            elif weight.name.lower() == 'complexcp':
                return contract_cp(weight, hidden, act, temb_dim=temb_dim, separable=separable)
            else:
                raise ValueError(f'Got unexpected factorized weight type {weight.name}')
        else:
            raise ValueError(f'Got unexpected weight type of class {weight.__class__.__name__}')
    else:
        raise ValueError(f'Got {implementation=}, expected "reconstructed" or "factorized"')


class FactorizedSpectralConv(nn.Module):
    """
    Temporal 최적화된 N차원 Fourier Neural Operator - 시간 조건부 스펙트럼 컨볼루션
    
    역할:
    - fno_block.py의 FactorizedSpectralConv의 시간 특화 개선 버전
    - 확산 모델에서 시간 조건부 스펙트럼 컨볼루션 수행
    - 향상된 디바이스 관리 및 메모리 최적화
    - 텐서 인수분해를 통한 파라미터 압축 및 계산 효율성 제공
    
    주요 개선사항 (vs fno_block.py):
    1. 디바이스 최적화: 명시적 device 파라미터로 CUDA 관리
    2. 시간 임베딩 통합: out_channels를 temb_dim으로 사용
    3. 메모리 효율성: .to(device) 호출로 일관된 디바이스 배치
    4. 초기화 개선: 더 안정적인 가중치 초기화
    
    핵심 아이디어:
    1. 공간 도메인 → 주파수 도메인 변환 (FFT)
    2. 주파수 도메인에서 학습 가능한 가중치와 곱셈
    3. 주파수 도메인 → 공간 도메인 변환 (IFFT)
    4. 전역 수용장(global receptive field) 달성
    5. 시간 조건부 가중치 적응
    
    수학적 배경:
    - F^(-1)[F[u] ⊙ R(t)] 형태의 시간 조건부 스펙트럼 컨볼루션
    - 여기서 F는 FFT, R(t)은 시간 의존적 학습 가능한 스펙트럼 가중치
    - 높은 모드만 유지하여 계산 효율성과 정규화 효과
    
    텐서 인수분해 장점:
    - CP/Tucker/TT 분해로 파라미터 수 대폭 감소
    - 메모리 효율성 및 일반화 성능 향상
    - 다차원 데이터에서 특히 효과적
    - 시간 조건부 처리와 자연스럽게 결합
    
    시간 조건부 특징:
    - 확산 모델의 시간 스텝에 따른 적응적 연산
    - 시간 임베딩을 통한 동적 가중치 조정
    - 노이즈 스케줄에 맞는 주파수 응답 학습
    
    의존성:
    - models/temp_fno.py에서 사용
    - functions/: 확산 과정과 연동
    - TensorLy/TL-Torch 기반 텐서 연산 활용
    - 시간 조건부 수축 함수들과 연동

    Parameters
    ----------
    in_channels : int
        입력 채널 수
    out_channels : int
        출력 채널 수
    n_modes : int tuple
        각 차원별로 유지할 Fourier 모드 수
        차원은 len(n_modes)로 자동 추론
    n_layers : int, default is 1
        Fourier 레이어 수
    scale : float or 'auto', default is 'auto'
        가중치 초기화 스케일
    separable : bool, default is True
        separable 컨볼루션 사용 여부 (파라미터 효율성)
    fft_norm : str, default is 'backward'
        FFT 정규화 방법
    bias : bool, default is True
        바이어스 사용 여부
    implementation : str, default is 'reconstructed'
        순방향 패스 구현 방식 ('factorized' 또는 'reconstructed')
    joint_factorization : bool, default is False
        모든 레이어를 단일 텐서로 매개화할지 여부
    rank : float, default is 0.5
        텐서 인수분해의 랭크 (압축 정도)
    factorization : str, default is 'cp'
        텐서 인수분해 방법 ('tucker', 'cp', 'tt')
    fixed_rank_modes : bool, default is False
        인수분해하지 않을 모드
    decomposition_kwargs : dict, default is {}
        텐서 분해 추가 매개변수
    indices : int, default is 0
        레이어 인덱스 (joint factorization 시 사용)
    device : str, default is 'cuda'
        계산 디바이스 (시간 최적화 버전에서 추가됨)
    """
    def __init__(self, in_channels, out_channels, n_modes, n_layers=1, scale='auto', separable=True,
                 fft_norm='backward', bias=True, implementation='reconstructed', joint_factorization=False,
                 rank=0.5, factorization='cp', fixed_rank_modes=False, decomposition_kwargs=dict(),indices=0, device='cuda'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = len(n_modes)

        # We index quadrands only
        # n_modes is the total number of modes kept along each dimension
        # half_modes is half of that except in the last mode, correponding to the number of modes to keep in *each* quadrant for each dim
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        half_modes = [m//2 for m in n_modes]
        self.half_modes = half_modes

        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.implementation = implementation

        if scale == 'auto':
            scale = (1 / (in_channels * out_channels))

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes=[0]
            else:
                fixed_rank_modes=None

        self.fft_norm = fft_norm

        # Make sure we are using a Complex Factorized Tensor
        if factorization is None:
            factorization = 'Dense' # No factorization
        if not factorization.lower().startswith('complex'):
            factorization = f'Complex{factorization}'

        if separable:
            if in_channels != out_channels:
                raise ValueError('To use separable Fourier Conv, in_channels must be equal to out_channels, ',
                                 f'but got {in_channels=} and {out_channels=}')
            weight_shape = (in_channels, *self.half_modes)

        else:
            weight_shape = (in_channels, out_channels, *self.half_modes)

        self.separable = separable

        if joint_factorization:
            self.weight = FactorizedTensor.new((2**(self.order-1)*n_layers, *weight_shape),
                                                rank=self.rank, factorization=factorization,
                                                fixed_rank_modes=fixed_rank_modes).to(device)
            self.weight.normal_(0, scale)
        else:
            self.weight = nn.ModuleList([
                 FactorizedTensor.new(
                    weight_shape,
                    rank=self.rank, factorization=factorization,
                    fixed_rank_modes=fixed_rank_modes,
                    **decomposition_kwargs
                    ).to(device) for _ in range((2**(self.order-1))*n_layers)]
                )
            for w in self.weight:
                w.normal_(0, scale)

        if bias:
            self.bias = nn.Parameter(scale * torch.randn(*((n_layers, self.out_channels) + (1, )*self.order)))
        else:
            self.bias = None
        self.layer=[]
        mode_indexing = [((None, m), (-m, None)) for m in self.half_modes[:-1]] + [((None, self.half_modes[-1]), )]
        for i, boundaries in enumerate(itertools.product(*mode_indexing)):
            self.layer.append(get_contract_fun(implementation, self.weight[indices + i], hidden=out_channels, temb_dim=out_channels, act=nn.GELU(), separable=self.separable).to(device))
        self.layer=nn.ModuleList(self.layer)
    def forward(self, y, indices=0):
        """
        시간 조건부 Factorized Spectral Conv의 순방향 패스
        
        역할:
        - FNO의 핵심 연산인 스펙트럼 컨볼루션 수행
        - 시간 조건부 가중치를 사용한 확산 모델 지원
        - 주파수 도메인에서 효율적인 전역 컨볼루션 구현
        - fno_block.py와 동일한 알고리즘, 시간 임베딩 통합
        
        알고리즘 단계:
        1. 입력 → 주파수 도메인 변환 (Real FFT)
        2. 주파수 모드별 시간 조건부 가중치 곱셈 (스펙트럼 컨볼루션)
        3. 주파수 도메인 → 출력 변환 (Inverse Real FFT)
        4. 바이어스 추가 (선택적)
        
        주파수 모드 처리:
        - 높은 주파수 모드만 유지 (저역 통과 필터 효과)
        - 대칭성 활용으로 계산 효율성 향상
        - 각 모드별 독립적 시간 조건부 가중치 학습
        
        시간 조건부 특징:
        - temb를 통한 동적 스펙트럼 가중치 조정
        - 확산 과정의 시간 스텝에 따른 적응적 주파수 응답
        - 각 주파수 영역별 독립적 시간 조건 처리

        Parameters
        ----------
        y : tuple
            (x, temb) 형태의 입력
            x : torch.Tensor
                입력 텐서 [batch_size, channels, d1, ..., dN]
            temb : torch.Tensor  
                시간 임베딩 [batch_size, temb_dim]
        indices : int, default is 0
            joint_factorization 시 레이어 인덱스

        Returns
        -------
        torch.Tensor
            스펙트럼 컨볼루션 결과 [batch_size, out_channels, d1, ..., dN]
        """
        # 입력 분리: 공간 데이터와 시간 임베딩
        x = y[0]        # 공간 데이터
        temb = y[1]     # 시간 임베딩

        # 입력 텐서 모양 분석
        batchsize, channels, *mode_sizes = x.shape
        fft_size = list(mode_sizes)
        
        # Real FFT의 마지막 차원 크기 조정 (대칭성 활용)
        fft_size[-1] = fft_size[-1]//2 + 1  # Redundant last coefficient

        # 1. 주파수 도메인 변환: 공간 → 스펙트럼
        fft_dims = list(range(-self.order, 0))  # 마지막 order개 차원에 대해 FFT
        x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)

        # 출력용 주파수 텐서 초기화 (복소수)
        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size], 
                             device=x.device, dtype=torch.cfloat)

        # 2. 스펙트럼 컨볼루션: 주파수 모드별 시간 조건부 가중치 곱셈
        # 모든 주파수 사분면(quadrant)에 대해 처리
        # 마지막 모드는 대칭성으로 인해 양의 주파수만 처리
        mode_indexing = [((None, m), (-m, None)) for m in self.half_modes[:-1]] + [((None, self.half_modes[-1]), )]

        for i, boundaries in enumerate(itertools.product(*mode_indexing)):
            # 각 주파수 영역에 대한 인덱스 튜플 구성
            # 배치와 채널 차원은 모두 유지: [slice(None), slice(None)]
            idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

            # 예시 (2D): [:, :, :height, :width], [:, :, -height:, :width] 등
            # 해당 주파수 영역에서 시간 조건부 스펙트럼 컨볼루션 수행
            out_fft[idx_tuple] = self.layer[i](x[idx_tuple], temb)

        # 3. 주파수 도메인 → 공간 도메인 변환
        x = torch.fft.irfftn(out_fft, s=(mode_sizes), norm=self.fft_norm)

        # 4. 바이어스 추가 (선택적, 디바이스 일관성 보장)
        if self.bias is not None:
            x = x + self.bias[indices, ...].to(x.device)

        return x

    def get_conv(self, indices):
        """
        Joint parametrization에서 서브 컨볼루션 레이어 반환
        
        역할:
        - fno_block.py와 동일한 구현
        - joint_factorization=True일 때 개별 레이어 접근
        - 파라미터 공유를 통한 메모리 효율성
        
        주의사항:
        - n_layers > 1일 때만 사용 가능
        - 단일 레이어인 경우 메인 클래스 직접 사용 권장
        """
        if self.n_layers == 1:
            raise ValueError('A single convolution is parametrized, directly use the main class.')

        return SubConv2d(self, indices)

    def __getitem__(self, indices):
        """
        인덱스 연산자 오버로드
        
        역할:
        - conv[i] 형태의 편리한 서브 레이어 접근
        - get_conv의 syntactic sugar
        """
        return self.get_conv(indices)



class SubConv2d(nn.Module):
    """
    Joint factorization에서 개별 컨볼루션을 나타내는 클래스
    
    역할:
    - fno_block.py의 SubConv2d와 동일한 구현
    - 메인 컨볼루션의 특정 인덱스에 해당하는 서브 레이어
    - 파라미터 공유를 통한 메모리 효율적 구현
    
    핵심 원리:
    - nn.Parameter는 중복되지 않음
    - 동일한 nn.Parameter가 여러 모듈에 할당되면 데이터 공유
    - 메인 컨볼루션과 서브 컨볼루션이 동일한 가중치 참조
    
    사용 컨텍스트:
    - joint_factorization=True인 FactorizedSpectralConv에서 사용
    - 여러 레이어를 단일 텐서로 매개화할 때 개별 접근 제공
    
    의존성:
    - 메인 FactorizedSpectralConv 인스턴스에 의존
    - 시간 임베딩 처리도 메인 클래스를 통해 수행
    """
    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv     # 메인 컨볼루션 참조
        self.indices = indices         # 서브 레이어 인덱스

    def forward(self, x, temb=None):
        """
        서브 컨볼루션 순방향 패스
        
        역할:
        - 메인 컨볼루션의 forward 메서드를 특정 인덱스로 호출
        - 시간 임베딩 처리 포함
        - 파라미터 공유를 통한 효율적 연산
        
        Parameters:
            x: 입력 텐서 (또는 (x, temb) 튜플)
            temb: 시간 임베딩 (선택적)
        
        Returns:
            서브 레이어의 스펙트럼 컨볼루션 결과
        """
        return self.main_conv.forward(x, self.indices)
