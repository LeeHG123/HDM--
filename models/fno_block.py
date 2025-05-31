# FNO (Fourier Neural Operator) 블록 모듈
# 과학적 컴퓨팅과 PDE 해결을 위한 신경망 연산자 구현
# 주파수 도메인에서 효율적인 컨볼루션 연산을 수행하는 핵심 구성 요소

"""
FNO 블록 모듈 개요
=================

이 모듈은 Fourier Neural Operator의 핵심 구성 요소들을 구현하며,
HDM(Hilbert Diffusion Model)과 결합하여 확산 기반 생성 모델을 지원합니다.

주요 구성 요소:
1. 가중치 초기화 (variance_scaling, default_init)
2. 위치 임베딩 (SinPositionEmbeddingsClass)
3. 텐서 수축 연산자들 (contract_dense, contract_cp, contract_tucker, contract_tt)
4. 메인 FNO 레이어 (FactorizedSpectralConv)

핵심 아이디어:
- 공간 도메인 대신 주파수 도메인에서 컨볼루션 수행
- 전역 수용장(global receptive field) 달성
- 텐서 인수분해를 통한 파라미터 효율성
- 시간 조건부 연산으로 확산 모델 지원

수학적 배경:
FNO는 다음과 같은 적분 연산자를 근사합니다:
(K(a)u)(x) = ∫ k(x,y,a(x),a(y)) u(y) dy

이를 주파수 도메인에서 효율적으로 구현:
F^(-1)[F[u] ⊙ R] 

여기서:
- F: Fourier 변환
- R: 학습 가능한 스펙트럼 가중치
- ⊙: 요소별 곱셈

텐서 인수분해 장점:
- CP 분해: 대칭 텐서에 효과적
- Tucker 분해: 고차원 텐서에 적합
- TT 분해: 초고차원 데이터용
- 파라미터 수 대폭 감소 및 일반화 성능 향상

확산 모델과의 통합:
- 시간 임베딩을 통한 노이즈 스케줄 조건부 연산
- 다중 해상도 주파수 처리
- 효율적인 역방향 샘플링 지원

의존성:
- TensorLy: 텐서 연산 및 인수분해
- TL-Torch: PyTorch 텐서 인수분해 확장
- functions/: 확산 과정 모듈들
- models/fno.py: FNO 아키텍처 구성

사용 컨텍스트:
- 1D/2D 과학 데이터 처리
- PDE 기반 물리 시뮬레이션
- 확산 기반 생성 모델링
- 고차원 함수 근사
"""

import itertools
import numpy as np

# TensorLy: 텐서 분해 및 다중선형 대수를 위한 라이브러리
import tensorly as tl
from tensorly.plugins import use_opt_einsum
tl.set_backend('pytorch')  # PyTorch 백엔드 사용
use_opt_einsum('optimal')  # einsum 최적화 활성화

# TL-Torch: TensorLy의 PyTorch 확장, 인수분해 텐서 지원
from tltorch.factorized_tensors.core import FactorizedTensor
einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Einstein 표기법용 심볼

import torch.nn as nn
import torch

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """
  분산 스케일링 기반 가중치 초기화 함수 (JAX에서 이식)
  
  역할:
  - 신경망 레이어의 가중치를 적절한 분산으로 초기화
  - Fan-in/Fan-out 기반으로 분산을 조절하여 그래디언트 소실/폭발 방지
  - 다양한 분포(정규분포, 균등분포)와 모드 지원
  
  수학적 배경:
  - Fan-in: 입력 뉴런 수 × receptive field 크기
  - Fan-out: 출력 뉴런 수 × receptive field 크기
  - 분산 = scale / denominator (fan_in, fan_out, 또는 평균)
  
  Parameters:
      scale: 스케일 팩터
      mode: 분산 계산 모드 ('fan_in', 'fan_out', 'fan_avg')
      distribution: 분포 타입 ('normal', 'uniform')
      in_axis: 입력 축 인덱스
      out_axis: 출력 축 인덱스
      dtype: 데이터 타입
      device: 디바이스
  
  Returns:
      init: 초기화 함수
  """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    """
    Fan-in과 Fan-out 계산
    
    Args:
        shape: 텐서 모양
        in_axis: 입력 채널 축
        out_axis: 출력 채널 축
    
    Returns:
        fan_in, fan_out: 입력/출력 팬 수
    """
    # receptive field 크기 = 전체 원소 수 / (입력 채널 × 출력 채널)
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    """
    실제 가중치 초기화 수행
    
    Args:
        shape: 초기화할 텐서 모양
        dtype: 데이터 타입
        device: 디바이스
    
    Returns:
        초기화된 텐서
    """
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    
    # 분산 계산을 위한 분모 결정
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
    
    # 분포에 따른 초기화
    if distribution == "normal":
      # 정규분포: N(0, variance)
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      # 균등분포: U(-sqrt(3*variance), sqrt(3*variance))
      # 균등분포의 분산이 variance가 되도록 범위 조정
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init


def default_init(scale=1.):
  """
  DDPM에서 사용하는 기본 초기화 방식
  
  역할:
  - 확산 모델에 최적화된 가중치 초기화
  - Fan 평균 기반 균등분포 초기화 사용
  - 0 스케일 방지를 위한 최소값 보장
  
  Parameters:
      scale: 초기화 스케일 (기본값: 1.0)
  
  Returns:
      variance_scaling 초기화 함수
  
  특징:
  - 확산 모델의 안정적인 학습을 위해 설계
  - 균등분포 사용으로 그래디언트 흐름 개선
  """
  # 0 스케일 방지: 최소 1e-10 보장
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')


# 위치 임베딩 (Position Embedding)
class SinPositionEmbeddingsClass(nn.Module):
    """
    사인/코사인 기반 위치 임베딩 클래스
    
    역할:
    - 확산 모델의 시간 스텝을 연속적인 임베딩 벡터로 변환
    - Transformer의 위치 인코딩을 시간 도메인에 적용
    - 시간 조건부 생성을 위한 시간 정보 인코딩
    
    수학적 원리:
    - PE(t, 2i) = sin(t / 10000^(2i/d))
    - PE(t, 2i+1) = cos(t / 10000^(2i/d))
    - 여기서 t는 시간 스텝, i는 차원 인덱스, d는 임베딩 차원
    
    특징:
    - 주기적 패턴으로 시간 간격 관계 학습 가능
    - 고정된 함수로 학습 불필요 (@torch.no_grad())
    - 다양한 주파수 성분으로 시간 정보 풍부하게 표현
    
    의존성:
    - FNO 블록에서 시간 조건부 연산에 사용
    - 확산 모델의 노이즈 스케줄링과 연동
    """
    def __init__(self,dim=256,T=10000):
        """
        Parameters:
            dim: 임베딩 차원 (기본값: 256)
            T: 주기 스케일링 팩터 (기본값: 10000)
        """
        super().__init__()
        self.dim = dim  # 임베딩 벡터 차원
        self.T = T      # 주파수 스케일링 상수

    @torch.no_grad()
    def forward(self,steps):
        """
        시간 스텝을 사인/코사인 임베딩으로 변환
        
        Parameters:
            steps: 시간 스텝 텐서 [batch_size]
        
        Returns:
            embeddings: 위치 임베딩 [batch_size, dim]
        
        구현 과정:
        1. 시간 스텝을 1000배 스케일링 (확산 모델 관례)
        2. 각 차원별 주파수 계산
        3. 시간과 주파수의 곱으로 위상 계산
        4. 사인/코사인 함수 적용하여 임베딩 생성
        """
        # 시간 스텝 스케일링 (확산 모델에서 일반적으로 사용)
        steps = 1000*steps
        device = steps.device
        
        # 임베딩 차원의 절반까지만 계산 (sin/cos 쌍으로 사용)
        half_dim = self.dim // 2
        
        # 주파수 계산: log(T) / (half_dim - 1)
        embeddings = np.log(self.T) / (half_dim - 1)
        
        # 지수 함수로 주파수 스케일 생성: exp(-i * log(T) / (half_dim - 1))
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # 시간 스텝과 주파수의 외적으로 위상 계산
        embeddings = steps[:, None] * embeddings[None, :]
        
        # 사인/코사인 함수 적용하여 최종 임베딩 생성
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class contract_dense(nn.Module):
    """
    Dense 텐서를 사용한 스펙트럼 컨볼루션 수축 연산
    
    역할:
    - 주파수 도메인에서 입력과 가중치 텐서 간의 계약(contraction) 수행
    - 시간 임베딩을 통한 조건부 연산 지원
    - Separable/Non-separable 모드 지원으로 유연한 연산 구조 제공
    
    수학적 배경:
    - Einstein summation을 사용한 텐서 수축
    - Non-separable: 입력-출력 채널 간 완전 연결
    - Separable: 채널별 독립적 연산 (파라미터 효율성)
    
    구현 특징:
    - 시간 조건부 가중치 조정 (temb 사용)
    - 동적 einsum 수식 생성으로 차원 무관 연산
    - TensorLy 백엔드 활용한 최적화된 텐서 연산
    
    의존성:
    - FactorizedSpectralConv에서 가중치 수축 연산으로 사용
    - 시간 임베딩과 결합하여 확산 모델에 적용
    """
    def __init__(self, weight, hidden, act,temb_dim = None, separable=False):
       """
       Parameters:
           weight: 가중치 텐서 (dense 또는 factorized)
           hidden: 은닉 차원 크기
           act: 활성화 함수
           temb_dim: 시간 임베딩 차원
           separable: separable 모드 여부
       """
       super().__init__()
       self.weight= weight          # 스펙트럼 컨볼루션 가중치
       self.separable =separable    # separable 연산 모드
       self.act =act               # 활성화 함수

       # 시간 임베딩을 위한 선형 레이어
       self.Dense_0 = nn.Linear(temb_dim, hidden)
       self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
       nn.init.zeros_(self.Dense_0.bias)
       self.separable = separable

    def forward(self,x,temb=None):
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
      3. 시간 임베딩 조건부 입력 조정
      4. TensorLy einsum으로 효율적 텐서 수축 수행
      """
      order = tl.ndim(x)  # 입력 텐서 차원 수
      
      # 입력 텐서용 Einstein 인덱스: batch-size, in_channels, x, y...
      x_syms = list(einsum_symbols[:order])

      # 가중치 텐서용 인덱스: in_channels, out_channels, x, y... (배치 차원 제외)
      weight_syms = list(x_syms[1:]) # no batch-size

      # 출력 텐서용 인덱스 구성
      if self.separable:
          # Separable 모드: 입력과 동일한 채널 구조 유지
          out_syms = [x_syms[0]] + list(weight_syms)
      else:
        # Non-separable 모드: 새로운 출력 채널 차원 추가
        weight_syms.insert(1, einsum_symbols[order]) # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]  # 배치 차원 유지

      # Einstein summation 수식 구성
      eq= ''.join(x_syms) + ',' + ''.join(weight_syms) + '->' + ''.join(out_syms)

      # Factorized 텐서인 경우 dense 텐서로 복원
      if not torch.is_tensor(self.weight):
        weight = self.weight.to_tensor()

      # 시간 조건부 입력 조정
      if temb is not None:
          # 시간 임베딩을 선형 변환하여 입력에 더함
          x+=self.Dense_0.to('cuda')(self.act.to('cuda')(temb))[:,:,None]

      # TensorLy einsum으로 최종 텐서 수축 수행
      return tl.einsum(eq, x, self.weight)

class contract_dense_separable(nn.Module):
  def __init__(self, weight,  hidden, act, temb_dim = None, separable=False):
    super().__init__()
    self.weight = weight
    self.act = act
    self.Dense_0 = nn.Linear(weight.shape[0], hidden)
    self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
    nn.init.zeros_(self.Dense_0.bias)
  def forward(self,x,temb=None):
    if self.separable == False:
        raise ValueError('This function is only for separable=True')
    if temb is not None:
        x+=self.Dense_0(self.act(temb))[:,None]
    h=x*self.weight
    return h


class contract_cp(nn.Module):
  def __init__(self, cp_weight,  hidden, act,temb_dim = None, separable=False):
      super().__init__()
      self.cp_weight= cp_weight

      self.Dense_0 = nn.Linear(temb_dim, hidden)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
      self.separable = separable
      self.act =act

  def forward(self, x,temb=None):
    order = tl.dim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order+1]
    out_syms = list(x_syms)
    if self.separable:
        factor_syms = [einsum_symbols[1]+rank_sym] #in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+rank_sym,out_sym+rank_sym] #in, out
    factor_syms += [xs+rank_sym for xs in x_syms[2:]] #x, y, ...
    eq = x_syms + ',' + rank_sym + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)
    if temb is not None:
        x+=self.Dense_0(self.act(temb))[:,None]
    h= tl.einsum(eq, x, self.cp_weight.weights, *self.cp_weight.factors)
    return h


class contract_tucker_1d(nn.Module):
  def __init__(self, tucker_weight,  hidden, act,temb_dim = None, separable=False):
    super().__init__()
    self.tucker_weight =tucker_weight
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(tucker_weight.shape[1], hidden)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
    self.separable = separable
    self.act =act

  def forward(self,x,temb=None):
    order = tl.ndim(x)
    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)

    if self.separable:
        core_syms = einsum_symbols[order+1:2*order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        factor_syms = [xs+rs for (xs, rs) in zip(x_syms[1:], core_syms)] #x, y, ...

    else:
        core_syms = einsum_symbols[order+1:2*order+1]
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+core_syms[0], out_sym+core_syms[1]] #out, in
        factor_syms += [xs+rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])] #x, y, ...

    eq = x_syms + ',' + core_syms + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)
    if temb is not None:
        x+=self.Dense_0(self.act(temb))[:,None]
    h= tl.einsum(eq, x, self.tucker_weight.core, *self.tucker_weight.factors)
    return h

class contract_tucker_2d(nn.Module):
  def __init__(self, tucker_weight,  hidden, act,temb_dim = None, separable=False):
    super().__init__()
    self.tucker_weight =tucker_weight
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(tucker_weight.shape[1], hidden)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
    self.separable = separable
    self.act =act

  def forward(self,x,temb=None):
    order = tl.ndim(x)
    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)

    if self.separable:
        core_syms = einsum_symbols[order+1:2*order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        factor_syms = [xs+rs for (xs, rs) in zip(x_syms[1:], core_syms)] #x, y, ...

    else:
        core_syms = einsum_symbols[order+1:2*order+1]
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+core_syms[0], out_sym+core_syms[1]] #out, in
        factor_syms += [xs+rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])] #x, y, ...

    eq = x_syms + ',' + core_syms + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)
    if temb is not None:
        x=x+self.Dense_0(self.act(temb))[:,:,None,None]
    h= tl.einsum(eq, x, self.tucker_weight.core, *self.tucker_weight.factors)
    return h


class contract_tt(nn.Module):

  def __init__(self, tt_weight,  hidden, act,temb_dim = None, separable=False):
    super().__init__()
    self.tt_weight=tt_weight
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(tt_weight.shape[0], hidden)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
      self.act =act
      self.separable = separable

  def forward(self,x,temb=None):
    order = tl.nladim(x)
    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:]) # no batch-size
    if not self.separable:
        weight_syms.insert(1, einsum_symbols[order]) # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order+1:])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i+1]])
    eq = ''.join(x_syms) + ',' + ','.join(''.join(f) for f in tt_syms) + '->' + ''.join(out_syms)
    if temb is not None:
        x+=self.Dense_0(self.act(temb))[:,None]
    return tl.einsum(eq, x, *self.tt_weight.factors)

def get_contract_fun(implementation, weight,  hidden, act,temb_dim = None, separable=False, is_2d=False):
    """
    Fourier 스펙트럼 컨볼루션 수축 함수 팩토리
    
    역할:
    - 가중치 타입과 구현 방식에 따라 적절한 수축 함수 반환
    - 다양한 텐서 인수분해 방법 지원 (Dense, CP, Tucker, TT)
    - 1D/2D 데이터와 separable/non-separable 모드 지원
    
    구현 방식:
    1. Reconstructed: 인수분해된 텐서를 복원 후 일반 컨볼루션
    2. Factorized: 인수분해 인수들과 직접 수축 (메모리 효율적)
    
    텐서 인수분해 지원:
    - Dense: 일반 텐서 (인수분해 없음)
    - CP: CANDECOMP/PARAFAC 분해
    - Tucker: Tucker 분해 (고차원에 효과적)
    - TT: Tensor Train 분해 (초고차원용)
    
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
    is_2d : bool, default is False
        2D 데이터 여부 (Tucker 분해 시 사용)

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
            print('SEPARABLE')
            return contract_dense_separable
        else:
            return contract_dense
    elif implementation == 'factorized':
        if torch.is_tensor(weight):
            return contract_dense(weight,  hidden, act,temb_dim = temb_dim, separable=separable)
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower() == 'complexdense':
                return contract_dense(weight, hidden, act,temb_dim = temb_dim, separable=separable)
            elif weight.name.lower() == 'complextucker':
                if is_2d:
                   return contract_tucker_2d(weight,  hidden, act,temb_dim = temb_dim, separable=separable)
                else:
                    return contract_tucker_1d(weight,  hidden, act,temb_dim = temb_dim, separable=separable)
            elif weight.name.lower() == 'complextt':
                return contract_tt(weight,  hidden, act,temb_dim = temb_dim, separable=separable)
            elif weight.name.lower() == 'complexcp':
                return contract_cp(weight,  hidden, act,temb_dim = temb_dim, separable=separable)
            else:
                raise ValueError(f'Got unexpected factorized weight type {weight.name}')
        else:
            raise ValueError(f'Got unexpected weight type of class {weight.__class__.__name__}')
    else:
        raise ValueError(f'Got {implementation=}, expected "reconstructed" or "factorized"')


class FactorizedSpectralConv(nn.Module):
    """
    인수분해된 스펙트럼 컨볼루션 레이어 - FNO의 핵심 구성 요소
    
    역할:
    - N차원 Fourier Neural Operator의 범용 구현
    - 주파수 도메인에서 효율적인 스펙트럼 컨볼루션 수행
    - 텐서 인수분해를 통한 파라미터 압축 및 계산 효율성 제공
    - 확산 모델과 결합하여 시간 조건부 연산 지원
    
    핵심 아이디어:
    1. 공간 도메인 → 주파수 도메인 변환 (FFT)
    2. 주파수 도메인에서 학습 가능한 가중치와 곱셈
    3. 주파수 도메인 → 공간 도메인 변환 (IFFT)
    4. 전역 수용장(global receptive field) 달성
    
    수학적 배경:
    - F^(-1)[F[u] ⊙ R] 형태의 스펙트럼 컨볼루션
    - 여기서 F는 FFT, R은 학습 가능한 스펙트럼 가중치
    - 높은 모드만 유지하여 계산 효율성과 정규화 효과
    
    텐서 인수분해 장점:
    - CP/Tucker/TT 분해로 파라미터 수 대폭 감소
    - 메모리 효율성 및 일반화 성능 향상
    - 다차원 데이터에서 특히 효과적
    
    시간 조건부 특징:
    - 확산 모델의 시간 스텝에 따른 적응적 연산
    - 시간 임베딩을 통한 동적 가중치 조정
    - 노이즈 스케줄에 맞는 주파수 응답 학습
    
    의존성:
    - models/fno.py에서 FNO 아키텍처 구성 요소로 사용
    - functions/의 확산 과정과 연동
    - TensorLy/TL-Torch 기반 텐서 연산 활용

    Parameters
    ----------
    in_channels : int
        입력 채널 수
    out_channels : int  
        출력 채널 수
    n_modes : int 또는 tuple
        각 차원별로 유지할 Fourier 모드 수
    separable : bool, default is False
        separable 컨볼루션 사용 여부 (파라미터 효율성)
    scale : float or 'auto', default is 'auto'
        가중치 초기화 스케일
    n_layers : int, default is 1
        Fourier 레이어 수
    joint_factorization : bool, default is False
        모든 레이어를 단일 텐서로 매개화할지 여부
    rank : float, default is 0.5
        텐서 인수분해의 랭크 (압축 정도)
    factorization : str, default is 'cp'
        텐서 인수분해 방법 ('tucker', 'cp', 'tt')
    fixed_rank_modes : bool, default is False
        인수분해하지 않을 모드
    fft_norm : str, default is 'backward'
        FFT 정규화 방법
    implementation : str, default is 'reconstructed'
        순방향 패스 구현 방식 ('factorized' 또는 'reconstructed')
    decomposition_kwargs : dict, default is {}
        텐서 분해 추가 매개변수
    indices : int, default is 0
        레이어 인덱스 (joint factorization 시 사용)
    """
    def __init__(self, in_channels, out_channels, n_modes, n_layers=1, scale='auto', separable=False,
                 fft_norm='backward', bias=True, implementation='reconstructed', joint_factorization=False,
                 rank=0.5, factorization='cp', fixed_rank_modes=False, decomposition_kwargs=dict(),indices=0):
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

        # self.mlp = None


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
            self.weight = FactorizedTensor.new(((2**(self.order-1))*n_layers, *weight_shape),
                                                rank=self.rank, factorization=factorization,
                                                fixed_rank_modes=fixed_rank_modes,
                                                **decomposition_kwargs).cuda()
            self.weight.normal_(0, scale)
        else:
            self.weight = nn.ModuleList([
                 FactorizedTensor.new(
                    weight_shape,
                    rank=self.rank, factorization=factorization,
                    fixed_rank_modes=fixed_rank_modes,
                    **decomposition_kwargs
                    ).cuda() for _ in range((2**(self.order-1))*n_layers)]
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
            if len(n_modes) == 1: 
                self.layer.append(get_contract_fun(implementation, self.weight[indices + i], hidden=out_channels, temb_dim=256, act=nn.SiLU(), separable=self.separable, is_2d=False))
            elif len(n_modes) == 2:
                self.layer.append(get_contract_fun(implementation, self.weight[indices + i], hidden=out_channels, temb_dim=256, act=nn.SiLU(), separable=self.separable, is_2d=True))
            else:
               raise NotImplementedError(f'length of n_modes should be either 1 or 2, but got {len(n_modes)}')
        self.layer =nn.ModuleList(self.layer)

    def forward(self, y, indices=0):
        """
        인수분해된 스펙트럼 컨볼루션의 순방향 패스
        
        역할:
        - FNO의 핵심 연산인 스펙트럼 컨볼루션 수행
        - 주파수 도메인에서 효율적인 전역 컨볼루션 구현
        - 시간 조건부 가중치를 사용한 확산 모델 지원
        
        알고리즘 단계:
        1. 입력 → 주파수 도메인 변환 (Real FFT)
        2. 주파수 모드별 가중치 곱셈 (스펙트럼 컨볼루션)
        3. 주파수 도메인 → 출력 변환 (Inverse Real FFT)
        4. 바이어스 추가 (선택적)
        
        주파수 모드 처리:
        - 높은 주파수 모드만 유지 (저역 통과 필터 효과)
        - 대칭성 활용으로 계산 효율성 향상
        - 각 모드별 독립적 가중치 학습
        
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
        # 입력 분리 및 GPU 이동
        x = y[0].to('cuda')      # 공간 데이터
        temb = y[1].to('cuda')   # 시간 임베딩

        # 입력 텐서 모양 분석
        batchsize, channels, *mode_sizes = x.shape
        fft_size = list(mode_sizes)
        
        # Real FFT의 마지막 차원 크기 조정 (대칭성 활용)
        fft_size[-1] = fft_size[-1]//2 + 1 # Redundant last coefficient

        # 1. 주파수 도메인 변환: 공간 → 스펙트럼
        fft_dims = list(range(-self.order, 0))  # 마지막 order개 차원에 대해 FFT
        x = torch.fft.rfftn(x.float(), norm=self.fft_norm, dim=fft_dims)

        # 출력용 주파수 텐서 초기화 (복소수)
        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size], 
                             device=x.device, dtype=torch.cfloat)

        # 2. 스펙트럼 컨볼루션: 주파수 모드별 가중치 곱셈
        # 모든 주파수 사분면(quadrant)에 대해 처리
        # 마지막 모드는 대칭성으로 인해 양의 주파수만 처리
        mode_indexing = [((None, m), (-m, None)) for m in self.half_modes[:-1]] + [((None, self.half_modes[-1]), )]

        for i, boundaries in enumerate(itertools.product(*mode_indexing)):
            # 각 주파수 영역에 대한 인덱스 튜플 구성
            # 배치와 채널 차원은 모두 유지: [slice(None), slice(None)]
            idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

            # 예시 (2D): [:, :, :height, :width], [:, :, -height:, :width] 등
            # 해당 주파수 영역에서 스펙트럼 컨볼루션 수행
            out_fft[idx_tuple] = self.layer[i](x[idx_tuple], temb)
        
        # 3. 주파수 도메인 → 공간 도메인 변환
        x = torch.fft.irfftn(out_fft, s=(mode_sizes), norm=self.fft_norm)

        # 4. 바이어스 추가 (선택적)
        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError('A single convolution is parametrized, directly use the main class.')

        return SubConv2d(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)



class SubConv2d(nn.Module):
    """Class representing one of the convolutions from the mother joint factorized convolution

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data,
    which is shared.
    """
    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices

    def forward(self, x,temb=None):
        return self.main_conv.forward(x, self.indices)
