
# Temporal FNO 모듈 - 시간 조건부 Fourier Neural Operator 아키텍처
# NVIDIA Corporation의 저작권 기반으로 확산 모델에 특화된 FNO 구현
# HDM(Hilbert Diffusion Model)의 메인 백본 네트워크

"""
Temporal FNO 모듈 개요
====================

이 모듈은 확산 모델을 위한 시간 조건부 Fourier Neural Operator를 구현합니다.
기본 FNO 아키텍처에 시간 임베딩과 AFNO(Adaptive Fourier Neural Operator) 
기능을 추가하여 확산 과정에서 시간에 따른 적응적 처리를 지원합니다.

주요 구성 요소:
1. AFNO2D: 적응적 2D Fourier 연산자 (주파수 선택적 처리)
2. FNO: 기본 시간 조건부 FNO 아키텍처
3. FNO2: 설정 기반 개선된 FNO (AFNO 통합)
4. 시간 임베딩 함수들
5. Lifting/Projection 레이어들

핵심 특징:
- 시간 조건부 확산 모델 지원
- AFNO를 통한 적응적 주파수 처리
- 텐서 인수분해를 통한 파라미터 효율성
- 다양한 정규화 및 스킵 연결 옵션
- 도메인 패딩 지원

아키텍처 흐름:
입력 → Lifting → [시간 임베딩 주입] → FNO 블록들 → Projection → 출력
                    ↓
              AFNO + 스펙트럼 컨볼루션
                    ↓
              MLP + 스킵 연결

수학적 배경:
- 시간 조건부 스펙트럼 컨볼루션: F^(-1)[F[u(t)] ⊙ R(t)]
- AFNO: 적응적 주파수 선택 및 희소성 제어
- 확산 과정: ε_θ(x_t, t) 예측

의존성:
- models/fno_block.py: 기본 FNO 구성 요소
- models/temp_fno_block.py: 시간 최적화된 블록들 (FNO2에서 사용)
- models/mlp.py: MLP 및 스킵 연결
- models/padding.py: 도메인 패딩
- functions/: 확산 과정 모듈들

사용 컨텍스트:
- HDM의 메인 노이즈 예측 네트워크
- 시간 조건부 이미지/데이터 생성
- 과학적 컴퓨팅 및 PDE 해결
- 고해상도 데이터 처리
"""

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

from functools import partial, partialmethod

import math
from models.fno_block import *      # 기본 FNO 구성 요소
from models.padding import *        # 도메인 패딩 모듈
from models.mlp import *           # MLP 및 스킵 연결

import torch
import torch.fft                   # FFT 연산
import torch.nn as nn
import torch.nn.functional as F

class AFNO2D(nn.Module):
    """
    Adaptive Fourier Neural Operator 2D - 적응적 2D Fourier 신경망 연산자
    
    역할:
    - 2D 데이터를 위한 적응적 주파수 도메인 처리
    - 블록 대각 구조를 통한 효율적인 파라미터 사용
    - 희소성 제어를 통한 주파수 선택적 처리
    - 하드 임계값을 통한 계산량 제어
    
    핵심 특징:
    1. 블록 대각 구조: 채널을 블록으로 나누어 독립적 처리
    2. 주파수 마스킹: 중요하지 않은 고주파 성분 제거
    3. 소프트 임계값: 희소성 유도를 위한 정규화
    4. 복소수 처리: 실수/허수부 분리 연산
    
    수학적 배경:
    - 2D FFT: F(x) = FFT2D(x)
    - 블록별 MLP: y = MLP_block(F(x))
    - 소프트 임계값: SoftShrink(y, λ)
    - 역변환: IFFT2D(y)
    
    장점:
    - 전역 수용장을 가지면서 O(N log N) 복잡도
    - 적응적 주파수 선택으로 효율성
    - 블록 구조로 파라미터 수 조절 가능
    
    Parameters:
        hidden_size: 채널 차원 크기
        num_blocks: 블록 대각 가중치 행렬의 블록 수 
                   (높을수록 복잡도 감소, 파라미터 감소)
        sparsity_threshold: softshrink를 위한 λ 값
        hard_thresholding_fraction: 완전히 마스크할 주파수 비율
                                  (낮을수록 연산량 제곱에 비례하여 감소)
    """
    def __init__(self, config):
        """
        AFNO2D 초기화
        
        Parameters:
            config: 설정 객체 (config.model 하위 속성 사용)
        """
        super().__init__()
        
        # 설정에서 하이퍼파라미터 추출
        self.hidden_size = config.model.hidden_size                    # 채널 차원
        self.sparsity_threshold = config.model.sparsity_threshold      # 희소성 임계값
        self.num_blocks = config.model.num_blocks                      # 블록 수
        self.block_size = self.hidden_size // self.num_blocks          # 블록당 채널 수
        self.hard_thresholding_fraction = config.model.hard_thresholding_fraction  # 하드 임계값 비율
        self.hidden_size_factor = config.model.hidden_size_factor      # MLP 확장 팩터
        self.scale = 0.02                                              # 가중치 초기화 스케일

        # 복소수 처리를 위한 2세트 가중치 (실수부/허수부)
        # w1: 첫 번째 MLP 레이어 가중치 [실수/허수, 블록, 입력, 출력]
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, 
                                   self.block_size * self.hidden_size_factor)
        ).to('cuda')
        
        # b1: 첫 번째 MLP 레이어 바이어스
        self.b1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, 
                                   self.block_size * self.hidden_size_factor)
        ).to('cuda')
        
        # w2: 두 번째 MLP 레이어 가중치 (역방향)
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, 
                                   self.block_size * self.hidden_size_factor, 
                                   self.block_size)
        ).to('cuda')
        
        # b2: 두 번째 MLP 레이어 바이어스
        self.b2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size)
        ).to('cuda')

    def forward(self, x):
        """
        AFNO2D 순방향 패스 - 적응적 2D Fourier 변환 및 블록 MLP 처리
        
        역할:
        - 2D 입력에 대해 적응적 주파수 도메인 처리 수행
        - 블록 대각 MLP를 통한 효율적 변환
        - 하드/소프트 임계값을 통한 주파수 선택 및 희소성 제어
        
        알고리즘 흐름:
        1. 2D FFT로 주파수 도메인 변환
        2. 채널을 블록으로 재구성
        3. 하드 임계값으로 주파수 마스킹
        4. 블록별 2층 MLP 적용 (복소수 연산)
        5. 소프트 임계값으로 희소성 유도
        6. 역 FFT로 공간 도메인 복원
        7. 잔차 연결
        
        Parameters:
            x: 입력 텐서 [batch, channels, height, width]
        
        Returns:
            처리된 출력 [batch, channels, height, width]
        """
        # 잔차 연결을 위한 원본 저장
        bias = x

        # 데이터 타입 보존 (정확도 위해 float로 계산)
        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        N = H * W
        
        # 1. 2D FFT 준비: [B, C, H, W] → [B, H, W, C]
        x = x.reshape(B, H, W, C)
        
        # 2. 2D FFT 변환 (주파수 도메인으로 이동)
        x = torch.fft.fft2(x, dim=(1, 2), norm="ortho")
        
        # 3. 블록 구조로 재구성: [B, H, W, num_blocks, block_size]
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        # 4. 출력 텐서 초기화 (실수부/허수부 분리)
        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, 
                              self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, 
                              self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # 5. 하드 임계값: 유지할 주파수 모드 계산
        total_modes = N // 2 + 1  # Real FFT 특성상 절반 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # 6. 첫 번째 MLP 레이어 (확장): 복소수 곱셈 구현
        # 복소수 곱셈: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        # 7. 두 번째 MLP 레이어 (축소): 원래 차원으로 복원
        o2_real[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        # 8. 복소수 재구성 및 소프트 임계값 적용
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)  # 희소성 유도
        x = torch.view_as_complex(x)
        
        # 9. 원래 채널 구조로 복원
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        
        # 10. 역 2D FFT (주파수 → 공간 도메인)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        
        # 11. 원래 형태로 재구성 및 데이터 타입 복원
        x = x.reshape(B, C, H, W)
        x = x.type(dtype)
        
        # 12. 잔차 연결
        return x + bias



def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """
  분산 스케일링 기반 가중치 초기화 함수 (JAX에서 이식)
  
  역할:
  - Temporal FNO의 안정적인 학습을 위한 가중치 초기화
  - fno.py, fno_block.py와 동일한 함수 (코드 중복)
  - 확산 모델에 최적화된 초기화 제공
  
  참고:
  - 모든 FNO 관련 모듈에서 일관된 초기화 보장
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
  - Temporal FNO의 시간 임베딩 레이어 초기화
  """
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """
    확산 모델용 시간 스텝 임베딩 생성 (메인 버전)
    
    역할:
    - fno.py의 get_timestep_embedding과 동일한 구현
    - 확산 과정의 시간 t를 연속적인 벡터로 인코딩
    - Transformer의 위치 인코딩을 시간 도메인에 적용
    
    구현 특징:
    - 1000배 스케일링으로 확산 모델 관례 따름
    - 사인/코사인 주기 함수로 시간 관계 표현
    - max_positions=10000 (Transformer 표준)
    
    수학적 공식:
    - PE(t, 2i) = sin(t / 10000^(2i/d))
    - PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """
    assert len(timesteps.shape) == 1  # 1D 시간 스텝 배열 필요
    half_dim = embedding_dim // 2
    timesteps = 1000 * timesteps  # 확산 모델 스케일링

    # 주파수 스케일 생성
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None].to(timesteps.device) * emb[None, :].to(timesteps.device)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    # 홀수 차원일 경우 제로 패딩
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def get_timestep_embedding2(timesteps, embedding_dim, max_positions=111):
    """
    조건부 확산 모델용 시간 스텝 임베딩 (보조 버전)
    
    역할:
    - fno.py의 get_timestep_embedding_2와 동일한 구현
    - 클래스 조건부 생성을 위한 추가 임베딩
    - 기본 시간 임베딩과 결합하여 다중 조건 지원
    
    차이점:
    - max_positions=111 (기본 10000과 다름)
    - 1000배 스케일링 없음 (원본 시간 사용)
    - 클래스 레이블이나 추가 조건 인코딩용
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2

    # 다른 주파수 스케일 사용
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None].to(timesteps.device) * emb[None, :].to(timesteps.device)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Lifting(nn.Module):
    """
    FNO의 입력 차원 변환 레이어 (Lifting Layer)
    
    역할:
    - fno.py의 Lifting 클래스와 동일한 구현
    - 저차원 입력을 고차원 잠재 공간으로 변환
    - 후속 Fourier 레이어가 작업할 충분한 표현력 제공
    
    수학적 의미:
    - P: R^d → R^h (d-차원 입력을 h-차원으로 lifting)
    - 일반적으로 h >> d (예: 3 → 256)
    
    구현 특징:
    - N차원 데이터 지원 (1D, 2D 등)
    - 1×1 컨볼루션 사용으로 공간 구조 보존
    """
    def __init__(self, in_channels, out_channels, n_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 동적으로 적절한 컨볼루션 레이어 선택
        Conv = getattr(nn, f'Conv{n_dim}d')  # Conv1d 또는 Conv2d
        self.fc = Conv(in_channels, out_channels, 1)  # 1×1 컨볼루션

    def forward(self, x):
        """
        입력을 고차원 잠재 공간으로 변환
        
        Args:
            x: 입력 텐서 [batch, in_channels, spatial_dims...]
        
        Returns:
            변환된 텐서 [batch, out_channels, spatial_dims...]
        """
        return self.fc(x)


class Projection(nn.Module):
    """
    FNO의 출력 차원 변환 레이어 (Projection Layer)
    
    역할:
    - fno.py의 Projection 클래스와 동일한 구현
    - 고차원 잠재 표현을 최종 출력 차원으로 변환
    - 2단계 MLP 구조로 비선형 변환 제공
    
    수학적 의미:
    - Q: R^h → R^o (h-차원 잠재에서 o-차원 출력으로 projection)
    - 비선형 변환: Q(x) = W₂·σ(W₁·x)
    
    구현 특징:
    - 2층 MLP 구조 (fc1 → activation → fc2)
    - 1×1 컨볼루션으로 공간 구조 보존
    """
    def __init__(self, in_channels, out_channels, hidden_channels=None, n_dim=2, non_linearity=F.gelu):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.non_linearity = non_linearity
        
        # 동적 컨볼루션 레이어 선택
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc1 = Conv(in_channels, hidden_channels, 1)      # 첫 번째 변환
        self.fc2 = Conv(hidden_channels, out_channels, 1)     # 두 번째 변환

    def forward(self, x):
        """
        고차원 잠재 표현을 최종 출력으로 변환
        
        Args:
            x: 잠재 표현 [batch, in_channels, spatial_dims...]
        
        Returns:
            최종 출력 [batch, out_channels, spatial_dims...]
        """
        x = self.fc1(x)                    # 중간 차원으로 변환
        x = self.non_linearity(x)          # 비선형 활성화
        x = self.fc2(x)                    # 최종 출력 차원으로 변환
        return x

class FNO(nn.Module):
    """
    Temporal FNO - N차원 시간 조건부 Fourier Neural Operator (기본 버전)
    
    역할:
    - fno.py의 FNO 클래스와 거의 동일한 구현
    - 시간 조건부 확산 모델을 위한 Fourier 기반 백본 네트워크
    - 1D 데이터 특화 처리 ([:,:,None] 브로드캐스팅)
    - 기본적인 시간 임베딩 통합과 스펙트럼 컨볼루션
    
    주요 특징:
    1. 시간 조건부: 확산 과정의 시간 스텝에 따른 적응적 처리
    2. 스펙트럼 컨볼루션: 주파수 도메인에서 전역 수용장 달성
    3. 텐서 인수분해: 파라미터 효율성과 일반화 성능 향상
    4. 유연한 아키텍처: 다양한 정규화, 스킵 연결, MLP 옵션
    5. 도메인 패딩: 경계 조건 처리 지원
    
    vs FNO2 차이점:
    - 기본 구현: 단순하고 안정적
    - 1D 특화: [:,:,None] 브로드캐스팅
    - AFNO 미포함: 기본 스펙트럼 컨볼루션만 사용
    - 설정 방식: 직접 파라미터 전달
    
    아키텍처 흐름:
    입력 → Lifting → [시간 임베딩] → FNO 블록들 → Projection → 출력
           ↓                          ↓
        차원 확장              스펙트럼 컨볼루션 + MLP
    
    수학적 배경:
    - 시간 조건부 연산자: T(u, t) → v
    - 스펙트럼 컨볼루션: F^(-1)[F[u] ⊙ R(t)]
    - 시간 임베딩: t → embedding → 네트워크 전반에 주입
    
    사용 컨텍스트:
    - 기본적인 시간 조건부 작업
    - 1D 시계열 데이터 처리
    - 안정적인 확산 모델 백본
    - 프로토타입 및 실험용

    Parameters
    ----------
    n_modes : int tuple
        각 차원별로 유지할 Fourier 모드 수
        TFNO의 차원은 len(n_modes)로 추론됨
    hidden_channels : int
        FNO의 폭 (채널 수)
    in_channels : int, optional
        입력 채널 수, 기본값 3
    out_channels : int, optional
        출력 채널 수, 기본값 1
    lifting_channels : int, optional
        Lifting 블록의 은닉 채널 수, 기본값 256
    projection_channels : int, optional
        Projection 블록의 은닉 채널 수, 기본값 256
    n_layers : int, optional
        Fourier 레이어 수, 기본값 4
    use_mlp : bool, optional
        각 FNO 블록 후 MLP 레이어 사용 여부, 기본값 False
    mlp : dict, optional
        MLP 파라미터, 기본값 None
        {'expansion': float, 'dropout': float}
    non_linearity : nn.Module, optional
        사용할 비선형 활성화, 기본값 F.gelu
    norm : F.module, optional
        사용할 정규화 레이어, 기본값 None
    preactivation : bool, default is False
        True일 경우 ResNet 스타일 사전 활성화 사용
    skip : {'linear', 'identity', 'soft-gating'}, optional
        스킵 연결 타입, 기본값 'soft-gating'
    separable : bool, default is False
        True일 경우 깊이별 분리 가능한 스펙트럼 컨볼루션 사용
    factorization : str or None, {'tucker', 'cp', 'tt'}
        파라미터 가중치에 사용할 텐서 인수분해, 기본값 None
        * None: dense 텐서로 스펙트럼 컨볼루션 매개화
        * 그 외: 지정된 텐서 인수분해 사용
    joint_factorization : bool, optional
        모든 Fourier 레이어를 단일 텐서로 매개화할지 여부, 기본값 False
    rank : float or rank, optional
        Fourier 가중치의 텐서 인수분해 랭크, 기본값 1.0
    fixed_rank_modes : bool, optional
        인수분해하지 않을 모드, 기본값 False
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        인수분해가 None이 아닐 경우 사용할 순방향 모드:
        * 'reconstructed': 인수분해에서 전체 가중치 텐서를 복원하여 순방향 패스에 사용
        * 'factorized': 분해 인수들과 직접 입력을 수축
    decomposition_kwargs : dict, optional, default is {}
        텐서 분해에 전달할 추가 매개변수 (선택적)
    domain_padding : None or float, optional
        None이 아닐 경우 사용할 패딩 비율, 기본값 None
    domain_padding_mode : {'symmetric', 'one-sided'}, optional
        도메인 패딩 수행 방법, 기본값 'one-sided'
    fft_norm : str, optional
        FFT 정규화 방법, 기본값 'forward'
    """
    def __init__(self, n_modes, hidden_channels,
                 act=nn.SiLU(),
                 in_channels=3,
                 out_channels=1,
                 lifting_channels=256,
                 projection_channels=256,
                 n_layers=4,
                 use_mlp=True, mlp= {'expansion': 4.0, 'dropout': 0.},
                 non_linearity=F.gelu,
                 norm='group_norm', preactivation=True,
                 skip='soft-gating',
                 separable=True,
                 factorization=None,
                 rank=1.0,
                 joint_factorization=False,
                 fixed_rank_modes=False,
                 implementation='factorized',
                 decomposition_kwargs=dict(),
                 domain_padding=None,
                 domain_padding_mode='one-sided',
                 fft_norm='ortho',
                 **kwargs):
        super().__init__()
        self.act = act
        self.n_dim = len(n_modes)
        self.act = act
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.skip = skip,
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation


        Dense= [nn.Linear(self.lifting_channels , self.lifting_channels ).to('cuda')]
        Dense[0].weight.data = default_init()(Dense[0].weight.data.shape)
        nn.init.zeros_(Dense[0].bias)
        Dense.append(nn.Linear(self.lifting_channels , self.lifting_channels ).to('cuda'))
        Dense[1].weight.data = default_init()(Dense[1].weight.data.shape)
        nn.init.zeros_(Dense[1].bias)
        self.Dense=nn.ModuleList(Dense)
        # self.attns = nn.ModuleList([ AFNO2D( hidden_size=).to('cuda')]*self.n_layers)
        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(domain_padding=domain_padding, padding_mode=domain_padding_mode)
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        self.convs = FactorizedSpectralConv(
            self.hidden_channels, self.hidden_channels, self.n_modes,
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
        )

        self.fno_skips = nn.ModuleList([skip_connection(self.hidden_channels, self.hidden_channels, type=skip, n_dim=self.n_dim) for _ in range(n_layers)])

        if use_mlp:
            self.mlp = nn.ModuleList(
                [MLP(in_channels=self.hidden_channels, hidden_channels=int(round(self.hidden_channels*mlp['expansion'])),
                     dropout=mlp['dropout'], n_dim=self.n_dim,temb_dim=self.hidden_channels) for _ in range(n_layers)]
            )
            self.mlp_skips = nn.ModuleList([skip_connection(self.hidden_channels, self.hidden_channels, type=skip, n_dim=self.n_dim) for _ in range(n_layers)])
        else:
            self.mlp = None

        if norm is None:
            self.norm = None
        elif norm == 'instance_norm':
            self.norm = nn.ModuleList([getattr(nn, f'InstanceNorm{self.n_dim}d')(num_features=self.hidden_channels) for _ in range(n_layers)])
        elif norm == 'group_norm':
            self.norm = nn.ModuleList([nn.GroupNorm(num_groups=4, num_channels=self.hidden_channels) for _ in range(n_layers)])
        elif norm == 'layer_norm':
            self.norm = nn.ModuleList([nn.LayerNorm() for _ in range(n_layers)])
        else:
            raise ValueError(f'Got {norm=} but expected None or one of [instance_norm, group_norm, layer_norm]')

        self.lifting = Lifting(in_channels=in_channels, out_channels=self.hidden_channels, n_dim=self.n_dim)
        self.projection = Projection(in_channels=self.hidden_channels, out_channels=out_channels, hidden_channels=projection_channels,
                                     non_linearity=non_linearity, n_dim=self.n_dim)


    def forward(self, x, t):
        """
        Temporal FNO의 순방향 패스 - 시간 조건부 Fourier 신경망 연산
        
        역할:
        - 시간 조건부 확산 모델의 핵심 연산 수행
        - 입력 데이터를 주파수 도메인에서 처리하여 전역 수용장 달성
        - 시간 임베딩을 통한 확산 과정의 시간 스텝 조건부 처리
        - 1D 데이터에 특화된 브로드캐스팅 ([:,:,None])
        
        처리 과정:
        1. 1D 데이터 차원 확장 (필요시)
        2. Lifting: 저차원 → 고차원 잠재 공간 변환
        3. 도메인 패딩 적용 (선택적)
        4. 시간 임베딩 생성 및 네트워크에 주입
        5. FNO 블록들을 통한 스펙트럼 컨볼루션
        6. MLP 블록들을 통한 비선형 변환 (선택적)
        7. 도메인 패딩 제거 (선택적)
        8. Projection: 고차원 → 출력 차원 변환
        9. 1D 데이터 차원 축소
        
        시간 조건부 특징:
        - get_timestep_embedding으로 시간 t를 연속 벡터로 인코딩
        - Dense 레이어들을 통한 시간 임베딩 처리
        - 각 FNO 블록에 시간 정보 주입
        - 확산 과정의 시간에 따른 적응적 연산
        
        Parameters:
            x: 입력 텐서 [batch, channels, spatial_dims...]
            t: 시간 스텝 [batch] (확산 모델의 타임스텝)
        
        Returns:
            처리된 출력 텐서 [batch, out_channels, spatial_dims...]
        """
        if x.dim()==2:
            x =x.unsqueeze(1)
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        m_idx=0

        temb = get_timestep_embedding(t, self.lifting_channels)
        temb = self.Dense[m_idx].to('cuda')(temb.to('cuda'))

        m_idx += 1
        temb = self.Dense[m_idx].to('cuda')(self.act.to('cuda')(temb.to('cuda')))
        m_idx += 1
        
        x = x+temb[:,:,None]
        for i in range(self.n_layers):

            if self.preactivation:
                x = self.non_linearity(x)


                if self.norm is not None:
                    x = self.norm[i](x)

            x_fno = self.convs[i]((x,temb))


            if not self.preactivation and self.norm is not None:
                x_fno = self.norm[i](x_fno)

            x_skip = self.fno_skips[i](x)
            x = x_fno + x_skip


            if not self.preactivation and i < (self.n_layers - 1):
                x = self.non_linearity(x)

            if self.mlp is not None:
                x_skip = self.mlp_skips[i](x)

                if self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)

                x = self.mlp[i](x) + x_skip

                if not self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)



        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        x=x.squeeze(1)
        return x

def partialclass(new_name, cls, *args, **kwargs):
    """
    다른 기본값을 가진 새로운 클래스 생성 유틸리티
    
    역할:
    - fno.py의 partialclass와 동일한 구현
    - 기존 클래스에서 일부 매개변수를 사전 설정한 새 클래스 생성
    - functools.partial 대신 진정한 클래스 상속 제공
    - TFNO 같은 특화된 FNO 변형 생성에 사용
    
    functools.partial 대신 사용하는 이유:
    1. 클래스 이름 문제:
       >>> new_class = partial(cls, **kwargs)
       >>> new_class.__name__ = new_name  # 명시적 설정 필요

    2. 상속 불가 문제:
       >>> functools 객체는 상속 불가능
       >>> 새 클래스 정의가 필요

    해결 방법:
    - 동적으로 새 클래스 정의
    - 기존 클래스를 상속하여 진정한 클래스 생성
    - partialmethod로 __init__ 사전 설정
    
    사용 예:
    >>> TFNO = partialclass('TFNO', FNO, factorization='Tucker')
    >>> # TFNO는 Tucker 인수분해를 기본으로 하는 FNO

    Parameters:
        new_name: 새 클래스 이름
        cls: 기존 클래스
        *args, **kwargs: 사전 설정할 인수들

    Returns:
        새로운 클래스 (기존 클래스 상속, 일부 매개변수 사전 설정)
    """
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    new_class = type(new_name, (cls,),  {
        '__init__': __init__,
        '__doc__': cls.__doc__,
        'forward': cls.forward,
    })
    return new_class


TFNO   = partialclass('TFNO', FNO, factorization='Tucker')

class FNO2(nn.Module):
    """
    향상된 Temporal FNO - 설정 기반 N차원 시간 조건부 Fourier Neural Operator
    
    역할:
    - FNO 클래스의 개선된 버전으로 AFNO 통합과 설정 기반 초기화 제공
    - config 객체를 통한 체계적인 하이퍼파라미터 관리
    - AFNO2D 블록 통합으로 적응적 주파수 처리 지원
    - 2D 이미지 데이터에 특화된 시간 조건부 처리 ([:,:,None,None] 브로드캐스팅)
    - 조건부 생성 모델 지원 (y 매개변수)
    
    주요 개선사항 (vs FNO):
    1. AFNO 통합: self.attns로 적응적 주파수 처리
    2. 설정 기반: config 객체로 체계적 파라미터 관리
    3. 2D 특화: 이미지 생성 모델에 최적화
    4. 조건부 지원: 클래스 조건부 생성 (get_timestep_embedding2)
    5. 디바이스 최적화: 명시적 device 관리
    6. 향상된 초기화: 더 안정적인 가중치 초기화
    
    AFNO 특징:
    - 각 FNO 레이어마다 AFNO2D 블록 적용
    - 적응적 주파수 선택 및 희소성 제어
    - 블록 대각 구조로 파라미터 효율성
    - 하드/소프트 임계값으로 계산량 제어
    
    아키텍처 흐름:
    입력 → Lifting → [시간+조건 임베딩] → FNO+AFNO 블록들 → Projection → 출력
           ↓                               ↓
        차원 확장                    적응적 스펙트럼 처리
    
    조건부 생성 지원:
    - t: 시간 스텝 (주 조건)
    - y: 클래스 레이블 등 (보조 조건, 선택적)
    - 두 임베딩을 결합하여 다중 조건 처리
    
    사용 컨텍스트:
    - 고품질 이미지 생성
    - 클래스 조건부 확산 모델
    - 복잡한 2D 패턴 학습
    - 프로덕션 레벨 확산 모델

    Parameters
    ----------
    config : Config 객체
        모든 하이퍼파라미터를 담은 설정 객체
        config.model 하위에 FNO 관련 설정 포함:
        - n_modes: 각 차원별 Fourier 모드 수
        - hidden_channels: FNO 폭 (채널 수)
        - in_channels: 입력 채널 수
        - out_channels: 출력 채널 수
        - lifting_channels: Lifting 블록 은닉 채널
        - projection_channels: Projection 블록 은닉 채널
        - n_layers: Fourier 레이어 수
        - use_mlp: MLP 사용 여부
        - mlp_expansion: MLP 확장 비율
        - mlp_dropout: MLP 드롭아웃 비율
        - norm: 정규화 타입
        - preactivation: 사전 활성화 여부
        - skip: 스킵 연결 타입
        - separable: 분리 가능한 컨볼루션 여부
        - factorization: 텐서 인수분해 방법
        - rank: 인수분해 랭크
        - joint_factorization: 조인트 인수분해 여부
        - fixed_rank_modes: 고정 랭크 모드
        - implementation: 구현 방식
        - domain_padding: 도메인 패딩 비율
        - domain_padding_mode: 패딩 모드
        - fft_norm: FFT 정규화 방법
        - sparsity_threshold: AFNO 희소성 임계값
        - num_blocks: AFNO 블록 수
        - hard_thresholding_fraction: AFNO 하드 임계값 비율
        - hidden_size_factor: AFNO MLP 확장 팩터
    device : str, default is 'cuda'
        계산 디바이스
    """
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.n_modes = n_modes = config.model.n_modes
        self.n_dim = len(n_modes)
        self.act = act=nn.GELU().to(device)
        self.fft_norm = fft_norm= config.model.fft_norm
        self.norm=norm = config.model.norm
        self.mlp_expension = config.model.mlp_expansion
        self.mlp_dropout = config.model.mlp_dropout
        self.hidden_channels = hidden_channels= config.model.hidden_channels
        self.lifting_channels =lifting_channels= config.model.lifting_channels
        self.projection_channels = projection_channels= config.model.projection_channels
        self.in_channels = in_channels =config.model.in_channels
        self.out_channels =out_channels  =config.model. out_channels
        self.n_layers = n_layers=config.model.n_layers
        self.joint_factorization = joint_factorization=config.model.joint_factorization
        self.non_linearity = non_linearity =nn.GELU().to(device)
        self.rank = rank =config.model.rank
        self.use_mlp = use_mlp = config.model.use_mlp
        self.factorization =factorization= config.model.factorization
        self.fixed_rank_modes = fixed_rank_modes= config.model.fixed_rank_modes
        self.skip = skip = config.model.skip
        self.implementation = implementation= config.model.implementation
        self.separable = separable = config.model.separable
        self.preactivation = config.model.preactivation
        domain_padding = config.model.domain_padding
        domain_padding_mode = config.model.domain_padding_mode

        Dense = [nn.Linear(self.lifting_channels, self.hidden_channels).to(device)]
        Dense[0].weight.data = default_init()(Dense[0].weight.data.shape)
        nn.init.zeros_(Dense[0].bias)
        Dense.append(nn.Linear( self.hidden_channels ,  self.hidden_channels).to(device))
        Dense[1].weight.data = default_init()(Dense[1].weight.data.shape)
        nn.init.zeros_(Dense[1].bias)
        self.Dense = nn.ModuleList(Dense)
        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(domain_padding=domain_padding, padding_mode=domain_padding_mode).to(device)
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode
        self.attns = nn.ModuleList([ AFNO2D(config).to(device)]*self.n_layers)
        self.convs = FactorizedSpectralConv(
            self.hidden_channels, self.hidden_channels, self.n_modes,
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            joint_factorization=joint_factorization,
            n_layers=n_layers, device=device
        )

        self.fno_skips = nn.ModuleList([skip_connection(self.hidden_channels, self.hidden_channels, type=skip, n_dim=self.n_dim).to(device) for _ in range(n_layers)])

        if use_mlp:
            self.mlp = nn.ModuleList(
                [MLP(in_channels=self.hidden_channels, hidden_channels=int(round(self.hidden_channels*self.mlp_expension)),
                     dropout=self.mlp_dropout, n_dim=self.n_dim,temb_dim=self.hidden_channels).to(device) for _ in range(n_layers)]
            )
            self.mlp_skips = nn.ModuleList([skip_connection(self.hidden_channels, self.hidden_channels, type=skip, n_dim=self.n_dim).to(device) for _ in range(n_layers)])
        else:
            self.mlp = None

        if norm is None:
            self.norm = None
        elif norm == 'instance_norm':
            self.norm = nn.ModuleList([getattr(nn, f'InstanceNorm{self.n_dim}d')(num_features=self.hidden_channels).to(device) for _ in range(n_layers)])
        elif norm == 'group_norm':
            self.norm = nn.ModuleList([nn.GroupNorm(num_groups=1, num_channels=self.hidden_channels).to(device) for _ in range(n_layers)])
        elif norm == 'layer_norm':
            self.norm = nn.ModuleList([nn.LayerNorm().to(device) for _ in range(n_layers)])
        else:
            raise ValueError(f'Got {norm=} but expected None or one of [instance_norm, group_norm, layer_norm]')

        self.lifting = Lifting(in_channels=in_channels, out_channels=self.hidden_channels, n_dim=self.n_dim).to(device)
        self.projection = Projection(in_channels=self.hidden_channels, out_channels=out_channels, hidden_channels=projection_channels,
                                     non_linearity=non_linearity, n_dim=self.n_dim).to(device)


    def forward(self, x, t, y=None):
        """
        향상된 Temporal FNO의 순방향 패스 - AFNO 통합 및 조건부 생성 지원
        
        역할:
        - AFNO2D를 포함한 고급 시간 조건부 Fourier 신경망 연산
        - 클래스 조건부 생성 모델 지원 (y 매개변수)
        - 2D 이미지 데이터에 특화된 처리 ([:,:,None,None] 브로드캐스팅)
        - 적응적 주파수 처리와 스펙트럼 컨볼루션의 결합
        
        vs FNO.forward 차이점:
        1. AFNO 미사용: 기본 FNO는 AFNO 블록이 주석 처리됨
        2. 조건부 지원: y 매개변수로 추가 조건 임베딩
        3. 2D 브로드캐스팅: [:,:,None,None]로 이미지 차원 처리
        4. 디바이스 최적화: 동적 디바이스 이동으로 메모리 효율성
        
        처리 과정:
        1. Lifting: 입력 차원 확장
        2. 도메인 패딩 적용 (선택적)
        3. 다중 조건 임베딩: 시간(t) + 클래스(y) 결합
        4. 시간 임베딩 Dense 레이어 처리
        5. 2D 브로드캐스팅으로 이미지 전체에 조건 주입
        6. FNO 블록들: 스펙트럼 컨볼루션 + 정규화 + 스킵 연결
        7. MLP 블록들: 비선형 변환 + 시간 조건부 처리
        8. 도메인 패딩 제거 (선택적)
        9. Projection: 최종 출력 차원 변환
        
        AFNO 처리 (현재 미사용):
        - 각 레이어에서 self.attns[i] 호출 가능
        - 적응적 주파수 선택 및 희소성 제어
        - 하드/소프트 임계값으로 계산량 최적화
        
        조건부 생성:
        - t: 확산 과정의 시간 스텝 (필수)
        - y: 클래스 레이블 등 보조 조건 (선택적)
        - 두 임베딩 결합: temb = get_timestep_embedding(t) + get_timestep_embedding2(y)
        
        Parameters:
            x: 입력 텐서 [batch, channels, height, width]
            t: 시간 스텝 [batch] (확산 모델의 주 조건)
            y: 보조 조건 [batch] (클래스 레이블 등, 선택적)
        
        Returns:
            처리된 출력 텐서 [batch, out_channels, height, width]
        """

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        m_idx=0

        temb = get_timestep_embedding(t, self.lifting_channels)
        if y is not None:
            cemb = get_timestep_embedding2(y, self.lifting_channels)
            temb = temb+cemb

        temb = self.Dense[m_idx].to(temb.device)(temb)

        m_idx += 1

        temb = self.Dense[m_idx].to(temb.device)(self.act.to(temb.device)(temb))
        m_idx += 1

        x = x + temb[:,:,None,None]

        for i in range(self.n_layers):

            if self.preactivation:
                x = self.non_linearity(x)

                if self.norm is not None:
                    x = self.norm[i](x)

            x_fno = self.convs[i].to(x.device)((x,temb))

            if not self.preactivation and self.norm is not None:
                x_fno = self.norm[i].to(x.device)(x_fno)

            x_skip = self.fno_skips[i].to(x.device)(x)

            x = x_fno + x_skip

            if not self.preactivation and i < (self.n_layers - 1):
                x = self.non_linearity(x)

            if self.mlp is not None:
                x_skip = self.mlp_skips[i](x)

                if self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)

                x = self.mlp[i].to(x.device)(x, temb) + x_skip

                if not self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)


        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x



def partialclass(new_name, cls, *args, **kwargs):
    """
    새로운 클래스 생성 유틸리티 (FNO2 버전)
    
    역할:
    - 첫 번째 partialclass와 동일한 기능
    - FNO2를 기반으로 한 특화 클래스 생성용
    - TFNO2 생성에 사용됨
    
    참고:
    - 코드 중복: 동일한 함수가 두 번 정의됨
    - 마지막 return문 누락 (버그)
    - 코드 정리 시 통합 고려 필요
    
    버그 수정 필요:
    - return new_class 추가해야 함
    """
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    new_class = type(new_name, (cls,),  {
        '__init__': __init__,
        '__doc__': cls.__doc__,
        'forward': cls.forward,
    })
    return

TFNO2 = partialclass('TFNO2', FNO2, factorization='Tucker')
# Tucker 인수분해를 기본으로 하는 FNO2 변형
# 파라미터 압축과 성능의 균형이 우수한 설정
