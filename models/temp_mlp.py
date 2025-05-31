# Temporal MLP 모듈 - 시간 조건부 ResNet 기반 MLP 네트워크
# 기본 MLP와 달리 ResNet 블록을 활용한 더 강력한 비선형 처리
# 2D 이미지 데이터에 특화된 시간 조건부 아키텍처

"""
Temporal MLP 모듈 개요
=====================

이 모듈은 models/mlp.py의 확장 버전으로, ResNet 블록을 기반으로 한
더 강력한 시간 조건부 MLP를 구현합니다. 특히 2D 이미지 데이터 처리에
최적화되어 있으며, 각 레이어마다 독립적인 시간 임베딩을 지원합니다.

주요 차이점 (vs mlp.py):
1. 1×1 Conv 대신 ResNet 블록 사용
2. 레이어별 독립적 시간 임베딩 처리
3. 2D 데이터 특화 ([:,:,None,None] 브로드캐스팅)
4. 더 강력한 비선형성과 잔차 학습

아키텍처 구조:
x → (+time_emb₁) → ResNetBlock₁ → (+time_emb₂) → ResNetBlock₂ → ... → output

핵심 특징:
- 레이어별 시간 조건부: 각 레이어마다 다른 시간 처리
- ResNet 기반: 깊은 네트워크에서 안정적 학습
- 2D 특화: 이미지 생성 모델에 최적화
- 강력한 표현력: 복잡한 패턴 학습 가능

수학적 표현:
h₀ = x + Dense₀(time_emb)
h₁ = ResNetBlock₁(h₀) + Dense₁(time_emb)
...
y = ResNetBlockₙ(hₙ₋₁)

의존성:
- models/fno.py: FNO2 클래스에서 사용
- ResNet 블록과 시간 임베딩 시스템 연동
- 2D 확산 모델의 표현력 강화

사용 컨텍스트:
- 이미지 생성 작업
- 복잡한 비선형 변환이 필요한 경우
- 깊은 MLP 네트워크 구성
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tltorch import TensorizedTensor      # 텐서 분해 라이브러리
from tltorch.utils import get_tensorized_shape
from einops import rearrange              # 텐서 차원 조작 라이브러리

class LinearAttention(nn.Module):
    """
    선형 어텐션 메커니즘 (미완성 구현)
    
    역할:
    - mlp.py와 동일한 미완성 어텐션 클래스
    - forward 메서드가 구현되지 않음
    - 향후 확장을 위한 플레이스홀더
    
    참고:
    - 실제 사용되지 않는 코드
    - 코드 정리 시 제거 고려 대상
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))
        

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """
  분산 스케일링 기반 가중치 초기화 (JAX에서 이식)
  
  역할:
  - ResNet 블록의 안정적인 학습을 위한 가중치 초기화
  - 시간 임베딩 Dense 레이어 초기화에 사용
  
  참고:
  - mlp.py, fno.py, fno_block.py와 동일한 중복 구현
  - 모든 모듈에서 일관된 초기화 보장
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


def exists(x):
    """
    None 체크 유틸리티 함수
    
    역할:
    - mlp.py와 동일한 헬퍼 함수
    - 조건부 연산 간소화
    """
    return x is not None

class Block(nn.Module):
    """
    기본 컨볼루션 블록 - ResNet 구성 요소
    
    역할:
    - mlp.py의 Block 클래스와 동일한 구현
    - Conv → GroupNorm → (Scale/Shift) → Activation 구조
    - ResnetBlock의 빌딩 블록으로 사용
    
    참고:
    - mlp.py와 완전히 동일한 코드 (중복)
    - 코드 통합 고려 필요
    """
    def __init__(self, dim, dim_out, groups=2):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)  # 3×3 컨볼루션
        self.norm = nn.GroupNorm(groups, dim_out)           # 그룹 정규화
        self.act = nn.GELU()                                # GELU 활성화

    def forward(self, x, scale_shift=None):
        """
        순방향 패스 - 적응적 정규화 지원
        
        Parameters:
            x: 입력 [batch, dim, height, width]
            scale_shift: (scale, shift) 튜플 (선택적)
        
        Returns:
            처리된 특징 [batch, dim_out, height, width]
        """
        x = self.proj(x)      # 컨볼루션
        x = self.norm(x)      # 정규화

        # 시간 임베딩 기반 적응적 변조
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift  # AdaGN 스타일

        x = self.act(x)       # 활성화
        return x
    
    

class ResnetBlock(nn.Module):
    """
    시간 조건부 ResNet 블록
    
    참고 논문: "Deep Residual Learning for Image Recognition" (https://arxiv.org/abs/1512.03385)
    
    역할:
    - mlp.py의 ResnetBlock과 동일한 구현
    - Temporal MLP의 핵심 구성 요소
    - 강력한 비선형 변환 및 잔차 학습 제공
    
    특징:
    - 2D 이미지 처리에 최적화
    - 시간 임베딩 중간 주입
    - 그래디언트 흐름 개선
    
    참고:
    - mlp.py와 완전히 동일한 코드 (중복)
    - 모듈 간 코드 공유 고려 필요
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=2):
        """
        Parameters:
            dim: 입력 차원
            dim_out: 출력 차원
            time_emb_dim: 시간 임베딩 차원
            groups: GroupNorm 그룹 수
        """
        super().__init__()
        
        # 시간 임베딩 처리 MLP
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        # 두 개의 컨볼루션 블록
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        
        # 잔차 연결용 차원 매칭
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        시간 조건부 순방향 패스
        
        Parameters:
            x: 입력 [batch, dim, height, width]
            time_emb: 시간 임베딩 [batch, time_emb_dim]
        
        Returns:
            출력 [batch, dim_out, height, width]
        """
        h = self.block1(x)

        # 시간 임베딩 주입
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h  # 2D 브로드캐스팅

        h = self.block2(h)
        return h + self.res_conv(x)  # 잔차 연결

def default_init(scale=1.):
  """
  DDPM 스타일 기본 초기화
  
  역할:
  - 시간 임베딩 Dense 레이어 초기화
  - variance_scaling의 래퍼 함수
  
  참고:
  - 다른 모듈들과 동일한 중복 구현
  """
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')


class MLP(nn.Module):
    """
    Temporal Multi-Layer Perceptron - ResNet 블록 기반 시간 조건부 MLP
    
    역할:
    - mlp.py의 MLP보다 강력한 ResNet 기반 비선형 네트워크
    - 각 레이어마다 독립적인 시간 임베딩 처리
    - 2D 이미지 데이터에 특화된 아키텍처
    - FNO와 결합하여 표현력 강화
    
    주요 차이점 (vs mlp.py MLP):
    1. ResNet 블록 사용: 1×1 Conv 대신 더 강력한 ResNet 블록
    2. 레이어별 시간 임베딩: 각 레이어마다 별도의 Dense 레이어
    3. 2D 특화: [:,:,None,None] 브로드캐스팅으로 이미지 처리
    4. 더 깊은 네트워크: ResNet 블록으로 안정적 깊은 학습
    
    아키텍처:
    x → (+Dense₀(temb)) → ResNetBlock₀ → (+Dense₁(temb)) → ResNetBlock₁ → ... → output
    
    핵심 특징:
    - 레이어별 적응적 시간 조건부
    - 잔차 학습 기반 안정적 그래디언트 흐름
    - 2D 공간 구조 보존
    - 강력한 비선형 변환 능력
    
    수학적 표현:
    h₀ = x + Dense₀(act(temb))
    h₁ = ResNetBlock₀(h₀) + activation(h₀)
    h₂ = h₁ + Dense₁(act(temb))
    h₃ = ResNetBlock₁(h₂) + activation(h₂)
    ...
    
    의존성:
    - ResnetBlock: 잔차 학습 빌딩 블록
    - 시간 임베딩 시스템과 연동
    - FNO 네트워크의 MLP 구성 요소로 사용

    Parameters
    ----------
    in_channels : int
        입력 채널 수
    out_channels : int, default is None
        출력 채널 수 (None이면 in_channels와 동일)
    hidden_channels : int, default is None
        은닉 채널 수 (None이면 in_channels와 동일)
    n_layers : int, default is 2
        ResNet 블록 레이어 수
    n_dim : int, default is 2
        데이터 차원 (기본 2D)
    non_linearity : nn.Module, default is nn.SiLU()
        활성화 함수
    dropout : float, default is 0
        드롭아웃 확률
    hidden : int, default is 256
        시간 임베딩 처리용 은닉 차원
    act : nn.Module, default is nn.GELU()
        시간 임베딩용 활성화 함수
    temb_dim : int, default is None
        시간 임베딩 차원
    """
    def __init__(self, in_channels, out_channels=None, hidden_channels=None, 
                 n_layers=2, n_dim=2, non_linearity=nn.SiLU(), dropout=0., 
                 hidden=256, act=nn.GELU(),temb_dim = None,
                 **kwargs,
                 ):
        """
        Temporal MLP 초기화
        
        구현 과정:
        1. 기본 파라미터 설정
        2. 레이어별 시간 임베딩 Dense 레이어 생성
        3. ResNet 블록들로 MLP 구성
        4. 드롭아웃 레이어 추가 (선택적)
        """
        super().__init__()
        
        # 기본 파라미터 설정
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.act = act
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels 
        self.non_linearity = non_linearity
        
        # 레이어별 드롭아웃 (선택적)
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)]) if dropout > 0. else None
        
        # 레이어별 시간 임베딩 처리 Dense 레이어들
        self.Dense = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                # 첫 번째 레이어: 입력 채널 수에 맞춤
                self.Dense.append(nn.Linear(temb_dim, in_channels))
            else:
                # 나머지 레이어들: 은닉 채널 수에 맞춤
                self.Dense.append(nn.Linear(temb_dim, hidden_channels))
            
            # 확산 모델 최적화 초기화 적용
            self.Dense[i].weight.data = default_init()(self.Dense[i].weight.data.shape)
        
        # ResNet 블록들로 MLP 구성 (1×1 Conv 대신)
        Conv = getattr(nn, f'Conv{n_dim}d')  # 실제로는 사용되지 않음 (주석 처리된 코드)
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                # 첫 번째 레이어: 입력 → 은닉 차원
                # self.fcs.append(Conv(self.in_channels, self.hidden_channels, 1))  # 기존 방식
                self.fcs.append(ResnetBlock(self.in_channels, self.hidden_channels, time_emb_dim=hidden_channels))
            elif i == (n_layers - 1):
                # 마지막 레이어: 은닉 → 출력 차원
                # self.fcs.append(Conv(self.hidden_channels, self.out_channels, 1))  # 기존 방식
                self.fcs.append(ResnetBlock(self.hidden_channels, self.out_channels, time_emb_dim=hidden_channels))
            else:
                # 중간 레이어들: 은닉 → 은닉 차원
                # self.fcs.append(Conv(self.out_channels, self.out_channels, 1))  # 기존 방식 (버그?)
                self.fcs.append(ResnetBlock(self.hidden_channels, self.hidden_channels, time_emb_dim=hidden_channels))
            

    def forward(self, x, temb=None):
        """
        Temporal MLP의 순방향 패스 - 레이어별 시간 임베딩과 ResNet 블록 처리
        
        역할:
        - 시간 조건부 MLP 네트워크의 핵심 연산 수행
        - 각 ResNet 블록마다 독립적인 시간 임베딩 주입
        - 2D 공간 데이터에 최적화된 처리 ([:,:,None,None] 브로드캐스팅)
        - 깊은 네트워크에서 안정적인 그래디언트 흐름 제공
        
        주요 차이점 (vs mlp.py):
        1. 레이어별 시간 임베딩: 각 층마다 별도의 Dense 레이어 사용
        2. ResNet 블록: 1×1 Conv 대신 더 강력한 ResNet 구조
        3. 2D 브로드캐스팅: 이미지 데이터 특화 (4차원 텐서)
        4. 디바이스 동적 이동: GPU 메모리 최적화
        
        처리 과정:
        1. 각 레이어에서 시간 임베딩 주입
        2. ResNet 블록을 통한 비선형 변환
        3. 활성화 함수 적용 (마지막 층 제외)
        4. 드롭아웃으로 과적합 방지
        
        시간 임베딩 주입 방식:
        temb → Dense[i] → activation → [:,:,None,None] → x에 더하기
        
        Parameters:
            x: 입력 특징 맵 [batch, channels, height, width]
            temb: 시간 임베딩 [batch, temb_dim]
        
        Returns:
            변환된 특징 맵 [batch, out_channels, height, width]
        
        구현 특징:
        - 동적 디바이스 이동으로 GPU 메모리 효율성
        - 2D 이미지 데이터에 최적화된 브로드캐스팅
        - ResNet 블록 기반 안정적 학습
        
        주의사항:
        - 조건문 버그: i < self.n_layers → i < self.n_layers-1 이어야 함
        - 드롭아웃 인덱싱: dropout[i] 대신 dropout 사용 (mlp.py 참조)
        """
        
        # 각 ResNet 레이어 순차 처리
        for i, fc in enumerate(self.fcs):
            # 1. 시간 임베딩 주입 (레이어별 독립적 처리)
            if temb is not None:
                # 시간 임베딩을 해당 레이어의 Dense로 변환
                time_emb_processed = self.Dense[i].to(x.device)(self.act.to(x.device)(temb))
                
                # 2D 공간 차원에 브로드캐스팅: [batch, ch] → [batch, ch, 1, 1]
                # 이미지의 모든 픽셀에 동일한 시간 정보 적용
                x = x + time_emb_processed[:, :, None, None]
            
            # 2. ResNet 블록 적용
            # ResNet 블록 내부에서 잔차 연결과 비선형 변환 수행
            x = fc(x)
            
            # 3. 활성화 함수 적용 (마지막 레이어 제외)
            # 주의: 조건문이 잘못됨 (i < self.n_layers-1 이어야 함)
            if i < self.n_layers:
                x = self.non_linearity(x)
            
            # 4. 드롭아웃 적용 (과적합 방지)
            if self.dropout is not None:
                # 주의: 인덱싱 오류 가능성 (mlp.py와 구현 방식 다름)
                x = self.dropout(x)  # 수정 필요: self.dropout[i](x)

        return x

import torch
from torch import nn


def skip_connection(in_features, out_features, n_dim=2, bias=False, type="soft-gating"):
    """
    다양한 타입의 스킵 연결을 위한 팩토리 함수
    
    역할:
    - Temporal MLP와 FNO 블록에서 잔차 학습 지원
    - 세 가지 스킵 연결 전략 제공으로 유연한 아키텍처 구성
    - 그래디언트 흐름 개선 및 학습 안정성 향상
    - 파라미터 효율성과 표현력 간의 균형 조절
    
    스킵 연결 타입별 특징:
    1. Identity: y = x
       - 입출력 차원이 동일할 때 사용
       - 파라미터 없음, 직접적 그래디언트 전파
       - 가장 간단하고 안정적인 연결
    
    2. Linear: y = Conv1×1(x)
       - 1×1 컨볼루션을 통한 차원 변환
       - 입출력 차원이 다를 때 필수
       - 선형 변환으로 채널 수 조정
    
    3. Soft-gating: y = w ⊙ x + b
       - 학습 가능한 채널별 가중치
       - 적응적 특징 선택 및 억제
       - 가장 표현력이 풍부하지만 파라미터 추가
    
    수학적 배경:
    - ResNet: y = F(x) + skip(x)
    - 그래디언트: ∂y/∂x = ∂F(x)/∂x + ∂skip(x)/∂x
    - Identity/Linear: 직접적 그래디언트 전파
    - Soft-gating: 적응적 특징 가중치 학습
    
    의존성:
    - models/mlp.py와 동일한 구현 (코드 중복)
    - SoftGating 클래스와 연동
    - FNO 아키텍처의 스킵 연결로 사용
    
    Parameters
    ----------
    in_features : int
        입력 특징 수 (채널 수)
    out_features : int
        출력 특징 수 (채널 수)
    n_dim : int, default is 2
        데이터 차원 (배치와 채널 제외)
        n_dim=1: 1D 데이터, n_dim=2: 2D 데이터
    bias : bool, default is False
        바이어스 사용 여부
    type : str, default is "soft-gating"
        스킵 연결 타입 {'identity', 'linear', 'soft-gating'}

    Returns
    -------
    nn.Module
        스킵 연결 모듈 (x를 받아 skip(x) 반환)
    """
    if type.lower() == 'soft-gating':
        # 학습 가능한 채널별 가중치 스킵 연결
        return SoftGating(in_features=in_features, out_features=out_features, bias=bias, n_dim=n_dim)
    elif type.lower() == 'linear':
        # 1×1 컨볼루션 기반 선형 스킵 연결
        return getattr(nn, f'Conv{n_dim}d')(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=bias)
    elif type.lower() == 'identity':
        # 항등 매핑 스킵 연결 (입출력 차원 동일 시)
        return nn.Identity()
    else:
        raise ValueError(f"Got skip-connection {type=}, expected one of {'soft-gating', 'linear', 'identity'}.")


class SoftGating(nn.Module):
    """
    소프트 게이팅 스킵 연결 - 학습 가능한 채널별 가중치
    
    역할:
    - 채널별로 독립적인 학습 가능한 가중치 적용
    - 적응적 특징 선택 및 억제 지원
    - 잔차 학습에서 중요한 특징 강조
    - Temporal MLP의 표현력 향상을 위한 유연한 스킵 연결
    
    수학적 원리:
    - 기본: y = w ⊙ x (element-wise multiplication)
    - 바이어스: y = w ⊙ x + b
    - 여기서 w, b는 채널별 학습 가능한 파라미터
    
    특징:
    - 초기값: w=1, b=1 (항등 매핑으로 시작)
    - 학습 과정에서 중요한 채널에 큰 가중치 할당
    - 불필요한 채널은 자동으로 억제
    - 각 채널마다 독립적인 게이팅 제어
    
    장점:
    1. 파라미터 효율적: 채널당 1-2개 파라미터만 추가
    2. 해석 가능성: 가중치로 채널 중요도 확인 가능
    3. 안정성: 항등 매핑으로 초기화하여 안전한 학습
    4. 유연성: 선택적 바이어스로 오프셋 조정
    
    입력 차원별 가중치 모양:
    - 1D: (batch, channels, length) → w: (1, channels, 1)
    - 2D: (batch, channels, height, width) → w: (1, channels, 1, 1)
    - 3D: (batch, channels, depth, height, width) → w: (1, channels, 1, 1, 1)
    
    의존성:
    - models/mlp.py의 SoftGating과 동일한 구현 (코드 중복)
    - skip_connection 함수에서 생성
    - FNO 및 Temporal MLP의 스킵 연결로 사용
    
    Parameters
    ----------
    in_features : int
        입력 채널 수
    out_features : int or None
        출력 채널 수 (API 호환성을 위해 제공, in_features와 동일해야 함)
    n_dim : int, default is 2
        데이터 차원 (배치와 채널 제외)
        n_dim=2: 2D 이미지 데이터
    bias : bool, default is False
        바이어스 파라미터 사용 여부
    """
    def __init__(self, in_features, out_features=None, n_dim=2, bias=False):
        super().__init__()
        
        # API 호환성 검증: soft-gating은 입출력 차원이 동일해야 함
        if out_features is not None and in_features != out_features:
            raise ValueError(f"Got {in_features=} and {out_features=} "
                             "but these two must be the same for soft-gating")
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 가중치 파라미터: 채널별로 독립적, 공간 차원에는 브로드캐스팅
        # 모양: (1, channels, 1, 1, ...) - n_dim개의 1
        # 초기값 1로 설정하여 항등 매핑부터 시작
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,)*n_dim))
        
        # 바이어스 파라미터 (선택적)
        if bias:
            # 바이어스도 1로 초기화 (mlp.py와 동일)
            self.bias = nn.Parameter(torch.ones(1, self.in_features, *(1,)*n_dim))
        else:
            self.bias = None

    def forward(self, x):
        """
        소프트 게이팅 적용
        
        역할:
        - 입력 활성화에 채널별 가중치 적용
        - 학습 가능한 게이팅으로 적응적 특징 조절
        - 잔차 학습에서 중요한 정보 선택적 전달
        
        Parameters:
            x: 입력 활성화 [batch, channels, spatial_dims...]
        
        Returns:
            게이팅된 활성화 [batch, channels, spatial_dims...]
        
        연산:
        - 바이어스 있음: y = w * x + b (아핀 변환)
        - 바이어스 없음: y = w * x (스케일링만)
        
        브로드캐스팅:
        - 가중치 (1, ch, 1, 1, ...)가 모든 배치와 공간 차원에 확장
        - 각 채널마다 동일한 가중치가 모든 공간 위치에 적용
        """
        if self.bias is not None:
            return self.weight * x + self.bias  # 아핀 변환
        else:
            return self.weight * x              # 스케일링만
