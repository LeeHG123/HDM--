# MLP (Multi-Layer Perceptron) 및 보조 네트워크 모듈
# FNO와 결합하여 비선형성을 강화하고 스킵 연결을 제공하는 핵심 구성 요소
# 확산 모델에서 시간 조건부 처리와 잔차 학습을 지원

"""
MLP 모듈 개요
=============

이 모듈은 FNO 아키텍처의 보조 네트워크들을 구현하며,
주파수 도메인 처리와 결합하여 강력한 하이브리드 모델을 구성합니다.

주요 구성 요소:
1. MLP: 시간 조건부 Multi-Layer Perceptron
2. 스킵 연결 함수들 (skip_connection, SoftGating)
3. ResNet 스타일 블록들 (Block, ResnetBlock)
4. 선형 어텐션 (LinearAttention) - 미완성
5. 유틸리티 함수들

핵심 역할:
- FNO 블록 간 비선형성 강화
- 시간 조건부 정보 처리
- 잔차 학습 및 그래디언트 흐름 개선
- 다양한 스킵 연결 전략 제공

아키텍처 통합:
FNO Block → MLP Block → Skip Connection
    ↓           ↓           ↓
 스펙트럼    비선형성     잔차학습
 컨볼루션     강화        지원

수학적 배경:
- MLP: y = σ(W_n(...σ(W_1x + b_1)...) + b_n)
- Soft Gating: y = w ⊙ x + b (채널별 가중치)
- ResNet: y = F(x) + x (잔차 연결)

의존성:
- models/fno.py: FNO 메인 아키텍처에서 사용
- models/fno_block.py: 초기화 함수 공유
- 확산 모델 시간 임베딩과 연동
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from tltorch import TensorizedTensor  # 텐서 분해 라이브러리
from tltorch.utils import get_tensorized_shape
from models.fno_block import *  # 초기화 함수 및 유틸리티 공유

class LinearAttention(nn.Module):
    """
    선형 어텐션 메커니즘 (미완성 구현)
    
    역할:
    - 어텐션 기반 특징 강화 (현재 forward 메서드 없음)
    - 2D 컨볼루션 기반 Q, K, V 생성
    - 멀티헤드 어텐션 구조
    
    참고:
    - 현재 초기화만 구현되어 있음
    - forward 메서드가 누락되어 실제 사용 불가
    - 향후 FNO와 어텐션 결합 시 사용 예정
    
    구조:
    - Query, Key, Value를 1×1 컨볼루션으로 생성
    - 멀티헤드로 분할하여 병렬 처리
    - GroupNorm으로 정규화
    """
    def __init__(self, dim, heads=4, dim_head=32):
        """
        Parameters:
            dim: 입력 차원
            heads: 어텐션 헤드 수
            dim_head: 각 헤드의 차원
        """
        super().__init__()
        self.scale = dim_head ** -0.5  # 스케일링 팩터
        self.heads = heads
        hidden_dim = dim_head * heads
        
        # Q, K, V를 동시에 생성하는 1×1 컨볼루션
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # 출력 프로젝션 및 정규화
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))
    
    # 주의: forward 메서드가 누락됨


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """
  분산 스케일링 기반 가중치 초기화 (JAX에서 이식)
  
  역할:
  - MLP 레이어의 안정적인 학습을 위한 가중치 초기화
  - 확산 모델에 최적화된 초기화 제공
  
  참고:
  - fno.py, fno_block.py와 중복 구현 (코드 정리 필요)
  - 일관된 초기화를 위해 동일한 함수 사용
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
    - 조건부 연산에서 None 검사를 간소화
    - 코드 가독성 향상
    
    사용 예:
    - if exists(time_emb): ...
    - scale_shift if exists(scale_shift) else None
    """
    return x is not None

class Block(nn.Module):
    """
    기본 컨볼루션 블록 - ResNet 스타일 빌딩 블록
    
    역할:
    - Conv → GroupNorm → (Scale/Shift) → Activation 순서 처리
    - 시간 임베딩 기반 적응적 정규화 지원
    - ResnetBlock의 구성 요소로 사용
    
    구조:
    1. 3×3 컨볼루션으로 특징 추출
    2. GroupNorm으로 정규화
    3. Scale/Shift 변조 (선택적)
    4. GELU 활성화
    
    수학적 표현:
    - 기본: y = GELU(GroupNorm(Conv(x)))
    - 변조: y = GELU((GroupNorm(Conv(x)) * (scale + 1)) + shift)
    
    의존성:
    - ResnetBlock에서 사용
    - 시간 임베딩과 연동하여 적응적 처리
    """
    def __init__(self, dim, dim_out, groups=2):
        """
        Parameters:
            dim: 입력 차원
            dim_out: 출력 차원
            groups: GroupNorm의 그룹 수
        """
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)  # 3×3 컨볼루션
        self.norm = nn.GroupNorm(groups, dim_out)           # 그룹 정규화
        self.act = nn.GELU()                                # GELU 활성화

    def forward(self, x, scale_shift=None):
        """
        순방향 패스
        
        Parameters:
            x: 입력 텐서 [batch, dim, height, width]
            scale_shift: 적응적 정규화 파라미터 (scale, shift) 튜플
        
        Returns:
            처리된 특징 [batch, dim_out, height, width]
        """
        x = self.proj(x)      # 컨볼루션 적용
        x = self.norm(x)      # 그룹 정규화

        # 시간 임베딩 기반 적응적 정규화
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift  # AdaGN 스타일 변조

        x = self.act(x)       # 활성화 함수
        return x



class ResnetBlock(nn.Module):
    """
    시간 조건부 ResNet 블록 (ResNet v1 기반)
    
    참고 논문: "Deep Residual Learning for Image Recognition" (https://arxiv.org/abs/1512.03385)
    
    역할:
    - 확산 모델에서 시간 조건부 잔차 학습 구현
    - 두 개의 컨볼루션 블록과 스킵 연결로 구성
    - 시간 임베딩을 통한 적응적 특징 변조
    
    아키텍처:
    x → Block1 → (+time_emb) → Block2 → (+residual) → output
    ↘_____________________________________________↗
                    (residual connection)
    
    핵심 특징:
    1. 시간 조건부 처리: time_emb를 중간에 주입
    2. 잔차 연결: 그래디언트 흐름 개선
    3. 차원 매칭: 입출력 차원이 다를 때 1×1 conv로 조정
    
    수학적 표현:
    h = Block1(x) + time_emb
    y = Block2(h) + res_conv(x)
    
    의존성:
    - Block 클래스를 빌딩 블록으로 사용
    - 시간 임베딩과 연동
    - 확산 모델 아키텍처의 표준 구성 요소
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=2):
        """
        Parameters:
            dim: 입력 차원
            dim_out: 출력 차원
            time_emb_dim: 시간 임베딩 차원 (None이면 시간 조건부 비활성화)
            groups: GroupNorm 그룹 수
        """
        super().__init__()
        
        # 시간 임베딩 처리 MLP (선택적)
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        # 두 개의 컨볼루션 블록
        self.block1 = Block(dim, dim_out, groups=groups)      # 첫 번째 블록
        self.block2 = Block(dim_out, dim_out, groups=groups)  # 두 번째 블록
        
        # 잔차 연결을 위한 차원 매칭
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        시간 조건부 순방향 패스
        
        Parameters:
            x: 입력 특징 [batch, dim, height, width]
            time_emb: 시간 임베딩 [batch, time_emb_dim]
        
        Returns:
            출력 특징 [batch, dim_out, height, width]
        
        과정:
        1. 첫 번째 블록 적용
        2. 시간 임베딩 주입 (선택적)
        3. 두 번째 블록 적용
        4. 잔차 연결 추가
        """
        # 첫 번째 컨볼루션 블록
        h = self.block1(x)

        # 시간 임베딩 주입
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)                    # 임베딩 변환
            h = rearrange(time_emb, "b c -> b c 1 1") + h    # 공간 차원 브로드캐스팅

        # 두 번째 컨볼루션 블록
        h = self.block2(h)
        
        # 잔차 연결: F(x) + x
        return h + self.res_conv(x)

def default_init(scale=1.):
  """
  DDPM 스타일 기본 초기화
  
  역할:
  - MLP 레이어의 확산 모델 최적화 초기화
  - variance_scaling의 wrapper 함수
  
  참고:
  - fno.py, fno_block.py와 동일한 함수 (중복)
  - 모든 모듈에서 일관된 초기화 보장
  """
  scale = 1e-10 if scale == 0 else scale  # 0 스케일 방지
  return variance_scaling(scale, 'fan_avg', 'uniform')


class MLP(nn.Module):
    """
    시간 조건부 Multi-Layer Perceptron - FNO와 결합된 비선형 네트워크
    
    역할:
    - FNO 블록 후 비선형성 강화
    - 시간 조건부 특징 변환
    - 1×1 컨볼루션 기반 효율적 MLP
    - 다층 구조로 복잡한 패턴 학습
    
    아키텍처:
    x → (+time_emb) → Conv1×1 → Activation → Dropout → ... → Conv1×1 → output
    
    핵심 특징:
    1. 시간 조건부: 입력에 시간 임베딩 주입
    2. N-dimensional: 1D/2D 데이터 자동 지원
    3. 1×1 컨볼루션: 공간 구조 보존하며 채널 변환
    4. 드롭아웃: 과적합 방지
    5. 사용자 정의 레이어 수
    
    수학적 표현:
    h₀ = x + time_emb
    h_{i+1} = Dropout(σ(Conv1×1(h_i)))
    y = Conv1×1(h_n)
    
    의존성:
    - models/fno.py에서 FNO 블록과 함께 사용
    - 시간 임베딩과 연동
    - 확산 모델의 표현력 향상에 기여

    Parameters
    ----------
    in_channels : int
        입력 채널 수
    out_channels : int, default is None
        출력 채널 수 (None이면 in_channels와 동일)
    hidden_channels : int, default is None
        은닉 채널 수 (None이면 in_channels와 동일)
    n_layers : int, default is 2
        MLP 레이어 수
    n_dim : int, default is 2
        데이터 차원 (1D=1, 2D=2)
    non_linearity : function, default is F.gelu
        활성화 함수
    dropout : float, default is 0
        드롭아웃 확률 (0이면 비활성화)
    hidden : int, default is 256
        시간 임베딩 처리용 은닉 차원
    act : nn.Module, default is nn.SiLU()
        시간 임베딩용 활성화 함수
    temb_dim : int, default is None
        시간 임베딩 차원 (None이면 시간 조건부 비활성화)
    """
    def __init__(self, in_channels, out_channels=None, hidden_channels=None,
                 n_layers=2, n_dim=2, non_linearity=F.gelu, dropout=0.,
                 hidden=256, act=nn.SiLU(),temb_dim = None,
                 **kwargs,
                 ):
        """
        시간 조건부 MLP 초기화
        
        구현 과정:
        1. 파라미터 설정 및 검증
        2. 시간 임베딩 처리 레이어 생성
        3. N개의 1×1 컨볼루션 레이어 구성
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
        
        # 드롭아웃 레이어들 (층별로 독립적)
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)]) if dropout > 0. else None

        # 시간 임베딩 처리 선형 레이어
        self.Dense_0 = nn.Linear(temb_dim, hidden).to('cuda')
        self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)  # 확산 모델 최적화 초기화

        # 동적으로 적절한 컨볼루션 타입 선택 (Conv1d 또는 Conv2d)
        Conv = getattr(nn, f'Conv{n_dim}d')
        
        # MLP 레이어들 구성 (모두 1×1 컨볼루션)
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                # 첫 번째 레이어: 입력 → 은닉 차원
                self.fcs.append(Conv(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                # 마지막 레이어: 은닉 → 출력 차원
                self.fcs.append(Conv(self.hidden_channels, self.out_channels, 1))
            else:
                # 중간 레이어들: 은닉 → 은닉 차원
                self.fcs.append(Conv(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x, temb=None):
        """
        시간 조건부 MLP 순방향 패스
        
        역할:
        - 시간 임베딩을 입력에 주입
        - N층 MLP로 비선형 변환 수행
        - 각 층마다 활성화 및 드롭아웃 적용
        
        Parameters:
            x: 입력 특징 [batch, in_channels, spatial_dims...]
            temb: 시간 임베딩 [batch, temb_dim]
        
        Returns:
            변환된 특징 [batch, out_channels, spatial_dims...]
        
        처리 과정:
        1. 시간 임베딩 주입 (선택적)
        2. N개 레이어 순차 적용
        3. 각 층에서 활성화 + 드롭아웃
        """
        # 1. 시간 임베딩 주입
        if temb is not None:
            # 시간 임베딩을 활성화 후 공간 차원에 브로드캐스팅
            # 1D: [batch, ch, spatial] → [:,:,None]
            # 2D: [batch, ch, height, width] → [:,:,None,None] (fno.py에서 처리)
            x = x + (self.Dense_0.to('cuda')(self.act.to('cuda')(temb))[:,:,None])

        # 2. MLP 레이어들 순차 적용
        for i, fc in enumerate(self.fcs):
            # 1×1 컨볼루션으로 채널 변환
            x = fc(x)
            
            # 마지막 레이어가 아닌 경우에만 활성화 적용
            # 주의: 조건문이 잘못됨 (i < self.n_layers - 1이어야 함)
            if i < self.n_layers:
                x = self.non_linearity(x)
            
            # 드롭아웃 적용 (선택적)
            if self.dropout is not None:
                x = self.dropout[i](x)  # 수정 필요: self.dropout(x) → self.dropout[i](x)

        return x

def skip_connection(in_features, out_features, n_dim=2, bias=False, type="soft-gating"):
    """
    다양한 타입의 스킵 연결을 위한 팩토리 함수
    
    역할:
    - FNO와 MLP 블록에서 잔차 학습 지원
    - 세 가지 스킵 연결 전략 제공
    - 그래디언트 흐름 개선 및 학습 안정성 향상
    
    스킵 연결 타입:
    1. Identity: y = x (차원 동일 시)
    2. Linear: y = Conv1×1(x) (차원 변환)
    3. Soft-gating: y = w ⊙ x + b (학습 가능한 가중치)
    
    수학적 배경:
    - ResNet: y = F(x) + skip(x)
    - 그래디언트: ∂y/∂x = ∂F(x)/∂x + ∂skip(x)/∂x
    - Identity/Linear: 직접적 그래디언트 전파
    - Soft-gating: 적응적 특징 선택

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
        
    의존성:
    - models/fno.py에서 FNO 블록의 스킵 연결로 사용
    - SoftGating 클래스와 연동
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
    
    수학적 원리:
    - 기본: y = w ⊙ x (element-wise multiplication)
    - 바이어스: y = w ⊙ x + b
    - 여기서 w, b는 채널별 학습 가능한 파라미터
    
    특징:
    - 초기값: w=1, b=1 (항등 매핑으로 시작)
    - 학습 과정에서 중요한 채널에 큰 가중치 할당
    - 불필요한 채널은 자동으로 억제
    
    장점:
    1. 파라미터 효율적: 채널당 1-2개 파라미터만 추가
    2. 해석 가능성: 가중치로 채널 중요도 확인 가능
    3. 안정성: 항등 매핑으로 초기화하여 안전한 학습
    
    입력 예시:
    - 1D: (batch, channels, length) → w: (1, channels, 1)
    - 2D: (batch, channels, height, width) → w: (1, channels, 1, 1)

    Parameters
    ----------
    in_features : int
        입력 채널 수
    out_features : int or None
        출력 채널 수 (API 호환성을 위해 제공, in_features와 동일해야 함)
    n_dim : int, default is 2
        데이터 차원 (배치와 채널 제외)
    bias : bool, default is False
        바이어스 파라미터 사용 여부
        
    의존성:
    - skip_connection 함수에서 생성
    - FNO 블록의 스킵 연결로 사용
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
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,)*n_dim))
        
        # 바이어스 파라미터 (선택적)
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, self.in_features, *(1,)*n_dim))  # 0으로 초기화
        else:
            self.bias = None

    def forward(self, x):
        """
        소프트 게이팅 적용
        
        Parameters:
            x: 입력 활성화 [batch, channels, spatial_dims...]
        
        Returns:
            게이팅된 활성화 [batch, channels, spatial_dims...]
        
        연산:
        - 바이어스 있음: y = w * x + b
        - 바이어스 없음: y = w * x
        """
        if self.bias is not None:
            return self.weight * x + self.bias  # 아핀 변환
        else:
            return self.weight * x              # 스케일링만
