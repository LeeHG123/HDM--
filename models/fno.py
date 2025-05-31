# FNO (Fourier Neural Operator) 메인 모듈
# HDM에서 사용하는 완전한 FNO 아키텍처 구현
# 시간 조건부 확산 모델과 통합된 Fourier 기반 신경망 연산자

"""
FNO 메인 모듈 개요
=================

이 모듈은 HDM(Hilbert Diffusion Model)의 백본 네트워크로 사용되는
완전한 FNO 아키텍처를 구현합니다. 주파수 도메인에서의 전역적 
컨볼루션 연산을 통해 효율적인 PDE 기반 생성 모델링을 지원합니다.

주요 구성 요소:
1. 시간 임베딩 함수들 (get_timestep_embedding, get_timestep_embedding_2)
2. Lifting/Projection 레이어 (차원 변환)
3. FNO 메인 클래스 (시간 조건부 Fourier 연산자)
4. 설정 기반 FNO2 클래스 (구성 파일 기반 초기화)

핵심 특징:
- 시간 조건부 확산 모델 지원
- 텐서 인수분해를 통한 효율적 구현
- 다양한 정규화 및 스킵 연결 옵션
- MLP와 결합된 하이브리드 아키텍처
- 1D/2D 데이터 처리 지원

수학적 배경:
FNO는 다음 연산자 학습 문제를 해결합니다:
G†: u₀ → u_T (확산 과정의 역변환)

여기서 각 레이어는:
x ↦ σ(W·x + F⁻¹(R·F(x)))

- W: 로컬 변환 (1×1 컨볼루션)
- F: Fourier 변환
- R: 학습 가능한 스펙트럼 가중치
- σ: 비선형 활성화

시간 조건부 확장:
x_t ↦ σ(W·x_t + temb + F⁻¹(R(temb)·F(x_t)))

의존성:
- models/fno_block.py: FNO 핵심 구성 요소
- models/mlp.py: MLP 보조 네트워크
- functions/: 확산 과정과 연동
- 설정 파일을 통한 하이퍼파라미터 관리
"""

import torch.nn as nn
import torch.nn.functional as F
from functools import partial, partialmethod
import torch
import math
from models.fno_block import *  # FactorizedSpectralConv, SinPositionEmbeddingsClass 등
from models.mlp import *        # MLP, skip_connection 등

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """
  분산 스케일링 기반 가중치 초기화 (JAX에서 이식)
  
  역할:
  - FNO 네트워크의 안정적인 학습을 위한 가중치 초기화
  - Fan-in/Fan-out 기반 분산 조절로 그래디언트 흐름 최적화
  - 확산 모델에 적합한 초기화 제공
  
  참고: fno_block.py의 동일 함수와 중복 (코드 정리 필요)
  """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    """팬 수 계산: 입력/출력 뉴런과 receptive field 크기"""
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
  DDPM 스타일 기본 초기화
  
  역할:
  - 확산 모델에 최적화된 가중치 초기화
  - 시간 임베딩 레이어 초기화에 주로 사용
  """
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  """
  확산 모델용 시간 스텝 임베딩 생성 (기본 버전)
  
  역할:
  - 확산 과정의 시간 t를 연속적인 벡터로 인코딩
  - Transformer의 위치 인코딩을 시간 도메인에 적용
  - 시간 조건부 생성 모델의 핵심 구성 요소
  
  구현 특징:
  - 1000배 스케일링으로 확산 모델 관례 따름
  - 사인/코사인 주기 함수로 시간 관계 표현
  - max_positions=10000 (Transformer 표준)
  
  Parameters:
      timesteps: 시간 스텝 텐서 [batch_size]
      embedding_dim: 임베딩 차원
      max_positions: 최대 위치 (주파수 스케일 결정)
  
  Returns:
      시간 임베딩 벡터 [batch_size, embedding_dim]
  
  수학적 공식:
  - PE(t, 2i) = sin(t / 10000^(2i/d))
  - PE(t, 2i+1) = cos(t / 10000^(2i/d))
  """
  assert len(timesteps.shape) == 1  # 1D 시간 스텝 배열 필요
  half_dim = embedding_dim // 2
  timesteps = 1000*timesteps  # 확산 모델 스케일링
  
  # Transformer에서 사용하는 매직 넘버 10000
  emb = math.log(max_positions) / (half_dim - 1)
  
  # 주파수 스케일 생성: exp(-i * log(max_pos) / (half_dim - 1))
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  
  # 시간과 주파수의 외적으로 위상 계산
  emb = timesteps.float()[:, None] * emb[None, :]
  
  # 사인/코사인 임베딩 생성
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  
  # 홀수 차원일 경우 제로 패딩
  if embedding_dim % 2 == 1:
    emb = F.pad(emb, (0, 1), mode='constant')
  
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


def get_timestep_embedding_2(timesteps, embedding_dim, max_positions=111):
    """
    조건부 확산 모델용 시간 스텝 임베딩 (보조 버전)
    
    역할:
    - 클래스 조건부 생성을 위한 추가 임베딩
    - 기본 시간 임베딩과 결합하여 다중 조건 지원
    - 다른 max_positions로 주파수 다양성 제공
    
    차이점:
    - max_positions=111 (기본 10000과 다름)
    - 1000배 스케일링 없음 (원본 시간 사용)
    - 클래스 레이블이나 추가 조건 인코딩용
    
    사용 예:
    - FNO2.forward()에서 y(클래스) 조건과 함께 사용
    - temb = get_timestep_embedding(t) + get_timestep_embedding_2(y)
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2

    # 다른 주파수 스케일 사용 (max_positions=111)
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    
    # 디바이스 일관성 보장
    emb = timesteps.float()[:, None].to(timesteps.device)* emb[None, :].to(timesteps.device)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class Lifting(nn.Module):
    """
    FNO의 입력 차원 변환 레이어 (Lifting Layer)
    
    역할:
    - 저차원 입력을 고차원 잠재 공간으로 변환
    - 후속 Fourier 레이어가 작업할 충분한 표현력 제공
    - 1×1 컨볼루션을 통한 채널 차원 확장
    
    수학적 의미:
    - P: R^d → R^h (d-차원 입력을 h-차원으로 lifting)
    - 일반적으로 h >> d (예: 3 → 256)
    
    구현 특징:
    - N차원 데이터 지원 (1D, 2D 등)
    - 1×1 컨볼루션 사용으로 공간 구조 보존
    - 파라미터 효율적 (kernel_size=1)
    
    의존성:
    - FNO/FNO2의 첫 번째 레이어로 사용
    - FactorizedSpectralConv 이전에 적용
    """
    def __init__(self, in_channels, out_channels, n_dim=2):
        """
        Parameters:
            in_channels: 입력 채널 수 (예: 1 for 1D data)
            out_channels: 출력 채널 수 (예: 256)
            n_dim: 데이터 차원 (1D=1, 2D=2)
        """
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
    - 고차원 잠재 표현을 최종 출력 차원으로 변환
    - 2단계 MLP 구조로 비선형 변환 제공
    - FNO 처리 후 원하는 출력 형태로 복원
    
    수학적 의미:
    - Q: R^h → R^o (h-차원 잠재에서 o-차원 출력으로 projection)
    - 비선형 변환: Q(x) = W₂·σ(W₁·x)
    
    구현 특징:
    - 2층 MLP 구조 (fc1 → activation → fc2)
    - 1×1 컨볼루션으로 공간 구조 보존
    - 사용자 정의 활성화 함수 지원
    - 은닉 차원 조절 가능
    
    의존성:
    - FNO/FNO2의 마지막 레이어로 사용
    - FactorizedSpectralConv 처리 후 적용
    """
    def __init__(self, in_channels, out_channels, hidden_channels=None, n_dim=2, non_linearity=F.gelu):
        """
        Parameters:
            in_channels: 입력 채널 수 (잠재 차원)
            out_channels: 출력 채널 수 (최종 출력)
            hidden_channels: 은닉 차원 (None이면 in_channels 사용)
            n_dim: 데이터 차원
            non_linearity: 활성화 함수 (기본: GELU)
        """
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
        
        과정:
        1. 첫 번째 1×1 컨볼루션
        2. 비선형 활성화
        3. 두 번째 1×1 컨볼루션 (최종 출력)
        """
        x = self.fc1(x)                    # 중간 차원으로 변환
        x = self.non_linearity(x)          # 비선형 활성화
        x = self.fc2(x)                    # 최종 출력 차원으로 변환
        return x


class FNO(nn.Module):
    """
    N차원 Fourier Neural Operator - HDM의 메인 백본 네트워크
    
    역할:
    - 확산 모델의 노이즈 예측 네트워크로 사용
    - 주파수 도메인에서 전역적 컨볼루션 수행
    - 시간 조건부 생성 모델링 지원
    - 다차원 과학 데이터 처리
    
    아키텍처 구조:
    1. Lifting: 입력을 고차원 잠재 공간으로 변환
    2. FNO Blocks: N개의 Fourier 레이어 스택
       - FactorizedSpectralConv: 주파수 도메인 컨볼루션
       - Skip connections: 그래디언트 흐름 개선
       - MLP layers: 비선형성 강화 (선택적)
       - Normalization: 안정적 학습 (선택적)
    3. Projection: 잠재 표현을 출력 차원으로 변환
    
    시간 조건부 특징:
    - 시간 임베딩을 통한 확산 스텝 인코딩
    - 각 FNO 블록에서 시간 정보 주입
    - 적응적 주파수 응답 학습
    
    수학적 표현:
    각 FNO 레이어: v_{l+1} = σ(W·v_l + (F⁻¹·R_l·F)(v_l) + temb)
    
    여기서:
    - v_l: l번째 레이어의 출력
    - W: 로컬 변환 행렬
    - F: Fourier 변환
    - R_l: 학습 가능한 스펙트럼 가중치
    - temb: 시간 임베딩
    - σ: 활성화 함수
    
    핵심 장점:
    - 전역 수용장(global receptive field)
    - O(N log N) 계산 복잡도
    - 텐서 인수분해를 통한 파라미터 효율성
    - 다해상도 처리 가능
    
    의존성:
    - models/fno_block.py: FactorizedSpectralConv
    - models/mlp.py: MLP, skip_connection
    - functions/: 확산 과정 모듈들
    
    Parameters
    ----------
    n_modes : int tuple
        각 차원별 유지할 Fourier 모드 수
        차원은 len(n_modes)로 자동 추론
    hidden_channels : int
        FNO의 너비 (채널 수)
    act : nn.Module, default=nn.SiLU()
        활성화 함수
    in_channels : int, default=3
        입력 채널 수
    out_channels : int, default=1
        출력 채널 수
    lifting_channels : int, default=256
        Lifting 블록의 은닉 채널 수
    projection_channels : int, default=256
        Projection 블록의 은닉 채널 수
    n_layers : int, default=4
        Fourier 레이어 수
    use_mlp : bool, default=True
        각 FNO 블록 후 MLP 사용 여부
    mlp : dict, default={'expansion': 4.0, 'dropout': 0.}
        MLP 파라미터 {'expansion': float, 'dropout': float}
    non_linearity : nn.Module, default=F.gelu
        비선형 활성화 함수
    norm : str, default='group_norm'
        정규화 레이어 타입 ('instance_norm', 'group_norm', 'layer_norm', None)
    preactivation : bool, default=True
        ResNet 스타일 pre-activation 사용 여부
    skip : str, default='soft-gating'
        스킵 연결 타입 ('linear', 'identity', 'soft-gating')
    separable : bool, default=True
        Depthwise separable 스펙트럼 컨볼루션 사용 여부
    factorization : str or None, default=None
        텐서 인수분해 방법 ('tucker', 'cp', 'tt', None)
    rank : float, default=1.0
        텐서 인수분해 랭크
    joint_factorization : bool, default=False
        모든 레이어를 단일 텐서로 매개화할지 여부
    fixed_rank_modes : bool, default=False
        인수분해하지 않을 모드
    implementation : str, default='factorized'
        순방향 패스 구현 방식 ('factorized', 'reconstructed')
    decomposition_kwargs : dict, default={}
        텐서 분해 추가 파라미터
    fft_norm : str, default='ortho'
        FFT 정규화 방법
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
                 fft_norm='ortho',
                 **kwargs):
        super().__init__()
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
        self.domain_padding = None

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
        FNO의 시간 조건부 순방향 패스
        
        역할:
        - 확산 모델의 노이즈 예측 함수 ε_θ(x_t, t) 구현
        - 시간 스텝 t에 조건화된 Fourier 연산자 적용
        - 주파수 도메인에서 전역적 패턴 학습
        
        알고리즘 흐름:
        1. 입력 전처리 및 차원 조정
        2. Lifting: 저차원 → 고차원 잠재 공간
        3. 시간 임베딩 생성 및 주입
        4. N개 FNO 블록 순차 처리:
           - Pre/Post-activation 선택적 적용
           - 스펙트럼 컨볼루션 (핵심 연산)
           - 스킵 연결로 잔차 학습
           - MLP로 비선형성 강화 (선택적)
        5. Projection: 잠재 공간 → 출력 차원
        
        Parameters:
            x: 노이즈가 추가된 입력 [batch, channels, spatial_dims...]
            t: 확산 시간 스텝 [batch]
        
        Returns:
            예측된 노이즈 또는 스코어 [batch, out_channels, spatial_dims...]
        """
        # 1. 입력 차원 조정 (1D 데이터의 경우)
        if x.dim()==2:
            x = x.unsqueeze(1)  # [batch, spatial] → [batch, 1, spatial]

        # 2. Lifting: 입력을 고차원 잠재 공간으로 변환
        x = self.lifting(x)  # [batch, in_ch, ...] → [batch, hidden_ch, ...]

        # 3. 시간 임베딩 생성 및 처리
        m_idx = 0
        
        # 시간 스텝을 사인/코사인 임베딩으로 변환
        temb = get_timestep_embedding(t, self.lifting_channels)
        
        # 첫 번째 MLP 레이어: 임베딩 차원 → 은닉 차원
        temb = self.Dense[m_idx].to('cuda')(temb.to('cuda'))
        m_idx += 1
        
        # 두 번째 MLP 레이어: 활성화 후 은닉 차원 유지
        temb = self.Dense[m_idx].to('cuda')(self.act.to('cuda')(temb.to('cuda')))
        m_idx += 1

        # 4. 시간 정보 주입: 공간 차원 브로드캐스팅
        x = x + temb[:,:,None]  # 1D의 경우: [batch, ch, spatial]
        
        # 5. FNO 블록들 순차 처리
        for i in range(self.n_layers):

            # Pre-activation 패턴 (ResNet 스타일)
            if self.preactivation:
                x = self.non_linearity(x)       # 활성화 먼저
                if self.norm is not None:
                    x = self.norm[i](x)          # 정규화 적용

            # 핵심 스펙트럼 컨볼루션 연산
            x_fno = self.convs[i]((x,temb))      # Fourier 도메인 처리

            # Post-activation 정규화 (필요시)
            if not self.preactivation and self.norm is not None:
                x_fno = self.norm[i](x_fno)

            # 스킵 연결: 잔차 학습 지원
            x_skip = self.fno_skips[i](x)
            x = x_fno + x_skip

            # Post-activation 패턴
            if not self.preactivation and i < (self.n_layers - 1):
                x = self.non_linearity(x)

            # MLP 블록 (선택적): 비선형성 강화
            if self.mlp is not None:
                x_skip = self.mlp_skips[i](x)    # MLP용 스킵 연결

                # Pre-activation MLP
                if self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)

                # MLP 적용 및 잔차 연결
                x = self.mlp[i](x) + x_skip

                # Post-activation MLP
                if not self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)

        # 6. Projection: 잠재 표현을 최종 출력으로 변환
        x = self.projection(x)  # [batch, hidden_ch, ...] → [batch, out_ch, ...]

        # 7. 출력 차원 조정 (1D 데이터의 경우)
        x = x.squeeze(1)        # [batch, 1, spatial] → [batch, spatial]
        return x


class FNO2(nn.Module):
    """
    설정 기반 N차원 Fourier Neural Operator (Config-driven FNO)
    
    역할:
    - 설정 파일을 통한 하이퍼파라미터 관리
    - 더 유연한 초기화 및 디바이스 관리
    - 조건부 생성 지원 (클래스 조건 등)
    - 실험 재현성 및 확장성 향상
    
    FNO와의 차이점:
    1. 설정 파일 기반 초기화 (config.model.*)
    2. 명시적 디바이스 관리 ('cuda' 강제)
    3. 조건부 임베딩 지원 (y 파라미터)
    4. 더 나은 디바이스 일관성 보장
    5. 2D 데이터 특화 설계
    
    아키텍처:
    - 기본 FNO와 동일한 구조
    - 시간 + 클래스 조건부 임베딩
    - 2D 공간 데이터 처리 최적화
    
    사용 컨텍스트:
    - HDM의 메인 백본 (FNO 대신)
    - 설정 파일 기반 실험 관리
    - 2D 이미지 생성 모델
    
    의존성:
    - configs/ 디렉토리의 YAML 설정 파일
    - FNO와 동일한 구성 요소 재사용
    
    Parameters
    ----------
    config : object
        설정 객체 (config.model 하위 속성들 사용):
        - n_modes : Fourier 모드 수
        - hidden_channels : 은닉 채널 수
        - in_channels : 입력 채널 수
        - out_channels : 출력 채널 수
        - n_layers : 레이어 수
        - 기타 FNO 하이퍼파라미터들
    device : str, default='cuda'
        계산 디바이스 (강제 CUDA 사용)
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

        Dense = [nn.Linear(self.lifting_channels, self.hidden_channels).to(device)]
        Dense[0].weight.data = default_init()(Dense[0].weight.data.shape)
        nn.init.zeros_(Dense[0].bias)
        Dense.append(nn.Linear( self.hidden_channels ,  self.hidden_channels).to(device))
        Dense[1].weight.data = default_init()(Dense[1].weight.data.shape)
        nn.init.zeros_(Dense[1].bias)
        self.Dense = nn.ModuleList(Dense)
        self.domain_padding = None
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


    def forward(self, x,t,y=None):
        """
        FNO2의 조건부 순방향 패스
        
        역할:
        - 시간 + 클래스 조건부 확산 모델 지원
        - 2D 데이터 특화 처리 (이미지 등)
        - 더 강력한 디바이스 일관성 보장
        
        주요 개선사항:
        - 조건부 임베딩 (y) 지원
        - 동적 디바이스 이동 (.to(device))
        - 2D 공간 브로드캐스팅 (None,None)
        
        Parameters:
            x: 입력 데이터 [batch, channels, height, width]
            t: 시간 스텝 [batch]
            y: 조건 라벨 [batch] (선택적)
        
        Returns:
            출력 [batch, out_channels, height, width]
        """

        # 1. Lifting: 입력 차원 변환
        x = self.lifting(x)

        # 2. 시간 및 조건부 임베딩 생성
        m_idx = 0

        # 기본 시간 임베딩
        temb = get_timestep_embedding(t, self.lifting_channels)
        
        # 조건부 임베딩 추가 (클래스 조건 등)
        if y is not None:
            cemb = get_timestep_embedding_2(y, self.lifting_channels)
            temb = temb + cemb  # 시간 + 조건 임베딩 결합

        # 임베딩 MLP 처리 (디바이스 동적 이동)
        temb = self.Dense[m_idx].to(temb.device)(temb)
        m_idx += 1

        temb = self.Dense[m_idx].to(temb.device)(self.act.to(temb.device)(temb))
        m_idx += 1

        # 3. 2D 공간에 시간 정보 브로드캐스팅
        x = x + temb[:,:,None,None]  # [batch, ch, height, width]

        # 4. FNO 블록들 순차 처리
        for i in range(self.n_layers):

            # Pre-activation
            if self.preactivation:
                x = self.non_linearity(x)
                if self.norm is not None:
                    x = self.norm[i](x)

            # 스펙트럼 컨볼루션 (동적 디바이스 이동)
            x_fno = self.convs[i].to(x.device)((x,temb))

            # Post-activation 정규화
            if not self.preactivation and self.norm is not None:
                x_fno = self.norm[i].to(x.device)(x_fno)

            # 스킵 연결 (동적 디바이스 이동)
            x_skip = self.fno_skips[i].to(x.device)(x)
            x = x_fno + x_skip

            # Post-activation
            if not self.preactivation and i < (self.n_layers - 1):
                x = self.non_linearity(x)

            # MLP 블록 (선택적)
            if self.mlp is not None:
                x_skip = self.mlp_skips[i](x)

                if self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)

                # MLP + 스킵 연결 (시간 임베딩 전달)
                x = self.mlp[i].to(x.device)(x, temb) + x_skip

                if not self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)

        # 5. Projection: 최종 출력 생성
        x = self.projection(x)

        return x
    
    
def partialclass(new_name, cls, *args, **kwargs):
    """
    다른 기본값을 가진 새 클래스 생성 유틸리티
    
    역할:
    - 기존 클래스의 특화 버전 생성
    - 특정 하이퍼파라미터가 고정된 변형 클래스 제공
    - 코드 재사용성 및 실험 편의성 향상
    
    배경:
    functools.partial을 사용할 수 있지만 문제점들이 있음:
    1. 클래스에 이름이 없음 (명시적 설정 필요)
    2. functools 객체가 되어 상속 불가
    
    해결책:
    - 동적으로 새 클래스 정의
    - 기존 클래스에서 상속
    - 모든 메서드와 속성 유지
    
    Parameters:
        new_name: 새 클래스의 이름
        cls: 기반 클래스
        *args, **kwargs: 고정할 초기화 인수들
    
    Returns:
        새로운 클래스 타입
    
    사용 예:
        TFNO = partialclass('TFNO', FNO, factorization='Tucker')
        # Tucker 분해가 기본인 FNO 변형
    """
    # 부분 적용된 __init__ 메서드 생성
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    
    # 동적으로 새 클래스 정의
    new_class = type(new_name, (cls,),  {
        '__init__': __init__,      # 새로운 초기화 메서드
        '__doc__': cls.__doc__,    # 기존 문서 유지
        'forward': cls.forward,    # 기존 forward 메서드 유지
    })
    return new_class

# Tucker 분해를 기본으로 하는 FNO 변형들
TFNO   = partialclass('TFNO', FNO, factorization='Tucker')
"""
Tucker Fourier Neural Operator - Tucker 텐서 분해 특화 FNO

특징:
- factorization='Tucker'가 기본값으로 설정
- 고차원 텐서에 효과적인 Tucker 분해 사용
- 파라미터 압축 및 계산 효율성 향상
- FNO의 모든 기능 유지

사용 컨텍스트:
- 메모리 제약이 있는 환경
- 고차원 데이터 처리
- 파라미터 효율적 모델 필요 시
"""

TFNO2   = partialclass('TFNO2', FNO2, factorization='Tucker')
"""
Tucker Fourier Neural Operator 2 - 설정 기반 Tucker FNO

특징:
- 설정 파일 기반 초기화
- Tucker 분해 기본 적용
- 조건부 생성 지원
- 2D 데이터 특화

사용 컨텍스트:
- HDM의 메인 백본으로 사용
- 실험 관리 및 재현성
- 이미지 생성 작업
"""
