# 유틸리티 모듈
# HDM에 필요한 다양한 보조 함수들을 제공
# Hilbert 공간 노이즈 생성, DCT 변환, 모델 파라미터 카운팅 등

import os

import torch
from torch import nn, einsum
import tqdm
import numpy as np
from einops import rearrange

from tqdm.asyncio import trange, tqdm

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from scipy.spatial.distance import cdist

def kernel(x, y, gain=1.0, lens=1, metric='seuclidean', device='cuda'):
    """
    1D 데이터를 위한 커널 함수 계산
    
    역할:
    - Hilbert 공간에서의 노이즈 생성을 위한 커널 행렬 계산
    - Squared Exponential 커널을 사용하여 데이터 점들 간의 유사도 측정
    - 거리 기반 커널로 공간적 상관관계 모델링
    
    구현 방식:
    1. 입력 데이터를 CPU로 이동 및 reshape
    2. scipy의 cdist로 거리 행렬 계산
    3. SE 커널 공식 적용: K(x,y) = gain * exp(-dist/lens)
    4. 결과를 GPU로 이동
    
    Parameters:
        x, y: 1D 데이터 점들
        gain: 커널의 최대 값 (기본값: 1.0)
        lens: 커널의 길이 스케일 (기본값: 1)
        metric: 거리 측정 방법 (기본값: 'seuclidean')
        device: 계산 디바이스
    
    Returns:
        K: 커널 행렬 [len(x), len(y)]
    
    의존성:
        - hilbert_noise 클래스에서 커널 행렬 생성 시 사용
    """
    # 수치 안정성을 위해 CPU에서 계산
    x = x.cpu()
    y = y.cpu()
    
    # 1D 데이터를 2D로 reshape ([N] -> [N, 1])
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    
    # 모든 점 쌍 간의 거리 행렬 계산
    dist = cdist(x, y, metric=metric)
    
    # numpy에서 PyTorch tensor로 변환 및 스케일링
    K = torch.from_numpy(dist / lens)
    
    # Squared Exponential 커널 적용
    K = gain * torch.exp(-K).to(torch.float32)
    
    # 지정된 디바이스로 이동
    return K.to(device).to(torch.float32)


class hilbert_noise:
    """
    1D 데이터를 위한 Hilbert 공간 노이즈 생성 클래스
    
    역할:
    - 커널 기반 구조화된 노이즈를 생성
    - Gaussian Process의 샘플링과 유사하지만 HDM에 맞게 최적화
    - 서로 다른 해상도에서의 노이즈 생성 지원
    
    구현 방식:
    1. 커널 행렬 K 계산
    2. 고유값 분해로 커널의 주성분 추출
    3. sqrt(D) * V^T 형태로 변환 행렬 M 생성
    4. 백색 노이즈에 M을 곱하여 구조화된 노이즈 생성
    
    수학적 배경:
    - K = V * D * V^T (고유값 분해)
    - M = V * sqrt(D)
    - 샘플 = M * 백색노이즈 (Cholesky 분해와 유사)
    
    의존성:
        - runner/hilbert_runner.py에서 HilbertNoise로 사용
        - functions/sampler.py에서 샘플링 시 노이즈 생성
    """
    
    def __init__(self, config, device='cuda'):
        """
        Hilbert 노이즈 생성기 초기화
        
        Parameters:
            config: 설정 객체 (diffusion 관련 파라미터 포함)
            device: 계산 디바이스
        """
        # 기본 파라미터 설정
        self.grid = grid = config.diffusion.grid  # 격자 점 개수
        self.device = device
        self.metric = metric = config.diffusion.metric  # 거리 측정 방법
        self.initial_point = config.diffusion.initial_point  # 시작점
        self.end_point = config.diffusion.end_point  # 끝점

        # 1D 도메인 격자 생성
        self.x = torch.linspace(config.diffusion.initial_point, config.diffusion.end_point, grid).to(self.device)
        
        # 커널 파라미터
        self.lens = lens = config.diffusion.lens  # 길이 스케일
        self.gain = gain = config.diffusion.gain  # 강도 스케일
        
        # 1D 데이터를 위한 커널 행렬 계산
        K = kernel(self.x, self.x, lens=lens, gain=gain, metric=metric, device=device)
        
        # 고유값 분해 (Eigendecomposition)
        # 1e-6 * I 추가로 수치 안정성 향상 (양정치 행렬 보장)
        eig_val, eig_vec = torch.linalg.eigh(K + 1e-6 * torch.eye(K.shape[0], K.shape[0]).to(self.device))
        
        self.eig_val = eig_val.to(self.device) 
        self.eig_vec = eig_vec.to(torch.float32).to(self.device) 
        
        # 고유값 범위 확인 (디버깅용)
        print('eig_val', eig_val.min(), eig_val.max())
        
        # 고유값 대각 행렬 생성
        self.D = torch.diag(self.eig_val).to(torch.float32).to(self.device) 
        
        # 변환 행렬 M = V * sqrt(D) 계산
        # 이후 백색 노이즈에 곱하여 구조화된 노이즈 생성
        self.M = torch.matmul(self.eig_vec, torch.sqrt(self.D)).to(self.device)
 
    def sample(self, size):
        """
        Hilbert 공간에서 1D 데이터 샘플 생성
        
        역할:
        - 백색 가우시안 노이즈를 구조화된 노이즈로 변환
        - 커널에 의해 정의된 공간적 상관관계를 가진 노이즈 생성
        
        구현 방식:
        1. 지정된 크기의 백색 노이즈 생성
        2. einsum을 사용하여 효율적인 행렬 곱셈 수행
        3. 결과적으로 커널에 의해 정의된 노이즈 분포 획득
        
        Parameters:
            size: 샘플 크기 [batch, channels, grid]
            
        Returns:
            output: 구조화된 노이즈 샘플 [batch, channels, grid]
            
        사용 예:
            - functions/sampler.py에서 샘플링 과정에서 호출
            - runner/hilbert_runner.py에서 학습 데이터 생성 시 사용
        """
        size = list(size)  # batch*ch*grid
        
        # 백색 가우시안 노이즈 생성
        x_0 = torch.randn(size).to(self.device)  

        # Hilbert 공간으로 변환
        # 'g k, b c k -> b c g': 격자 차원에서 변환 행렬 M과 노이즈를 곱셈
        output = einsum('g k, b c k -> b c g', self.M, x_0)

        return output  # (batch, ch, grid)

    def free_sample(self, resolution_grid):
        """
        다른 해상도에서 1D 데이터 샘플 생성
        
        역할:
        - 기존 격자와 다른 해상도에서 노이즈 생성
        - 고해상도 샘플링이나 다양한 도메인 크기에서 사용
        - 커널의 보간 성질을 활용한 확장
        
        구현 방식:
        1. 새로운 해상도로 도메인 격자 생성
        2. 기존 격자와 새 격자 간의 커널 행렬 계산
        3. 고유벡터를 사용하여 새 도메인에서의 변환 행렬 생성
        
        Parameters:
            resolution_grid: 새로운 해상도 (격자 점 개수)
            
        Returns:
            N: 새로운 해상도에서의 변환 행렬
            
        사용 예:
            - 다양한 해상도에서 노이즈 생성
            - Super-resolution 작업
        """
        # 새로운 해상도로 도메인 격자 생성
        y = torch.linspace(self.initial_point, self.end_point, resolution_grid).to(self.device)
        
        # 기존 격자 x와 새 격자 y 간의 커널 행렬 계산
        K = kernel(self.x, y, lens=self.lens, gain=self.gain, device=self.device)
        
        # 고유벡터를 사용하여 새 도메인에서의 변환 행렬 생성
        # 'g k, g r -> r k': 고유벡터와 커널 행렬의 행렬 곱셈
        N = einsum('g k, g r -> r k', self.eig_vec, K)
        
        return N

"""
DCT (Discrete Cosine Transform) 모듈

Taken from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
Some modifications have been made to work with newer versions of Pytorch

역할:
- 1D 신호 처리를 위한 DCT/IDCT 변환 구현
- 주파수 도메인에서의 신호 분석 및 처리
- PyTorch 기반으로 최적화되어 GPU 가속 지원

의존성:
- 이 프로젝트에서는 1D 데이터용으로만 사용
- functions/ihdm.py에서 참조되지만 실제로는 사용되지 않음
"""

import numpy as np
import torch
import torch.nn as nn


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    
    역할:
    - 1D 신호를 주파수 도메인으로 변환
    - 신호의 주파수 성분 분석을 위해 사용
    - FFT를 사용하여 효율적으로 구현
    
    구현 방식:
    1. 입력 신호를 짝수/홀수 인덱스로 재배열
    2. FFT를 적용하여 주파수 도메인으로 변환
    3. 코사인/사인 가중치를 적용하여 DCT 계수 계산
    4. 정규화 옵션 적용
    
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    
    Parameters:
        x: the input signal (마지막 차원에 대해 DCT 수행)
        norm: the normalization, None or 'ortho'
        
    Returns:
        the DCT-II of the signal over the last dimension
    """
    # 입력 텐서의 모양 저장
    x_shape = x.shape
    N = x_shape[-1]  # 마지막 차원의 크기
    
    # 2D로 리쉬이프하여 배치 처리 가능하게 만듦
    x = x.contiguous().view(-1, N)

    # DCT-II를 위한 전처리: 짝수/홀수 인덱스 재배열
    # 짝수 인덱스는 순서대로, 홀수 인덱스는 역순으로 배치
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # FFT 적용 (복소수 결과를 실수/허수 부로 분리)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    
    # DCT 계수를 위한 코사인/사인 가중치 계산
    k = - torch.arange(N, dtype=x.dtype,
                       device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)  # 코사인 가중치
    W_i = torch.sin(k)  # 사인 가중치

    # 복소수 곱셈을 실수 연산으로 구현: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    # 직교 정규화 옵션
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2      # DC 성분 정규화
        V[:, 1:] /= np.sqrt(N / 2) * 2  # 나머지 성분 정규화

    # 최종 스케일링 및 원래 모양으로 복원
    V = 2 * V.view(*x_shape)

    return V

def dct_shift(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    #Vc = torch.fft.rfft(v, 1)
    Vc = torch.view_as_real(torch.fft.fftshift(torch.fft.fft(v, dim=1), dim=1))
    
    k = - torch.arange(N, dtype=x.dtype,
                       device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype,
                     device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    #v = torch.fft.irfft(V, 1)
    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

def idct_shift(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype,
                     device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    #v = torch.fft.irfft(V, 1)
    v = torch.fft.irfft(torch.fft.fftshift(torch.view_as_complex(V), dim=1), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


# Removed 2D/3D DCT functions as they are not needed for 1D data processing


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""

    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False  # don't learn this!


if __name__ == '__main__':
    x = torch.Tensor(1000, 4096)
    x.normal_(0, 1)
    linear_dct = LinearDCT(4096, 'dct')
    error = torch.abs(dct(x) - linear_dct(x))
    assert error.max() < 1e-3, (error, error.max())
    linear_idct = LinearDCT(4096, 'idct')
    error = torch.abs(idct(x) - linear_idct(x))
    assert error.max() < 1e-3, (error, error.max())

# Removed all image-related classes and functions (DCTBlur, Snow, rgb2hsv, hsv2rgb, rgb2lab, lab2rgb, etc.)
# as they are not needed for 1D data processing
from prettytable import PrettyTable

def count_parameters(model):
    """
    모델의 파라미터 수를 계산하고 표시하는 유틸리티 함수
    
    역할:
    - 모델의 각 모듈별 파라미터 수 계산
    - 학습 가능한 파라미터만 카운트 (requires_grad=True)
    - 가독성 좋은 표 형태로 결과 출력
    
    사용 예:
    - 모델 비교 및 분석
    - 메모리 사용량 추정
    - 모델 복잡도 평가
    
    Parameters:
        model: PyTorch 모델 객체
        
    Returns:
        total_params: 전체 학습 가능 파라미터 수
        
    의존성:
        - 모델 분석 및 디버깅 시 사용
        - prettytable 라이브러리 필요
    """
    # 모듈별 파라미터 정보를 담을 표 생성
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    
    # 모든 모듈을 순회하며 파라미터 수 계산
    for name, parameter in model.named_parameters():
        # 학습 가능한 파라미터만 카운트
        if not parameter.requires_grad: 
            continue
        
        # 해당 파라미터의 원소 수 계산
        param = parameter.numel()
        
        # 표에 모듈 이름과 파라미터 수 추가
        table.add_row([name, param])
        total_params += param
    
    # 결과 출력
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
