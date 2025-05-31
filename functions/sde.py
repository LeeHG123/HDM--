# SDE (Stochastic Differential Equations) 모듈
# 확산 과정을 정의하는 핵심 모듈로, forward/reverse diffusion process를 구현
# HDM의 이론적 기반이 되는 VP-SDE (Variance Preserving SDE)를 1D 데이터용으로 구현

import numpy as np
import torch
import math


class VPSDE1D:
    """
    1D 데이터를 위한 Variance Preserving SDE (VP-SDE) 클래스
    
    역할:
    - Forward diffusion process: 원본 데이터에 점진적으로 노이즈를 추가
    - 각 시간 t에서의 평균과 분산을 계산하여 샘플링에 사용
    - Cosine schedule을 사용하여 노이즈 추가 속도를 조절
    
    구현 방식:
    - beta(t): 시간에 따른 노이즈 스케줄 함수
    - marginal_log_mean_coeff: log(alpha(t)) 계산
    - marginal_std: 시간 t에서의 표준편차 계산
    
    의존성:
    - functions/loss.py: 손실 함수 계산 시 사용
    - functions/sampler.py: 샘플링 과정에서 사용
    - runner/hilbert_runner.py: 학습 및 샘플링에서 사용
    """

    def __init__(self, schedule='cosine'):
        # VP-SDE 하이퍼파라미터
        self.beta_0 = 0.01  # 시작 노이즈 레벨
        self.beta_1 = 20    # 최종 노이즈 레벨
        self.cosine_s = 0.008  # Cosine schedule의 offset 파라미터
        self.schedule = schedule  # 노이즈 스케줄 타입
        self.cosine_beta_max = 999.  # Cosine schedule의 최대 beta 값

        # Cosine schedule을 위한 최대 시간 계산
        self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. \
                            * (1. + self.cosine_s) / math.pi - self.cosine_s

        # 종료 시간 T 설정 (cosine schedule은 수치 안정성을 위해 0.9946 사용)
        if schedule == 'cosine':
            self.T = 0.9946
        else:
            self.T = 1.

        # 기타 파라미터
        self.sigma_min = 0.01  # 최소 표준편차
        self.sigma_max = 20    # 최대 표준편차
        self.eps = 1e-5       # 수치 안정성을 위한 작은 값
        
        # Cosine schedule의 초기 log alpha 값
        self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))

    def beta(self, t):
        """
        시간 t에서의 노이즈 추가 비율(beta) 계산
        
        beta(t)는 확산 과정의 속도를 결정하는 핵심 함수로,
        시간에 따라 얼마나 빠르게 노이즈를 추가할지 결정
        
        Parameters:
            t: 시간 텐서 (0 <= t <= T)
        
        Returns:
            beta(t): 시간 t에서의 노이즈 스케일
        
        스케줄 종류:
        - linear: 선형적으로 증가
        - cosine: cosine 함수 기반으로 부드럽게 증가 (더 나은 샘플 품질)
        - 기타: 상수 비율
        """
        if self.schedule == 'linear':
            # 선형 스케줄: beta_0에서 beta_1로 선형 증가
            beta = (self.beta_1 - self.beta_0) * t + self.beta_0
        elif self.schedule == 'cosine':
            # Cosine 스케줄: tan 함수를 사용하여 부드러운 증가
            # 초반에는 천천히, 후반에는 빠르게 노이즈 추가
            beta = math.pi / 2 * 2 / (self.cosine_s + 1) * torch.tan(
                (t + self.cosine_s) / (1 + self.cosine_s) * math.pi / 2)
        else:
            # 상수 스케줄
            beta = 2 * np.log(self.sigma_max / self.sigma_min) * (t * 0 + 1)

        return beta

    def marginal_log_mean_coeff(self, t):
        """
        시간 t에서의 평균 계수의 로그값 계산 (log(alpha(t)))
        
        alpha(t)는 원본 신호가 시간 t에서 얼마나 남아있는지를 나타내는 계수
        x_t = alpha(t) * x_0 + sigma(t) * epsilon 형태로 사용됨
        
        Parameters:
            t: 시간 텐서
        
        Returns:
            log(alpha(t)): 평균 계수의 로그값
        
        구현 세부사항:
        - linear: beta의 적분을 통해 계산
        - cosine: cos 함수를 사용하여 부드러운 감소
        - 기타: 지수 함수 기반
        """
        if self.schedule == 'linear':
            # 선형 스케줄의 경우 beta의 적분으로 계산
            # log_alpha_t = -∫beta(s)ds from 0 to t
            log_alpha_t = - 1 / (2 * 2) * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / 2 * t * self.beta_0

        elif self.schedule == 'cosine':
            # Cosine 스케줄: cos 함수 사용
            # clamp로 수치 안정성 보장 (-1 <= cos <= 1)
            log_alpha_fn = lambda s: torch.log(
                torch.clamp(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.), -1, 1))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0

        else:
            # 지수 스케줄
            log_alpha_t = -torch.exp(np.log(self.sigma_min) + t * np.log(self.sigma_max / self.sigma_min))

        return log_alpha_t

    def diffusion_coeff(self, t):
        """
        시간 t에서의 확산 계수 계산 (alpha(t))
        
        원본 신호의 스케일링 계수로, 시간이 지날수록 감소
        
        Returns:
            alpha(t) = exp(log_alpha(t))
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        시간 t에서의 표준편차 계산 (sigma(t))
        
        노이즈의 스케일을 결정하는 함수로,
        variance preserving 조건을 만족: alpha(t)^2 + sigma(t)^2 = 1
        
        Parameters:
            t: 시간 텐서
        
        Returns:
            sigma(t) = sqrt(1 - alpha(t)^2)
        """
        return torch.pow(1. - torch.exp(self.marginal_log_mean_coeff(t) * 2), 1 / 2)

    def inverse_a(self, a):
        """
        Cosine schedule에서 alpha 값으로부터 시간 t를 역산
        
        샘플링 과정에서 특정 노이즈 레벨에 해당하는 시간을 찾을 때 사용
        
        Parameters:
            a: alpha 값
        
        Returns:
            t: 해당하는 시간
        """
        return 2 / np.pi * (1 + self.cosine_s) * torch.acos(a) - self.cosine_s