# 샘플링 모듈
# SDE 기반 역확산 과정을 구현하여 노이즈로부터 1D 함수를 생성
# Euler-Maruyama 방법을 사용한 수치적 SDE 해법 구현

import torch
import torch.nn.functional as F
import numpy as np
import tqdm

def sampler(x, model, sde, device, W, eps, dataset, steps=1000, sampler_input=None):
    """
    1D 함수 데이터를 위한 SDE 기반 샘플러
    
    역할:
    - 노이즈로부터 시작하여 점진적으로 의미있는 1D 함수를 생성
    - 학습된 score function을 사용하여 역확산 과정 수행
    - Hilbert 공간에서의 구조화된 노이즈를 활용
    
    구현 방식:
    - Euler-Maruyama 방법으로 reverse SDE 수치 해법
    - 각 스텝에서 score function을 사용하여 gradient 계산
    - Hilbert noise를 추가하여 확률적 특성 유지
    - 안정성을 위한 gradient clipping 적용
    
    Parameters:
        x: 초기 노이즈 텐서 [batch, dimension]
        model: 학습된 FNO 기반 score 예측 모델
        sde: VPSDE1D 객체 (확산 계수 제공)
        device: 연산 수행 디바이스
        W: HilbertNoise 객체 (구조화된 노이즈 생성)
        eps: 수치 안정성을 위한 작은 값
        dataset: 데이터셋 이름 (gradient clipping 임계값 결정용)
        steps: 샘플링 스텝 수 (기본값: 1000)
        sampler_input: 사전 생성된 노이즈 (선택적)
    
    Returns:
        x: 생성된 1D 함수 샘플들 [batch, dimension]
    
    의존성:
        - runner/hilbert_runner.py의 sample 함수에서 호출
        - VPSDE1D의 beta, marginal_std 함수 사용
        - HilbertNoise의 sample 함수 사용
    """
    def sde_score_update(x, s, t):
        """
        SDE의 한 스텝 업데이트를 수행하는 내부 함수
        
        역할:
        - 시간 s에서 t로의 역확산 과정을 한 스텝 수행
        - Euler-Maruyama 방법으로 reverse SDE 근사 해법
        - score function을 사용하여 drift term 계산
        
        구현 세부사항:
        1. 모델로부터 score 예측
        2. score를 적절히 스케일링 (marginal_std 고려)
        3. SDE 계수들 계산 (drift, diffusion)
        4. Hilbert 노이즈 추가
        5. 다음 시간 스텝의 상태 계산
        
        Parameters:
            x: 현재 상태 x_s [batch, dimension]
            s: 현재 시간
            t: 다음 시간
            
        Returns:
            x_t: 다음 시간 스텝의 상태
            
        수학적 배경:
        dx = [f(x,t) - g(t)²∇log p_t(x)]dt + g(t)dW
        여기서 ∇log p_t(x)가 학습된 score function
        """
        # 모델로부터 score 예측
        models = model(x, s)
        
        # Score scaling: score를 적절한 스케일로 조정
        # marginal_std^(-1)을 곱하여 정규화
        score_s = models * torch.pow(sde.marginal_std(s), -(2.0 - 1))[:, None].to(device)

        # 시간 스텝 크기에 따른 beta 계수
        beta_step = sde.beta(s) * (s - t)
        
        # 드리프트 항의 계수 (1차 근사)
        x_coeff = 1 + beta_step / 2.0

        # 확산 항의 계수 (노이즈 강도)
        noise_coeff = torch.pow(beta_step, 1 / 2.0)
        
        # Hilbert 공간 노이즈 생성
        if sampler_input == None:
            # 일반적인 경우: 동일한 해상도로 노이즈 생성
            e = W.sample(x.shape)
        else:
            # 특별한 경우: 다른 해상도로 노이즈 생성
            e = W.free_sample(free_input=sampler_input)

        # Score 항의 계수
        score_coeff = beta_step
        
        # 다음 스텝 계산: x_t = x_coeff*x + score_coeff*score + noise_coeff*noise
        x_t = x_coeff[:, None].to(device) * x + score_coeff[:, None].to(device) * score_s + noise_coeff[:, None].to(device) * e.to(device)

        return x_t

    # 샘플링을 위한 시간 스케줄 생성
    # T(=0.9946)에서 eps(작은 값)까지 선형적으로 감소
    timesteps = torch.linspace(sde.T, eps, steps + 1).to(device)

    # 그래디언트 계산 비활성화 (추론 모드)
    with torch.no_grad():
        # 진행 상황 표시와 함께 역확산 과정 수행
        for i in tqdm.tqdm(range(steps)):
            # 현재 시간과 다음 시간을 배치 크기만큼 복제
            vec_s = torch.ones((x.shape[0],)).to(device) * timesteps[i]      # 현재 시간
            vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i + 1]  # 다음 시간

            # SDE 한 스텝 업데이트 수행
            x = sde_score_update(x, vec_s, vec_t)

            # 수치 안정성을 위한 그래디언트 클리핑
            # 생성된 함수가 너무 극단적인 값을 가지지 않도록 제한
            size = x.shape
            l = x.shape[0]
            
            # 각 샘플의 L2 norm 계산을 위해 평탄화
            x = x.reshape((l, -1))
            
            # norm이 임계값을 초과하는 샘플들 찾기
            indices = x.norm(dim=1) > 10
            
            # 데이터셋에 따라 다른 임계값 적용
            if dataset == 'Gridwatch':
                # Gridwatch 데이터셋: 더 큰 값 허용 (17)
                x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * 17
            else:
                # 기본 데이터셋 (Quadratic 포함): 10으로 제한
                x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * 10
            
            # 원래 shape으로 복원
            x = x.reshape(size)

    return x