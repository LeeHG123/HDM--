# 손실 함수 모듈
# Score-based generative model의 학습을 위한 손실 함수 정의
# Denoising score matching을 사용하여 score function을 학습

import torch.nn.functional as F

def loss_fn(model, sde, x_0, t, e):
    """
    2D 이미지를 위한 손실 함수 (이 프로젝트에서는 사용되지 않음)
    
    참고: 이 함수는 2D 이미지 데이터를 위한 것으로,
    1D Quadratic 데이터셋에서는 hilbert_loss_fn을 사용함
    
    Parameters:
        model: Score 예측 모델
        sde: SDE 객체
        x_0: 원본 데이터
        t: 타임스텝
        e: 노이즈
    
    Returns:
        loss: MSE 손실
    """
    x_mean = sde.diffusion_coeff(x_0, t)
    noise = sde.marginal_std(e, t)

    x_t = x_mean + noise
    score = -noise

    output = model(x_t, t)

    loss = (output - score).square().sum(dim=(1,2,3)).mean(dim=0)
    return loss

def hilbert_loss_fn(model, sde, x_0, t, e):
    """
    1D 함수 데이터를 위한 Hilbert 공간 손실 함수
    
    역할:
    - HDM의 핵심 손실 함수로, score function을 학습
    - Denoising score matching 방법을 사용
    - 모델이 노이즈가 추가된 데이터로부터 score를 예측하도록 학습
    
    구현 방식:
    1. Forward process: x_t = alpha(t) * x_0 + sigma(t) * e
    2. Score: ∇ log p(x_t) = -e / sigma(t) (단순화된 형태로 -e 사용)
    3. 모델 예측과 score 간의 L2 손실 계산
    
    Parameters:
        model: FNO 기반 score 예측 모델
        sde: VPSDE1D 객체 (diffusion 계수 제공)
        x_0: 원본 1D 함수 데이터 [batch, dimension]
        t: 타임스텝 텐서 [batch]
        e: Hilbert 공간 노이즈 [batch, dimension]
    
    Returns:
        loss: 평균 MSE 손실 (스칼라 값)
    
    의존성:
        - runner/hilbert_runner.py의 train 함수에서 호출
        - sde.diffusion_coeff와 sde.marginal_std 사용
    """
    # 시간 t에서의 확산 계수 계산
    x_mean = sde.diffusion_coeff(t)  # alpha(t)
    noise = sde.marginal_std(t)      # sigma(t)

    # Forward diffusion process: 노이즈가 추가된 데이터 생성
    # x_t = alpha(t) * x_0 + sigma(t) * e
    x_t = x_0 * x_mean[:, None] + e * noise.view(-1, 1)
    
    # Ground truth score = -e (단순화된 형태)
    # 실제로는 -e/sigma(t)이지만, 모델이 직접 e를 예측하도록 학습
    score = -e

    # 모델을 통해 score 예측
    output = model(x_t, t.float())

    # L2 손실 계산: ||output - score||^2
    # dim=1: 각 함수의 모든 점에 대해 합산
    # mean: 배치에 대해 평균
    loss = (output - score).square().sum(dim=(1)).mean(dim=0)
    return loss
