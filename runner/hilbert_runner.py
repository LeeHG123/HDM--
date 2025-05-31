# Hilbert Runner 모듈 - HDM(Hilbert Diffusion Model) 학습 및 샘플링 실행기
# 확산 모델의 전체 파이프라인을 관리하는 메인 실행 모듈
# 1D 시계열 데이터를 위한 Hilbert 공간 기반 확산 모델 구현

"""
Hilbert Runner 모듈 개요
========================

이 모듈은 HDM(Hilbert Diffusion Model)의 핵심 실행기로, 확산 모델의 
학습(training)과 샘플링(sampling) 전체 파이프라인을 관리합니다.

주요 역할:
1. HilbertNoise: Gaussian Process 기반 구조화된 노이즈 생성
2. HilbertDiffusion: 확산 모델의 학습 및 샘플링 통합 관리
3. 다양한 백본 모델 지원: FNO, DDPM, UNet
4. 데이터셋별 특화 처리: Quadratic, Melbourne, Gridwatch
5. 분산 학습 및 체크포인트 관리
6. 성능 평가 및 시각화

핵심 특징:
- Hilbert 공간에서의 구조화된 노이즈 모델링
- SDE(Stochastic Differential Equation) 기반 확산 과정
- 해상도 자유(resolution-free) 생성 지원
- 통계적 신뢰구간 계산을 통한 성능 평가
- 실시간 텐서보드 로깅 및 모니터링

아키텍처 의존성:
- models/: FNO, DDPM 등 백본 네트워크
- functions/: SDE, 손실함수, 샘플러
- datasets/: 데이터 전처리 및 스케일링
- evaluate/: 통계적 성능 평가

수학적 배경:
- Gaussian Process: GP(x) ~ N(0, K(x,x'))
- SDE: dx = f(x,t)dt + g(t)dW
- Diffusion: p(x_t|x_0) = N(x_t; α_t x_0, σ_t²I)
- Reverse Process: x_{t-1} = μ(x_t, t) + σ(t)z

사용 컨텍스트:
- 1D 시계열 데이터 생성
- 함수 공간에서의 확산 모델링
- 조건부/무조건부 생성 모델
- 과학적 컴퓨팅 및 시뮬레이션
"""

import os
import logging
from scipy.spatial import distance  # 커널 함수 거리 계산용
import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt           # 시각화를 위한 matplotlib

# 성능 평가 및 데이터 처리
from evaluate.power import calculate_ci  # 신뢰구간 기반 통계적 검정력 계산
from datasets import data_scaler, data_inverse_scaler  # 데이터 정규화/역정규화

from collections import OrderedDict  # 분산 학습 시 모델 state dict 정리용

# PyTorch 관련 모듈
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler  # 분산 학습용 데이터 샘플러

# HDM 모델 아키텍처들
from models import *  # FNO, DDPM, UNet 등 백본 네트워크

# HDM 핵심 함수들
from functions.utils import *  # 유틸리티 함수들
from functions.loss import hilbert_loss_fn  # Hilbert 공간 손실함수
from functions.sde import VPSDE1D  # 1D Variance Preserving SDE
from functions.sampler import sampler  # 확산 모델 샘플링 함수

# 그래디언트 이상치 감지 활성화 (디버깅용)
torch.autograd.set_detect_anomaly(True)

def kernel_se(x1, x2, hyp={'gain':1.0,'len':1.0}):
    """
    Squared Exponential (SE) 커널 함수 - Gaussian Process의 핵심 구성 요소
    
    역할:
    - Gaussian Process의 공분산 함수로 사용
    - 입력 간의 유사도를 매끄러운 지수 함수로 모델링
    - Hilbert 공간에서 구조화된 노이즈 생성의 기반
    - 두 점 간의 거리에 따른 상관관계 정의
    
    수학적 배경:
    - SE 커널: k(x₁, x₂) = σ² exp(-||x₁ - x₂||² / (2ℓ²))
    - 여기서 σ² = gain (신호 분산), ℓ = len (길이 스케일)
    - 무한히 미분 가능한 매끄러운 함수 생성
    - 길이 스케일이 클수록 더 매끄러운 함수
    
    특징:
    - 매끄러움: 무한히 미분 가능
    - 지역성: 가까운 점들이 높은 상관관계
    - 파라미터화: gain과 len으로 조절 가능
    - 회전 불변성: 모든 방향에서 동일한 매끄러움
    
    Parameters:
        x1, x2: 입력 좌표 텐서 [N, D], [M, D]
        hyp: 하이퍼파라미터 딕셔너리
            - gain: 신호 분산 (σ²), 커널 출력의 스케일
            - len: 길이 스케일 (ℓ), 상관관계가 감소하는 거리
    
    Returns:
        K: 공분산 행렬 [N, M], 각 원소는 k(x1[i], x2[j])
    
    구현 세부사항:
    - CPU 변환: NumPy 기반 거리 계산 활용
    - 제곱 유클리드 거리: ||x₁ - x₂||²
    - 길이 스케일 정규화: x/ℓ로 입력 스케일링
    - 지수 변환: exp(-D)로 매끄러운 감소
    """
    # GPU 텐서를 CPU NumPy로 변환 (scipy 호환성)
    x1 = x1.cpu().numpy()
    x2 = x2.cpu().numpy()

    # 제곱 유클리드 거리 행렬 계산
    # D[i,j] = ||x1[i] - x2[j]||² / len²
    D = distance.cdist(x1/hyp['len'], x2/hyp['len'], 'sqeuclidean')
    
    # SE 커널 적용: K[i,j] = gain * exp(-D[i,j])
    K = hyp['gain'] * np.exp(-D)
    
    # PyTorch 텐서로 변환 후 반환
    return torch.from_numpy(K).to(torch.float32)

class HilbertNoise:
    def __init__(self, grid, x=None, hyp_len=1.0, hyp_gain=1.0, use_truncation=False):
        x = torch.linspace(-10, 10, grid)
        self.hyp = {'gain': hyp_gain, 'len': hyp_len}
        x = torch.unsqueeze(x, dim=-1)
        self.x = x
        if x is not None:
            self.x=x

        K = kernel_se(x, x, self.hyp)
        K = K.cpu().numpy()
        eig_val, eig_vec = np.linalg.eigh(K + 1e-6 * np.eye(K.shape[0], K.shape[0]))

        self.eig_val = torch.from_numpy(eig_val)
        self.eig_vec = torch.from_numpy(eig_vec).to(torch.float32)
        self.D = torch.diag(self.eig_val).to(torch.float32)
        self.M = torch.matmul(self.eig_vec, torch.sqrt(self.D))

    def sample(self, size):
        size = list(size)  # batch*grid
        x_0 = torch.randn(size)

        output = (x_0 @ self.M.transpose(0, 1))  # batch grid x grid x grid
        return output  # bath*grid

    def free_sample(self, free_input):  # input (batch,grid)

        y = torch.randn(len(free_input), self.x.shape[0]) @ self.eig_vec.T @ kernel_se(self.x, free_input[0].unsqueeze(-1), self.hyp)
        return y

class HilbertDiffusion(object):
    def __init__(self, args, config, dataset, test_dataset, device=None):
        self.args = args
        self.config = config
        self.W = HilbertNoise(grid=config.data.dimension, hyp_len=config.data.hyp_len, hyp_gain=config.data.hyp_gain)

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                    else torch.device("cpu")
            )
        self.device = device
        self.num_timesteps = config.diffusion.num_diffusion_timesteps

        self.sde = VPSDE1D(schedule='cosine')

        self.dataset = dataset
        self.test_dataset = test_dataset

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        if args.distributed:
            sampler = DistributedSampler(self.dataset, shuffle=True,
                                     seed=args.seed if args.seed is not None else 0)
        else:
            sampler = None
        train_loader = data.DataLoader(
            self.dataset,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            sampler=sampler
        )

        # Model
        if config.model.model_type == "ddpm_mnist":
            model = Unet(dim=config.data.image_size,
                         channels=config.model.channels,
                         dim_mults=config.model.dim_mults,
                         is_conditional=config.model.is_conditional)
        elif config.model.model_type == "FNO":
            model = FNO(n_modes=config.model.n_modes, hidden_channels=config.model.hidden_channels, in_channels=config.model.in_channels, out_channels=config.model.out_channels,
                      lifting_channels=config.model.lifting_channels, projection_channels=config.model.projection_channels,
                      n_layers=config.model.n_layers, joint_factorization=config.model.joint_factorization,
                      norm=config.model.norm, preactivation=config.model.preactivation, separable=config.model.separable)
        elif config.model.model_type == "ddpm":
            model = Model(config)

        model = model.to(self.device)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.local_rank],)
                                                            #   find_unused_parameters=True)
        logging.info("Model loaded.")

        # Optimizer, LR scheduler
        optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)

        # lr_scheduler = get_scheduler(
        #     "linear",
        #     optimizer=optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=2000000,
        # )

        start_epoch, step = 0, 0
        # if args.resume:
        #     states = torch.load(os.path.join(args.log_path, "ckpt.pth"), map_location=self.device)
        #     model.load_state_dict(states[0], strict=False)
        #     start_epoch = states[2]
        #     step = states[3]

        for epoch in range(config.training.n_epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            data_start = time.time()
            data_time = 0

            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device).squeeze(-1)
                y = y.to(self.device).squeeze(-1)

                data_time += time.time() - data_start
                model.train()
                step += 1

                if config.data.dataset == 'Melbourne':
                    y = data_scaler(y)

                t = torch.rand(y.shape[0], device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                e = self.W.sample(y.shape).to(self.device).squeeze(-1)

                loss = hilbert_loss_fn(model, self.sde, y, t, e).to(self.device)
                tb_logger.add_scalar("train_loss", torch.abs(loss), global_step=step)

                optimizer.zero_grad()
                loss.backward()

                if args.local_rank == 0:
                    logging.info(
                        f"step: {step}, loss: {torch.abs(loss).item()}, data time: {data_time / (i+1)}"
                    )

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass

                optimizer.step()
                # lr_scheduler.step()

                if step % config.training.ckpt_store == 0:
                    self.ckpt_dir = os.path.join(args.log_path, 'ckpt.pth')
                    torch.save(model.state_dict(), self.ckpt_dir)

                data_start = time.time()

    def sample(self, score_model=None):
        args, config = self.args, self.config

        if config.model.model_type == "ddpm_mnist":
            model = Unet(dim=config.data.image_size,
                         channels=config.model.channels,
                         dim_mults=config.model.dim_mults,
                         is_conditional=config.model.is_conditional,)
        elif config.model.model_type == "FNO":
            model = FNO(n_modes=config.model.n_modes, hidden_channels=config.model.hidden_channels, in_channels=config.model.in_channels, out_channels=config.model.out_channels,
                      lifting_channels=config.model.lifting_channels, projection_channels=config.model.projection_channels,
                      n_layers=config.model.n_layers, joint_factorization=config.model.joint_factorization,
                      norm=config.model.norm, preactivation=config.model.preactivation, separable=config.model.separable)
        elif config.model.model_type == "ddpm":
            model = Model(config)

        model = model.to(self.device)

        if score_model is not None:
            model = score_model

        elif "ckpt_dir" in config.model.__dict__.keys():
            # First try the checkpoint from training
            ckpt_path = os.path.join(args.log_path, 'ckpt.pth')
            if os.path.exists(ckpt_path):
                ckpt_dir = ckpt_path
            else:
                ckpt_dir = config.model.ckpt_dir
            
            states = torch.load(
                ckpt_dir,
                map_location=config.device,
            )

            if args.distributed:
                state_dict = OrderedDict()
                for k, v in states.items():
                    if 'module' in k:
                        name = k[7:]
                        state_dict[name] = v
                    else:
                        state_dict[k] = v

                model.load_state_dict(state_dict)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            else:
                model.load_state_dict(states, strict=False)
        else:
            raise Exception("Fail to load model due to invalid ckpt_dir")

        logging.info("Done loading model")
        model.eval()

        test_loader = torch.utils.data.DataLoader(self.test_dataset, config.sampling.batch_size, shuffle=False)

        x_0, y_0 = next(iter(test_loader))

        if config.data.dataset == 'Quadratic':
            free_input = torch.rand((config.sampling.batch_size, y_0.shape[1])) * 20 - 10
            free_input = torch.sort(free_input)[0]

            a = torch.randint(low=0, high=2, size=(free_input.shape[0], 1)).repeat(1, 100) * 2 - 1
            eps = torch.normal(mean=0., std=1., size=(free_input.shape[0], 1)).repeat(1, 100)
            y00 = a * (free_input ** 2) + eps

            with torch.no_grad():
                for _ in tqdm(range(1), desc="Generating image samples"):
                    y_shape = (config.sampling.batch_size, config.data.dimension)
                    t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T

                    y = self.W.free_sample(free_input).to(self.device) * self.sde.marginal_std(t)[:, None]
                    y = sampler(y, model, self.sde, self.device, self.W,  self.sde.eps, config.data.dataset, sampler_input=free_input)

            y_0 = y_0 * 50
            y = y * 50

            _, ax = plt.subplots(1, 2, figsize=(10, 5))

            for i in range(config.sampling.batch_size):
                ax[0].plot(x_0[i, :].cpu(), y_0[i, :].cpu())

            ax[0].set_title(f'Ground truth, len:{config.data.hyp_len:.2f}')

            n_tests = config.sampling.batch_size // 10

            for i in range(y.shape[0]):
                ax[1].plot(free_input[i, :].cpu(), y[i, :].cpu(), alpha=1)
            print('Calculate Confidence Interval:')
            power_res = calculate_ci(y, y_0, n_tests=n_tests)
            print(f'Calculate Confidence Interval: resolution-free, power(avg of 30 trials): {power_res}')
            # power_res2 = calculate_ci(y, y00, n_tests=n_tests)
            # print(f'Calculate Confidence Interval: resolution-free test2, power(avg of 30 trials): {power_res2}')
            logging.info(f'Calculate Confidence Interval: resolution-free, power(avg of 30 trials): {power_res}')
            # logging.info(f'Calculate Confidence Interval: resolution-free test2, power(avg of 30 trials): {power_res2}')
            ax[1].set_title(f'resolution-free, power(avg of 30 trials): {power_res}')
            # ax[1].set_title(f'resfree 1: {power_res}, resfree 2: {power_res2}')
            # plt.savefig('result.png')
            # np.savez(args.log_path + '/rawdata', x_0=x_0.cpu().numpy(), y_0=y_0.cpu().numpy(), free_input=free_input.cpu().numpy(), y=y.cpu().numpy())

        else:
            y_0 = y_0.squeeze(-1)
            with torch.no_grad():
                for _ in tqdm(range(1), desc="Generating image samples"):
                    y_shape = (config.sampling.batch_size, config.data.dimension)
                    t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T

                    y = self.W.sample(y_shape).to(self.device) * self.sde.marginal_std(t)[:, None]
                    y = sampler(y, model, self.sde, self.device, self.W,  self.sde.eps, config.data.dataset)

            _, ax = plt.subplots(1, 2, figsize=(10, 5))

            if config.data.dataset == 'Melbourne':
                lp = 10
                n_tests = y.shape[0] // 10
                y = data_inverse_scaler(y)
            if config.data.dataset == 'Gridwatch':
                lp = y.shape[0]
                n_tests = y.shape[0] // 10
                plt.ylim([-2, 3])

            for i in range(lp):
                ax[0].plot(x_0[i, :].cpu(), y[i, :].cpu())
                ax[1].plot(x_0[i, :].cpu(), y_0[i, :].cpu(), c='black', alpha=1)


            ax[0].set_title(f'Ground truth, len:{config.data.hyp_len:.2f}')

            for i in range(lp):
                ax[1].plot(x_0[i, :].cpu(), y[i, :].cpu(), alpha=1)


            power = calculate_ci(y, y_0, n_tests=n_tests)
            print(f'Calculate Confidence Interval: grid, 0th: {power}')

            ax[1].set_title(f'grid, power(avg of 30 trials):{power}')

        # Visualization figure save
        plt.savefig('visualization_default.png')
        print("Saved plot fig to {}".format('visualization_default.png'))
        plt.clf()
        plt.figure()
