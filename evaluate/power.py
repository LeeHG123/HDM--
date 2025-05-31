# Power Test 모듈: 생성된 데이터와 원본 데이터 간의 통계적 차이를 평가
# MMD (Maximum Mean Discrepancy) 기반 two-sample test를 사용하여 두 분포의 차이를 검정

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore 
from sklearn.metrics import pairwise_distances

from statistics import NormalDist
from tqdm import tqdm

def K_ID(X, Y, gamma=1):
    """
    Forms the kernel matrix K for the two sample test using the SE-T kernel with bandwidth gamma
    where T is the identity operator
    
    역할:
    - 두 데이터 분포 간의 유사도를 측정하기 위한 커널 행렬 생성
    - Squared Exponential (SE) 커널을 사용하여 데이터 점들 간의 유사도 계산
    - MMD 계산의 기초가 되는 커널 행렬 K를 반환
    
    구현 방식:
    - 유클리드 거리를 계산하고 차원 수로 정규화 (1/sqrt(n_obs))
    - median heuristic: gamma=-1이면 거리 행렬의 중앙값을 bandwidth로 사용
    - SE 커널 공식: K(x,y) = exp(-0.5 * ||x-y||^2 / gamma^2)
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the first distribution 
    Y - (n_samples,n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel, if -1 then median heuristic is used to pick gamma
    
    Returns:
    K - matrix formed from the kernel values of all pairs of samples from the two distributions
    
    의존성:
    - two_sample_test 함수에서 호출
    - MMD_K 함수에서 K 행렬을 사용하여 MMD 값 계산
    """
    # 차원 수 추출 (각 샘플이 가진 관측치 개수)
    n_obs = X.shape[1]
    
    # X와 Y를 하나의 행렬로 결합
    XY = np.vstack((X,Y))
    
    # 모든 샘플 쌍 간의 유클리드 거리 계산
    # 1/sqrt(n_obs)로 정규화하여 차원 수에 대한 영향 감소
    dist_mat = (1/np.sqrt(n_obs))*pairwise_distances(XY, metric='euclidean')
    
    # Median heuristic: bandwidth를 자동으로 선택
    if gamma == -1:
        # 0이 아닌 거리들의 중앙값을 bandwidth로 사용
        gamma = np.median(dist_mat[dist_mat > 0])
   
    # Squared Exponential 커널 계산
    K = np.exp(-0.5 * (1 / gamma ** 2) * (dist_mat ** 2))
    return K

def MMD_K(K, M, N):
    """
    Calculates the empirical MMD^{2} given a kernel matrix computed from the samples and the sample sizes of each distribution.
    
    역할:
    - Maximum Mean Discrepancy (MMD)의 제곱 값을 계산
    - 두 분포 간의 차이를 수치화하는 통계량
    - MMD = 0이면 두 분포가 동일, 값이 클수록 분포가 다름
    
    구현 방식:
    - MMD^2 = E[k(X,X')] - 2E[k(X,Y)] + E[k(Y,Y')]
    - 각 항을 커널 행렬 K의 부분행렬으로부터 계산
    - 대각선 원소는 제외 (self-similarity)
    
    Parameters:
    K - kernel matrix of all pairwise kernel values of the two distributions
    M - number of samples from first distribution
    N - number of samples from second distribution
    
    Returns:
    MMDsquared - empirical estimate of MMD^{2}
    
    의존성:
    - two_sample_test와 _power_test에서 호출
    """
    # 커널 행렬 K를 부분행렬로 분할
    # Kxx: X 샘플들 간의 커널 값
    Kxx = K[:N,:N]
    # Kyy: Y 샘플들 간의 커널 값
    Kyy = K[N:,N:]
    # Kxy: X와 Y 샘플들 간의 커널 값
    Kxy = K[:N,N:]
    
    # MMD^2의 각 항 계산
    # t1: E[k(X,X')] - X 분포 내 샘플 쌍의 평균 커널 값
    t1 = (1. / (M * (M-1))) * np.sum(Kxx - np.diag(np.diagonal(Kxx)))
    # t2: 2E[k(X,Y)] - X와 Y 분포 간 샘플 쌍의 평균 커널 값
    t2 = (2. / (M * N)) * np.sum(Kxy)
    # t3: E[k(Y,Y')] - Y 분포 내 샘플 쌍의 평균 커널 값
    t3 = (1. / (N * (N-1))) * np.sum(Kyy - np.diag(np.diagonal(Kyy)))
    
    # 최종 MMD^2 값 계산
    MMDsquared = (t1-t2+t3)
    
    return MMDsquared


def two_sample_test(X, Y, hyp, n_perms, z_alpha = 0.05, make_K = K_ID, return_p = False):
    """
    Performs the two sample test and returns an accept or reject statement
    
    역할:
    - 두 데이터 분포가 동일한지 검정하는 통계적 가설 검정
    - 귀무가설(null hypothesis): 두 분포가 동일하다
    - 대립가설: 두 분포가 다르다
    
    구현 방식:
    1. 원본 데이터의 MMD^2 계산
    2. Permutation test로 null distribution 생성
    3. 임계값과 비교하여 가설 기각/채택 결정
    
    Parameters:
    X - (n_samples, n_obs) array of samples from the first distribution 
    Y - (n_samples, n_obs) array of samples from the second distribution 
    hyp - hyperparameter dictionary for kernel (gamma value)
    n_perms - number of permutations performed when bootstrapping the null
    z_alpha - rejection threshold of the test
    return_p - option to return the p-value of the test
    make_K - function called to construct the kernel matrix used to compute the empirical MMD
    
    Returns:
    rej - 1 if null rejected, 0 if null accepted
    p-value - p_value of test (if return_p=True)
    
    의존성:
    - _power_test 함수에서 호출
    - plot_reject_null_hyp에서 시각화와 함께 사용
    """
    
    # 각 분포의 샘플 수 파악
    M = X.shape[0]  # 첫 번째 분포(X)의 샘플 수
    N = Y.shape[0]  # 두 번째 분포(Y)의 샘플 수

    # 커널 행렬 생성
    K = make_K(X, Y, hyp)

    # 원본 데이터의 MMD^2 계산
    MMD_test = MMD_K(K, M, N)
    
    # Permutation test: null distribution 생성
    # 귀무가설 하에서 MMD^2의 분포를 시뮬레이션
    shuffled_tests = np.zeros(n_perms)
    for i in range(n_perms):
            # X와 Y를 섮어서 두 분포가 동일한 경우를 시뮬레이션
            idx = np.random.permutation(M+N)
            K = K[idx, idx[:, None]]
            # 섮인 데이터의 MMD^2 계산
            shuffled_tests[i] = MMD_K(K,M,N)
    
    # 귀무가설 분포의 임계값 계산 (1-z_alpha quantile)
    q = np.quantile(shuffled_tests, 1.0-z_alpha)
    
    # 가설 검정: MMD_test > q이면 귀무가설 기각
    rej = int(MMD_test > q)
    
    if return_p:
        # p-value 계산: 섮은 데이터 중 원본 MMD^2보다 큰 비율
        p_value = 1-(percentileofscore(shuffled_tests,MMD_test)/100)
        return rej, p_value
    else:
        return rej


def _power_test(X_samples,Y_samples,gamma,n_tests,n_perms,z_alpha = 0.05,make_K = K_ID,return_p = False):
    """
    Computes multiple two-sample tests and returns the rejection rate
    
    역할:
    - 여러 번의 two-sample test를 수행하고 통계적 검정력(power) 계산
    - 검정력: 대립가설이 참일 때 귀무가설을 기각할 확률
    - 모델의 생성 품질을 평가하는 중요한 지표
    
    구현 방식:
    - 전체 데이터를 n_tests 개의 부분집합으로 분할
    - 각 부분집합에 대해 독립적으로 two-sample test 수행
    - 기각률을 평균하여 검정력 계산
    
    Parameters:
    X_samples - (n_samples*n_tests,n_obs) array of samples from the first distribution 
    Y_samples - (n_samples*n_tests,n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel
    n_tests - number of tests to perform
    n_perms - number of permutations performed when bootstrapping the null
    z_alpha - rejection threshold of the test
    make_K - function called to construct the kernel matrix used to compute the empirical MMD
    return_p - option to return the p-value of the test
    
    Returns:
    power - the rate of rejection of the null
    
    의존성:
    - power_test 함수에서 호출
    - calculate_ci에서 신뢰구간 계산 시 사용
    """
    
    # 각 테스트에 사용할 샘플 수 계산
    M = int(X_samples.shape[0]/n_tests)  # X 분포에서 각 테스트당 샘플 수
    N = int(Y_samples.shape[0]/n_tests)  # Y 분포에서 각 테스트당 샘플 수
    
    # 각 테스트의 기각 여부를 저장할 배열
    rej = np.zeros(n_tests)

    # n_tests번 독립적인 two-sample test 수행
    for t in range(n_tests):
        # t번째 테스트를 위한 데이터 추출
        X_t = np.array(X_samples[t*M:(t+1)*M,:])
        Y_t = np.array(Y_samples[t*N:(t+1)*N,:])
        # two-sample test 수행 및 결과 저장
        rej[t] = two_sample_test(X_t,Y_t,gamma,n_perms,z_alpha = z_alpha,make_K = make_K,return_p = return_p)

    # 검정력 계산: 귀무가설 기각 비율의 평균
    power = np.mean(rej)
    return power


def plot_reject_null_hyp(X_samples, Y_samples, rejection):
    """
    Two-sample test 결과를 시각화하는 함수
    
    역할:
    - 원본 데이터와 생성된 데이터를 시각적으로 비교
    - 가설 검정 결과(reject/not reject)를 표시
    
    구현 방식:
    - 빨간색: 원본 데이터 (Ground truth)
    - 파란색: 생성된 데이터 (Generation)
    - x축: -10에서 10 사이의 도메인
    - y축: 함수 값 (-105에서 105 범위로 고정)
    
    Parameters:
        X_samples: 원본 데이터 샘플들
        Y_samples: 생성된 데이터 샘플들
        rejection: 가설 검정 결과 (1=reject, 0=not reject)
    
    의존성:
    - 단독으로 호출되어 결과를 시각화
    """
    # 가설 검정 결과를 문자열로 변환
    reject = 'reject' if int(rejection) == 1 else 'not reject'
    print(reject)
    
    plt.figure(figsize=(12,4))
    X_t_label = 'Ground truth'
    Y_t_label = 'Generation'

    M = X_samples.shape[0]
    # Quadratic 데이터셋의 도메인과 동일
    domain = np.linspace(-10, 10, 100)
    
    # 모든 샘플을 그래프에 표시
    for i in range(M):
        plt.plot(domain, X_samples[i, :],'-',color='red', label=X_t_label)
        plt.plot(domain, Y_samples[i, :],'-',color='blue', label=Y_t_label)
        # 범례에 중복 표시 방지
        X_t_label = '_nolegend_'
        Y_t_label = '_nolegend_'
    
    plt.legend(loc='upper right')  
    plt.ylim([-105, 105])  # y축 범위 고정
    plt.show()


def power_test(y_0, y_t, n_tests=100, n_perms=100, gamma=-1):
    """
    생성된 데이터와 원본 데이터의 통계적 검정력 계산
    
    역할:
    - HDM이 생성한 함수들이 원본 분포와 얼마나 유사한지 평가
    - 검정력이 높을수록 두 분포가 다르다는 의미 (모델 품질이 낮음)
    - 검정력이 낮을수록 두 분포가 유사하다는 의미 (모델 품질이 높음)
    
    구현 방식:
    - 입력 텐서를 CPU로 이동 및 차원 축소
    - 전체 데이터를 10개의 부분집합으로 분할하여 테스트
    - Median heuristic을 사용하여 커널 bandwidth 자동 선택
    
    Parameters:
        y_0: 원본 데이터 (ground truth) [B, N, D] 형태
        y_t: 생성된 데이터 (generation) [B, N, D] 형태
        n_tests: 수행할 테스트 개수 (기본값: 100, 실제로는 데이터 크기/10 사용)
        n_perms: 각 테스트당 permutation 개수 (기본값: 100)
        gamma: 커널 bandwidth (기본값: -1로 median heuristic 사용)
    
    Returns:
        power: 검정력 백분율 (0-100)
    
    의존성:
        - runner/hilbert_runner.py의 sample 함수에서 호출
        - calculate_ci에서 신뢰구간 계산 시 사용
    """
    # 데이터를 CPU로 이동하고 차원 축소 [B, N, D] -> [B, N]
    X = y_0.cpu().squeeze(-1)  # ground truth
    Y = y_t.cpu().squeeze(-1)  # prediction

    # 테스트 개수를 데이터 크기에 따라 자동 조정
    n_tests = X.shape[0] // 10
    
    # Median heuristic 사용
    gamma = -1

    # 검정력 계산
    power = _power_test(X_samples=X, Y_samples=Y, gamma=gamma, n_tests=n_tests, n_perms=n_perms)
    
    # 백분율로 변환하여 반환
    return power * 100

def confidence_interval(data, N=10, confidence=0.95):
    '''
    Computes confidence interval (default 95% => confidence=0.95)
    
    역할:
    - 주어진 데이터의 신뢰구간 계산
    - 통계적 불확실성을 정량화하여 결과의 신뢰도 표현
    
    구현 방식:
    - 정규분포 가정 하에 z-score를 사용한 신뢰구간 계산
    - 표본 평균 ± (z * 표준오차) 형태로 표현
    
    Parameters:
     - data: list input of data
     - N: number of data
     - confidence: confidence percent
     
    Returns:
        mean: 데이터의 평균
        h: 신뢰구간의 폭 (±h)
    
    의존성:
        - calculate_ci 함수에서 호출
    '''
    # 데이터로부터 정규분포 생성
    dist = NormalDist.from_samples(data)
    
    # 신뢰수준에 해당하는 z-score 계산
    z = NormalDist().inv_cdf((1 + confidence) / 2.)
    
    # 신뢰구간 폭 계산: 표준오차 = 표준편차 / sqrt(N-1)
    h = dist.stdev * z / ((N - 1) ** 0.5)
    
    return dist.mean, h

def calculate_ci(y1, y0, gamma=-1, n_tests=10, n_perms=1000, N=30):
    """
    검정력의 신뢰구간을 계산하는 함수
    
    역할:
    - 여러 번의 power test를 반복하여 검정력의 통계적 불확실성 측정
    - 결과를 "mean ± interval" 형식으로 제공
    - 모델 성능의 일관성과 신뢰도를 평가
    
    구현 방식:
    1. N번 독립적으로 power test 수행
    2. 결과들의 평균과 신뢰구간 계산
    3. "평균 ± 구간" 형식으로 문자열 생성
    
    Parameters:
        y1: 첫 번째 데이터셋 (일반적으로 원본 데이터)
        y0: 두 번째 데이터셋 (일반적으로 생성된 데이터)
        gamma: 커널 bandwidth (기본값: -1로 median heuristic 사용)
        n_tests: 각 power test당 수행할 테스트 개수
        n_perms: 각 two-sample test당 permutation 개수 (기본값: 1000)
        N: 신뢰구간 계산을 위해 반복할 power test 횟수 (기본값: 30)
    
    Returns:
        test_power_interval: "평균 ± 구간" 형식의 문자열
    
    의존성:
        - runner/hilbert_runner.py의 sample 함수에서 호출
    """
    # 테스트 개수를 데이터 크기에 따라 자동 조정
    n_tests = y1.shape[0] // 10
    n_perms = 1000
    
    # N번의 power test 결과를 저장할 리스트
    F = []
    
    # 진행 상황을 표시하면서 N번 반복
    for _ in tqdm(range(N)):
        F.append(power_test(y1, y0, gamma=gamma, n_tests=n_tests, n_perms=n_perms))
    
    # 신뢰구간 계산
    mean, interval = confidence_interval(F, N)
    
    # "평균 ± 구간" 형식으로 문자열 생성
    test_power_interval = str(np.round(mean, 3)) + u"\u00B1" + str(np.round(interval, 3))
    
    return test_power_interval
