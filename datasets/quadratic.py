import torch

class QuadraticDataset(torch.utils.data.Dataset):
    """
    2차 함수 데이터셋 클래스
    
    역할:
    - Hilbert Diffusion Model 학습을 위한 2차 함수 데이터 생성
    - y = a * x^2 + ε 형태의 함수를 생성하여 모델이 학습할 수 있는 다양한 패턴 제공
    
    구현 방식:
    - x: -10에서 10 사이의 균등한 격자 위의 점들
    - a: 각 함수마다 무작위로 -1 또는 1 선택 (함수의 방향 결정)
    - ε: 가우시안 노이즈로 현실적인 데이터 패턴 모사
    - 최종 출력은 50으로 나누어 정규화
    
    의존성:
    - datasets/__init__.py의 get_dataset 함수에서 생성
    - runner/hilbert_runner.py에서 DataLoader로 로드
    """
    
    def __init__(self, num_data, num_points, seed=42):
        """
        데이터셋 초기화
        
        Parameters:
            num_data: 생성할 함수의 개수 (기본값: 1000)
            num_points: 각 함수당 x 축의 점 개수 (기본값: 100)
            seed: 랜덤 시드 (학습/테스트 데이터 분리를 위해 다른 값 사용)
        """
        super().__init__()

        self.num_data = num_data
        self.num_points = num_points
        self.seed = seed
        
        # x 축의 격자 생성: -10에서 10 사이를 num_points개로 분할
        # unsqueeze(0): [num_points] -> [1, num_points]
        # repeat: [1, num_points] -> [num_data, num_points]
        self.x = torch.linspace(start=-10., end=10., steps=self.num_points).unsqueeze(0).repeat(self.num_data, 1)
        
        # 실제 2차 함수 데이터 생성
        self.dataset = self._create_dataset()

    def _create_dataset(self):
        """
        2차 함수 데이터를 실제로 생성하는 내부 메서드
        
        구현 세부사항:
        1. 시드 고정으로 재현 가능한 데이터 생성
        2. a 값을 {-1, 1} 중에서 무작위 선택
        3. 가우시안 노이즈 추가
        4. y = a * x^2 + ε 계산
        
        Returns:
            y: [num_data, num_points] 크기의 2차 함수 데이터
        """
        # 재현 가능한 랜덤 데이터 생성을 위해 시드 고정
        torch.manual_seed(self.seed)
        
        # a 값 생성: 0 또는 1을 랜덤하게 선택 후, 2를 곱하고 1을 빼서 {-1, 1}로 변환
        # size=(self.x.shape[0], 1): [num_data, 1] 크기로 생성
        # repeat(1, self.num_points): 모든 x 위치에 대해 같은 a 값 사용
        a = torch.randint(low=0, high=2, size=(self.x.shape[0], 1)).repeat(1, self.num_points) * 2 - 1
        
        # 가우시안 노이즈 생성: 평균 0, 표준편차 1
        # 모든 x 위치에 대해 독립적인 노이즈 추가
        eps = torch.normal(mean=0., std=1., size=(self.x.shape[0], 1)).repeat(1, self.num_points)

        # 2차 함수 생성: y = a * x^2 + ε
        y = a * (self.x ** 2) + eps
        return y

    def __len__(self):
        """
        데이터셋의 크기 반환 (PyTorch Dataset 인터페이스 구현)
        
        Returns:
            데이터셋에 포함된 함수의 개수
        """
        return self.num_data

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 데이터 반환 (PyTorch Dataset 인터페이스 구현)
        
        Parameters:
            idx: 데이터 인덱스
        
        Returns:
            x: [num_points, 1] 크기의 입력 도메인 데이터
            y: [num_points, 1] 크기의 함수 값 (정규화됨)
        
        중요 사항:
        - unsqueeze(-1): 마지막 차원 추가하여 FNO 모델의 입력 형식에 맞춤
        - /50: 출력 값을 적절한 범위로 정규화하여 학습 안정성 향상
        """
        return self.x[idx, :].unsqueeze(-1), self.dataset[idx, :].unsqueeze(-1) / 50
