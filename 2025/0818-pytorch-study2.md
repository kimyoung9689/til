# PyTorch 현업 완벽 가이드

## 1. 기초 개념 (현업에서 반드시 알아야 할 것들)

### 1.1 Tensor 기본 - 메모리와 성능 중심
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ⭐ 현업 핵심: device 관리는 기본 중의 기본
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tensor 생성 - 현업에서 자주 쓰는 패턴들
x = torch.zeros(1000, 1000, device=device)  # GPU에서 바로 생성
x = torch.randn(32, 3, 224, 224, device=device)  # 배치 이미지 형태

# ⚠️ 절대 피해야 할 실수: CPU↔GPU 이동 최소화
# 나쁜 예
bad_tensor = torch.randn(1000, 1000).to(device)  # CPU에서 생성 후 이동
# 좋은 예
good_tensor = torch.randn(1000, 1000, device=device)  # GPU에서 바로 생성
```

### 1.2 현업 필수 Tensor 조작
```python
# 배치 처리 - 현업의 99%는 배치로 처리
batch_size = 32
seq_len = 128
hidden_dim = 768

# reshape vs view - 메모리 레이아웃 이해 필수
x = torch.randn(batch_size, seq_len, hidden_dim)
# view: contiguous해야 함 (메모리 효율적)
x_view = x.view(batch_size * seq_len, hidden_dim)
# reshape: 항상 작동하지만 때로 복사 발생
x_reshape = x.reshape(batch_size * seq_len, hidden_dim)

# 현업에서 자주 쓰는 indexing 패턴들
indices = torch.tensor([0, 2, 4])
selected = x[indices]  # 특정 배치만 선택

# Broadcasting - 메모리 절약의 핵심
attention_mask = torch.ones(batch_size, 1, seq_len)  # (32, 1, 128)
scores = torch.randn(batch_size, seq_len, seq_len)   # (32, 128, 128)
masked_scores = scores * attention_mask  # broadcasting 활용
```

## 2. 신경망 구조 설계 (현업 베스트 프랙티스)

### 2.1 모듈 설계 원칙
```python
class ProductionModel(nn.Module):
    """현업에서 사용하는 모델 설계 패턴"""
    
    def __init__(self, config):
        super().__init__()
        # ⭐ config 객체로 하이퍼파라미터 관리
        self.config = config
        
        # 레이어 정의 - 명확한 네이밍
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        
        # ⭐ 현업 필수: weight 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier/He 초기화 등 적용"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None):
        # ⭐ 현업에서는 항상 배치 처리 가정
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        hidden_states = self.embedding(input_ids)
        
        # Encoder layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Classification
        pooled = hidden_states.mean(dim=1)  # 간단한 pooling
        logits = self.classifier(pooled)
        
        return logits
```

### 2.2 현업에서 자주 쓰는 레이어들
```python
class EncoderLayer(nn.Module):
    """Transformer 스타일 인코더 레이어"""
    
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.intermediate_dim),
            nn.GELU(),  # ReLU보다 GELU가 현업에서 많이 쓰임
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, hidden_states, attention_mask=None):
        # ⭐ Residual connection + LayerNorm 패턴
        # Self-attention
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + ff_output)
        
        return hidden_states

# CNN에서 자주 쓰는 패턴
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)  # 현업 필수
        self.relu = nn.ReLU(inplace=True)  # 메모리 절약
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

## 3. 학습 과정 (현업 중심)

### 3.1 데이터 로더 - 성능 최적화
```python
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp

class ProductionDataset(Dataset):
    """현업에서 사용하는 데이터셋 패턴"""
    
    def __init__(self, data_path, transform=None):
        self.data = self.load_data(data_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# ⭐ 현업 데이터로더 설정 - 성능 최적화
def create_dataloader(dataset, batch_size, is_training=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=4,  # CPU 코어 수에 맞게 조정
        pin_memory=True,  # GPU 전송 속도 향상
        drop_last=is_training,  # 마지막 배치 크기 일정하게
        persistent_workers=True  # worker 재사용으로 속도 향상
    )
```

### 3.2 학습 루프 - 현업 표준
```python
class Trainer:
    """현업에서 사용하는 학습 클래스"""
    
    def __init__(self, model, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler()  # Mixed Precision
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # GPU로 이동
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # ⭐ Mixed Precision Training (현업 필수)
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids)
                loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # ⭐ Gradient Clipping (현업에서 자주 사용)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Scheduler step (학습률 조정)
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # ⭐ 현업에서는 주기적 로깅 필수
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / len(dataloader)
```

### 3.3 최적화 - 현업에서 자주 쓰는 설정
```python
# Optimizer 설정 - 현업 베스트 프랙티스
def create_optimizer(model, config):
    # AdamW가 현업에서 가장 많이 쓰임
    no_decay = ['bias', 'LayerNorm.weight']  # weight decay 제외할 파라미터
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    
    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )

# Learning Rate Scheduler - 현업에서 자주 쓰는 패턴
def create_scheduler(optimizer, num_training_steps, warmup_steps=None):
    if warmup_steps is None:
        warmup_steps = num_training_steps * 0.1  # 전체의 10%
    
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        total_steps=num_training_steps,
        pct_start=warmup_steps / num_training_steps
    )
```

## 4. 모델 저장/로드 - 현업 필수 패턴

### 4.1 체크포인트 관리
```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """현업에서 사용하는 체크포인트 저장 패턴"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'pytorch_version': torch.__version__,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, path):
    """체크포인트 로드"""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']

# ⭐ 현업에서는 모델만 따로 저장도 많이 함
def save_model_only(model, path):
    """추론용 모델만 저장"""
    torch.save(model.state_dict(), path)

def load_model_for_inference(model_class, model_path, config):
    """추론용 모델 로드"""
    model = model_class(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
```

## 5. 현업 디버깅 & 모니터링

### 5.1 메모리 관리 - 현업 필수
```python
import gc

def clear_memory():
    """메모리 정리 - 현업에서 자주 사용"""
    torch.cuda.empty_cache()
    gc.collect()

def monitor_memory():
    """GPU 메모리 모니터링"""
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Context manager로 메모리 추적
class MemoryTracker:
    def __enter__(self):
        self.start_memory = torch.cuda.memory_allocated()
        return self
    
    def __exit__(self, *args):
        self.end_memory = torch.cuda.memory_allocated()
        print(f"Memory used: {(self.end_memory - self.start_memory) / 1024**2:.2f} MB")

# 사용 예
with MemoryTracker():
    output = model(input_tensor)
```

### 5.2 성능 프로파일링
```python
# ⭐ 현업에서 모델 속도 측정하는 방법
def benchmark_model(model, input_tensor, num_runs=100):
    """모델 성능 벤치마크"""
    model.eval()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    
    torch.cuda.synchronize()  # GPU 동기화 중요!
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"Average inference time: {avg_time*1000:.2f} ms")
```

## 6. 현업 배포 준비

### 6.1 모델 최적화
```python
# TorchScript 변환 - 프로덕션 배포용
def convert_to_torchscript(model, example_input):
    """TorchScript로 변환하여 배포 최적화"""
    model.eval()
    
    # Trace 방식
    traced_model = torch.jit.trace(model, example_input)
    
    # Script 방식 (제어 흐름이 있는 경우)
    # scripted_model = torch.jit.script(model)
    
    return traced_model

# 양자화 - 모델 크기 줄이기
def quantize_model(model):
    """동적 양자화로 모델 최적화"""
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model
```

## 7. 현업에서 알아둬야 할 필수 사항들

### 7.1 성능 최적화 팁
- **GPU 메모리 관리**: 배치 크기 조정, gradient checkpointing 사용
- **Mixed Precision**: AMP 사용으로 2배 빠른 학습
- **DataLoader 최적화**: num_workers, pin_memory 튜닝
- **학습률 스케줄링**: Warmup + Cosine decay 조합

### 7.2 디버깅 체크리스트
- **NaN 체크**: `torch.isnan()` 사용
- **Gradient 확인**: `torch.nn.utils.clip_grad_norm_()` 후 norm 값 로깅
- **Shape 불일치**: `tensor.shape` 자주 확인
- **Device 불일치**: 모든 tensor가 같은 device에 있는지 확인

### 7.3 현업 코딩 스타일
- **Config 클래스**: 모든 하이퍼파라미터 중앙 관리
- **로깅**: wandb, tensorboard 적극 활용
- **재현성**: `torch.manual_seed()` 설정
- **타입 힌트**: 함수 시그니처에 타입 명시

### 7.4 협업 시 주의사항
- **모델 버전 관리**: Git LFS 사용
- **실험 추적**: 실험 설정과 결과 체계적 기록
- **코드 리뷰**: 메모리 누수, 성능 이슈 중점 체크
- **문서화**: 모델 아키텍처, 학습 과정 상세 기록

---

**🚀 현업 생존 가이드**

1. **메모리가 생명**: GPU 메모리 부족은 현업에서 가장 흔한 문제
2. **배치 처리 마스터**: 모든 것을 배치로 생각하라
3. **실험 기록**: 무엇을 했는지 기록하지 않으면 재현 불가
4. **성능 측정**: "빠른 것 같다"는 금물, 정확히 측정하라
5. **점진적 개발**: 작은 단위로 테스트하며 개발하라


