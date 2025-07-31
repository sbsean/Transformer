# train.py ( WMT14 Translation )
import os
import sys
import random
import inspect
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_checkpoint_folder(base_path='checkpoints', log=False):
    """체크포인트 및 로그 저장을 위한 폴더를 생성합니다."""
    setting_number = 1
    while True:
        setting_folder = f'setting_#{setting_number}'
        path = os.path.join(base_path, setting_folder)
        if log:
            path = os.path.join(path, 'logs')
        
        if not os.path.exists(path):
            os.makedirs(path)
            
            return os.path.dirname(path) if log else path
        
        
        if os.path.exists(os.path.join(base_path, f'setting_#{setting_number}')):
             setting_number += 1
        else: 
            os.makedirs(path)
            return os.path.dirname(path)


class Trainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get('model')
        if not self.model:
            raise ValueError("Model is required.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        self.src_vocab = kwargs.get('src_vocab')
        self.tgt_vocab = kwargs.get('tgt_vocab')
        if not self.src_vocab or not self.tgt_vocab:
            raise ValueError("Source and target vocabularies are required.")

       
        self.lr = kwargs.get('lr', 1e-4)
        self.num_epochs = kwargs.get('num_epochs', 10)
        
       
        self.criterion = kwargs.get('criterion')
        if not self.criterion:
            raise ValueError("Criterion (loss function) is required.")

       
        self.train_loader = kwargs.get('train_data')
        self.val_loader = kwargs.get('val_data')
        if not self.train_loader or not self.val_loader:
            raise ValueError("DataLoaders are required.")

       
        self.seed = kwargs.get('seed', 42)
        self.checkpoint_dir = create_checkpoint_folder(base_path=kwargs.get('checkpoint_dir', './checkpoints'))
        log_dir = os.path.join(self.checkpoint_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self._set_seed(self.seed)
        self.model.to(self.device)

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _init_optimizer(self):
        opt_name = self.kwargs.get('optimizer', 'Adam')
        optimizer_class = getattr(torch.optim, opt_name)
        
        
        if opt_name == 'Adam':
            self.optimizer = optimizer_class(
                self.model.parameters(), 
                lr=self.lr,
                weight_decay=self.kwargs.get('weight_decay', 1e-4)
            )
        else:
            
            valid_kwargs = {
                k: v for k, v in self.kwargs.items() 
                if k in inspect.signature(optimizer_class).parameters
            }
            self.optimizer = optimizer_class(self.model.parameters(), **valid_kwargs)

    def _init_scheduler(self):
        """스케줄러 초기화"""
        scheduler_name = self.kwargs.get('scheduler')
        if scheduler_name and self.optimizer:
            # 기본 스케줄러 (StepLR 등)
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
            valid_kwargs = {
                k: v for k, v in self.kwargs.items() 
                if k in inspect.signature(scheduler_class).parameters and k != 'optimizer'
            }
            self.scheduler = scheduler_class(self.optimizer, **valid_kwargs)
        else:
            self.scheduler = None

    def _calculate_bleu_score(self, outputs, targets):
        """BLEU 스코어 계산"""
        # NLTK 데이터 다운로드 (처음 실행 시에만)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("NLTK punkt 데이터를 다운로드합니다...")
            nltk.download('punkt', quiet=True)
            print("다운로드 완료!")
        
        # BLEU 계산을 위한 토큰화 및 디코딩
        smoothing = SmoothingFunction()
        bleu_scores = []
        
        # 배치의 각 샘플에 대해 BLEU 계산
        preds = torch.argmax(outputs, dim=-1)
        
        for i in range(preds.size(0)):
            # 예측 시퀀스 디코딩 (패딩 토큰 제거)
            pred_seq = preds[i].cpu().numpy().flatten()  # 1차원으로 평탄화
            target_seq = targets[i].cpu().numpy().flatten()  # 1차원으로 평탄화
            
            # 패딩 토큰(<pad>)과 EOS 토큰(<eos>) 제거
            pred_tokens = []
            for token in pred_seq:
                if token == self.tgt_vocab['<pad>'] or token == self.tgt_vocab['<eos>']:
                    break
                pred_tokens.append(token)
            
            target_tokens = []
            for token in target_seq:
                if token == self.tgt_vocab['<pad>'] or token == self.tgt_vocab['<eos>']:
                    break
                target_tokens.append(token)
            
            # BLEU 계산 (sentence_bleu 사용)
            if len(pred_tokens) > 0 and len(target_tokens) > 0:
                bleu = sentence_bleu([target_tokens], pred_tokens, 
                                   smoothing_function=smoothing.method1)
                bleu_scores.append(bleu)
            else:
                bleu_scores.append(0.0)
        
        return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    def plot_training_curves(self, train_losses, val_losses, train_bleus, val_bleus):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # BLEU curves
        ax2.plot(train_bleus, label='Train BLEU', color='blue')
        ax2.plot(val_bleus, label='Validation BLEU', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('BLEU Score')
        ax2.set_title('Training and Validation BLEU Score')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        img_dir = os.path.join(self.checkpoint_dir, 'img')
        os.makedirs(img_dir, exist_ok=True)
        fig.savefig(os.path.join(img_dir, 'training_curves.png'), dpi=150)
        plt.close(fig)

    def train_step(self, pbar):
        """한 에포크의 학습 단계"""
        self.model.train()
        total_loss = 0.0
        total_bleu = 0.0
        
        for i, (src, tgt) in enumerate(self.train_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # Teacher forcing을 위한 입력/출력 분리
            tgt_input = tgt[:, :-1]  # Remove last token
            tgt_output = tgt[:, 1:]  # Remove first token

            self.optimizer.zero_grad()
            outputs = self.model(src, tgt_input)
            
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.contiguous().view(-1, vocab_size)
            tgt_output = tgt_output.contiguous().view(-1)
            
            loss = self.criterion(outputs, tgt_output)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            bleu = self._calculate_bleu_score(outputs, tgt_output)
            total_bleu += bleu

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'BLEU': f"{bleu:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.1e}"
            })
            pbar.update(1)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_bleu = total_bleu / len(self.train_loader)
        return avg_loss, avg_bleu

    @torch.no_grad()
    def evaluate(self):
        """검증 단계"""
        self.model.eval()
        total_loss = 0.0
        total_bleu = 0.0
        
        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        
        for src, tgt in pbar:
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # Teacher forcing을 위한 입력/출력 분리
            # Teacher forcing: 훈련 시 디코더가 이전에 예측한 토큰 대신 실제 정답 토큰을 입력으로 받는 기법
            # 장점: 훈련 초기에 안정적인 학습 가능, 수렴 속도 향상
            # 단점: 훈련과 추론 시 입력 분포 차이로 인한 exposure bias 문제 발생 가능
            # tgt_input: 디코더 입력 (시퀀스의 마지막 토큰 제외)
            # tgt_output: 예측해야 할 정답 (시퀀스의 첫 번째 토큰 제외)
            tgt_input = tgt[:, :-1]  # [B, seq_len-1] - <sos> 토큰부터 마지막 토큰 직전까지
            tgt_output = tgt[:, 1:]  # [B, seq_len-1] - 두 번째 토큰부터 <eos> 토큰까지
            
            outputs = self.model(src, tgt_input)
            
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.contiguous().view(-1, vocab_size)
            tgt_output = tgt_output.contiguous().view(-1)
            
            loss = self.criterion(outputs, tgt_output)
            
            total_loss += loss.item()
            bleu = self._calculate_bleu_score(outputs, tgt_output)
            total_bleu += bleu

            pbar.set_postfix({'val_loss': f"{loss.item():.4f}", 'val_BLEU': f"{bleu:.4f}"})
            pbar.update(1)

        avg_loss = total_loss / len(self.val_loader)
        avg_bleu = total_bleu / len(self.val_loader)
        return avg_loss, avg_bleu

    def train(self):
        """전체 학습 과정 실행"""
        self._init_optimizer()
        self._init_scheduler()

        ckpt_dir = os.path.join(self.checkpoint_dir, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)

        epoch_train_losses, epoch_val_losses = [], []
        epoch_train_bleus, epoch_val_bleus = [], []
        
        best_val_loss = float('inf')

        total_steps = len(self.train_loader) * self.num_epochs
        pbar = tqdm(total=total_steps, desc="Training", unit="step", dynamic_ncols=True)

        for epoch in range(self.num_epochs):
            # Training phase
            train_loss, train_bleu = self.train_step(pbar)
            epoch_train_losses.append(train_loss)
            epoch_train_bleus.append(train_bleu)

            # Validation phase
            val_loss, val_bleu = self.evaluate()
            epoch_val_losses.append(val_loss)
            epoch_val_bleus.append(val_bleu)

            if self.scheduler:
                self.scheduler.step()

           
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/BLEU', train_bleu, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/BLEU', val_bleu, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'src_vocab_size': len(self.src_vocab),
                    'tgt_vocab_size': len(self.tgt_vocab),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_bleu': train_bleu,
                    'val_bleu': val_bleu
                }, os.path.join(ckpt_dir, 'best_model.pth'))

            # 매 에포크마다 학습 곡선 저장
            self.plot_training_curves(epoch_train_losses, epoch_val_losses, epoch_train_bleus, epoch_val_bleus)

            tqdm.write(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train BLEU: {train_bleu:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val BLEU: {val_bleu:.4f}"
            )
        
        pbar.close()
        self.writer.close()
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")