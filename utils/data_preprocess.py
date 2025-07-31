import torch
from torch.utils.data import DataLoader, Dataset
import os
import pickle
import re

class SimpleVocab:
    def __init__(self, vocab_dict):
        self.vocab_dict = vocab_dict
        self.stoi = vocab_dict
        self.itos = {v: k for k, v in vocab_dict.items()}
    
    def __call__(self, tokens):
        return [self.stoi.get(token, self.stoi['<unk>']) for token in tokens]
    
    def __getitem__(self, key):
        return self.stoi[key]
    
    def __len__(self):
        return len(self.stoi)

class TranslationDataset(Dataset):
    def __init__(self, data_list, src_vocab, tgt_vocab, max_len=64):  # max_len을 64로 변경
        self.data = []
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
        # 데이터를 메모리에 로드
        for item in data_list:
            src_text = item['translation']['en']
            tgt_text = item['translation']['de']
            
            src_tokens = src_vocab(src_text.split())
            tgt_tokens = tgt_vocab(tgt_text.split())
            
            # 최대 길이 제한 (BOS, EOS 토큰 고려)
            if len(src_tokens) <= max_len - 2 and len(tgt_tokens) <= max_len - 2:
                self.data.append((src_tokens, tgt_tokens))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def simple_tokenizer(text):
    """간단한 토크나이저"""
    # 특수문자 제거하고 소문자로 변환
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

def yield_tokens(data_list, tokenizer, field='en'):
    """토큰을 yield하는 함수"""
    for item in data_list:
        text = item['translation'][field]
        yield tokenizer(text)

def create_vocab(data_list, tokenizer, min_freq=2, special_symbols=['<unk>', '<pad>', '<bos>', '<eos>']):
    """어휘 사전 생성"""
    from collections import Counter
    
    # 모든 토큰 수집
    all_tokens = []
    for item in data_list:
        text = item['translation']['en']
        tokens = tokenizer(text)
        all_tokens.extend(tokens)
    
    # 빈도 계산
    counter = Counter(all_tokens)
    
    # 최소 빈도 필터링
    filtered_tokens = [token for token, count in counter.items() if count >= min_freq]
    
    # 어휘 사전 생성
    vocab = {token: idx + len(special_symbols) for idx, token in enumerate(filtered_tokens)}
    
    # 특수 토큰 추가
    for idx, symbol in enumerate(special_symbols):
        vocab[symbol] = idx
    
    return SimpleVocab(vocab_dict=vocab)

def collate_batch(batch, src_vocab, tgt_vocab, max_len=64):  # max_len 줄임
    """배치 데이터를 처리하는 함수"""
    src_list, tgt_list = [], []
    
    for src_sample, tgt_sample in batch:
        # BOS, EOS 토큰 추가
        src_tokens = [src_vocab['<bos>']] + src_sample.tolist() + [src_vocab['<eos>']]
        tgt_tokens = [tgt_vocab['<bos>']] + tgt_sample.tolist() + [tgt_vocab['<eos>']]
        
        # 패딩 (정확한 길이로)
        src_padding = [src_vocab['<pad>']] * (max_len - len(src_tokens))
        tgt_padding = [tgt_vocab['<pad>']] * (max_len - len(tgt_tokens))
        
        src_tokens = src_tokens + src_padding
        tgt_tokens = tgt_tokens + tgt_padding
        
        src_list.append(src_tokens[:max_len])
        tgt_list.append(tgt_tokens[:max_len])
    
    return torch.tensor(src_list, dtype=torch.long), torch.tensor(tgt_list, dtype=torch.long)

def create_dummy_data(num_samples=1000):
    """더미 번역 데이터 생성"""
    dummy_data = []
    
    # 간단한 영어-독일어 번역 쌍들
    translation_pairs = [
        ("Hello world", "Hallo Welt"),
        ("How are you", "Wie geht es dir"),
        ("Good morning", "Guten Morgen"),
        ("Thank you", "Danke"),
        ("Please help me", "Bitte hilf mir"),
        ("I love you", "Ich liebe dich"),
        ("What is your name", "Wie ist dein Name"),
        ("Where are you going", "Wohin gehst du"),
        ("The weather is nice", "Das Wetter ist schön"),
        ("I want to learn German", "Ich möchte Deutsch lernen"),
        ("This is a test", "Das ist ein Test"),
        ("Machine learning is interesting", "Maschinelles Lernen ist interessant"),
        ("Artificial intelligence", "Künstliche Intelligenz"),
        ("Deep learning models", "Deep Learning Modelle"),
        ("Neural networks", "Neuronale Netze"),
        ("Transformers are powerful", "Transformer sind mächtig"),
        ("Attention mechanism", "Aufmerksamkeitsmechanismus"),
        ("Natural language processing", "Verarbeitung natürlicher Sprache"),
        ("Computer vision", "Computersehen"),
        ("Data science", "Datenwissenschaft")
    ]
    
    for i in range(num_samples):
        # 기본 번역 쌍에서 랜덤 선택
        base_pair = translation_pairs[i % len(translation_pairs)]
        
        # 약간의 변형 추가
        if i > len(translation_pairs):
            suffix = f" {i}"
            en_text = base_pair[0] + suffix
            de_text = base_pair[1] + suffix
        else:
            en_text = base_pair[0]
            de_text = base_pair[1]
        
        dummy_data.append({
            'translation': {
                'en': en_text,
                'de': de_text
            }
        })
    
    return dummy_data

def load_wmt14_data(batch_size=8, max_len=64, min_freq=2, max_samples=None):
    """WMT14 데이터 로드 및 전처리 (실제 WMT14 데이터 사용)"""
    print("Loading WMT14 dataset...")
    
    try:
        from datasets import load_dataset
        
        # WMT14 데이터셋 로드 (영어-독일어)
        dataset = load_dataset("wmt14", "de-en", split="train")
        val_dataset = load_dataset("wmt14", "de-en", split="validation")
        
        # 샘플 수 제한 (메모리 및 시간 고려)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            val_dataset = val_dataset.select(range(min(max_samples // 10, len(val_dataset))))
        
        print(f"Loaded {len(dataset)} training samples and {len(val_dataset)} validation samples")
        
        # 데이터 형식 변환
        train_data = []
        for item in dataset:
            train_data.append({
                'translation': {
                    'en': item['translation']['en'],
                    'de': item['translation']['de']
                }
            })
        
        val_data = []
        for item in val_dataset:
            val_data.append({
                'translation': {
                    'en': item['translation']['en'],
                    'de': item['translation']['de']
                }
            })
        
        # 토크나이저 생성
        tokenizer_en = simple_tokenizer
        tokenizer_de = simple_tokenizer
        
        print("Building vocabulary...")
        
        # 어휘 사전 생성
        src_vocab = create_vocab(train_data, tokenizer_en, min_freq)
        tgt_vocab = create_vocab(train_data, tokenizer_de, min_freq)
        
        print(f"Source vocabulary size: {len(src_vocab)}")
        print(f"Target vocabulary size: {len(tgt_vocab)}")
        
        # 데이터셋 생성
        train_dataset = TranslationDataset(train_data, src_vocab, tgt_vocab, max_len)
        val_dataset = TranslationDataset(val_data, src_vocab, tgt_vocab, max_len)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # DataLoader 생성
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=lambda batch: collate_batch(batch, src_vocab, tgt_vocab, max_len)
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, src_vocab, tgt_vocab, max_len)
        )
        
        return train_loader, val_loader, src_vocab, tgt_vocab
        
    except ImportError:
        print("datasets 라이브러리가 설치되지 않았습니다. 더미 데이터를 사용합니다.")
        return load_dummy_data(batch_size, max_len, min_freq, max_samples)
    except Exception as e:
        print(f"WMT14 데이터 로드 중 오류 발생: {e}")
        print("더미 데이터를 사용합니다.")
        return load_dummy_data(batch_size, max_len, min_freq, max_samples)

def load_dummy_data(batch_size=8, max_len=64, min_freq=2, max_samples=None):
    """더미 데이터 로드 (fallback)"""
    print("Loading dummy translation dataset...")
    
    # 더미 데이터 생성
    dataset = create_dummy_data(max_samples or 2000)
    val_dataset = create_dummy_data(max_samples // 10 if max_samples else 200)
    
    print(f"Created {len(dataset)} training samples and {len(val_dataset)} validation samples")
    
    # 토크나이저 생성
    tokenizer_en = simple_tokenizer
    tokenizer_de = simple_tokenizer
    
    print("Building vocabulary...")
    
    # 어휘 사전 생성
    src_vocab = create_vocab(dataset, tokenizer_en, min_freq)
    tgt_vocab = create_vocab(dataset, tokenizer_de, min_freq)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # 데이터셋 생성
    train_dataset = TranslationDataset(dataset, src_vocab, tgt_vocab, max_len)
    val_dataset = TranslationDataset(val_dataset, src_vocab, tgt_vocab, max_len)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, src_vocab, tgt_vocab, max_len)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, src_vocab, tgt_vocab, max_len)
    )
    
    return train_loader, val_loader, src_vocab, tgt_vocab

def save_vocab(vocab, filepath):
    """어휘 사전 저장"""
    with open(filepath, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(filepath):
    """어휘 사전 로드"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
