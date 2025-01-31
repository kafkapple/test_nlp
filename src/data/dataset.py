from torch.utils.data import Dataset
import pandas as pd
import os
import wget
import tarfile
import zipfile

def download_and_extract(url, output_dir="./"):
    """
    URL에서 파일을 다운로드하고 압축을 해제합니다.
    
    Args:
        url (str): 다운로드할 파일의 URL.
        output_dir (str): 파일을 저장하고 압축을 풀 디렉토리 경로.
    """
    # 다운로드 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 파일 이름 추출
    filename = os.path.basename(url)

    # 다운로드 파일 경로
    download_path = os.path.join(output_dir, filename)

    # 파일 다운로드
    print(f"Downloading {url}...")
    wget.download(url, download_path)
    print(f"\nDownloaded to {download_path}")

    # 압축 해제
    print("Extracting files...")
    if filename.endswith(".tar.gz") or filename.endswith(".tgz"):
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(output_dir)
            print(f"Extracted to {output_dir}")
    elif filename.endswith(".zip"):
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
            print(f"Extracted to {output_dir}")
    else:
        print("Unsupported file type. No extraction performed.")

    # 원본 압축 파일 제거 (선택)
    os.remove(download_path)
    print(f"Removed compressed file {download_path}")

class DialogueDataset(Dataset):
    def __init__(self, encoder_input, decoder_input=None, labels=None, tokenizer=None):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        
        if self.decoder_input is not None:
            decoder_item = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
            item['decoder_input_ids'] = decoder_item['input_ids']
            item['decoder_attention_mask'] = decoder_item['attention_mask']
            
        if self.labels is not None:
            labels = self.labels['input_ids'][idx].clone().detach()
            # 패딩 토큰을 -100으로 변환
            if self.tokenizer is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            item['labels'] = labels
            
        return item
    
    def __len__(self):
        return len(next(iter(self.encoder_input.values())))

class DataProcessor:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        
    def prepare_data(self, data_path, is_train=True):
        if not os.path.exists("data"):
            print("Downloading data...")
            data_dir = "https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000342/data/data.tar.gz"
            code_dir = "https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000342/data/code.tar.gz"
            download_and_extract(data_dir, "./")
            download_and_extract(code_dir, "./")
            print("Downloaded data")
        df = pd.read_csv(data_path)
        if is_train:
            return self._prepare_train_data(df)
        return self._prepare_test_data(df)
        
    def _prepare_train_data(self, df):
        # 입력 텍스트 인코딩
        model_inputs = self.tokenizer(
            df['dialogue'].tolist(),
            max_length=self.config.tokenizer.encoder_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 타겟 텍스트 인코딩
        labels = self.tokenizer(
            df['summary'].tolist(),  # BOS/EOS 토큰은 tokenizer가 자동으로 추가
            max_length=self.config.tokenizer.decoder_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # DialogueDataset 인스턴스 반환
        return DialogueDataset(
            encoder_input=model_inputs,
            labels=labels,
            tokenizer=self.tokenizer  # tokenizer 전달
        )
        
    def _prepare_test_data(self, df):
        # 입력 텍스트 인코딩
        model_inputs = self.tokenizer(
            df['dialogue'].tolist(),
            max_length=self.config.tokenizer.encoder_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # DialogueDataset 인스턴스 반환 (labels 없음)
        return DialogueDataset(
            encoder_input=model_inputs
        ) 