import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import  get_linear_schedule_with_warmup
# AdamW from torch 
from torch.optim import AdamW

# dataset
def create_sample_dataset(num_samples=200):
    np.random.seed(42)
    """創建一個小型的範例數據集"""
    vocab = {
        'astronomy': ['The galaxy contains billions of stars.', 'A planet orbits a star.', 'Our universe is constantly expanding.', 'The probe entered a stable orbit.'],
        'physics': ['Quantum mechanics describes nature at the smallest scales.', 'Gravity is the force that attracts two bodies.', 'Energy cannot be created or destroyed.', 'The object has a significant mass.'],
        'cs': ['We wrote the script in Python.', 'The algorithm has a time complexity of O(n log n).', 'A new server was deployed to the database.', 'Clean code is important for maintenance.'],
        'biology': ['DNA carries genetic instructions.', 'Evolution by natural selection is a key concept.', 'The cell is the basic unit of life.', 'Every organism is made of cells.']
    }
    labels = list(vocab.keys())
    num_labels = len(labels)
    label_map = {name: i for i, name in enumerate(labels)}

    documents, y_true = [], np.zeros((num_samples, num_labels), dtype=int)
    for i in range(num_samples):
        num_doc_topics = np.random.randint(1, 3)
        doc_topics = np.random.choice(labels, num_doc_topics, replace=False)
        doc_sentences = []
        for topic in doc_topics:
            y_true[i, label_map[topic]] = 1
            doc_sentences.append(np.random.choice(vocab[topic]))
        np.random.shuffle(doc_sentences)
        documents.append(" ".join(doc_sentences))
    print(f"Created a sample dataset with {num_samples} documents and {num_labels} labels.")
    return documents, y_true, labels
def load_data_from_tsv(file_path):
    """
    從指定的 .tsv 檔案讀取資料集。
    檔案格式: '010101...[Tab]document text'
    """
    documents = []
    labels_list = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                # 使用 tab 作為分隔符
                label_str, text = line.split('\t', 1)
                
                # 將 0/1 字串轉換為 numpy 數字陣列
                label_vector = np.array([int(char) for char in label_str])
                
                documents.append(text)
                labels_list.append(label_vector)
            except ValueError:
                print(f"Skipping malformed line: {line}")
                continue

    if not documents:
        raise ValueError("No valid data found in the file.")

    # 將標籤列表堆疊成一個大的 numpy 矩陣
    y_true = np.vstack(labels_list)
    
    
    num_labels = y_true.shape[1]
    import json
    with open("dataset/AAPD/label_to_index.json", "r") as f:
            label_to_index = json.load(f)
    index_to_name = {v: k for k, v in label_to_index.items()}
    label_names = [index_to_name[i] for i in range(num_labels)]
                
    print(f"Loaded {len(documents)} documents from '{file_path}'.")
    print(f"Labels are vectors of size {num_labels}.")
    
    return documents, y_true, label_names


class DictDataset(Dataset):
    def __init__(self, encodings, labels):
        self.enc = encodings
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        item = {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": self.labels[idx],
            "index": idx
        }
        if "token_type_ids" in self.enc:
            item["token_type_ids"] = self.enc["token_type_ids"][idx]
        else:
            item["token_type_ids"] = torch.zeros_like(self.enc["input_ids"][idx])
        return item
    
class DatasetReassembler:
    """
    職責：將原始的輸入特徵 (Input IDs, Mask) 與 '修正後的標籤' 重新組合成新的 DataLoader。
    """
    @staticmethod
    def create_retraining_loader(encoded_inputs, corrected_labels, batch_size):
        """
        encoded_inputs: 包含 input_ids 和 attention_mask 的字典或 tuple (依原本 encoded_batch_train 格式而定)
        corrected_labels: Tensor, 形狀為 [num_samples, num_labels]
        """
        # 假設 encoded_inputs 是一個包含 (input_ids, attention_mask) 的 tuple 或 list
        # 這裡根據你原本 load_and_preprocess_data 的回傳格式進行適配
        if isinstance(encoded_inputs, (tuple, list)):
            input_ids = encoded_inputs[0]
            attention_masks = encoded_inputs[1]
            
        else:
            # 若是 dict 格式
            input_ids = encoded_inputs['input_ids']
            attention_masks = encoded_inputs['attention_mask']

        # 確保 corrected_labels 是 Tensor
        if not isinstance(corrected_labels, torch.Tensor):
            corrected_labels = torch.tensor(corrected_labels)

        # 建立新的 Dataset
        new_encodings = {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': corrected_labels,
            # 'labels': labels,
            # 'index': indexs
        }
        retrain_data = DictDataset(new_encodings, corrected_labels)

        retrain_sampler = RandomSampler(retrain_data)
        retrain_dataloader = DataLoader(retrain_data, sampler=retrain_sampler, batch_size=batch_size)
        
        return retrain_dataloader