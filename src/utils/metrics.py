from rouge import Rouge
import numpy as np

def compute_metrics(tokenizer, pred, config):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    # 디코딩
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    # 특수 토큰 제거
    for token in config.train.inference.remove_tokens:
        pred_str = [pred.replace(token, '').strip() for pred in pred_str]
        label_str = [label.replace(token, '').strip() for label in label_str]
    
    # 빈 문자열 처리
    pred_str = [pred if pred else "empty" for pred in pred_str]
    label_str = [label if label else "empty" for label in label_str]
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores(pred_str, label_str, avg=True)
        
        return {
            'rouge1_f1': scores['rouge-1']['f'],
            'rouge2_f1': scores['rouge-2']['f'],
            'rougeL_f1': scores['rouge-l']['f']
        }
    except ValueError as e:
        print(f"ROUGE 계산 중 오류 발생: {str(e)}")
        print(f"Sample predictions: {pred_str[:3]}")
        print(f"Sample labels: {label_str[:3]}")
        # 오류 발생 시 0점 반환
        return {
            'rouge1_f1': 0.0,
            'rouge2_f1': 0.0,
            'rougeL_f1': 0.0
        } 