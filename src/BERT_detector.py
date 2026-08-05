# BERT_detector.py
import numpy as np
from transformers import BertTokenizer, DistilBertTokenizer, TFBertModel
from tensorflow import keras
from tensorflow.keras import layers

class XSSDetector_BERT:
    def __init__(self, model_path, model_type="tinyBERT", max_length=256):
       
        self.model_path = model_path
        self.model_type = model_type
        self.max_length = max_length

        if model_type == "tinyBERT":
            self.tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
        elif model_type == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif model_type == "DistilBERT":
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # 載入訓練好的 keras 模型
        self.model = keras.models.load_model(model_path, custom_objects={"TFBertModel": TFBertModel})

    def preprocess(self, payload):
        
        tokens = self.tokenizer.encode(payload, add_special_tokens=True, truncation=True, max_length=self.max_length)
        attention_mask = [1] * len(tokens)

        pad_len = self.max_length - len(tokens)
        if pad_len > 0:
            tokens += [0] * pad_len
            attention_mask += [0] * pad_len
        else:
            tokens = tokens[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        return np.array([tokens]), np.array([attention_mask])
    
    def preprocess_batch(self, payloads):
        """批次預處理多個 payloads"""
        all_tokens = []
        all_masks = []
        
        for payload in payloads:
            # 確保 payload 是字串
            if not isinstance(payload, str):
                payload = str(payload)
                
            tokens = self.tokenizer.encode(payload, add_special_tokens=True, 
                                          truncation=True, max_length=self.max_length)
            attention_mask = [1] * len(tokens)
            
            pad_len = self.max_length - len(tokens)
            if pad_len > 0:
                tokens += [0] * pad_len
                attention_mask += [0] * pad_len
            else:
                tokens = tokens[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            
            all_tokens.append(tokens)
            all_masks.append(attention_mask)
        
        return np.array(all_tokens), np.array(all_masks)


    def is_xss(self, payload):
        token_ids, att_mask = self.preprocess(payload)
        prediction = self.model([token_ids, att_mask], training=False)
        return int(np.argmax(prediction, axis=1)[0])
    
    def predict_batch(self, payloads, batch_size=32):
        """批次預測，大幅提升速度"""
        all_preds = []
        
        for i in range(0, len(payloads), batch_size):
            batch = payloads[i:i+batch_size]
            token_ids, att_masks = self.preprocess_batch(batch)
            predictions = self.model.predict([token_ids, att_masks], verbose=0)
            batch_preds = np.argmax(predictions, axis=1).tolist()
            all_preds.extend(batch_preds)
        
        return all_preds

    def predict_proba(self, payload):
    
        token_ids, att_mask = self.preprocess(payload)
        return self.model.predict([token_ids, att_mask], verbose=0)[0]
    
    def predict_proba_batch(self, payloads, batch_size=32):
        """批次預測機率"""
        all_probas = []
        
        for i in range(0, len(payloads), batch_size):
            batch = payloads[i:i+batch_size]
            token_ids, att_masks = self.preprocess_batch(batch)
            probas = self.model.predict([token_ids, att_masks], verbose=0)
            all_probas.extend(probas)
        return np.array(all_probas)

def build_bert_bilstm_model(bert_layer, max_length=256):
    
    inputs = keras.Input(shape=(max_length,), dtype="int64", name="input_ids")
    attention_mask = keras.Input(shape=(max_length,), dtype="int64", name="attention_mask")

    bert_outputs = bert_layer(inputs, attention_mask=attention_mask)
    sequence_output = bert_outputs.last_hidden_state
    pooler_output = bert_outputs.pooler_output

    bilstm_output = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False)
    )(sequence_output)

    merged = layers.Concatenate()([pooler_output, bilstm_output])
    outputs = layers.Dense(2, activation='softmax')(merged)

    model = keras.Model(inputs=[inputs, attention_mask], outputs=outputs)
    return model