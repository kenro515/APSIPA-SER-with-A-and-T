# APSIPA-SER-with-A-and-T

This code is the implementation of Speech Emotion Recognition (SER) with acoustic and linguistic features.
The network model is Convolutional Neural Network (CNN) + Bidirectional Long Short Term Memory (BLSTM) + Self-Attention and BERT.
Before running this code, you should get model parameters from "APSIPA-SER-with-A" and "APSIPA-SER-with-T."

## How to use

0. Run main.py in "APSIPA-SER-with-A" and "APSIPA-SER-with-T" 
1. Edit hyper_param.yaml
2. Run main.py
```
python3 main.py
```
## Reference

[1] GitHub_cl-tohoku/bert-japanese, https://github.com/cl-tohoku/bert-japanese (Last View: 2022-02-07)

[2] Huggingface_bert-japanese, https://huggingface.co/docs/transformers/model_doc/bert-japanese (Last View: 2022-02-07)

[3] Qiita-自然言語処理モデル（BERT）を利用した日本語の文章分類, https://qiita.com/takubb/items/fd972f0ac3dba909c293#bertforsequenceclassification (Last View: 2022-02-07)
