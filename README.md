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
## Paper

Ryotaro Nagase, Takahiro Fukumori and Yoichi Yamashita: ``Speech Emotion Recognition with Fusion of Acoustic- and Linguistic-Feature-Based Decisions, '' Proc. Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pp. 725 -- 730, 2021.
