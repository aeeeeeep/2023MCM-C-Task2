# 2023MCM-C-Task2

The second question of 2023 MCM-ICM Problem C

## Tree

```bash
├── data
│   ├── df.xlsx     # processed data after run features.py
│   ├── Problem_C_Data_Wordle.xlsx  # raw data
│   ├── unigram_freq.csv    # download from https://www.kaggle.com/datasets/rtatman/english-word-frequency
│   └── words.json  # raw words
├── 2.py			# train & pred
├── features.py		# extract 6 features
├── find.py         # return features
├── requirements.txt
└── tries.py		# normal distribution curve fit, convert 7 percent features into 2 features
```

## Run

```bash
conda create --name hanlp python=3.8
conda activate hanlp
pip install -r requirements.txt
python features.py # features.py use tensorflow
python tries.py
conda deactivate
python 2.py # 2.py use torch
```

## Blog

https://aeeeeeep.top/p/c1bfab13/
