from BiLSTM.data_preprocessor import DataPreprocessor
data = [
    ("床前明月光", "疑是地上霜"),
    ("举头望明月", "低头思故乡"),
    ("欲出未出光辣達","千山萬山如火發")
    # 更多数据...
]
data_preprocessor = DataPreprocessor()
data_preprocessor.preprocess_data()
print(data_preprocessor.data_pairs)
