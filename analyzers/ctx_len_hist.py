from datasets import load_dataset
import matplotlib.pyplot as plt

# KorQuAD 데이터셋 로드하기
dataset = load_dataset("squad_kor_v1")

# context의 길이를 저장할 리스트 초기화
context_lengths = []

# 훈련 데이터셋의 각 항목에 대해 반복
for item in dataset["train"]:
    # context의 길이 계산 후 리스트에 추가
    context_length = len(item["context"])
    context_lengths.append(context_length)

# 히스토그램 작성
plt.figure(figsize=(5, 6))
plt.hist(context_lengths, bins=200, color="blue", alpha=0.7)
plt.title("Context Lengths in KorQuAD Dataset")
plt.xlabel("Length of context")
plt.ylabel("Frequency")
plt.xlim(0, 2000)
plt.show()
