import pandas as pd
import nltk
from rank_bm25 import BM25Okapi

# Danh sách các tệp CSV
files = [
    "data/viwiki-test-syl-qa.csv",
    "data/viwiki-dev-syl-qa.csv",
    "data/viwiki-train-syl-qa.csv",
    "data/ise-dsc-test-syl-qa.csv",
    "data/ise-dsc-dev-syl-qa.csv",
    "data/ise-dsc-train-syl-qa.csv"
]

# Đọc và nối các cột 'context'
context = pd.concat([pd.read_csv(file)['context'] for file in files])

# Merge context and split into sentences
context = context.drop_duplicates()
context.reset_index(drop=True, inplace=True)
combined_text = " ".join(context)
sentences = nltk.sent_tokenize(combined_text)

#Xóa những sentence bị trùng
sentences = list(set(sentences))

# Preprocess sentences: tokenize and lowercase for BM25
tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]

# Initialize BM25 model
bm25 = BM25Okapi(tokenized_sentences)

def ER_BM25(claim, top_best_evidence):

    # Tokenize the claim
    tokenized_claim = nltk.word_tokenize(claim.lower())
    # Get BM25 scores for the claim against all sentences
    scores = bm25.get_scores(tokenized_claim)
    # Get the indices of the top-k most relevant sentences
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_best_evidence]
    # Get the top-k most relevant sentences
    top_similar_sentences = [sentences[i] for i in top_indices]
    
    return ' '.join(top_similar_sentences)
