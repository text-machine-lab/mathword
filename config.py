import sys

# add path for transformer
PROJECT_PATH = '..'
TRANSFORMER_PATH = '../transformerpy'
FORMATTED_DATA = '/data2/ymeng/dolphin18k/eval_dataset/eval_dataset_formatted.json'

if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
if TRANSFORMER_PATH not in sys.path:
    sys.path.append(TRANSFORMER_PATH)

WORD_VECTORS = '../embeddings/crawl-300d-2M.vec'
EMBEDDING_DIM = 300