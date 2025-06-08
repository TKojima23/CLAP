import sys
sys.path.append("src")

from laion_clap import CLAP_Module
import torch
import torch.nn.functional as F

# モデル初期化・重み読み込み
model = CLAP_Module(enable_fusion=False)
model.load_ckpt("retrieval/630k-audioset-best.pt")

# 音声ファイルとテキストクエリ
text_prompts = ["a dog barking", "a baby crying", "a car passing"]
audio_files = ["retrieval/dog_bark.wav"]

# 埋め込み取得（NumPyで返ってくる → torch.Tensorに変換）
text_embed_np = model.get_text_embedding(text_prompts)
audio_embed_np = model.get_audio_embedding_from_filelist(audio_files)

text_embed = torch.tensor(text_embed_np, dtype=torch.float32)
audio_embed = torch.tensor(audio_embed_np, dtype=torch.float32)

# cosine類似度計算のためにL2正規化
text_embed = F.normalize(text_embed, dim=-1)
audio_embed = F.normalize(audio_embed, dim=-1)

# 類似度計算
scores = torch.matmul(audio_embed, text_embed.T)[0]  # [N_text]
scores = scores.cpu().tolist()

# 結果表示
for prompt, score in zip(text_prompts, scores):
    print(f"Text: '{prompt}' → Score: {score:.4f}")
