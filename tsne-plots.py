
from matplotlib import pyplot as plt
import pandas as pd

df_audio_features_comb = pd.read_csv("audio_neural_features_combined.csv")

df_audio_zs = df_audio_features_comb.copy().iloc[:, 1020:2160]
df_audio_zs_mean = df_audio_zs.mean(axis=0)
df_audio_zs_std = df_audio_zs.std(axis=0)
df_audio_zs = (df_audio_zs - df_audio_zs_mean) / df_audio_zs_std
df_audio_zs["subject"] = df_audio_features_comb["subject"]
# delete empty rows
df_audio_zs = df_audio_zs.dropna(how='all')

# run tsne on df_audio_features_comb
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
df_tsne = df_audio_zs.copy()

# drop nan rows
df_tsne = df_tsne.dropna()
tsne_results = tsne.fit_transform(df_tsne.drop(columns=["score", "subject"], errors="ignore"))
df_tsne["tsne-2d-one"] = tsne_results[:, 0]
df_tsne["tsne-2d-two"] = tsne_results[:, 1]


plt.figure(figsize=(4, 4))
plt.scatter(df_tsne["tsne-2d-one"], df_tsne["tsne-2d-two"], c=pd.factorize(df_tsne["subject"])[0], cmap='Accent', alpha=0.7)
plt.colorbar(label='Subject')
# add the subject names to the colorbar
cbar = plt.colorbar(ticks=range(len(df_tsne["subject"].unique())))
cbar.ax.set_yticklabels(df_tsne["subject"].unique())
plt.tight_layout()
plt.title("t-SNE Subject")
plt.savefig("tsne_subject.pdf", bbox_inches='tight')