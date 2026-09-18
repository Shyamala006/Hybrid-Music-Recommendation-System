# IMPORTS
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, auc_score
import random
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings("ignore")

# LOAD DATA
songs_df = pd.read_csv("data/data.csv")
p = 0.1
playlist_df = pd.read_csv(
    "data/spotify_dataset.csv",
    error_bad_lines=False,
    warn_bad_lines=False,
    skiprows=lambda i: i > 0 and random.random() > p
)

# SONG DATA PRE-PROCESSING (CONTENT-BASED)
songs_df = songs_df.reset_index(drop=True)
songs_df["song_id"] = songs_df.index
AUDIO_FEATURES = [
    "danceability", "energy", "valence",
    "speechiness", "instrumentalness", "acousticness"
]
song_features = songs_df[AUDIO_FEATURES].fillna(0)
song_features_norm = normalize(song_features)

# PLAYLIST PRE-PROCESSING (COLLABORATIVE)
playlist_df.columns = playlist_df.columns.str.replace('"', '').str.strip()
playlist_df = playlist_df[['user_id', 'trackname', 'artistname']]
playlist_df = (
    playlist_df.groupby(['user_id', 'trackname', 'artistname'])
    .size()
    .reset_index(name='freq')
)
playlist_df = playlist_df[
    playlist_df.groupby('trackname').freq.transform('sum') >= 30
]
playlist_df = playlist_df[
    playlist_df.groupby('user_id').trackname.transform('nunique') >= 5
]
users = playlist_df['user_id'].unique()
tracks = playlist_df['trackname'].unique()

# TRAINING & TESTING
dataset = Dataset()
dataset.fit(users, tracks)
(interactions, weights) = dataset.build_interactions(
    [(row.user_id, row.trackname, row.freq) for _, row in playlist_df.iterrows()]
)
train, test = random_train_test_split(
    interactions, test_percentage=0.2, random_state=42
)
# Train Model
model = LightFM(loss='warp')
model.fit(train, epochs=20, num_threads=4)
# Evaluation
precision = precision_at_k(model, test, k=5).mean()
auc = auc_score(model, test).mean()
print(f"Precision@5: {precision:.4f}")
print(f"AUC Score: {auc:.4f}")

# CONTENT-BASED RECOMMENDATION
def recommend_songs(song_id, top_n):
    if song_id < 0 or song_id >= len(songs_df):
        raise ValueError("Invalid Song ID!")    
    similarity = cosine_similarity(
        song_features_norm[song_id].reshape(1, -1),
        song_features_norm
    ).flatten()
    indices = np.argsort(similarity)[::-1][1:top_n + 1]
    similar_songs = songs_df.loc[indices, ['name', 'artists']].apply(
        lambda x: f"{x['name']} — {x['artists']}", axis=1
    )
    scores = similarity[indices]
    return similar_songs.tolist(), scores.tolist()

# COLLABORATIVE RECOMMENDATION
def recommend_tracks(user_id, top_n):
    user_map, _, item_map, _ = dataset.mapping()
    user_index = user_map[user_id]
    item_indices = np.arange(len(item_map))
    scores = model.predict(user_index, item_indices)
    top_items = np.argsort(scores)[::-1][:top_n]
    reverse_item_map = {v: k for k, v in item_map.items()}
    results = []
    score_vals = []
    for idx in top_items:
        track = reverse_item_map[idx]
        artist = playlist_df[playlist_df['trackname'] == track]['artistname'].iloc[0]
        results.append(f"{track} — {artist}")
        score_vals.append(scores[idx])
    return results, score_vals

# CLUSTERED BAR PLOT
def plot_clustered_scores(collab_items, collab_scores, content_items, content_scores, frame):
    n = len(collab_items)
    indices = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(indices - width/2, collab_scores, width, label='Collaborative', color='skyblue')
    ax.bar(indices + width/2, content_scores, width, label='Content-Based', color='pink')
    ax.set_xticks(indices)
    ax.set_xticklabels([str(i+1) for i in range(n)])
    ax.set_xlabel("Rank Position")
    ax.set_ylabel("Score / Similarity")
    ax.set_title("Hybrid Recommendations (Clustered)")
    ax.legend()
    fig.tight_layout()
    # Embed in Tkinter
    for widget in frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# HYBRID ENGINE
def hybrid_recommend(song_id, top_n):
    similar_songs, content_scores = recommend_songs(song_id, top_n)
    default_user = users[0]
    collab_tracks, collab_scores = recommend_tracks(default_user, top_n)
    return collab_tracks, collab_scores, similar_songs, content_scores

# GUI LOGIC
def run_recommendation():
    try:
        song_id = int(song_entry.get())
        top_n = int(topn_entry.get())
        output_box.delete("1.0", tk.END)
        collab_tracks, collab_scores, similar_songs, content_scores = hybrid_recommend(song_id, top_n)
        output_box.insert(tk.END, "🎧Collaborative Recommendations\n")
        output_box.insert(tk.END, "-"*55 + "\n")
        for i, (track, score) in enumerate(zip(collab_tracks, collab_scores),1):
            output_box.insert(tk.END, f"{i}. {track} (score: {score:.2f})\n")
        output_box.insert(tk.END, "\n🎵Content-Based Recommendations\n")
        output_box.insert(tk.END, "-"*55 + "\n")
        for i, (song, score) in enumerate(zip(similar_songs, content_scores),1):
            output_box.insert(tk.END, f"{i}. {song} (similarity: {score:.2f})\n")
        # Plot clustered bar chart
        plot_clustered_scores(collab_tracks, collab_scores, similar_songs, content_scores, plot_frame)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# TKINTER GUI
root = tk.Tk()
root.title("Recommendation Engine")
root.geometry("1000x750")
tk.Label(
    root,
    text="🎶Music Recommendation Engine",
    font=("Arial", 18, "bold")
).pack(pady=10)
# Input frame
input_frame = tk.Frame(root)
input_frame.pack(pady=10)
tk.Label(input_frame, text="Song ID").grid(row=0, column=0, padx=5)
song_entry = tk.Entry(input_frame, width=10)
song_entry.grid(row=0, column=1, padx=5)
tk.Label(input_frame, text="Top N").grid(row=0, column=2, padx=5)
topn_entry = tk.Entry(input_frame, width=10)
topn_entry.grid(row=0, column=3, padx=5)
tk.Button(
    root,
    text="Get Recommendations",
    command=run_recommendation
).pack(pady=5)
# Output box
output_box = tk.Text(root, font=("Arial", 11), height=15)
output_box.pack(fill="both", expand=False, padx=10, pady=10)
# Frame for clustered plot
plot_frame = tk.Frame(root)
plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
root.mainloop()