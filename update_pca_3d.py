import json

with open('notebooks/05_visualization.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "from sklearn.decomposition import PCA\n",
    "import plotly.express as px\n",
    "\n",
    "print(\"Loading 169D features...\")\n",
    "train_data = np.load(\"../results/features_train.npz\")\n",
    "X_train = train_data[\"features_169d\"]\n",
    "labels = df_train['cluster_id'].values\n",
    "labels_text = df_train['cluster_label'].values\n",
    "\n",
    "# Lấy mẫu 10,000 điểm để chạy PCA nhanh và biểu đồ không bị rối\n",
    "np.random.seed(42)\n",
    "sample_idx = np.random.choice(len(X_train), 10000, replace=False)\n",
    "X_sample = X_train[sample_idx]\n",
    "labels_sample = labels_text[sample_idx]  # Hiển thị tên cụm thay vì số ID\n",
    "\n",
    "print(\"Running PCA projection to 3D...\")\n",
    "pca = PCA(n_components=3, random_state=42)\n",
    "X_pca = pca.fit_transform(X_sample)\n",
    "\n",
    "pca_df = pd.DataFrame({\n",
    "    'PC1': X_pca[:, 0],\n",
    "    'PC2': X_pca[:, 1],\n",
    "    'PC3': X_pca[:, 2],\n",
    "    'Cụm Nghề nghiệp': labels_sample\n",
    "})\n",
    "\n",
    "fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3',\n",
    "              color='Cụm Nghề nghiệp',\n",
    "              opacity=0.6,\n",
    "              title='Bản đồ Phân mảnh Không gian Cụm (PCA 3D Projection)',\n",
    "              color_discrete_sequence=px.colors.qualitative.Alphabet)\n",
    "\n",
    "fig.update_traces(marker=dict(size=3))\n",
    "fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))\n",
    "fig.write_html('../plots/05_pca_clusters_3d.html')\n",
    "fig.show()\n",
    "print(\"Visualization phase complete! 3D plot saved as HTML.\")"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'source' in cell and len(cell['source']) > 0 and 'sklearn.decomposition import PCA' in cell['source'][0]:
        cell['source'] = new_source

with open('notebooks/05_visualization.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Updated cell to 3D Plotly!")
