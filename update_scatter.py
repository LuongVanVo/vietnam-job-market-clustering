import json

with open('notebooks/05_visualization.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "# Tính trung vị lương (min, max) và kinh nghiệm cho mỗi cụm\n",
    "cluster_profiles = df_train.groupby('cluster_label').agg(\n",
    "    salary_min_med=('salary_min_m_vnd', 'median'),\n",
    "    salary_max_med=('salary_max_m_vnd', 'median'),\n",
    "    exp_med=('exp_min_years', 'median'),\n",
    "    size=('cluster_id', 'count')\n",
    ").reset_index()\n",
    "\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))\n",
    "\n",
    "# Hàm hỗ trợ vẽ và gắn nhãn\n",
    "def plot_scatter(ax, x_col, y_col, title, ylabel):\n",
    "    scatter = ax.scatter(x=cluster_profiles[x_col], \n",
    "                y=cluster_profiles[y_col], \n",
    "                s=cluster_profiles['size'] / 40,  # Scale bong bóng\n",
    "                alpha=0.6, \n",
    "                c=range(len(cluster_profiles)), cmap='tab20', edgecolors='white', linewidth=2)\n",
    "    \n",
    "    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)\n",
    "    ax.set_xlabel('Kinh nghiệm trung vị (Năm)', fontsize=13)\n",
    "    ax.set_ylabel(ylabel, fontsize=13)\n",
    "    \n",
    "    # Gắn nhãn, thêm một chút nhiễu (offset) để giảm chồng lấn\n",
    "    for i, row in cluster_profiles.iterrows():\n",
    "        short_name = row['cluster_label'].split('-')[1].strip() if '-' in row['cluster_label'] else row['cluster_label'][:15]\n",
    "        # Nếu các điểm có cùng tọa độ x, dịch y lên/xuống luân phiên\n",
    "        offset_y = 12 if i % 2 == 0 else -16\n",
    "        ax.annotate(short_name, \n",
    "                     (row[x_col], row[y_col]),\n",
    "                     xytext=(0, offset_y), textcoords='offset points',\n",
    "                     ha='center', fontsize=9, fontweight='bold',\n",
    "                     bbox=dict(boxstyle=\"round,pad=0.2\", fc=\"white\", ec=\"gray\", alpha=0.7))\n",
    "    \n",
    "    ax.grid(True, linestyle='--', alpha=0.7)\n",
    "\n",
    "# Subplot 1: Min Salary\n",
    "plot_scatter(ax1, 'exp_med', 'salary_min_med', 'Lương Tối Thiểu vs Kinh Nghiệm', 'Lương tối thiểu trung vị (Triệu VNĐ)')\n",
    "\n",
    "# Subplot 2: Max Salary\n",
    "plot_scatter(ax2, 'exp_med', 'salary_max_med', 'Lương Tối Đa vs Kinh Nghiệm', 'Lương tối đa trung vị (Triệu VNĐ)')\n",
    "\n",
    "plt.suptitle('Bản đồ Định vị Nghề nghiệp Phân mảnh', fontsize=20, fontweight='bold', y=1.05)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../plots/05_value_position_map_split.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'source' in cell and len(cell['source']) > 0 and 'Bản đồ Định vị Nghề nghiệp: Lương tối thiểu vs Kinh nghiệm' in "".join(cell['source']):
        cell['source'] = new_source
        print("Updated cell!")

with open('notebooks/05_visualization.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Process finished.")
