import json

with open('notebooks/05_visualization.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12))\n",
    "\n",
    "# Sắp xếp các cụm theo mức lương trung vị tối thiểu tăng dần để dễ nhìn\n",
    "order = cluster_profiles.sort_values('salary_min_med')['cluster_label']\n",
    "\n",
    "# Subplot 1: Lương tối thiểu\n",
    "sns.boxplot(data=df_train, x='salary_min_m_vnd', y='cluster_label', order=order, \n",
    "            palette='viridis', showfliers=False, width=0.6, ax=ax1)\n",
    "ax1.set_title('Phổ Lương Tối Thiểu (Đã ẩn Outliers)', fontsize=16, fontweight='bold', pad=20)\n",
    "ax1.set_xlabel('Mức lương tối thiểu (Triệu VNĐ)', fontsize=13)\n",
    "ax1.set_ylabel('Cụm Nghề nghiệp', fontsize=13)\n",
    "\n",
    "# Subplot 2: Lương tối đa\n",
    "sns.boxplot(data=df_train, x='salary_max_m_vnd', y='cluster_label', order=order, \n",
    "            palette='magma', showfliers=False, width=0.6, ax=ax2)\n",
    "ax2.set_title('Phổ Lương Tối Đa (Đã ẩn Outliers)', fontsize=16, fontweight='bold', pad=20)\n",
    "ax2.set_xlabel('Mức lương tối đa (Triệu VNĐ)', fontsize=13)\n",
    "ax2.set_ylabel('')  # Ẩn nhãn trục Y ở biểu đồ thứ 2 để tránh lặp lại\n",
    "\n",
    "plt.suptitle('Phổ Dao Động Lương Tối Thiểu & Tối Đa Theo Cụm', fontsize=20, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../plots/05_salary_distribution_split.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'source' in cell and len(cell['source']) > 0 and 'sns.boxplot(data=df_train, x=\'salary_min_m_vnd\'' in "".join(cell['source']):
        cell['source'] = new_source
        print("Updated boxplot cell!")

with open('notebooks/05_visualization.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Process finished.")
