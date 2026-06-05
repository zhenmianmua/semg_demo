"""
多通道 sEMG 可视化指南
展示 4 种视角理解 16 通道数据
"""
import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from scipy import signal

mat = sio.loadmat('data/ninapro_db5/s1/S1_E1_A1.mat')
emg_raw = mat['emg'].astype(np.float64)
fs = 200

# 预处理
b, a = signal.butter(4, [20, 95], btype='bandpass', fs=fs)
emg_filtered = signal.filtfilt(b, a, emg_raw, axis=0)
b_notch, a_notch = signal.iirnotch(50, 15, fs=fs)
emg_clean = signal.filtfilt(b_notch, a_notch, emg_filtered, axis=0)

# 取一个手势片段（从标签 > 0 的位置开始）
labels = mat['stimulus'].squeeze()
gesture_start = np.where(labels > 0)[0][0]
window = slice(gesture_start, gesture_start + int(2.5 * fs))  # 2.5秒

t = np.arange(window.start, window.stop) / fs

# ============================================================
# 视角1: 多通道叠加图（垂直偏移） — 回答"多通道叠加有没有意义"
# ============================================================
fig1, ax1 = plt.subplots(figsize=(14, 8))

# 处理方法后的数据，每个通道加偏移
for ch in range(16):
    offset = (15 - ch) * 2.5  # 最上面=Ch1, 最下面=Ch16
    line = emg_clean[window, ch] + offset
    ax1.plot(t, line, linewidth=0.4, alpha=0.8)

ax1.set_yticks([(15 - ch) * 2.5 for ch in range(16)])
ax1.set_yticklabels([f'Ch{ch+1}' for ch in range(16)])
ax1.set_xlabel('Time (s)')
ax1.set_title(
    '视角1: 多通道叠加图（垂直偏移）\n'
    '可以看到：哪些通道同时活跃、波形是否同相/反相、基线上是否有干扰通道',
    fontsize=11, loc='left'
)
ax1.grid(True, alpha=0.2)

# 标注 Myo 分组
for myo_y, myo_label, color in [(17, 'Myo #1\n(近端)', '#2196F3'), (8, 'Myo #2\n(远端)', '#FF9800')]:
    ax1.annotate(myo_label, xy=(0.02, myo_y * 2.5), fontsize=9, color=color,
                 fontweight='bold', va='center',
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.1))

plt.tight_layout()
plt.savefig('viz1_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
print('viz1 已保存')

# ============================================================
# 视角2: 16 通道子图网格 + 原始信号对比 — 细看每个通道
# ============================================================
fig2, axes2 = plt.subplots(8, 2, figsize=(16, 10), sharex=True, sharey=False)

for ch in range(16):
    row = ch // 2       # 0-7 (同一 pod)
    col = ch % 2         # 0,1  (pod 内的两个电极)

    ax = axes2[row, col]

    # 浅灰=原始, 彩色=处理后
    ax.plot(t, emg_raw[window, ch], color='lightgray', linewidth=0.3, label='Raw')
    ax.plot(t, emg_clean[window, ch], color='steelblue', linewidth=0.5, label='Processed')

    pod_id = row + 1
    electrode = 'A' if col == 0 else 'B'
    ax.set_ylabel(f'Pod{pod_id}\nElec{electrode}', fontsize=7, rotation=0, labelpad=25)
    ax.yaxis.set_label_position('right')

    ax.set_ylim(-40, 40)
    ax.grid(True, alpha=0.2)
    if row == 0 and col == 0:
        ax.legend(fontsize=6, loc='upper right')

axes2[-1, 0].set_xlabel('Time (s)')
axes2[-1, 1].set_xlabel('Time (s)')
fig2.suptitle(
    '视角2: 逐通道原始 vs 处理后对比（8 pods × 2 electrodes）\n'
    '每行 = 一个 pod（前后方向不同），两列 = pod 上的两个电极',
    fontsize=12
)
plt.tight_layout()
plt.savefig('viz2_channel_grid.png', dpi=150, bbox_inches='tight')
plt.close()
print('viz2 已保存')

# ============================================================
# 视角3: 改进的热力图 — 回答"怎么看热力图"
# ============================================================
fig3, axes3 = plt.subplots(3, 1, figsize=(16, 10),
                           gridspec_kw={'height_ratios': [2, 1, 0.5]})

# 计算每通道的包络线（RMS envelope）
rms_envelope = np.zeros_like(emg_clean)
win_len = int(0.05 * fs)  # 50ms 窗口
for ch in range(16):
    squared = emg_clean[:, ch] ** 2
    rms_envelope[:, ch] = np.sqrt(
        np.convolve(squared, np.ones(win_len) / win_len, mode='same')
    )

# === 3A: 处理后信号热力图 ===
im1 = axes3[0].imshow(
    emg_clean[window, :16].T,
    aspect='auto', cmap='coolwarm', interpolation='bilinear',
    vmin=-30, vmax=30,
    extent=[t[0], t[-1], 16, 0]
)
axes3[0].set_title(
    '3A: 处理后 sEMG 热力图\n'
    'X轴=时间，Y轴=通道号，颜色=电压幅值。红=正，蓝=负，白=0。\n'
    '横向条纹=该通道持续活跃，纵向条带=多通道同时激活（手势动作）',
    fontsize=10, loc='left'
)
axes3[0].set_ylabel('Channel')
plt.colorbar(im1, ax=axes3[0], label='Amplitude', shrink=0.9)

# --- 标注 Myo 分界线 ---
axes3[0].axhline(y=8, color='black', linewidth=1.5, linestyle='-')
axes3[0].text(t[0] + 0.02, 4, 'Myo #2 (远端)', color='black', fontsize=9,
              va='center', fontweight='bold')
axes3[0].text(t[0] + 0.02, 12, 'Myo #1 (近端)', color='black', fontsize=9,
              va='center', fontweight='bold')

# === 3B: RMS 包络线叠加 ===
for ch in range(16):
    offset = (15 - ch) * 5
    axes3[1].fill_between(t, offset,
                          rms_envelope[window, ch] + offset,
                          alpha=0.6, linewidth=0.3, color=f'C{ch % 10}')

axes3[1].set_yticks([(15 - ch) * 5 for ch in range(16)])
axes3[1].set_yticklabels([f'Ch{ch+1}' for ch in range(16)])
axes3[1].set_title(
    '3B: RMS 包络线（叠加）\n'
    'Y偏移显示各通道的肌肉激活包络。鼓起的地方=肌肉在收缩',
    fontsize=10, loc='left'
)
axes3[1].set_xlabel('Time (s)')
axes3[1].grid(True, alpha=0.2)

# === 3C: 标签 ===
axes3[2].step(t, labels[window], linewidth=1, color='black', where='post')
axes3[2].set_title('3C: 手势标签 (stimulus)', fontsize=10, loc='left')
axes3[2].set_xlabel('Time (s)')
axes3[2].set_ylabel('Gesture ID')
axes3[2].set_ylim(-0.5, 13)
axes3[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('viz3_heatmap_explained.png', dpi=150, bbox_inches='tight')
plt.close()
print('viz3 已保存')

# ============================================================
# 视角4: 相关性矩阵 — 通道间关系
# ============================================================
fig4, ax4 = plt.subplots(figsize=(10, 9))

corr = np.corrcoef(emg_clean.T[:16])

im4 = ax4.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1,
                 interpolation='nearest', aspect='equal')
ax4.set_xticks(range(16))
ax4.set_yticks(range(16))
ax4.set_xticklabels([f'Ch{i+1}' for i in range(16)], fontsize=8)
ax4.set_yticklabels([f'Ch{i+1}' for i in range(16)], fontsize=8)

# 画 pod 分块线
for boundary in [7.5]:
    ax4.axhline(y=boundary, color='black', linewidth=2)
    ax4.axvline(x=boundary, color='black', linewidth=2)

# 标注
ax4.text(4, -1.5, 'Myo #2 (远端, Ch1-8)', ha='center', fontsize=10, fontweight='bold')
ax4.text(12, -1.5, 'Myo #1 (近端, Ch9-16)', ha='center', fontsize=10, fontweight='bold')

plt.colorbar(im4, ax=ax4, label='Pearson r', shrink=0.85)
ax4.set_title(
    '视角4: 16通道相关性矩阵\n'
    '每个格子 = 两个通道的皮尔逊相关系数。红色=正相关（同向激活），蓝色=负相关（拮抗）\n'
    '对角线上的方块=pod内部电极对（如Ch1↔Ch2），方块外的=不同pod之间的交叉',
    fontsize=11, loc='left'
)
plt.tight_layout()
plt.savefig('viz4_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print('viz4 已保存')

print("\n全部 4 张图已保存！")
print("  viz1_overlay.png       — 多通道叠加图")
print("  viz2_channel_grid.png  — 逐通道原始vs处理后")
print("  viz3_heatmap_explained.png — 热力图+包络+标签")
print("  viz4_correlation.png   — 通道间相关性矩阵")
