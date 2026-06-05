"""
改进版多通道可视化 — 长时间窗口 + 手势名称标注 + 去重标签
"""
import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from scipy import signal

# ================================================================
# 手势标签映射（NinaPro DB5 Exercise 3 — 基础手指动作）
# ================================================================
GESTURE_NAMES = {
    0:  'Rest',
    1:  'Thumb flexion\n拇指屈曲',
    2:  'Thumb extension\n拇指伸展',
    3:  'Index flexion\n食指屈曲',
    4:  'Index extension\n食指伸展',
    5:  'Middle flexion\n中指屈曲',
    6:  'Middle extension\n中指伸展',
    7:  'Ring+Little flex\n无名+小指屈',
    8:  'Ring+Little ext\n无名+小指伸',
    9:  'Thumb adduction\n拇指内收',
    10: 'Thumb abduction\n拇指外展',
    11: 'Index-Mid abduct\n食中指外展',
    12: 'Index-Mid adduct\n食中指内收',
}

# 加载和预处理 =====================================================
mat = sio.loadmat('data/ninapro_db5/s1/S1_E1_A1.mat')
emg_raw = mat['emg'].astype(np.float64)
labels = mat['stimulus'].squeeze().astype(int)
repetitions = mat['repetition'].squeeze().astype(int)
fs = 200

b, a = signal.butter(4, [20, 95], btype='bandpass', fs=fs)
emg_filtered = signal.filtfilt(b, a, emg_raw, axis=0)
b_notch, a_notch = signal.iirnotch(50, 15, fs=fs)
emg_clean = signal.filtfilt(b_notch, a_notch, emg_filtered, axis=0)

# ================================================================
# 大窗口：取 300 秒展示
# ================================================================
start_sample = int(2 * fs)
window_duration = 600
window = slice(start_sample, start_sample + int(window_duration * fs))

t = np.arange(window.start, window.stop) / fs
labels_in_window = labels[window]
reps_in_window = repetitions[window]

# 找到"手势组"边界 — 手势 ID 变化时（rest → gesture 不算新组，gesture → rest 才算）
# 策略：只在手势 ID 从 A 变为 B (B≠0, A≠B) 时标记 — 即非零标签变化
label_changes = np.where(np.diff(labels_in_window) != 0)[0]

# 提取"第一次出现每个 (手势ID, repetition) 组合"用于去重标注
# 即对于同手势同重复次数的连续段，只标一次
gesture_groups = []  # (start_idx, end_idx, gesture_id, rep)
in_seg = False
seg_start = 0
for i in range(len(labels_in_window)):
    if labels_in_window[i] > 0 and not in_seg:
        seg_start = i
        in_seg = True
    elif labels_in_window[i] != labels_in_window[max(0, seg_start)] and in_seg:
        gid = labels_in_window[max(0, seg_start)]
        if gid > 0:
            gesture_groups.append((seg_start, i, gid, reps_in_window[seg_start]))
        seg_start = i
        in_seg = (labels_in_window[i] > 0)
if in_seg:
    gid = labels_in_window[seg_start]
    if gid > 0:
        gesture_groups.append((seg_start, len(labels_in_window), gid, reps_in_window[seg_start]))

print(f'窗口内手势段数: {len(gesture_groups)}')

# ================================================================
# 绘图
# ================================================================
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1.5, 1.5], hspace=0.06)

# ================================================================
# A: 多通道 RMS 包络叠加
# ================================================================
ax1 = fig.add_subplot(gs[0])

rms_win = int(0.03 * fs)
for ch in range(16):
    squared = emg_clean[:, ch] ** 2
    rms = np.sqrt(np.convolve(squared, np.ones(rms_win)/rms_win, mode='same'))
    offset = (15 - ch) * 2.5
    ax1.fill_between(t, offset, rms[window] + offset,
                     alpha=0.5, linewidth=0.2, color=f'C{ch%10}', edgecolor='none')

ax1.axhline(y=20, color='black', linewidth=1, linestyle='--', alpha=0.4)
ax1.text(t[0] + 0.5, 21.5, 'Myo #2 (远端 Ch1-8)', fontsize=8, va='bottom', alpha=0.7)
ax1.text(t[0] + 0.5, 18.5, 'Myo #1 (近端 Ch9-16)', fontsize=8, va='top', alpha=0.7)
ax1.set_yticks([(15-ch)*2.5 for ch in range(16)])
ax1.set_yticklabels([f'Ch{ch+1}' for ch in range(16)], fontsize=7)
ax1.set_xlim(t[0], t[-1])
ax1.grid(True, alpha=0.12)
ax1.tick_params(labelbottom=False)

# ================================================================
# B: 手势标签（去重标注）
# ================================================================
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# 先画底层：所有手势事件的颜色块（rest 不画）
unique_gids = sorted(g for g in np.unique(labels_in_window) if g > 0)
gid_colors = {gid: plt.cm.tab20(i % 20) for i, gid in enumerate(unique_gids)}

for seg_start, seg_end, gid, rep in gesture_groups:
    ax2.axvspan(t[seg_start], t[min(seg_end, len(t))-1],
                alpha=0.35, color=gid_colors[gid], linewidth=0)

# 去重：同一手势的连续 block 只标注一次（第一个）
already_labeled = set()
y_base = 1
for seg_start, seg_end, gid, rep in gesture_groups:
    key = gid
    seg_t = t[seg_start:min(seg_end, len(t))]
    mid = (seg_start + seg_end) // 2
    if mid >= len(t):
        mid = len(t) - 1

    if key not in already_labeled:
        # 画竖线标记 + 文字
        ax2.axvline(x=t[mid], color=gid_colors[gid], linewidth=1.2, alpha=0.9, linestyle='-')
        ax2.annotate(
            GESTURE_NAMES.get(gid, f'G{gid}'),
            xy=(t[mid], 0.5), fontsize=7.5, ha='center', va='center',
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=gid_colors[gid], alpha=0.3,
                      edgecolor=gid_colors[gid], linewidth=1)
        )
        already_labeled.add(key)
    else:
        # 后续重复：只画细灰线
        ax2.axvline(x=t[mid], color='gray', linewidth=0.4, alpha=0.4, linestyle=':')

# 标注 "×6 repetitions" 提示 — 在每个手势的第一个 block 底部
# (已经在上面去重逻辑中通过 already_labeled 处理)

ax2.set_ylabel('Gesture', fontsize=10)
ax2.set_ylim(0, 1)
ax2.set_xlim(t[0], t[-1])
ax2.grid(True, alpha=0.12, axis='y')
ax2.tick_params(labelleft=False, labelbottom=False)

# ================================================================
# C: 代表性的单通道原始波形（展示滤波效果）
# ================================================================
ax3 = fig.add_subplot(gs[2], sharex=ax1)

rep_channels = [2, 5, 10, 14]
offsets = [25, 10, -5, -20]
for ch, off in zip(rep_channels, offsets):
    ax3.plot(t, emg_clean[window, ch] + off,
             linewidth=0.25, alpha=0.8, label=f'Ch{ch+1}')
    # 灰底覆盖原始信号
    ax3.plot(t, emg_raw[window, ch] + off,
             linewidth=0.15, alpha=0.2, color='gray')

ax3.set_yticks(offsets)
ax3.set_yticklabels([f'Ch{ch+1}' for ch in rep_channels])
ax3.set_xlim(t[0], t[-1])
ax3.set_xlabel('Time (s)', fontsize=11)
ax3.legend(fontsize=7, loc='upper right', ncol=4)
ax3.grid(True, alpha=0.12)

plt.tight_layout()
plt.savefig('improved_multichannel_viz.png', dpi=150, bbox_inches='tight')
plt.show()
print('已保存 improved_multichannel_viz.png')

# ================================================================
# 补充：打印窗口内的 gesture 统计，帮助理解数据结构
# ================================================================
print('\n窗口内手势统计:')
from collections import Counter
gesture_counts = Counter()
for seg_start, seg_end, gid, rep in gesture_groups:
    gesture_counts[gid] += 1
for gid in sorted(gesture_counts):
    name = GESTURE_NAMES.get(gid, f'G{gid}').replace('\n', ' ')
    print(f'  Gesture {gid:2d} ({name:25s}): {gesture_counts[gid]} 段')
print(f'  共计 {len(gesture_groups)} 段手势，{len(gesture_counts)} 种手势')
