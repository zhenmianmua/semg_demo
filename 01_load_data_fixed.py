import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import StandardScaler

# 加载MAT文件
mat = sio.loadmat('data/ninapro_db5/s1/S1_E1_A1.mat')

print("EMG shape:", mat['emg'].shape)
print("Stimulus shape:", mat['stimulus'].shape)
print("Unique gestures:", np.unique(mat['stimulus']))

emg_raw = mat['emg'].astype(np.float64)
fs = 200  # Hz (NinaPro DB5 采样率)
nyquist = fs / 2

# ===== 预处理 =====
# 1. 带通滤波 (20-95Hz) — sEMG有效频段，高截止必须在Nyquist以下
#    BUG修复: 原本采用 [20, 500] 能更全面的收集到有效信号，但数据集（硬件）采样频率似乎只有200Hz
b, a = signal.butter(4, [20, 95], btype='bandpass', fs=fs)
emg_filtered = signal.filtfilt(b, a, emg_raw, axis=0)

# 2. 50Hz陷波器 — 去除工频干扰
#    Q调整为更实际的15
b_notch, a_notch = signal.iirnotch(50, 15, fs=fs)
emg_clean = signal.filtfilt(b_notch, a_notch, emg_filtered, axis=0)

# 3. 归一化（每通道独立）
# 数据集是肘/腕关节各有一个臂环，每个臂环8个电极，因此一共16个通道
scaler = StandardScaler()
emg_normalized = scaler.fit_transform(emg_clean)

# ===== 可视化对比 =====
channel_to_plot = 0       # 第1个通道
time_start = 0             # 起始秒
time_duration = 2          # 显示2秒
start_sample = int(time_start * fs)
end_sample = start_sample + int(time_duration * fs)

t = np.arange(start_sample, end_sample) / fs

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# --- 原始信号 ---
axes[0].plot(t, emg_raw[start_sample:end_sample, channel_to_plot],
             color='gray', alpha=0.8, linewidth=0.6)
axes[0].set_ylabel('Amplitude (raw)')
axes[0].set_title(f'Raw EMG — Channel {channel_to_plot+1}')
axes[0].grid(True, alpha=0.3)

# --- 带通滤波后 ---
axes[1].plot(t, emg_filtered[start_sample:end_sample, channel_to_plot],
             color='steelblue', linewidth=0.6)
axes[1].set_ylabel('Amplitude')
axes[1].set_title(f'After Bandpass Filter (20–95 Hz) — Channel {channel_to_plot+1}')
axes[1].grid(True, alpha=0.3)

# --- 陷波+归一化后 ---
axes[2].plot(t, emg_clean[start_sample:end_sample, channel_to_plot],
             color='darkgreen', linewidth=0.6)
axes[2].set_ylabel('Amplitude')
axes[2].set_title(f'After Notch Filter (50 Hz) — Channel {channel_to_plot+1}')
axes[2].grid(True, alpha=0.3)

# --- 归一化后 ---
axes[3].plot(t, emg_normalized[start_sample:end_sample, channel_to_plot],
             color='darkred', linewidth=0.6)
axes[3].set_ylabel('Amplitude (σ)')
axes[3].set_title(f'After StandardScaler — Channel {channel_to_plot+1}')
axes[3].set_xlabel('Time (s)')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n可视化已保存至 preprocessing_comparison.png")


# ===== 频域对比（功率谱密度）=====
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

for ax, data, label, color in [
    (axes2[0], emg_raw[:, channel_to_plot], 'Raw', 'gray'),
    (axes2[1], emg_clean[:, channel_to_plot], 'Processed', 'darkgreen'),
]:
    freqs, psd = signal.welch(data, fs=fs, nperseg=1024)
    ax.semilogy(freqs, psd, color=color, linewidth=0.8)
    ax.set_title(f'{label} — PSD Channel {channel_to_plot+1}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.axvline(50, color='red', linestyle='--', alpha=0.4, label='50 Hz')
    ax.axvspan(20, 95, alpha=0.05, color='blue', label='Passband 20–95 Hz')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('psd_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("频域对比图已保存至 psd_comparison.png")


# ===== 多通道热力图（处理前后对比）=====
fig3, axes3 = plt.subplots(2, 1, figsize=(14, 6))

for ax, data, title in [
    (axes3[0], emg_raw, 'Raw EMG — All Channels'),
    (axes3[1], emg_clean, 'Processed EMG — All Channels'),
]:
    n_show = min(2000, data.shape[0])
    im = ax.imshow(data[:n_show, :16].T, aspect='auto', cmap='RdBu_r',
                    interpolation='none', vmin=-3, vmax=3,
                    extent=[0, n_show/fs, 16, 0])
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel #')
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig('multichannel_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("多通道对比图已保存至 multichannel_comparison.png")
