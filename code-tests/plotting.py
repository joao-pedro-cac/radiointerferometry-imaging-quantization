import matplotlib.pyplot as plt
import numpy as np

x_axis = np.linspace(0, 9, 1000)
y_axis = np.cos(20 * np.pi * x_axis)

fft_y_axis = np.fft.fft(y_axis)
freqs = np.fft.fftfreq(len(fft_y_axis))

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

ax1.plot(x_axis, y_axis)
ax1.set_title("Cosine function")
ax1.set_xlabel("time")
ax1.set_ylabel("function output")

ax2.plot(freqs, np.abs(fft_y_axis))
ax2.set_title("Fourier transform (magnitude)")
ax2.set_xlabel("frequency")
ax2.set_ylabel("magnitude")

print(x_axis)
print()
print(y_axis)

print("\n")

print(freqs)
print()
print(fft_y_axis)
print()
print(np.abs(fft_y_axis))

plt.show()