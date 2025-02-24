import numpy as np
import matplotlib.pyplot as plt

def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def fft_radix2(x):
    N = len(x)
    if N <= 1:
        return x
    if N % 2 != 0:
        raise ValueError("N deve ser potência de 2")
    
    even = fft_radix2(x[0::2])
    odd = fft_radix2(x[1::2])
    
    T = np.array([np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)])
    even = np.array(even)
    
    return np.concatenate((even[:N // 2] + T, even[:N // 2] - T))

def fft_radix3(x):
    N = len(x)
    if N <= 1:
        return x
    if N % 3 != 0:
        raise ValueError("N deve ser potência de 3")
    
    x0 = fft_radix3(x[0::3])
    x1 = fft_radix3(x[1::3])
    x2 = fft_radix3(x[2::3])
    
    W_N = np.exp(-2j * np.pi / N)
    W = np.array([W_N**k for k in range(N)])
    
    X = np.zeros(N, dtype=complex)
    for k in range(N // 3):
        X[k] = x0[k] + W[k] * x1[k] + W[2 * k] * x2[k]
        X[k + N // 3] = x0[k] + W[k + N // 3] * x1[k] + W[2 * (k + N // 3)] * x2[k]
        X[k + 2 * N // 3] = x0[k] + W[k + 2 * N // 3] * x1[k] + W[2 * (k + 2 * N // 3)] * x2[k]
    
    return X

# Parâmetros do sinal
fs = 1000  # Taxa de amostragem
T = 1.0    # Duração do sinal
N = 512    # Número de amostras (potência de 2 para FFT Radix-2)
t = np.linspace(0, T, N, endpoint=False)

# Sinal: senoide + ruído
freq1, freq2 = 50, 120  # Frequências do sinal
sinal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t) + 0.2 * np.random.randn(N)

# Aplicação das transformadas
X_dft = dft(sinal)
X_fft2 = fft_radix2(sinal)
X_fft3 = np.fft.fft(sinal)  # FFT de referência

# Frequências associadas
freqs = np.fft.fftfreq(N, 1/fs)

# Plotagem
plt.figure(figsize=(12, 12))

plt.subplot(6, 1, 1)
plt.plot(t, sinal)
plt.title('Sinal no Domínio do Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(6, 1, 2)
plt.plot(freqs[:N//2], np.abs(X_dft[:N//2]))
plt.title('DFT')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.grid()

plt.subplot(6, 1, 3)
plt.plot(freqs[:N//2], np.abs(X_fft2[:N//2]))
plt.title('Radix-2 FFT')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.grid()

plt.subplot(6, 1, 4)
plt.plot(freqs[:N//2], np.abs(X_fft3[:N//2]))
plt.title('Radix-3 FFT')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.grid()

plt.subplot(6, 1, 5)
plt.plot(freqs[:N//2], np.abs(np.fft.fft(sinal)[:N//2]))
plt.title('FFT (numpy)')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.grid()

plt.subplot(6, 1, 6)
plt.plot(freqs[:N//2], np.abs(X_dft[:N//2]), label='DFT')
plt.plot(freqs[:N//2], np.abs(X_fft2[:N//2]), label='Radix-2 FFT')
plt.plot(freqs[:N//2], np.abs(X_fft3[:N//2]), label='Radix-3 FFT')
plt.plot(freqs[:N//2], np.abs(np.fft.fft(sinal)[:N//2]), label='FFT (numpy)', linestyle='dashed')
plt.title('Comparação de Todas as FFTs')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
