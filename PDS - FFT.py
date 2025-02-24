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
        raise ValueError("N deve ser uma potência de 2")
    
    even = fft_radix2(x[0::2])
    odd = fft_radix2(x[1::2])
    
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    T = np.array(T)
    even = np.array(even)
    
    return np.concatenate([even + T, even - T])

def next_power_of_2(n):
    """Encontra a próxima potência de 2 maior ou igual a n."""
    return 1 if n == 0 else 2 ** (n - 1).bit_length()

# Parâmetros do sinal
fs = 1000  # Taxa de amostragem
T = 1.0    # Duração do sinal
N = 360    # Número de amostras (ajustado para ser uma potência de 2)
N = next_power_of_2(N)  # Ajusta N para ser uma potência de 2
t = np.linspace(0, T, N, endpoint=False)

# Sinal: senoide + ruído
freq1, freq2 = 50, 120  # Frequências do sinal
sinal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t) + 0.2 * np.random.randn(N)

# Aplicação das transformadas
X_dft = dft(sinal)
X_fft2 = fft_radix2(sinal)  # Agora com tamanho adequado
X_fft_np = np.fft.fft(sinal)  # FFT de referência usando numpy

freqs = np.fft.fftfreq(N, 1/fs)

# Plotagem
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t, sinal)
plt.title('Sinal no Domínio do Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(freqs[:N//2], np.abs(X_dft[:N//2]))
plt.title('DFT')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(freqs[:N//2], np.abs(X_fft2[:N//2]))
plt.title('Radix-2 FFT')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(freqs[:N//2], np.abs(X_fft_np[:N//2]), label='FFT (numpy)', linestyle='dashed')
plt.title('FFT (numpy)')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
