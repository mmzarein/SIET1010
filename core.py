import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from kivy.clock import Clock


class SignalProcessor:
    def __init__(self, sample_rate: int = 96000, n_samples: int = 2**17) -> None:
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.t = np.arange(self.n_samples) / self.sample_rate
        self.frequencies = self.sample_rate * np.arange(self.n_samples // 2 + 1) / self.n_samples

    def compute_fft(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''Compute the Fast Fourier Transform (FFT) of the signal.'''
        self.signal = signal
        self.fft_result = np.fft.fft(self.signal)
        magnitude = np.abs(self.fft_result / self.n_samples)
        self.single_side_magnitude = magnitude[:self.n_samples // 2 + 1]
        self.single_side_magnitude[1:-1] *= 2
        return self.fft_result, self.single_side_magnitude

    def find_peaks(self, distance: int = 1000) -> np.ndarray:
        '''Find the peaks in the single-sided magnitude spectrum.'''
        self.peaks, _ = find_peaks(self.single_side_magnitude, distance=distance)
        return self.peaks

    def plot_wave(self) -> plt.Figure:
        '''Plot the time-domain representation of the signal.'''
        wave_fig, wave_ax = plt.subplots(layout='constrained')
        wave_ax.plot(self.t[:200], self.signal[:200])
        wave_ax.grid()
        wave_ax.set_xlabel('Time (ms)', fontsize=13, labelpad=10)
        wave_ax.set_ylabel('Amplitude (v)', fontsize=13, labelpad=10)
        return wave_fig

    def plot_fft(self) -> plt.Figure:
        '''Plot the frequency-domain representation (FFT) and mark the detected peaks.'''
        fft_fig, fft_ax = plt.subplots(layout='constrained')
        fft_ax.plot(self.frequencies, self.single_side_magnitude)
        fft_ax.plot(self.frequencies[self.peaks], self.single_side_magnitude[self.peaks], 's')
        fft_ax.grid()
        fft_ax.set_xlabel('Frequency (kHz)', fontsize=13, labelpad=10)
        fft_ax.set_ylabel('Amplitude (v)', fontsize=13, labelpad=10)
        return fft_fig

    def generate_test_signal(self) -> np.ndarray:
        '''Generate a test signal (used internally).'''
        return (
            np.sin(2 * np.pi * 8000 * self.t) +
            0.5 * np.sin(2 * np.pi * 16000 * self.t) +
            0.25 * np.sin(2 * np.pi * 25000 * self.t)
        )

    def run(self, stop_event, callback):
        print('Starting signal processing...')

        # Step 1: Generate the signal
        print('Generating signal...')
        time.sleep(1)  # Simulate a 2-second task
        signal = self.generate_test_signal()
        if stop_event.is_set():
            print('Stop event detected after generating signal. Exiting...')
            return False

        if False: # TMP: Just to test the behavior in Unexpected situations!
            print('Holy shit, unexpected Error!!!! Fuck my life!!!')
            Clock.schedule_once(lambda dt: callback(False, [], []))
            return False

        # Step 2: Compute FFT
        print('Computing FFT...')
        time.sleep(1)  # Simulate a 2-second task
        fft_result, magnitude = self.compute_fft(signal)
        if stop_event.is_set():
            print('Stop event detected after computing FFT. Exiting...')
            return False

        # Step 3: Find peaks
        print('Finding peaks...')
        time.sleep(1)  # Simulate a 2-second task
        peaks = self.find_peaks()
        if stop_event.is_set():
            print('Stop event detected after finding peaks. Exiting...')
            return False

        # Step 4: Update UI or finalize
        print('Updating UI...')
        # time.sleep(1)  # Simulate a 2-second task
        if stop_event.is_set():
            print('Stop event detected before updating UI. Exiting...')
            return False

        Clock.schedule_once(lambda dt: callback(True, signal, peaks))
        print('Signal processing completed.')
        return True


class Calculator:

    @staticmethod
    def _calc_T1(L, t, P):
        # Correction factor for bar (flexural and poisson_ratio cases)
        r_lt = t / L
        if L / t >= 20:
            return 1 + 6.585 * r_lt**2
        else:
            num = 8.34 * (1 + 0.2023 * P + 2.173 * P**2) * r_lt**4
            den = 1 + 0.6338 * (1 + 0.1408 * P + 1.536 * P**2) * r_lt**2
            return 1 + 6.585 * (1 + 0.0752 * P + 0.8109 * P**2) * r_lt**2 - (num / den)

    @staticmethod
    def _calc_BA(b, t):
        # Helper to compute factors used in torsional and poisson_ratio cases for bar
        r_bt = b / t
        r_tb = t / b
        B = (r_bt + r_tb) / (4 * r_tb - 2.52 * r_tb**2 + 0.21 * r_tb**6)
        A = (0.5062 - 0.8776 * r_bt + 0.3504 * r_bt**2 - 0.0078 * r_bt**3) / (12.03 * r_bt + 9.892 * r_bt**2)
        return B, A

    @staticmethod
    def _calc_T1r(L, D, P):
        # Correction factor for rod (flexural and poisson_ratio cases)
        r_DL = D / L
        if L / D >= 20:
            return 1 + 4.939 * r_DL**2
        else:
            term1 = 4.939 * (1 + 0.0752 * P + 0.8109 * P**2) * r_DL**2
            term2 = 0.4883 * r_DL**4
            term3 = (4.691 * (1 + 0.2023 * P + 2.1730 * P**2) * r_DL**4) / (1 + 4.754 * (1 + 0.1408 * P + 1.536 * P**2) * r_DL**2)
            return 1 + term1 - term2 - term3

    @staticmethod
    def _bilinear_interpolation(in1, in2, X, Y, tab):
        # Optimized bilinear interpolation using NumPy searchsorted
        i = np.clip(np.searchsorted(X, in1) - 1, 0, len(X) - 2)
        j = np.clip(np.searchsorted(Y, in2) - 1, 0, len(Y) - 2)
        x1, x2 = X[i], X[i + 1]
        y1, y2 = Y[j], Y[j + 1]
        Q11 = tab[j, i]
        Q21 = tab[j, i + 1]
        Q12 = tab[j + 1, i]
        Q22 = tab[j + 1, i + 1]
        return ((Q11 * (x2 - in1) * (y2 - in2) +
                 Q21 * (in1 - x1) * (y2 - in2) +
                 Q12 * (x2 - in1) * (in2 - y1) +
                 Q22 * (in1 - x1) * (in2 - y1))
                / ((x2 - x1) * (y2 - y1)))

    @staticmethod
    def bar(**kwargs):
        L = float(kwargs['length'])
        b = float(kwargs['width'])
        t = float(kwargs['thickness'])
        m = float(kwargs['mass'])
        measurement = kwargs['measurement_type']
        if measurement == 'flexural':
            Ff = float(kwargs.get('flexural_frequency', 0)) * 1000
        if measurement == 'torsional':
            Ft = float(kwargs.get('torsional_frequency', 0)) * 1000
        P = float(kwargs['initial_poisson_ratio'])

        if measurement == 'flexural':
            T1 = Calculator._calc_T1(L, t, P)
            E = 0.9465 * (m * Ff**2 / b) * (L / t)**3 * T1 * 1e-9
            return {
                'dynamic_young_modulus_output': f'{E:.4f}',
                'dynamic_shear_modulus_output': '-',
                'poisson_ratio_output': '-'
            }
        elif measurement == 'torsional':
            B, A = Calculator._calc_BA(b, t)
            G = ((4 * L * m * Ft**2) / (b * t)) * (B / (1 + A)) * 1e-9
            return {
                'dynamic_young_modulus_output': '-',
                'dynamic_shear_modulus_output': f'{G:.4f}',
                'poisson_ratio_output': '-'
            }
        elif measurement == 'poisson':
            B, A = Calculator._calc_BA(b, t)
            G = ((4 * L * m * Ft**2) / (b * t)) * (B / (1 + A)) * 1e-9
            if L / t >= 20:
                T1 = 1 + 6.585 * (t / L)**2
                E = 0.9465 * (m * Ff**2 / b) * (L / t)**3 * T1 * 1e-9
                if 2 * G <= E <= 4 * G:
                    P_new = (E / (2 * G)) - 1
                    return {
                        'dynamic_young_modulus_output': f'{E:.4f}',
                        'dynamic_shear_modulus_output': f'{G:.4f}',
                        'poisson_ratio_output': f'{P_new:.4f}'
                    }
                else:
                    return {
                        'dynamic_young_modulus_output': 'Invalid',
                        'dynamic_shear_modulus_output': 'Invalid',
                        'poisson_ratio_output': 'Invalid'
                    }
            else:
                T1 = Calculator._calc_T1(L, t, P)
                E = 0.9465 * (m * Ff**2 / b) * (L / t)**3 * T1 * 1e-9
                if 2 * G <= E <= 4 * G:
                    P_new = (E / (2 * G)) - 1
                    while abs(P_new - P) / P_new > 0.02:
                        P = P_new
                        T1 = Calculator._calc_T1(L, t, P)
                        E = 0.9465 * (m * Ff**2 / b) * (L / t)**3 * T1 * 1e-9
                        P_new = (E / (2 * G)) - 1
                    return {
                        'dynamic_young_modulus_output': f'{E:.4f}',
                        'dynamic_shear_modulus_output': f'{G:.4f}',
                        'poisson_ratio_output': f'{P_new:.4f}'
                    }
                else:
                    return {
                        'dynamic_young_modulus_output': 'Invalid',
                        'dynamic_shear_modulus_output': 'Invalid',
                        'poisson_ratio_output': 'Invalid'
                    }

    @staticmethod
    def rod(**args):
        # Extract inputs
        L = float(args['length'])
        D = float(args['diameter'])
        m = float(args['mass'])
        measurement = args['measurement_type']
        if measurement == 'flexural':
            Ff = float(args.get('flexural_frequency', 0)) * 1000
        if measurement == 'torsional':
            Ft = float(args.get('torsional_frequency', 0)) * 1000
        P = float(args['initial_poisson_ratio'])

        if measurement == 'flexural':
            T1r = Calculator._calc_T1r(L, D, P)
            E = 1.6067 * (L**3 / D**4) * (m * Ff**2) * T1r * 1e-9
            return {
                'dynamic_young_modulus_output': f'{E:.4f}',
                'dynamic_shear_modulus_output': '-',
                'poisson_ratio_output': '-'
            }
        elif measurement == 'torsional':
            G = 16 * m * Ft**2 * (L / (np.pi * D**2)) * 1e-9
            return {
                'dynamic_young_modulus_output': '-',
                'dynamic_shear_modulus_output': f'{G:.4f}',
                'poisson_ratio_output': '-'
            }
        elif measurement == 'poisson_ratio':
            G = 16 * m * Ft**2 * (L / (np.pi * D**2)) * 1e-9
            if L / D >= 20:
                T1r = 1 + 4.939 * (D / L)**2
                E = 1.6067 * (L**3 / D**4) * (m * Ff**2) * T1r * 1e-9
                if 2 * G <= E <= 4 * G:
                    P_new = (E / (2 * G)) - 1
                    return {
                        'dynamic_young_modulus_output': f'{E:.4f}',
                        'dynamic_shear_modulus_output': f'{G:.4f}',
                        'poisson_ratio_output': f'{P_new:.4f}'
                    }
                else:
                    return {
                        'dynamic_young_modulus_output': 'Invalid',
                        'dynamic_shear_modulus_output': 'Invalid',
                        'poisson_ratio_output': 'Invalid'
                    }
            else:
                T1r = Calculator._calc_T1r(L, D, P)
                E = 1.6067 * (L**3 / D**4) * (m * Ff**2) * T1r * 1e-9
                if 2 * G <= E <= 4 * G:
                    P_new = (E / (2 * G)) - 1
                    while abs(P_new - P) / P_new > 0.02:
                        P = P_new
                        T1r = Calculator._calc_T1r(L, D, P)
                        E = 1.6067 * (L**3 / D**4) * (m * Ff**2) * T1r * 1e-9
                        P_new = (E / (2 * G)) - 1
                    return {
                        'dynamic_young_modulus_output': f'{E:.4f}',
                        'dynamic_shear_modulus_output': f'{G:.4f}',
                        'poisson_ratio_output': f'{P_new:.4f}'
                    }
                else:
                    return {
                        'dynamic_young_modulus_output': 'Invalid',
                        'dynamic_shear_modulus_output': 'Invalid',
                        'poisson_ratio_output': 'Invalid'
                    }
        else:
            return {
                'dynamic_young_modulus_output': 'Invalid',
                'dynamic_shear_modulus_output': 'Invalid',
                'poisson_ratio_output': 'Invalid'
            }

    @staticmethod
    def disc(**args):
        # Extract inputs
        D = float(args['diameter'])
        t = float(args['thickness'])
        m = float(args['mass'])
        F1 = float(args['first_frequency']) * 1000
        F2 = float(args['second_frequency']) * 1000

        # Define grid arrays (small arrays)
        FF = np.arange(1.35, 1.9 + 0.025, 0.025)
        tr_1 = np.arange(0.00, 0.50 + 0.05, 0.05)
        Poisson_1 = np.arange(0, 0.5 + 0.05, 0.05)
        tr_2 = np.arange(0.10, 0.20 + 0.01, 0.01)
        Poisson_2 = np.arange(0.14, 0.34 + 0.02, 0.02)

        PP = np.array([
            [0.015, 0.043, 0.070, 0.094, 0.118, 0.141, 0.163, 0.184, 0.205, 0.226, 0.247, 0.265, 0.282, 0.297, 0.312, 0.329, 0.346, 0.362, 0.378, 0.394, 0.409, 0.424, 0.438],
            [0.018, 0.044, 0.070, 0.094, 0.118, 0.141, 0.164, 0.185, 0.206, 0.226, 0.247, 0.265, 0.283, 0.298, 0.314, 0.331, 0.347, 0.363, 0.378, 0.394, 0.409, 0.424, 0.438],
            [0.020, 0.045, 0.070, 0.094, 0.118, 0.141, 0.164, 0.185, 0.206, 0.227, 0.247, 0.265, 0.283, 0.300, 0.316, 0.332, 0.348, 0.363, 0.378, 0.394, 0.409, 0.424, 0.438],
            [0.023, 0.049, 0.075, 0.100, 0.124, 0.148, 0.171, 0.192, 0.212, 0.233, 0.254, 0.271, 0.289, 0.306, 0.322, 0.338, 0.354, 0.368, 0.383, 0.398, 0.413, 0.427, 0.442],
            [0.025, 0.053, 0.080, 0.105, 0.130, 0.154, 0.178, 0.198, 0.218, 0.239, 0.260, 0.278, 0.295, 0.312, 0.328, 0.344, 0.359, 0.374, 0.388, 0.403, 0.417, 0.431, 0.445],
            [0.033, 0.060, 0.088, 0.114, 0.139, 0.162, 0.186, 0.206, 0.227, 0.247, 0.268, 0.286, 0.304, 0.320, 0.336, 0.351, 0.366, 0.380, 0.395, 0.409, 0.423, 0.437, 0.451],
            [0.040, 0.068, 0.096, 0.122, 0.148, 0.171, 0.193, 0.214, 0.235, 0.255, 0.275, 0.294, 0.312, 0.328, 0.344, 0.358, 0.372, 0.387, 0.402, 0.415, 0.428, 0.442, 0.456],
            [0.051, 0.078, 0.105, 0.130, 0.155, 0.179, 0.203, 0.224, 0.245, 0.264, 0.284, 0.302, 0.320, 0.336, 0.352, 0.367, 0.382, 0.398, 0.414, 0.428, 0.442, 0.456, 0.471],
            [0.062, 0.088, 0.113, 0.138, 0.162, 0.187, 0.212, 0.234, 0.255, 0.274, 0.292, 0.310, 0.328, 0.344, 0.360, 0.376, 0.392, 0.409, 0.425, 0.440, 0.455, 0.470, 0.485],
            [0.070, 0.096, 0.123, 0.148, 0.173, 0.197, 0.221, 0.242, 0.263, 0.281, 0.300, 0.318, 0.337, 0.354, 0.370, 0.387, 0.403, 0.420, 0.437, 0.452, 0.468, 0.485, 0.500],
            [0.078, 0.105, 0.132, 0.158, 0.183, 0.206, 0.229, 0.250, 0.270, 0.289, 0.307, 0.327, 0.346, 0.363, 0.380, 0.397, 0.414, 0.431, 0.448, 0.464, 0.480, 0.500, 0.500],
        ])

        K1_1 = np.array([
            [6.170, 6.144, 6.090, 6.012, 5.914, 5.800, 5.674, 5.540, 5.399, 5.255, 5.110],
            [6.076, 6.026, 5.968, 5.899, 5.816, 5.717, 5.603, 5.473, 5.331, 5.178, 5.019],
            [5.962, 5.905, 5.847, 5.782, 5.705, 5.613, 5.504, 5.377, 5.234, 5.079, 4.915],
            [5.830, 5.776, 5.720, 5.657, 5.581, 5.490, 5.382, 5.256, 5.115, 4.962, 4.800],
            [5.681, 5.639, 5.587, 5.524, 5.446, 5.351, 5.240, 5.114, 4.975, 4.826, 4.673],
            [5.517, 5.491, 5.445, 5.380, 5.297, 5.197, 5.083, 4.957, 4.822, 4.681, 4.537],
            [5.340, 5.331, 5.290, 5.223, 5.135, 5.030, 4.913, 4.787, 4.656, 4.523, 4.390],
            [5.192, 5.156, 5.120, 5.052, 4.961, 4.853, 4.734, 4.610, 4.483, 4.358, 4.234],
            [4.973, 4.964, 4.931, 4.865, 4.775, 4.668, 4.551, 4.429, 4.306, 4.186, 4.070],
            [4.781, 4.756, 4.723, 4.661, 4.576, 4.476, 4.365, 4.249, 4.131, 4.013, 3.899],
            [4.540, 4.525, 4.490, 4.436, 4.365, 4.280, 4.182, 4.075, 3.960, 3.841, 3.720]
        ])

        K1_2 = np.array([
            [5.746, 5.739, 5.722, 5.710, 5.696, 5.683, 5.670, 5.654, 5.642, 5.629, 5.608],
            [5.694, 5.687, 5.670, 5.664, 5.645, 5.632, 5.619, 5.602, 5.590, 5.576, 5.556],
            [5.641, 5.634, 5.617, 5.606, 5.592, 5.579, 5.566, 5.549, 5.537, 5.523, 5.502],
            [5.587, 5.576, 5.563, 5.551, 5.538, 5.524, 5.510, 5.495, 5.479, 5.463, 5.446],
            [5.531, 5.524, 5.507, 5.495, 5.481, 5.468, 5.455, 5.439, 5.427, 5.411, 5.388],
            [5.474, 5.467, 5.450, 5.438, 5.424, 5.410, 5.396, 5.379, 5.366, 5.351, 5.328],
            [5.415, 5.408, 5.391, 5.379, 5.364, 5.350, 5.336, 5.318, 5.304, 5.289, 5.266],
            [5.354, 5.347, 5.330, 5.317, 5.301, 5.287, 5.273, 5.255, 5.241, 5.225, 5.201],
            [5.290, 5.279, 5.266, 5.253, 5.238, 5.223, 5.207, 5.190, 5.173, 5.154, 5.135],
            [5.224, 5.217, 5.200, 5.187, 5.172, 5.157, 5.142, 5.123, 5.108, 5.091, 5.067],
            [5.156, 5.148, 5.131, 5.118, 5.103, 5.088, 5.073, 5.053, 5.037, 5.020, 4.997]
        ])

        K2_1 = np.array([
            [8.240, 8.226, 8.151, 8.027, 7.863, 7.670, 7.455, 7.227, 6.991, 6.754, 6.520],
            [8.378, 8.339, 8.252, 8.124, 7.963, 7.777, 7.570, 7.350, 7.120, 6.885, 6.649],
            [8.511, 8.459, 8.364, 8.233, 8.071, 7.885, 7.679, 7.459, 7.228, 6.991, 6.751],
            [8.640, 8.584, 8.485, 8.349, 8.182, 7.990, 7.779, 7.553, 7.316, 7.074, 6.830],
            [8.764, 8.712, 8.611, 8.469, 8.294, 8.092, 7.871, 7.635, 7.390, 7.141, 6.889],
            [8.884, 8.840, 8.738, 8.589, 8.403, 8.189, 7.954, 7.706, 7.450, 7.191, 6.931],
            [9.000, 8.962, 8.860, 8.705, 8.508, 8.280, 8.030, 7.767, 7.497, 7.226, 6.960],
            [9.111, 9.081, 8.977, 8.814, 8.605, 8.363, 8.098, 7.819, 7.535, 7.253, 6.979],
            [9.219, 9.193, 9.085, 8.913, 8.692, 8.436, 8.157, 7.865, 7.569, 7.276, 6.991],
            [9.321, 9.292, 9.178, 8.997, 8.766, 8.499, 8.208, 7.905, 7.598, 7.295, 7.001],
            [9.420, 9.376, 9.252, 9.063, 8.824, 8.550, 8.252, 7.940, 7.625, 7.313, 7.010]
        ])

        K2_2 = np.array([
            [8.460, 8.443, 8.411, 8.385, 8.355, 8.326, 8.297, 8.262, 8.234, 8.202, 8.160],
            [8.510, 8.493, 8.460, 8.433, 8.403, 8.373, 8.343, 8.308, 8.279, 8.248, 8.205],
            [8.560, 8.542, 8.509, 8.482, 8.451, 8.421, 8.391, 8.356, 8.327, 8.294, 8.249],
            [8.611, 8.586, 8.559, 8.530, 8.500, 8.469, 8.437, 8.403, 8.368, 8.331, 8.294],
            [8.662, 8.646, 8.613, 8.582, 8.548, 8.517, 8.487, 8.454, 8.425, 8.390, 8.338],
            [8.712, 8.694, 8.660, 8.630, 8.597, 8.565, 8.534, 8.498, 8.467, 8.432, 8.382],
            [8.762, 8.743, 8.708, 8.678, 8.645, 8.612, 8.580, 8.542, 8.510, 8.474, 8.425],
            [8.811, 8.791, 8.755, 8.726, 8.692, 8.659, 8.625, 8.585, 8.551, 8.515, 8.467],
            [8.860, 8.833, 8.804, 8.772, 8.739, 8.705, 8.668, 8.630, 8.591, 8.550, 8.508],
            [8.907, 8.885, 8.848, 8.818, 8.784, 8.750, 8.716, 8.675, 8.640, 8.601, 8.548],
            [8.954, 8.932, 8.894, 8.863, 8.827, 8.793, 8.758, 8.717, 8.681, 8.641, 8.586]
        ])

        ratio_F = F2 / F1
        ratio_tt = 2 * t / D

        # Validate input ranges
        if 1.35 <= ratio_F <= 1.9 and 0.00 <= ratio_tt <= 0.5:
            P_val = Calculator._bilinear_interpolation(ratio_F, ratio_tt, FF, tr_1, PP)
            # Choose K tables based on P_val and ratio_tt
            if (0.14 <= P_val <= 0.34) and (0.1 <= ratio_tt <= 0.2):
                K1 = Calculator._bilinear_interpolation(ratio_tt, P_val, tr_2, Poisson_2, K1_2)
                K2 = Calculator._bilinear_interpolation(ratio_tt, P_val, tr_2, Poisson_2, K2_2)
            else:
                K1 = Calculator._bilinear_interpolation(ratio_tt, P_val, tr_1, Poisson_1, K1_1)
                K2 = Calculator._bilinear_interpolation(ratio_tt, P_val, tr_1, Poisson_1, K2_1)
            E1 = (37.6991 * F1**2 * D**2 * m * (1 - P_val**2)) / (K1**2 * t**3) * 1e-9
            E2 = (37.6991 * F2**2 * D**2 * m * (1 - P_val**2)) / (K2**2 * t**3) * 1e-9
            E = (E1 + E2) / 2
            G = E / (2 * (1 + P_val))
            return {
                'dynamic_young_modulus_output': f'{E:.4f}',
                'dynamic_shear_modulus_output': f'{G:.4f}',
                'poisson_ratio_output': f'{P_val:.4f}'
            }
        else:
            return {
                'dynamic_young_modulus_output': 'Invalid',
                'dynamic_shear_modulus_output': 'Invalid',
                'poisson_ratio_output': 'Invalid'
            }
