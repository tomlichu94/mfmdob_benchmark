import numpy as np
from fractions import Fraction

def wave_generator(f_hz,amp,p_off,ts_ss,ts_fs,t_end):
    """
    Generate sinusoidal signals at continuous, slow, and fast sampling speeds
    
    Parameters:
    f_hz (array): frequency of the sin wave in Hz
    amp (array): amplitude of the sin wave
    p_off (array): phase offset of the sin wave in radians
    t_fs (float): sampling time for fast sampling in seconds
    t_ss (float): sampling time for slow sampling in seconds
    t_end (float): end time for the signal in seconds

    Returns:
    out_ct (ndarray): continuous time signal 
    out_ss (ndarray): slow sampling signal
    out_fs (ndarray): fast sampling signal
    """ 
    
    
    ts_ct = ts_fs/100 # continuous time sampling
    t_ct = np.arange(0,t_end + ts_ct,ts_ct)
    t_ss = np.arange(0, t_end + ts_ss, ts_ss)
    t_fs = np.arange(0, t_end + ts_fs, ts_fs)

    # preallocate output arrays
    out_ct = np.vstack((t_ct, np.zeros_like(t_ct)))
    out_ss = np.vstack((t_ss, np.zeros_like(t_ss)))
    out_fs = np.vstack((t_fs, np.zeros_like(t_fs)))

    # generate output signals
    for i in range(len(f_hz)):
        out_ct[1,:] += amp[i] * np.sin(2 * np.pi * f_hz[i] * t_ct + p_off[i])
        out_ss[1,:] += amp[i] * np.sin(2 * np.pi * f_hz[i] * t_ss + p_off[i])
        out_fs[1,:] += amp[i] * np.sin(2 * np.pi * f_hz[i] * t_fs + p_off[i])

    return out_ct, out_ss, out_fs

def multirate_model_coeff(f_hz, t_fs, L):
    """
    Determine coefficients for MMP for signal reconstruction.

    Parameters:
    f_hz (array): signal frequencies
    t_fs (float): sampling period
    L(float): scaling factor, can be non-integer

    Returns:
    MMP coefficients
    """
    
    frac = Fraction(L).limit_denominator(100) # identify fractional integers of L
    N_L, D_L = frac.numerator, frac.denominator # D_L is the number of slow samples per N_L fast samples
    N_L = int(N_L)
    
    k_max = N_L-1 # number of inter-samples
    m_d = len(f_hz) # number of disturbances

    Apara = np.array([1])  # Initialize Apara as a 1D array with one element
    for i in range(m_d):
        omega = 2 * np.pi * f_hz[i] * t_fs
        Apara = np.convolve(Apara, [1, -2 * np.cos(omega), 1])
    
    m_a = len(Apara)
    n_w = 2*m_d-1

    # preallocate dimensions for M_k
    m_k1 = 2*m_d*N_L
    m_k2 = 2*m_d*(N_L-1)

    # preallocate M_k
    M_kt = np.zeros((m_k1, m_k2))
    for i in range(m_k2):
        M_kt[i:i + m_a, i] = Apara

    # Construct M_kt (Toeplitz-like)
    for i in range(m_k2):
        M_kt[i:i + m_a, i] = Apara

    # Preallocate E_k and M_k
    E_k = np.zeros((m_k1, 2 * m_d, k_max))
    M_k = np.zeros((m_k1, m_k1, k_max))

    for ki in range(k_max):
        for j in range(2 * m_d):
            row_idx = ki + N_L * j
            E_k[row_idx, j, ki] = 1
        M_k[:, :m_k2, ki] = M_kt
        M_k[:, m_k2:, ki] = E_k[:, :, ki]

    # Construct a_sol vector
    a_sol = np.concatenate((-Apara[1:], np.zeros(m_k1 - m_a + 1)))

    # Solve M_k x = a_sol for each k
    x_sol = np.zeros((m_k1, k_max))
    for ki in range(k_max):
        x_sol[:, ki] = np.linalg.solve(M_k[:, :, ki], a_sol)

    # Extract w_k from the last (n_w + 1) rows
    w_k = x_sol[-(n_w + 1):, :]
    return w_k
    
def mmp_coeff(f_hz, t_fs, a_g, L, use_damped=True):
    """
    Determine FIR/IIR-MMP coefficients for signal reconstruction.

    Parameters:
        f_hz : array-like
            Signal frequencies to be recovered [Hz].
        t_fs : float
            Fast sampling period [s].
        a_g : float
            IIR bandwidth parameter. Use a_g = 0 for FIR, a_g > 0 for IIR.
        L : float
            Upsampling factor, can be non-integer.
        use_damped : bool
            If True, use damped pole models for known HDD fan modes.

    Returns:
        w_kiir : ndarray
            IIR-MMP numerator coefficients.
        Bpara : ndarray
            Denominator coefficients of the IIR-MMP.
    """

    frac = Fraction(L).limit_denominator(100)
    N_L, D_L = frac.numerator, frac.denominator

    N_L = int(N_L)
    D_L = int(D_L)

    k_max = N_L - 1
    m_d = len(f_hz)

    # Damping ratios from the continuous-time HDD fan disturbance models
    damping_map = {
        4220: 0.01,
        4380: 0.008,
        5072: 0.002,
        5850: 0.04,
        6660: 0.008,
        7670: 0.003,
        9200: 0.07,
    }

    def get_damping_ratio(freq):
        """Return damping ratio if freq matches a known damped HDD mode."""
        for f_key, zeta in damping_map.items():
            if np.isclose(freq, f_key):
                return zeta
        return None

    # Construct A(z) and B(z)
    Apara = np.array([1.0])
    Bpara = np.array([1.0])

    for i in range(m_d):
        f_i = float(f_hz[i])

        zeta = get_damping_ratio(f_i) if use_damped else None

        if zeta is not None:
            # Continuous-time damped mode:
            # s^2 + 2*zeta*omega_n*s + omega_n^2
            omega_n = 2 * np.pi * f_i
            omega_d = omega_n * np.sqrt(1 - zeta**2)

            # Fast-rate discrete pole:
            # z = r * exp(±j theta)
            r = np.exp(-zeta * omega_n * t_fs)
            theta = omega_d * t_fs

            # Fast-rate internal-model polynomial:
            # 1 - 2*r*cos(theta) z^-1 + r^2 z^-2
            A_factor = np.array([
                1.0,
                -2 * r * np.cos(theta),
                r**2
            ])

            # Same phase appears every N_L fast samples.
            # Therefore the slow/phase pole is z^N_L:
            r_slow = r**N_L
            theta_slow = N_L * theta

            # IIR-MMP denominator with bandwidth tuning a_g
            B_factor = np.array([
                1.0,
                -2 * a_g * r_slow * np.cos(theta_slow),
                (a_g * r_slow)**2
            ])

        else:
            # Standard undamped sinusoidal internal model
            omega = 2 * np.pi * f_i * t_fs

            A_factor = np.array([
                1.0,
                -2 * np.cos(omega),
                1.0
            ])

            B_factor = np.array([
                1.0,
                -2 * a_g * np.cos(N_L * omega),
                a_g**2
            ])

        Apara = np.convolve(Apara, A_factor)
        Bpara = np.convolve(Bpara, B_factor)

    m_a = len(Apara)
    n_w = 2 * m_d - 1

    # Preallocate dimensions
    m_k1 = 2 * m_d * N_L
    m_k2 = 2 * m_d * (N_L - 1)

    # Construct M_kt matrix
    M_kt = np.zeros((m_k1, m_k2))

    for i in range(m_k2):
        M_kt[i:i + m_a, i] = Apara

    # Preallocate E_k and M_k
    E_k = np.zeros((m_k1, 2 * m_d, k_max))
    M_k = np.zeros((m_k1, m_k1, k_max))

    for ki in range(k_max):
        for j in range(2 * m_d):
            row_idx = ki + N_L * j
            E_k[row_idx, j, ki] = 1

        M_k[:, :m_k2, ki] = M_kt
        M_k[:, m_k2:, ki] = E_k[:, :, ki]

    # Construct a_sol
    a_sol = np.concatenate((-Apara[1:], np.zeros(m_k1 - m_a + 1)))

    # Construct b_sol
    b_sol = np.zeros_like(a_sol)

    for i in range(2 * m_d):
        b_sol[(i + 1) * N_L - 1] = Bpara[i + 1]

    # Solve for x
    x_sol = np.zeros((m_k1, k_max))

    for ki in range(k_max):
        x_sol[:, ki] = np.linalg.solve(M_k[:, :, ki], a_sol + b_sol)

    # Extract IIR-MMP coefficients
    w_kiir = x_sol[-(n_w + 1):, :]

    return w_kiir, Bpara

def signal_recovery(input_signal, f_hz, t_fs, L):
    """ 
    Signal recovery alogorithm

    
    """
    w_k = multirate_model_coeff(f_hz, t_fs, L)
    length_fs = int(np.floor(len(input_signal)-1)*L + 1)
    y_fs = np.zeros(length_fs)
    n_w = w_k.shape[0] # define 2x number of frequencies
    n_ss = 0

    frac = Fraction(L).limit_denominator(100) # identify fractional integers of L
    N_L, D_L = frac.numerator, frac.denominator # D_L is the number of slow samples per N_L fast samples
    N_L = int(N_L)
    D_L = int(D_L)

    if not hasattr(signal_recovery,'phi'):
        signal_recovery.phi = None # place holder for phi, stored value in function

    if signal_recovery.phi is None:
        signal_recovery.phi = np.zeros(n_w) # create empty array for phi
    for n_fs in range(length_fs):
        k = int(n_fs % N_L) # determine k
       
        if n_ss < n_w:
            if k == 0: # fast and slow sampling align
                idx = n_ss*D_L
                if idx < len(input_signal):
                    signal_recovery.phi[1:] = signal_recovery.phi[:-1]
                    signal_recovery.phi[0] = input_signal[idx]
                y_fs[n_fs] = 0
                n_ss += 1
            else:
                y_fs[n_fs] = 0
        else:
            if k == 0:
                idx = n_ss*D_L
                if idx < len(input_signal):
                    signal_recovery.phi[1:] = signal_recovery.phi[:-1]
                    signal_recovery.phi[0] = input_signal[idx]
                    y_fs[n_fs] = input_signal[idx]
                n_ss += 1
            else:
                y_fs[n_fs] = np.dot(signal_recovery.phi, w_k[:, k-1])
    out = y_fs
    return out
    
def signal_recovery_iir(input_signal, f_hz, t_fs, a_g, L, use_damped=True):
    """
    IIR signal recovery using fractional-speed sampling.

    Parameters:
        input_signal : array-like
            Slow-sampled input signal.
        f_hz : array-like
            Frequencies to recover.
        t_fs : float
            Fast sampling time.
        a_g : float
            IIR bandwidth parameter.
        L : float
            Upsampling factor.

    Returns:
        out : ndarray
            Recovered fast-sampled signal.
    """

    input_signal = np.asarray(input_signal).flatten()

    frac = Fraction(L).limit_denominator(100)
    N_L = int(frac.numerator)
    D_L = int(frac.denominator)

    # Find IIR-MMP coefficients
    w_kiir, Bpara = mmp_coeff(f_hz, t_fs, a_g, L, use_damped=use_damped)

    # Determine signal lengths
    length_ss = len(input_signal)

    set_ss = np.arange(0, length_ss, D_L)
    length_set = len(set_ss) - 1
    length_fs = length_set * N_L + 1

    out = np.zeros(length_fs)

    n_w = w_kiir.shape[0]
    n_ss = 0

    # MATLAB:
    # buffer_indices = 2:N_L:((n_w - 1) * N_L + 2)
    #
    # Python zero-based equivalent:
    buffer_indices = np.arange(1, (n_w - 1) * N_L + 2, N_L)

    max_buffer_size = n_w * N_L + 1
    d_buff = np.zeros(max_buffer_size)

    phi = np.zeros(n_w)

    for n_fs in range(length_fs):

        d_temp = input_signal[n_ss * D_L]

        k = n_fs % N_L

        # Case 1: not enough slow samples
        if n_ss + 1 < n_w + 1:

            if k == 0:
                phi[1:] = phi[:-1]
                phi[0] = d_temp

                out[n_fs] = 0.0
                n_ss += 1
            else:
                out[n_fs] = 0.0

        # Case 2: enough for FIR
        elif (n_ss + 1 >= n_w) and (n_ss + 1 < n_w + D_L):

            if k == 0:
                phi[1:] = phi[:-1]
                phi[0] = d_temp

                out[n_fs] = d_temp
                n_ss += 1
            else:
                out[n_fs] = np.dot(phi, w_kiir[:, k - 1])

        # Case 3: IIR reconstruction
        else:

            if k == 0:
                phi[1:] = phi[:-1]
                phi[0] = d_temp

                out[n_fs] = d_temp
                n_ss += 1
            else:
                d_temp_buff = np.flip(d_buff)
                out[n_fs] = (np.dot(phi, w_kiir[:, k - 1]) - np.dot(d_temp_buff[buffer_indices], np.flip(Bpara[1:])))

        # Update buffer
        d_buff[1:] = d_buff[:-1]
        d_buff[0] = out[n_fs]

    return out

def multi_phase_recovery(input_signal, f_hz, t_fs, t_end, L):
    """    
        w_k (np.ndarray): weight matrix
        L (int): interpolation factor
        
    Returns:
        y_est (np.ndarray): reconstructed fast-rate signal array
    """
    # Calculate total length based on input and interpolation
    # Note: Adjust length estimate if needed based on your signal and w_k
    
    # We'll estimate max length conservatively:
    # Each iteration produces y_fcn of some length; max length * N_L covers indexing
    frac = Fraction(L).limit_denominator(100) # identify fractional integers of L
    N_L, D_L = frac.numerator, frac.denominator # D_L is the number of slow samples per N_L fast samples
    N_L = int(N_L)
    D_L = int(D_L)
    
    length_ss = len(input_signal)  # Length of the input signal
    y_est = np.zeros(int((length_ss-1)*N_L + 1))

    for i in range(1, D_L+1):
        y_fcn = signal_recovery(input_signal[i-1:], f_hz, t_fs, L) # signal recovery for each phase
        base_idx = (i - 1) * N_L # index starting point
        idx = base_idx + np.arange(len(y_fcn)) * D_L # range of indexed points
        y_est[idx] = y_fcn

    t_fast = np.arange(0, t_end + t_fs / D_L, t_fs / D_L) # fast sampling time, T_ss/(RL)
    out = np.vstack((t_fast, y_est))

    return out
