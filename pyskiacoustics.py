"""
PySKIacoustics - SKY Acoustics for Drone DOA Training
======================================================
Open-source outdoor acoustic propagation for airborne sources.

Features:
1. ISO 9613-1 Atmospheric Absorption
2. Delany-Bazley Ground Impedance Model  
3. Ground Reflection with Phase Shift (Ground Dip)
4. Doppler Effect for Moving Sources
5. Sensor Imperfections (gain mismatch, self-noise)

Comparison to pyroadacoustics:
- pyroadacoustics: Cars on roads (2.5D, asphalt, Doppler)
- pyskiacoustics: Drones in sky (3D, grass/soil, full elevation)
"""

import numpy as np
from scipy.signal import butter, sosfilt, resample
from scipy.interpolate import interp1d
import torch

# === 1. ISO 9613-1 ATMOSPHERIC ABSORPTION ===
def iso9613_attenuation(frequency_hz, distance_m, temperature_c=20.0, humidity_pct=50.0, pressure_kpa=101.325):
    """
    Calculate atmospheric absorption coefficient per ISO 9613-1.
    
    Returns attenuation in dB for given frequency and distance.
    
    Physics: High frequencies are absorbed by molecular relaxation of O2/N2.
    At 10kHz, 50m: ~5 dB loss. At 1kHz, 50m: ~0.1 dB loss.
    """
    T = temperature_c + 273.15  # Kelvin
    T0 = 293.15  # Reference (20°C)
    T01 = 273.16  # Triple point
    
    ps0 = 101.325  # Reference pressure (kPa)
    psat = ps0 * 10**(-6.8346 * (T01/T)**1.261 + 4.6151)  # Saturation vapor pressure
    h = humidity_pct * (psat / pressure_kpa)  # Molar concentration of water vapor
    
    # Relaxation frequencies
    frO = (pressure_kpa / ps0) * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (pressure_kpa / ps0) * (T / T0)**(-0.5) * (9 + 280 * h * np.exp(-4.170 * ((T / T0)**(-1/3) - 1)))
    
    f = np.asarray(frequency_hz)
    
    # Attenuation coefficient (dB/m)
    alpha = 8.686 * f**2 * (
        1.84e-11 * (ps0 / pressure_kpa) * (T / T0)**0.5 +
        (T / T0)**(-2.5) * (
            0.01275 * np.exp(-2239.1 / T) / (frO + f**2 / frO) +
            0.1068 * np.exp(-3352.0 / T) / (frN + f**2 / frN)
        )
    )
    
    return alpha * distance_m  # Total dB loss


def apply_atmospheric_absorption(signal, fs, distance_m, temperature_c=20.0, humidity_pct=50.0):
    """
    Apply frequency-dependent atmospheric absorption to audio signal.
    Uses FFT-based filtering.
    """
    # FFT
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1/fs)
    spectrum = np.fft.rfft(signal)
    
    # Calculate attenuation for each frequency bin
    atten_db = iso9613_attenuation(freqs + 1e-6, distance_m, temperature_c, humidity_pct)
    atten_linear = 10 ** (-atten_db / 20)
    
    # Apply filter
    filtered_spectrum = spectrum * atten_linear
    
    # Inverse FFT
    return np.fft.irfft(filtered_spectrum, n)


# === 2. DELANY-BAZLEY GROUND IMPEDANCE ===
def delany_bazley_impedance(frequency_hz, flow_resistivity=200.0):
    """
    Calculate normalized acoustic impedance of porous ground.
    
    Flow resistivity (σ, kPa·s/m²):
        - Grass/Forest floor: ~200
        - Loose soil: ~500  
        - Packed earth: ~2000
        - Asphalt: ~20000+ (essentially rigid)
    
    Returns complex impedance Z/Z0.
    """
    f = np.asarray(frequency_hz)
    sigma = flow_resistivity * 1000  # Convert kPa to Pa
    
    # X = f / sigma (normalized frequency)
    X = f / (sigma + 1e-9)
    
    # Delany-Bazley empirical formula
    Z_real = 1 + 9.08 * X**(-0.75)
    Z_imag = -11.9 * X**(-0.73)
    
    return Z_real + 1j * Z_imag


def ground_reflection_coefficient(frequency_hz, grazing_angle_rad, flow_resistivity=200.0):
    """
    Calculate complex reflection coefficient for ground reflection.
    
    For grazing angles near 0 (source/receiver near ground), this creates
    the "ground dip" effect - a notch in the 200-800 Hz range.
    """
    f = np.asarray(frequency_hz)
    theta = grazing_angle_rad
    
    # Normalized impedance
    Z = delany_bazley_impedance(f, flow_resistivity)
    
    # Characteristic impedance of air
    Z0 = 1.0  # Normalized
    
    # Reflection coefficient (Fresnel equation for acoustic)
    cos_theta = np.cos(theta)
    R = (Z * cos_theta - Z0) / (Z * cos_theta + Z0)
    
    return R


def apply_ground_reflection(signal, fs, direct_distance_m, reflected_distance_m, 
                            grazing_angle_rad=0.1, flow_resistivity=200.0):
    """
    Add ground-reflected signal with correct delay, attenuation, and phase shift.
    
    The reflected path is longer and goes through the porous ground impedance,
    creating interference (ground dip effect).
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1/fs)
    
    # Direct path spectrum
    direct_spec = np.fft.rfft(signal)
    
    # Reflected path
    path_diff = reflected_distance_m - direct_distance_m
    delay_samples = int(path_diff / 343.0 * fs)
    
    # Reflection coefficient (complex: magnitude + phase)
    R = ground_reflection_coefficient(freqs + 1e-6, grazing_angle_rad, flow_resistivity)
    
    # Delay in frequency domain
    delay_phase = np.exp(-2j * np.pi * freqs * path_diff / 343.0)
    
    # Distance attenuation (1/r)
    atten = direct_distance_m / (reflected_distance_m + 1e-6)
    
    # Combine: reflected = delayed, attenuated, phase-shifted
    reflected_spec = direct_spec * R * delay_phase * atten
    
    # Sum direct + reflected
    combined_spec = direct_spec + reflected_spec
    
    return np.fft.irfft(combined_spec, n)


# === 3. SENSOR IMPERFECTIONS ===
def add_sensor_imperfections(multichannel_signal, gain_std_db=1.0, position_noise_m=0.001):
    """
    Add realistic sensor imperfections:
    - Random gain mismatch between channels
    - Uncorrelated self-noise
    """
    n_channels = multichannel_signal.shape[0]
    
    # Random gain per channel
    gains = 10 ** (np.random.randn(n_channels) * gain_std_db / 20)
    
    # Apply
    for i in range(n_channels):
        multichannel_signal[i] *= gains[i]
        
    # Add uncorrelated thermal noise (very low level)
    noise_floor = np.std(multichannel_signal) * 0.001  # -60 dB
    multichannel_signal += np.random.randn(*multichannel_signal.shape) * noise_floor
    
    return multichannel_signal


# === 4. DOPPLER EFFECT (MOVING SOURCES) ===
def apply_doppler(signal, fs, source_velocity_mps, source_distance_m, speed_of_sound=343.0):
    """
    Apply Doppler frequency shift for a moving source.
    
    For a source approaching/receding at velocity v:
        f_observed = f_source * (c / (c - v_radial))
    
    This is implemented via time-varying resampling.
    
    Args:
        signal: Source audio
        fs: Sample rate
        source_velocity_mps: Radial velocity (positive = approaching)
        source_distance_m: Current distance (for time-varying calculation)
        speed_of_sound: Speed of sound (default 343 m/s)
    
    Returns:
        Doppler-shifted signal
    """
    # Doppler ratio
    v = source_velocity_mps
    c = speed_of_sound
    
    # Clamp to avoid singularities
    v = np.clip(v, -0.9 * c, 0.9 * c)
    
    doppler_ratio = c / (c - v)  # >1 for approaching, <1 for receding
    
    # Resample to simulate pitch shift
    n_in = len(signal)
    n_out = int(n_in / doppler_ratio)
    
    if n_out < 10:
        return signal  # Edge case protection
    
    # Use scipy resample for clean interpolation
    shifted = resample(signal, n_out)
    
    # Pad or trim to original length
    if len(shifted) < n_in:
        shifted = np.pad(shifted, (0, n_in - len(shifted)))
    else:
        shifted = shifted[:n_in]
    
    return shifted


# === 5. WIND NOISE (CORCOS TURBULENCE MODEL) ===
def generate_corcos_wind_noise(duration_s, fs, mic_positions, wind_speed_mps=5.0, 
                                convection_velocity_ratio=0.7):
    """
    Generate spatially-correlated wind noise using the Corcos turbulence model.
    
    Wind noise on microphone arrays is NOT uncorrelated - nearby mics see
    coherent low-frequency turbulence. The Corcos model captures this:
    
        γ(Δx, f) = exp(-α * |Δx| * f / U_c)
    
    where:
        Δx = mic separation
        f = frequency
        U_c = convection velocity = 0.7 * wind_speed
        α = decay constant (~0.5 for turbulent boundary layers)
    
    Args:
        duration_s: Duration in seconds
        fs: Sample rate
        mic_positions: [N, 3] array of mic positions (meters)
        wind_speed_mps: Wind speed (m/s)
        convection_velocity_ratio: U_c / U_wind (typically 0.6-0.8)
    
    Returns:
        [N, samples] correlated wind noise
    """
    n_mics = len(mic_positions)
    n_samples = int(duration_s * fs)
    n_fft = n_samples
    
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    
    # Convection velocity
    U_c = wind_speed_mps * convection_velocity_ratio
    if U_c < 0.1:
        U_c = 0.1  # Prevent division by zero
    
    # Decay constant (empirical)
    alpha = 0.5
    
    # Generate base pink noise spectrum (wind is predominantly low-freq)
    # 1/f spectrum with rolloff
    base_spectrum = 1.0 / (freqs + 1e-6) ** 0.5
    base_spectrum[0] = 0  # Kill DC
    
    # Build covariance matrix for each frequency bin
    output = np.zeros((n_mics, n_samples))
    
    # Generate in frequency domain with spatial correlation
    for i in range(n_mics):
        # Phase randomized pink noise
        phase = np.random.uniform(0, 2*np.pi, len(freqs))
        spectrum_i = base_spectrum * np.exp(1j * phase)
        
        # Apply coherence decay relative to other mics (simplified)
        # For speed, we model decay relative to array center
        center = np.mean(mic_positions, axis=0)
        dx = np.linalg.norm(mic_positions[i] - center)
        
        # Coherence decay per frequency
        coherence = np.exp(-alpha * dx * freqs / U_c)
        
        # Add independent noise for decorrelated high-freq
        spectrum_i = spectrum_i * coherence + base_spectrum * (1 - coherence) * np.exp(1j * np.random.uniform(0, 2*np.pi, len(freqs)))
        
        # IFFT
        output[i] = np.fft.irfft(spectrum_i, n_samples)
    
    # Normalize to reasonable level
    output = output / (np.max(np.abs(output)) + 1e-9) * wind_speed_mps * 0.01
    
    return output


# === 6. COMPLETE OUTDOOR PROPAGATION ===
def propagate_outdoor(source_signal, fs, distance_m, elevation_m, 
                      array_height_m=0.76,  # Table height
                      temperature_c=20.0, humidity_pct=50.0,
                      ground_type='grass'):
    """
    Full outdoor propagation model:
    1. Distance attenuation (1/r²)
    2. Atmospheric absorption (ISO 9613)
    3. Ground reflection (Delany-Bazley)
    
    Args:
        source_signal: Mono source audio
        fs: Sample rate
        distance_m: Horizontal distance to source
        elevation_m: Height of source above ground
        array_height_m: Height of microphone array
        temperature_c: Air temperature
        humidity_pct: Relative humidity
        ground_type: 'grass', 'soil', 'asphalt'
    """
    # Flow resistivity by ground type
    flow_resistivity = {
        'grass': 200,
        'soil': 500,
        'asphalt': 20000
    }.get(ground_type, 200)
    
    # 1. Distance attenuation (1/r, since amplitude not power)
    direct_path = np.sqrt(distance_m**2 + (elevation_m - array_height_m)**2)
    signal = source_signal / (direct_path + 1e-6)
    
    # 2. Atmospheric absorption
    signal = apply_atmospheric_absorption(signal, fs, direct_path, temperature_c, humidity_pct)
    
    # 3. Ground reflection (image source method)
    # Reflected path bounces off ground
    reflected_height = elevation_m + array_height_m  # Image source below ground
    reflected_path = np.sqrt(distance_m**2 + reflected_height**2)
    
    # Grazing angle (angle from horizontal)
    grazing_angle = np.arctan2(reflected_height, distance_m)
    
    signal = apply_ground_reflection(
        signal, fs, direct_path, reflected_path, 
        grazing_angle, flow_resistivity
    )
    
    return signal


# === 7. ARRAY STEERING (PER-MIC DELAYS) ===
# UMA-16 Default Geometry (4x4 grid, 48mm spacing)
UMA16_POSITIONS = np.array([
    [-0.066,  0.066, 0], [-0.024,  0.066, 0], [ 0.024,  0.066, 0], [ 0.066,  0.066, 0],
    [-0.066,  0.024, 0], [-0.024,  0.024, 0], [ 0.024,  0.024, 0], [ 0.066,  0.024, 0],
    [-0.066, -0.024, 0], [-0.024, -0.024, 0], [ 0.024, -0.024, 0], [ 0.066, -0.024, 0],
    [-0.066, -0.066, 0], [-0.024, -0.066, 0], [ 0.024, -0.066, 0], [ 0.066, -0.066, 0]
])


def simulate_array_response(source_signal, fs, source_position, 
                            mic_positions=None, 
                            array_center=np.array([0, 0, 0.76]),
                            apply_physics=True,
                            temperature_c=20.0, humidity_pct=50.0,
                            ground_type='grass',
                            wind_speed_mps=0.0):
    """
    Simulate the complete 16-channel microphone array response.
    
    *** SUB-SAMPLE PRECISION VERSION ***
    Uses frequency-domain phase shifting for exact fractional delays.
    Required for 0.01° DOA accuracy.
    
    Args:
        source_signal: Mono source audio (dry recording)
        fs: Sample rate
        source_position: [x, y, z] position of source in meters
        mic_positions: [N, 3] array of mic positions relative to array center
                       (defaults to UMA-16 geometry)
        array_center: [x, y, z] position of array center in world coords
        apply_physics: Whether to apply atmospheric absorption + ground reflection
        temperature_c: Air temperature for atmospheric model
        humidity_pct: Relative humidity
        ground_type: 'grass', 'soil', or 'asphalt'
        wind_speed_mps: Wind speed for noise injection (0 = no wind)
    
    Returns:
        [N_mics, N_samples] multi-channel array output
        [3] unit vector pointing from array to source (the DOA label)
    """
    if mic_positions is None:
        mic_positions = UMA16_POSITIONS
    
    n_mics = len(mic_positions)
    n_samples = len(source_signal)
    
    # === TEMPERATURE-DEPENDENT SPEED OF SOUND ===
    # c = 331.3 * sqrt(1 + T/273.15) m/s
    speed_of_sound = 331.3 * np.sqrt(1 + temperature_c / 273.15)
    
    # World positions of each microphone
    mic_world = mic_positions + array_center
    
    # Source vector (for label)
    source_vec = source_position - array_center
    source_distance = np.linalg.norm(source_vec)
    source_unit_vec = source_vec / (source_distance + 1e-9)
    
    # Flow resistivity by ground type
    flow_resistivity = {'grass': 200, 'soil': 500, 'asphalt': 20000}.get(ground_type, 200)
    
    # Reference delay (to array center) - for relative timing
    ref_delay_seconds = source_distance / speed_of_sound
    
    # Pre-compute FFT of source signal (for efficient phase shifting)
    n_fft = n_samples
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    source_spectrum = np.fft.rfft(source_signal)
    
    output = np.zeros((n_mics, n_samples))
    
    for i in range(n_mics):
        # === EXACT DISTANCE CALCULATION ===
        mic_vec = source_position - mic_world[i]
        distance = np.linalg.norm(mic_vec)
        
        # === FRACTIONAL DELAY IN SECONDS (not samples!) ===
        delay_seconds = distance / speed_of_sound - ref_delay_seconds
        
        # === FREQUENCY-DOMAIN PHASE SHIFTING (sub-sample precision) ===
        # This is EXACT - no interpolation artifacts
        # Phase shift = exp(-j * 2π * f * delay)
        delay_phase = np.exp(-2j * np.pi * freqs * delay_seconds)
        
        # Apply phase shift to spectrum
        shifted_spectrum = source_spectrum * delay_phase
        
        # 1. Distance attenuation (1/r)
        attenuation = 1.0 / (distance + 1e-6)
        shifted_spectrum = shifted_spectrum * attenuation
        
        if apply_physics:
            # 2. Atmospheric absorption (frequency-dependent)
            atten_db = iso9613_attenuation(freqs + 1e-6, distance, temperature_c, humidity_pct)
            atten_linear = 10 ** (-atten_db / 20)
            shifted_spectrum = shifted_spectrum * atten_linear
            
            # 3. Ground reflection (image source method)
            source_height = source_position[2]
            mic_height = mic_world[i, 2]
            horiz_dist = np.sqrt((source_position[0] - mic_world[i, 0])**2 + 
                                  (source_position[1] - mic_world[i, 1])**2)
            
            # Image source is below ground by source_height
            reflected_height = source_height + mic_height
            reflected_path = np.sqrt(horiz_dist**2 + reflected_height**2)
            grazing_angle = np.arctan2(reflected_height, horiz_dist + 1e-9)
            
            # Reflected wave: delay + attenuation + impedance phase shift
            reflected_delay = reflected_path / speed_of_sound - ref_delay_seconds
            reflected_phase = np.exp(-2j * np.pi * freqs * reflected_delay)
            
            # Ground reflection coefficient (complex: includes phase shift)
            R = ground_reflection_coefficient(freqs + 1e-6, grazing_angle, flow_resistivity)
            
            # Reflected path attenuation
            reflected_atten = 1.0 / (reflected_path + 1e-6)
            
            # Atmospheric absorption on reflected path
            reflected_atm_db = iso9613_attenuation(freqs + 1e-6, reflected_path, temperature_c, humidity_pct)
            reflected_atm_linear = 10 ** (-reflected_atm_db / 20)
            
            # Add reflected wave to direct wave
            reflected_spectrum = source_spectrum * reflected_phase * R * reflected_atten * reflected_atm_linear
            shifted_spectrum = shifted_spectrum + reflected_spectrum
        
        # Convert back to time domain
        sig = np.fft.irfft(shifted_spectrum, n_samples)
        output[i] = sig
    
    # 4. Add wind noise if requested
    if wind_speed_mps > 0:
        wind = generate_corcos_wind_noise(n_samples / fs, fs, mic_positions, wind_speed_mps)
        output = output + wind
    
    # 5. Add sensor imperfections (small for precision work)
    output = add_sensor_imperfections(output, gain_std_db=0.3)
    
    # Normalize
    max_val = np.max(np.abs(output))
    if max_val > 1e-9:
        output = output / max_val
    
    return output, source_unit_vec


def generate_training_sample(reference_clip, fs, 
                             azimuth_deg=None, elevation_deg=None, distance_m=None,
                             wind_speed_mps=None, apply_physics=True):
    """
    Convenience function to generate a single training sample.
    
    Randomizes parameters if not specified.
    
    Returns:
        audio: [16, N_samples] array audio
        label: [3] unit vector (x, y, z)
    """
    # Random parameters if not specified
    if azimuth_deg is None:
        azimuth_deg = np.random.uniform(0, 360)
    if elevation_deg is None:
        elevation_deg = np.random.uniform(5, 45)  # Typical drone angles
    if distance_m is None:
        distance_m = np.random.uniform(20, 100)
    if wind_speed_mps is None:
        wind_speed_mps = np.random.choice([0, 2, 5, 8], p=[0.3, 0.3, 0.3, 0.1])
    
    # Convert to Cartesian
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elevation_deg)
    
    x = distance_m * np.cos(el_rad) * np.cos(az_rad)
    y = distance_m * np.cos(el_rad) * np.sin(az_rad)
    z = distance_m * np.sin(el_rad) + 0.76  # Offset for array height
    
    source_position = np.array([x, y, z])
    
    # Simulate
    audio, label = simulate_array_response(
        reference_clip[:, 0] if reference_clip.ndim > 1 else reference_clip,
        fs,
        source_position,
        wind_speed_mps=wind_speed_mps,
        apply_physics=apply_physics
    )
    
    return audio, label


# === TEST ===
if __name__ == "__main__":
    # Quick validation
    import matplotlib.pyplot as plt
    
    # Test atmospheric absorption
    freqs = np.logspace(2, 4, 100)  # 100 Hz to 10 kHz
    atten_50m = iso9613_attenuation(freqs, 50)
    atten_100m = iso9613_attenuation(freqs, 100)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.semilogx(freqs, atten_50m, label='50m')
    plt.semilogx(freqs, atten_100m, label='100m')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation (dB)')
    plt.title('ISO 9613 Atmospheric Absorption')
    plt.legend()
    plt.grid()
    
    # Test ground reflection coefficient
    R = ground_reflection_coefficient(freqs, 0.1, 200)
    plt.subplot(1, 2, 2)
    plt.semilogx(freqs, np.abs(R), label='|R| Grass')
    R_asphalt = ground_reflection_coefficient(freqs, 0.1, 20000)
    plt.semilogx(freqs, np.abs(R_asphalt), label='|R| Asphalt')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|Reflection Coefficient|')
    plt.title('Delany-Bazley Ground Reflection')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('outdoor_physics_test.png')
    print("Saved outdoor_physics_test.png")
