"""
@serrano
--------------------------------------------------------------------------------------
Note: This file will be used to introduce different sweeps (this version works fully)

    -   In here we have created a class that contains the different functions for the sweeps.
    -   Waveguides have been included in this version.
    -   Different functions:
        *   __init__(self): Basic parameters for the transmitter
        *   transmitter_simple()
                Simple transmitter, no sweeps
        *   transmitter_rf_input_voltage_sweep()
                This function sweeps the different input power (from the synthesizer)  rf values, that are transformed
                into voltages.
        *   transmitter_rf_input_voltage_vpi_rf_sweep()
                This function loops the different rf input power(voltages) with the pi rf voltages from the MZM.
        *   transmitter_rf_input_voltage_vpi_dc_sweep()
                This function loops the different input power(voltages) with the pi dc voltages from the MZM.
        *   transmitter_rf_input_voltage_v_bias_sweep()
                This function loops the different input power voltages with the bias voltage from the MZM
        *   transmitter_rf_input_voltage_power_laser_sweep()
                This function loops the different input powers from the laser and the input rf powers (voltages).

    -   Tested and working file
--------------------------------------------------------------------------------------

Link 1 - Analogic Photonic Link

In here we describe a transmitter system composed by a laser, MZM, SOA, Attenuator, a photodiode, and a low pass filter.

The purpose is to modulate 2 RF frequencies, into an optical signal provided by the
Continuous Laser, modulated at a central wavelength of 1.55um (balanced-single drived - single push-pull ).

Balanced single-drive MZMs indeed operate in a push-pull configuration,
where the two arms of the interferometer are driven in opposite directions.
This configuration enhances modulation efficiency, speed, and linearity, making it
suitable for high-speed optical communication applications.
The push-pull design also helps in achieving high dynamic range and suppressing unwanted
distortions, ensuring high-quality signal transmission.

A gain has been included to amplify the signal.
Another MZM, is used with a Modulation signal from the Local Oscillator (LO), because
we want to modulate the signal to an IF signal
The PD will detect the signal
And the LPF, will filter only he signal with the wanted frequency

In this case the simulations will show a transmission that should go up to 0.50, as the component is divided in 2 outputs

    Note:
            In this file I wanted to only change the value of the voltages for the rf voltages, but in reality this was
        giving me different results, Even though you're calling ic.deleteall() at the end of each function call,
        some instruments (like analyzers or power meters) might hold internal state or cached data that isn't reset
        unless you explicitly reinstantiate them.

        Suggestion:
        Instead of only deleting the layout (deleteall()), fully rebuild every component inside the transmitter() function per loop.
        Don’t reuse handles like rf1, mzm1, etc., between iterations.

        Thus, it's better to use everything in a loop, as this example in here

"""

import importlib.util
import sys
import os

sys.path.insert(0, "/opt/lumerical/v252/api/python")
api_path = "/opt/lumerical/v252/api/python/lumapi.py"
if not os.path.exists(api_path):
    raise FileNotFoundError(f"Lumerical API not found at {api_path}")


# default path for current release
spec_lin = importlib.util.spec_from_file_location('lumapi', api_path)
# Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_lin)
spec_lin.loader.exec_module(lumapi)

print("Loaded:", lumapi.__file__)
print("Lumerical version:", lumapi.__doc__.splitlines()[0] if lumapi.__doc__ else "Unknown")

import numpy as np
import matplotlib.pyplot as plt
import pdb
import itertools as it
import pickle
from receiver import *

# Open an INTERCONNECT session
ic = lumapi.INTERCONNECT()
ic.switchtodesign()
ic.deleteall()  # Clear previous elements


# ic.closeall()

class Transceiver:
    """
    Class of transceiver, where you can find functions for different variations
    Simulation of the transmitter, receiver and simulates an objective by the addition of a fiber delay
    """

    # Constants
    def __init__(self):
        """
        Basic parameters for the transmitter
        """
        # Physical constants
        self.c = 3e8  # Speed of light (m/s)
        self.lambda_central = 1.55e-6  # Central wavelength (meters)
        self.lambda_range = 100e-9  # Wavelength range (meters)
        self.lambda_points = 100  # Number of points in the wavelength sweep

        # Waveguide parameters
        self.neff_te = 3.24  # Effective index
        self.ngroup_te = 3.7  # Group index
        self.wg_length = 200e-6  # Waveguide length (meters)
        self.wg_loss = 227  # Waveguide losses (dB/m)

        # Time and RF
        self.time_window = 1e-7  # Time for the simulation (s) 2e-6
        self.rf_freq1 = 10e9  # RF signal 1 (Hz)
        self.rf_freq2 = 10.2e9  # RF signal 2 (Hz)

        self.p_in_start = 0.0001  # Start value for the variation of the input rf power
        self.p_in_stop = 45.0  # Stop value for the variation of the input rf power
        self.n_values = 2  # number of values for the variation of the input rf power

        # Electrical and optical parameters
        self.impedance = 50  # Ohms

        self.sample_rate = 200e9  # Sampling rate (Hz)
        self.n_samples = int(5e5)  # Number of samples

        # Modulator parameters
        # Tx
        self.v_pi_dc_tx = 4  # Vpi for DC bias
        self.v_pi_rf_tx = 4  # Vpi for RF
        self.v_bias_tx = 2  # DC bias voltage - Minimum for carrier suppression
        self.mmi_loss = 3.0

        # Rx
        self.v_pi_dc_rx = 4  # Vpi for DC bias
        self.v_pi_rf_rx = 4  # Vpi for RF
        self.v_bias_rx = 1  # DC bias voltage - Biased at quadrature

        # SOA
        self.SOA_gain = 15  # Gain in dB
        self.SOA_gain_rx = 20 # On the receiver side

        # Waveguides enabled
        self.wg_enabled = False

        # Attenuation
        self.att_value = 0

        # Bandwidth of the ECN
        self.spectrum_bw = 50e6

        # Laser
        self.RIN = -160
        self.power_laser = 0.010  # 10 dBm - 10 mW -> It gets the values in W and transforms it into dBm

        # Thermal noise for the Photodiode
        self.thermal_noise = 3.185e-22  # The input is A^2/Hz, so this value 1.78462e-11 A/Hz^0.5 squared
        self.sat_power = 0.03162  # In Watt 15 dBm - If the saturation power is disabled, setting this up will give a problem

        # PSD Noise for Noise source
        # To have the noise floor at -174 dBm/Hz
        self.psd_noise = 1.99053e-19  # W/Hz from 4.46154e-10 V/Hz^0.5, the input is W/Hz

        # Spectrum Analyzers
        self.sensitivity_imd3 = 1e-23
        self.sensitivity_singleCarrier = 1e-18
        self.interpolation_offset = 300e6

        # reference position
        self.x_pos = 0
        self.y_pos = 0

        #MZI
        self.delta_l = 33e-6

        # Electrical amplifier for the target
        self.gain_target = 30 # dB electrical
        self.noise_sd = 1e-17 # W/Hz

        # Length of the fiber
        self.length_fiber_delay = 1e3  # m

        # Time delay of the target
        self.tau = 40e-9 # check the units of this

        # radar parameters
        self.Vp = 1.0   # before 0.25
        self.B = 0.5e9

    def modulator_s21(self,
                  v_bias_start=None,
                  v_bias_stop=None,
                  v_bias_n=None):
        """
        Modulator only simulation
        """
        # General root element
        root = "::Root Element"
        ic.setnamed(root, "time window", self.time_window)
        ic.setnamed(root, "sample rate", self.sample_rate)
        # ic.setnamed(root, "number of samples", self.n_samples)
        ic.setnamed(root, "output signal mode", "block")  # Block is giving errors
        ic.setnamed(root, "number of output signals", 1)
        ic.setnamed(root, "monitor data", "save to memory")

        # Add the Optical Network Analyzer (ONA)
        ic.addelement("Optical Network Analyzer")
        ona_1 = "ONA_1"
        ic.set("name", ona_1)
        ic.set("x position", -300)
        ic.set("y position", -500)
        ic.set("power", 1)
        ic.set("excitation", 1)
        ic.set("center frequency", self.c / self.lambda_central)
        ic.set("frequency range", self.sample_rate)
        ic.setnamed(ona_1, "number of points", self.lambda_points)
        ic.set("number of input ports", 2)
        ic.set("orthogonal identifier", 1)  # 1 TE, 2 TM
        ic.set("label", "X")  # X - TE, Y-TM
        ## When we are checking the TE mode te "orthogonal identifier" will be 1,
        ## with label "X"

        # Add the Modulator for the 2 RF signals
        ic.addelement("Mach-Zehnder Modulator")
        mzm1 = "MZM_RF"
        ic.set("name", mzm1)
        ic.set("x position", 0)
        ic.set("y position", 0)
        ic.set("modulator type", "balanced single drive")
        # ic.set("modulator type", "dual drive")
        # ic.set("modulator type", "unbalanced single drive")
        ic.set("dc bias source", "internal")  # internal because I do not need extra components
        ic.set("bias voltage 1", self.v_bias_tx)  # Why is it 1? Because is cuadrature and the it divides by 2, and then
        # Lumerical counts it as you would have a bias voltage in 2 arms, and that is why
        # you have to divide it by 2 again
        ic.set("pi dc voltage", self.v_pi_dc_tx)
        ic.set("pi rf voltage", self.v_pi_rf_tx)
        ic.set("extinction ratio", 100)  # dB
        ic.set("insertion loss", 5)  # dB
        ic.set("phase shift", 0)  # rad

        f0 = self.rf_freq1


        Vp = self.Vp  # peak voltage
        B = self.B  # sweep 50 MHz
        T = self.time_window  # 1 us chirp period, is this the same as the time window? -yes
        k = B / T  # [Hz/s] chirp slope
        t = np.linspace(0, T, 1000)  # 0.5 us for example
        t_mod = np.mod(t, T)  # modulo operation


        chirp_script = (
                f"f0 = {f0};" +
                f"B = {B};" +
                f"T = {T};" +
                f"k = B/T;" +
                f"t_mod = mod(TIME, T);" +
                "OUTPUT = " +
                f"{Vp}*sin(2*pi*(f0*t_mod + 0.5*k*t_mod^2));"
        )

        ic.addelement("Scripted Source")
        chirp1 = "chirp"
        ic.set("name", chirp1)
        ic.set("x position", -200)
        ic.set("y position", -200)
        ic.set('script', chirp_script)

        ic.connect(ona_1, "input 1", mzm1, 'input')
        ic.connect(ona_1, "output", mzm1, 'output')
        ic.connect(chirp1, "output", mzm1, "modulation 1")

        # ic.run()

        # Retrieve simulation results

        # t1 = ic.getresult(ona_1, "input 1/mode 1/transmission")
        # lambda_ona = t1.get("wavelength")
        # trans_mzm = t1.get("TE transmission")
        #
        # t1 = ic.getresult(ona_1, "input 1/mode 1/gain")
        # lambda_ona = t1.get("wavelength")
        # gain_mzm = t1.get("TE gain")
        #
        # # Transfer function
        # plt.figure()
        # plt.plot(lambda_ona * 1e9, np.abs(trans_mzm) ** 2)
        # plt.xlabel("Wavelength (nm)")
        # plt.ylabel("Amplitude")
        # plt.title("Transmission (TE)")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        #
        # # S21 (dB) vs Frequency (GHz)
        # plt.figure()
        # plt.plot(lambda_ona * 1e9, np.abs(gain_mzm))
        # plt.xlabel("Wavelength (nm)")
        # plt.ylabel("Gain (dB)")
        # plt.title("S21 (dB) vs Frequency (GHz)")
        # plt.legend()
        # plt.grid(True)
        # plt.show()


    def modulator_link(self,
                       v_bias_start=None,
                       v_bias_stop=None,
                       v_bias_n=None,
                       save_path='pin_vbias_results.pkl'):
        """
        Simple analog link simulation to graph the Vpi
        """
        # General root element
        root = "::Root Element"
        ic.setnamed(root, "time window", self.time_window)
        ic.setnamed(root, "sample rate", self.sample_rate)
        # ic.setnamed(root, "number of samples", self.n_samples)
        ic.setnamed(root, "output signal mode", "block")  # Block is giving errors
        ic.setnamed(root, "number of output signals", 1)
        ic.setnamed(root, "monitor data", "save to memory")

        # Add the laser
        ic.addelement("CW Laser")
        laser = "CW_Laser"
        ic.set("name", laser)
        ic.set("x position", 0)
        ic.set("y position", 0)
        ic.set("frequency", self.c / self.lambda_central)
        ic.set("power", self.power_laser)  # dBm for it to give me 10 dBm
        ic.set("enable RIN", True)
        ic.set("RIN", self.RIN)
        # ic.set("linewidth, 5 ") #MHz
        # ic.set("phase", 0) #rad
        ic.set("reference power", self.power_laser)

        # Add the Modulator
        ic.addelement("Mach-Zehnder Modulator")
        mzm1 = "MZM_RF"
        ic.set("name", mzm1)
        ic.set("x position", 200)
        ic.set("y position", 0)
        ic.set("modulator type", "balanced single drive")
        # ic.set("modulator type", "dual drive")
        # ic.set("modulator type", "unbalanced single drive")
        ic.set("dc bias source", "internal")  # internal because I do not need extra components
        ic.set("bias voltage 1", self.v_bias_tx)  # Why is it 1? Because is cuadrature and the it divides by 2, and then
        # Lumerical counts it as you would have a bias voltage in 2 arms, and that is why
        # you have to divide it by 2 again
        ic.set("pi dc voltage", self.v_pi_dc_tx)
        ic.set("pi rf voltage", self.v_pi_rf_tx)
        ic.set("extinction ratio", 100)  # dB
        ic.set("insertion loss", 5)  # dB
        ic.set("phase shift", 0)  # rad

        f0 = self.rf_freq1


        Vp = self.Vp  # peak voltage
        B = self.B  # sweep 50 MHz
        T = self.time_window  # 1 us chirp period, is this the same as the time window? -yes
        k = B / T  # [Hz/s] chirp slope
        t = np.linspace(0, T, 1000)  # 0.5 us for example
        t_mod = np.mod(t, T)  # modulo operation


        chirp_script = (
                f"f0 = {f0};" +
                f"B = {B};" +
                f"T = {T};" +
                f"k = B/T;" +
                f"t_mod = mod(TIME, T);" +
                "OUTPUT = " +
                f"{Vp}*sin(2*pi*(f0*t_mod + 0.5*k*t_mod^2));"
        )

        ic.addelement("Scripted Source")
        chirp1 = "chirp"
        ic.set("name", chirp1)
        ic.set("x position", -200)
        ic.set("y position", -200)
        ic.set('script', chirp_script)

        ic.addelement("Optical Power Meter")
        opwm1 = "opwm_1"
        ic.set("name", opwm1)
        ic.set("x position", 300)
        ic.set("y position", -200)
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        # Add the photodiode
        ic.addelement("PIN Photodetector")
        pd1 = "PIN_1"
        ic.set("name", pd1)
        ic.set("x position", 400)
        ic.set("y position", 0)
        ic.set("frequency at max power", 1)  # 0 meaning false
        # ic.set("frequency", c / lambda_central)
        ic.set("input parameter", "constant")
        ic.set("responsivity", 0.85)  # 0.85 A/W
        ic.set("dark current", 2.5e-8)  # A
        ic.set("enable power saturation", False)
        # ic.set("saturation power", self.sat_power)
        ic.set("enable thermal noise", True)
        ic.set("enable shot noise", False)
        ic.set("convert noise bins", False)
        ic.set("automatic seed", True)
        ic.set("thermal noise", self.thermal_noise)


        ic.connect(laser, "output", mzm1, "input")
        ic.connect(chirp1, "output", mzm1, "modulation 1")
        ic.connect(mzm1, "output", pd1, "input")
        ic.connect(mzm1, "output", opwm1, "input")


        ic.run()

        # Retrieve simulation results
        v_bias_values = np.linspace(v_bias_start, v_bias_stop, v_bias_n)

        results = {}

        for v1 in v_bias_values:
            ic.switchtodesign()

            # Change voltages from MZM
            ic.setnamed(mzm1, "bias voltage 1", v1)
            # ic.setnamed(mzm1, "pi dc voltage", v1)
            # ic.setnamed(mzm1, "pi rf voltage", v1)

            ic.run()

            # Retrieve the simulation results
            # ------------------------------------
            P_opt = ic.getresult(opwm1, "sum/power")

            entry = {   "Powermeter": {
                        "P_pd optical (dBm)": P_opt,
                        "V_bias_values (V)": v1}
            }

        with open(save_path, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


    def transceiver_simple(self,
                           R_in=None,
                           voltage=1,
                           rf_freq1=None,
                           wg_enabled=None):
        """
        Function that sets the architecture of the transmitter, target simulation and receiver.
        This transmitter does not include any specific technology
        In this specific case we have introduced waveguides in between the elements.

        This function simulates the target with an optical fiber of any length, situated
        Returns: Various graphs

        Simple simulation, without any loops

        """
        if R_in is None:
            R_in = self.impedance  # Default to 50 Ohms

        if wg_enabled is None:
            wg_enabled = self.wg_enabled  # Default False

        if rf_freq1 is None:
            rf_freq1 = self.rf_freq1  # Default False

        # General root element
        root = "::Root Element"
        # ic.setnamed(root, 'name', root)
        # Set up simulation to match time window and number of samples
        # The time windows should match because the INTERCONNECT simulation
        # will stop once the simulation time exceeds the time window.
        # The number of samples defines the INTERCONNECT time step, dt, by dt = time_window/(Nsamples+1).
        # The time steps do NOT have to match, although in this example they do. Indeed,
        # the time step of an external simulator can be variable
        ic.setnamed(root, "time window", self.time_window)
        ic.setnamed(root, "sample rate", self.sample_rate)
        # ic.setnamed(root, "number of samples", self.n_samples)
        ic.setnamed(root, "output signal mode", "block")  # Block is giving errors
        ic.setnamed(root, "number of output signals", 1)
        ic.setnamed(root, "monitor data", "save to memory")

        # Add the laser
        ic.addelement("CW Laser")
        laser = "CW_Laser"
        ic.set("name", laser)
        ic.set("x position", -600)
        ic.set("y position", 0)
        ic.set("frequency", self.c / self.lambda_central)
        ic.set("power", self.power_laser)  # dBm for it to give me 10 dBm
        ic.set("enable RIN", True)
        ic.set("RIN", self.RIN)
        # ic.set("linewidth, 5 ") #MHz
        # ic.set("phase", 0) #rad
        ic.set("reference power", self.power_laser)

        # Add the chirp signal

        # Chirp signal Sawtooth

        # t_mod is t wrapped into the interval [0, T]. Without mod, the instantaneous frequency keeps increasing
        # indefinitely: f(t) = f_0 + kt
        # With mod, it resets every chirp period:
        # f(t) = f_0 + k(t mod T)
        # This is the usual way to model a repeating FMCW chirp in radar.

        Vp = self.Vp  # peak voltage
        B = self.B  # sweep 50 MHz
        T = self.time_window  # 1 us chirp period, is this the same as the time window? -yes
        k = B / T  # [Hz/s] chirp slope
        t = np.linspace(0, T, 1000)  # 0.5 us for example
        t_mod = np.mod(t, T)  # modulo operation
        # signal_out = Vp * np.sin(2 * np.pi * (rf_freq1 * t_mod + 0.5 * k * t_mod ** 2))
        #
        # f_inst = rf_freq1 + k * t_mod # instantaneous frequency (analytic)  -> f_ins (t) = f0 + k * tm_mod
        #
        # # plot the waveform (as you had)
        # plt.figure()
        # plt.plot(t * 1e6, signal_out)
        # plt.xlabel("Time (µs)")
        # plt.ylabel("Amplitude (V)")
        # plt.title("Chirp waveform")
        # plt.grid(True)
        # plt.show()
        # plt.tight_layout()
        #
        # # plot instantaneous frequency
        # plt.figure()
        # plt.plot(t * 1e6, f_inst * 1e-9)
        # plt.xlabel("Time (µs)")
        # plt.ylabel("Instantaneous frequency (GHz)")
        # plt.title("FMCW: time vs instantaneous frequency")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        # # ic.set("Vp", Vp)
        # # ic.set("k", k)
        # # ic.set("rf_freq1", self.rf_freq1)
        # # ic.set("t_mod", t_mod)
        # # ic.set("T", T)
        f0 = rf_freq1
        chirp_script = (
                f"f0 = {f0};" +
                f"B = {B};" +
                f"T = {T};" +
                f"k = B/T;" +
                f"t_mod = mod(TIME, T);" +
                "OUTPUT = " +
                f"{Vp}*sin(2*pi*(f0*t_mod + 0.5*k*t_mod^2));"
        )

        # Triangle signal
        # It helps to extract both distance and Doppler (use a triangle r(t) among [-1, 1] scaled by B/2 as the deviation

        # # Vp = np.sqrt(0.25)  # peak voltage
        # B = 100e6  # sweep 100 MHz
        # tw = self.time_window
        # T = tw/2  # 1 us chirp period, is this the same as the time window? -yes
        # k = 1/T  # [Hz/s] chirp slope
        # t = np.linspace(0, T, 1000)  # 0.5 us for example
        # t_mod = np.mod(t, T)  # modulo operation
        #
        # chirp_script = (
        #         f"B = {B};" +
        #         f"tw = {tw};"
        #         f"T = {T};" +
        #         f"k = 1/{T};" +
        #         f"t_mod = mod(TIME, T);" +
        #         "OUTPUT = " +
        #         f"(TIME>T)*(1-k*(TIME-T))+(TIME<=T)*(k*TIME);"
        # )


        ic.addelement("Scripted Source")
        chirp1 = "chirp"
        ic.set("name", chirp1)
        ic.set("x position", -200)
        ic.set("y position", -200)
        ic.set('script', chirp_script)

        ic.addelement("Noise Source")
        noise1 = "NoiseIn"
        ic.set("name", noise1)
        ic.set("x position", -300)
        ic.set("y position", -600)
        ic.set("power spectral density", 1e-17)  # rad
        ic.set("enabled", False)

        # Add the sum of the signals
        ic.addelement("Fork 1xN")
        fork_adder = "Fork_sum_1"
        ic.set("name", fork_adder)
        ic.set("x position", 600)
        ic.set("y position", -300)
        ic.set("number of ports", 2)

        ic.addelement("Electrical Constant Multiplier")
        gain_elec = "GAIN_1"
        ic.set("name", gain_elec)
        ic.set("x position", 800)
        ic.set("y position", -300)
        ic.set("gain", 0.141421)  # Why 0.141421?

        ic.addelement("Spectrum Analyzer")
        esa_rf_in = "ESA_RF_in"
        ic.set("name", esa_rf_in)
        ic.set("x position", 800)
        ic.set("y position", -400)
        ic.set("resolution", "Gaussian function")
        ic.set("bandwidth", 100e6)
        ic.set("limit time range", 1)  # True
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        ic.addelement("Spectrum Analyzer")
        esa_rf_in_dBm_Hz = "ESA_RF_in_dBm/Hz"
        ic.set("name", esa_rf_in_dBm_Hz)
        ic.set("x position", 1000)
        ic.set("y position", -600)
        ic.set("power unit", "dBm/Hz")
        # ic.set("sensitivity", -180)  # dBm
        ic.set("resolution", "Gaussian function")
        ic.set("bandwidth", 100e6)
        ic.set("limit time range", 1)  # True
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        ic.addelement("Carrier Analyzer")
        ecn_1 = "ECN_1"
        ic.set("name", ecn_1)
        ic.set("x position", 1000)
        ic.set("y position", -500)
        ic.set("sensitivity", self.sensitivity_singleCarrier)  # in W -> -150 dBm
        ic.set("bandwidth", self.spectrum_bw)
        ic.set("interpolation offset", self.interpolation_offset)
        ic.set("limit time range", 1)  # True
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)
        ic.set("frequency carriers", "user defined")
        ic.set("carriers table", rf_freq1)

        ic.addelement("Electrical Adder")
        sum2 = "SUM_2"
        ic.set("name", sum2)
        ic.set("x position", 200)
        ic.set("y position", -200)
        ic.set("run diagnostic", 0)

        # Add waveguide
        ic.addelement("Straight Waveguide Unidirectional")
        wg1 = 'wg1'
        ic.set("name", wg1)
        ic.set("x position", -400)
        ic.set("y position", 0)
        ic.set("length", self.wg_length)
        ic.set("loss 1", self.wg_loss)
        ic.set("effective index 1", self.neff_te)
        ic.set("group index 1", self.ngroup_te)
        ic.set("enabled", wg_enabled)  # 1 or 0

        # Add the Modulator for the 2 RF signals
        ic.addelement("Mach-Zehnder Modulator")
        mzm1 = "MZM_RF"
        ic.set("name", mzm1)
        ic.set("x position", -200)
        ic.set("y position", 0)
        ic.set("modulator type", "balanced single drive")
        # ic.set("modulator type", "dual drive")
        # ic.set("modulator type", "unbalanced single drive")
        ic.set("dc bias source", "internal")  # internal because I do not need extra components
        ic.set("bias voltage 1", self.v_bias_tx)  # Why is it 1? Because is cuadrature and the it divides by 2, and then
        # Lumerical counts it as you would have a bias voltage in 2 arms, and that is why
        # you have to divide it by 2 again
        ic.set("pi dc voltage", self.v_pi_dc_tx)
        ic.set("pi rf voltage", self.v_pi_rf_tx)
        ic.set("extinction ratio", 100)  # dB
        ic.set("insertion loss", 5)  # dB
        ic.set("phase shift", 0)  # rad

        # Add waveguide
        ic.addelement("Straight Waveguide Unidirectional")
        wg2 = 'wg2'
        ic.set("name", wg2)
        ic.set("x position", 325)
        ic.set("y position", 0)
        ic.set("length", self.wg_length)
        ic.set("loss 1", self.wg_loss)
        ic.set("effective index 1", self.neff_te)
        ic.set("group index 1", self.ngroup_te)
        ic.set("enabled", wg_enabled)  # 1 or 0

        # Optical Amplifier
        ic.addelement("Optical Amplifier")
        gain1 = "AMP_1"
        ic.set("name", gain1)
        ic.set("x position", 500)
        ic.set("y position", 0)
        ic.set("gain", self.SOA_gain)
        ic.set("enable noise", True)
        ic.set("noise figure", 5)
        ic.set("enabled", True)

        # Add waveguide
        ic.addelement("Straight Waveguide Unidirectional")
        wg3 = 'wg3'
        ic.set("name", wg3)
        ic.set("x position", 700)
        ic.set("y position", 0)
        ic.set("length", self.wg_length)
        ic.set("loss 1", self.wg_loss)
        ic.set("effective index 1", self.neff_te)
        ic.set("group index 1", self.ngroup_te)
        ic.set("enabled", wg_enabled)  # 1 or 0

        # Add the Attenuator
        ic.addelement("Optical Attenuator")
        att_pd1 = "ATT_PD"
        ic.set("name", att_pd1)
        ic.set("x position", 900)
        ic.set("y position", 0)
        ic.set("attenuation", 21)  # dB

        # Add waveguide
        ic.addelement("Straight Waveguide Unidirectional")
        wg4 = 'wg4'
        ic.set("name", wg4)
        ic.set("x position", 1100)
        ic.set("y position", 0)
        ic.set("length", self.wg_length)
        ic.set("loss 1", self.wg_loss)
        ic.set("effective index 1", self.neff_te)
        ic.set("group index 1", self.ngroup_te)
        ic.set("enabled", wg_enabled)  # 1 or 0

        # Add the photodiode
        ic.addelement("PIN Photodetector")
        pd1 = "PIN_1"
        ic.set("name", pd1)
        ic.set("x position", 1300)
        ic.set("y position", 0)
        ic.set("frequency at max power", 1)  # 0 meaning false
        # ic.set("frequency", c / lambda_central)
        ic.set("input parameter", "constant")
        ic.set("responsivity", 0.85)  # 0.85 A/W
        ic.set("dark current", 2.5e-8)  # A
        ic.set("enable power saturation", False)
        # ic.set("saturation power", self.sat_power)
        ic.set("enable thermal noise", True)
        ic.set("enable shot noise", False)
        ic.set("convert noise bins", False)
        ic.set("automatic seed", True)
        ic.set("thermal noise", self.thermal_noise)

        # Add a Low-pass filter
        ic.addelement("BP Bessel Filter")
        bpf1 = 'BPF_1'
        ic.set("name", bpf1)
        ic.set("x position", 1600)
        ic.set("y position", 0)
        ic.set("frequency", self.rf_freq1)
        ic.set("enabled", False)

        # Compound element
        comp_1 = "PD_OUT_1"
        ic.addelement("PD_OUT_1")
        ic.set("name", comp_1)
        ic.set("x position", 1800)
        ic.set("y position", 0)

        #________________________________________________________
        # Measurement devices for transmitter
        # OSA y position
        osa_y = 200
        opw_y = 300

        # Power meter - RF_signal
        ic.addelement("Power Meter")
        pw1 = "PWM_1"
        ic.set("name", pw1)
        ic.set("x position", -0)
        ic.set("y position", -280)
        ic.set("input kind", "voltage")
        ic.set("impedance", self.impedance)
        ic.set("power unit", "dBm")
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        # Add the ESA for the chirp
        ic.addelement("Spectrum Analyzer")
        esa1_chirp = "ESA_chirp"
        ic.set("name", esa1_chirp)
        ic.set("x position", 0)
        ic.set("y position", -280-100)
        # ic.set("sensitivity", -100) #dBm
        ic.set("limit frequency range", 0)
        ic.set("limit time range", 1)
        ic.set("start time", 0)
        ic.set("stop time", 1e-8)
        ic.set("resolution", "rectangular function")
        ic.set("bandwidth", 1e6)
        ic.set("spectrogram", "average")


        # Oscilloscope SUM2
        ic.addelement("Oscilloscope")
        osc1 = "osc_1"
        ic.set("name", osc1)
        ic.set("x position", 200)
        ic.set("y position", -300)

        # Add the OSA for the laser
        ic.addelement("Optical Spectrum Analyzer")
        osa1_laser = "OSA_laser_1"
        ic.set("name", osa1_laser)
        ic.set("x position", -200)
        ic.set("y position", osa_y)
        # ic.set("sensitivity", -100) #dBm
        ic.set("limit frequency range", 0)
        ic.set("limit time range", 1)
        ic.set("start time", 0)
        ic.set("stop time", 1e-8)
        ic.set("plot kind", "wavelength")

        # Add the OSA for the MZM
        ic.addelement("Optical Spectrum Analyzer")
        osa1 = "OSA_MZM_1"
        ic.set("name", osa1)
        ic.set("x position", 0)
        ic.set("y position", osa_y)
        # ic.set("sensitivity", -100) #dBm
        ic.set("limit frequency range", 0)
        ic.set("limit time range", 1)
        ic.set("start time", 0)
        ic.set("stop time", 1e-8)
        ic.set("plot kind", "wavelength")

        # Optical power meter for the MZM-RF
        ic.addelement("Optical Power Meter")
        opw_mzm1 = "OPWM_SOAin"
        ic.set("name", opw_mzm1)
        ic.set("x position", 0)
        ic.set("y position", opw_y)
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        # Add the OSA - SOA
        ic.addelement("Optical Spectrum Analyzer")
        osa_soa1 = "OSA_AMP"
        ic.set("name", osa_soa1)
        ic.set("x position", 750)
        ic.set("y position", osa_y)
        ic.set("limit frequency range", 0)
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)
        ic.set("plot kind", "wavelength")

        # Optical power meter for the amplifier - SOA
        ic.addelement("Optical Power Meter")
        opw_soa1 = "PowerMeter_AMP1"
        ic.set("name", opw_soa1)
        ic.set("x position", 750)
        ic.set("y position", opw_y)

        # Add the OSA for Attenuator
        ic.addelement("Optical Spectrum Analyzer")
        osa_pd = "OSA_PD"
        ic.set("name", osa_pd)
        ic.set("x position", 1350)
        ic.set("y position", osa_y)
        ic.set("limit frequency range", 0)
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)
        ic.set("plot kind", "wavelength")

        # Optical power meter for the PD
        ic.addelement("Optical Power Meter")
        opw_pd = "OPWM_PD"
        ic.set("name", opw_pd)
        ic.set("x position", 1350)
        ic.set("y position", opw_y)
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        # Optical Oscilloscope - PIN_1
        ic.addelement("Optical Oscilloscope")
        osc_pd_opt = "OSC_PD_opt"
        ic.set("name", osc_pd_opt)
        ic.set("x position", 1350)
        ic.set("y position", opw_y + 100)


        # Add the Electrical Spectrum Analyzer - PIN_1
        ic.addelement("Spectrum Analyzer")
        rf_sa1 = "RF_SA1"
        ic.set("name", rf_sa1)
        ic.set("x position", 1600)
        ic.set("y position", osa_y)
        ic.set("limit frequency range", False)
        ic.set("remove dc", 0)
        ic.set("limit time range", 0)
        ic.set("resolution", "rectangular function")
        ic.set("bandwidth", 1e6)
        # ic.set("sensitivity", -130)
        ic.set("spectrogram", "average")

        # Power meter - PIN_1
        ic.addelement("Power Meter")
        pw_out_pd = "PWM_out_pd"
        ic.set("name", pw_out_pd)
        ic.set("x position", 1600)
        ic.set("y position", opw_y)
        ic.set("input kind", "voltage")
        ic.set("impedance", 1)
        ic.set("power unit", "dBm")
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        # Electrical Oscilloscope - PIN_1
        ic.addelement("Oscilloscope")
        osc_pd_elec = "OSC_PD_elec"
        ic.set("name", osc_pd_elec)
        ic.set("x position", 1600)
        ic.set("y position", opw_y + 100)

        # Elements for the PD matching network
        # Add the Electrical Spectrum Analyzer - After the matching PD network
        ic.addelement("Spectrum Analyzer")
        rf_out = "ESA_output"
        ic.set("name", rf_out)
        ic.set("x position", 2000)
        ic.set("y position", 400)
        ic.set("limit frequency range", False)
        ic.set("remove dc", 0)
        ic.set("limit time range", 0)
        ic.set("resolution", "rectangular function")
        ic.set("bandwidth", 1e6)
        # ic.set("sensitivity", -130)

        # Power meter
        ic.addelement("Power Meter")
        pw_out = "PWM_out"
        ic.set("name", pw_out)
        ic.set("x position", 2000)
        ic.set("y position", 200)
        ic.set("input kind", "voltage")
        ic.set("impedance", 1)
        ic.set("power unit", "dBm")
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        ic.addelement("Spectrum Analyzer")
        esa_rf_out_dBm_Hz = "ESA_RF_out_dBm/Hz"
        ic.set("name", esa_rf_out_dBm_Hz)
        ic.set("x position", 2000)
        ic.set("y position", 0)
        ic.set("power unit", "dBm/Hz")
        # ic.set("sensitivity", -180)  # dBm
        ic.set("resolution", "Gaussian function")
        ic.set("bandwidth", 200e6)
        ic.set("limit time range", 1)  # True
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        carriers = np.array([self.rf_freq1, self.rf_freq2])
        ic.addelement("Carrier Analyzer")
        ecn_sinc = "ECN_SingleCarrier"
        ic.set("name", ecn_sinc)
        ic.set("x position", 2000)
        ic.set("y position", -200)
        # ic.set("sensitivity", -150) # dBm
        ic.set("bandwidth", 100e6)
        ic.set("interpolation offset", 200e6)
        ic.set("limit time range", 1)  # True
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)
        ic.set("frequency carriers", "user defined")
        ic.set("carriers table", rf_freq1)
        ic.set("enabled", False)

        imd3_1 = 2 * self.rf_freq1 - self.rf_freq2
        imd3_2 = 2 * self.rf_freq2 - self.rf_freq1
        imd3 = np.array([imd3_1, imd3_2])

        ic.addelement("Carrier Analyzer")
        ecn_imd3 = "ECN_IMD3"
        ic.set("name", ecn_imd3)
        ic.set("x position", 2000)
        ic.set("y position", -400)
        ic.set("sensitivity", self.sensitivity_imd3)  # in W -> -150 dBm
        ic.set("bandwidth", self.spectrum_bw)
        ic.set("interpolation offset", self.interpolation_offset)
        ic.set("limit time range", 1)  # True
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)
        ic.set("frequency carriers", "user defined")
        ic.set("carriers table", imd3)
        ic.set("enabled", False)

        # Radar target emulator --------------------------------------------------
        # ------------------------------------------------------------------------

        y_target = 500
        x_target = 500

        # MMI
        ic.addelement("Waveguide Splitter")
        mmi1x2_1 = "MMI_1"
        ic.set("name", mmi1x2_1)
        ic.set("x position", x_target + 50)
        ic.set("y position", y_target)
        ic.set("insertion loss", 0)
        ic.set("coupling coefficient 1", 0.5)

        # In an ideal case, I should introduce the edge coupler here, that will be connected
        # to the fiber delay

        # Fiber delay
        ic.addelement("Optical Linear Fiber Unidirectional")
        fiber1 = "FIBER_1"
        ic.set("name", fiber1)
        ic.set("x position", x_target + 300)
        ic.set("y position", y_target)
        ic.set("length", self.length_fiber_delay) # in meters
        # ic.set("reference frequency", 1550 ) #nm
        ic.set("enabled", False)

        # Add the photodiode
        ic.addelement("PIN Photodetector")
        pd_target = "PIN_target"
        ic.set("name", pd_target)
        ic.set("x position", x_target + 500)
        ic.set("y position", y_target)
        ic.set("frequency at max power", 1)  # 0 meaning false
        # ic.set("frequency", c / lambda_central)
        ic.set("input parameter", "constant")
        ic.set("responsivity", 0.85)  # 0.85 A/W
        ic.set("dark current", 2.5e-8)  # A
        ic.set("enable power saturation", False)
        # ic.set("saturation power", self.sat_power)  # W
        ic.set("enable thermal noise", True)
        ic.set("enable shot noise", False)
        ic.set("convert noise bins", False)
        ic.set("automatic seed", True)
        ic.set("thermal noise", self.thermal_noise)

        # Add a Low-pass filter
        ic.addelement("BP Bessel Filter")
        bpf_target = 'BPF_target'
        ic.set("name", bpf_target)
        ic.set("x position", x_target + 700)
        ic.set("y position", y_target)
        ic.set("frequency", rf_freq1)
        ic.set("enabled", False)

        # Electrical amplifier
        ic.addelement("Electrical Amplifier")
        gain_target = "GAIN_target"
        ic.set("name", gain_target)
        ic.set("x position", x_target + 900)
        ic.set("y position", y_target)
        ic.set("gain", self.gain_target)
        ic.set("noise parameter", "disable")  # before , "output"
        # ic.set("noise spectral density", self.noise_sd)  # 1e-17 W/Hz
        ic.set("one db compression parameter", "disable")
        ic.set("saturation parameter", "disable")
        ic.set("third order intercept parameter", "disable")

        # Electrical time delay
        ic.addelement("Electrical Delay")
        t_delay = "DELAY_t1"
        ic.set("name", t_delay)
        ic.set("x position", x_target + 1100)
        ic.set("y position", y_target + 10)
        ic.set("delay", self.tau)

        # Variable Attenuator
        ic.addelement("Electrical Attenuator")
        att_elec = "VA_target"
        ic.set("name", att_elec)
        ic.set("x position", x_target + 1300)
        ic.set("y position", y_target)
        ic.set("attenuation", 0)  #dB
        ic.set("enabled", False)

        # ----------------------------------------------
        # Measurement devices for target

        # OSA_fiber
        ic.addelement("Optical Spectrum Analyzer")
        osa_target = "OSA_fiber1_rx"
        ic.set("name", osa_target)
        ic.set("x position", 500 + x_target)
        ic.set("y position", osa_y + y_target)
        # ic.set("sensitivity", -100) #dBm
        ic.set("limit frequency range", 0)
        ic.set("limit time range", 1)
        ic.set("start time", 0)
        ic.set("stop time", 1e-8)
        ic.set("plot kind", "wavelength")

        # Optical power meter for the fiber
        ic.addelement("Optical Power Meter")
        opw_target = "OPWM_fiber"
        ic.set("name", opw_target)
        ic.set("x position", 500 + x_target)
        ic.set("y position", opw_y + y_target)
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        # Optical power meter for the fiber
        ic.addelement("Optical Oscilloscope")
        osc1_target_opt = "OSC1_opt_target"
        ic.set("name", osc1_target_opt)
        ic.set("x position", 500 + x_target)
        ic.set("y position", opw_y + y_target + 100)

        # Electrical oscilloscope for the PD output
        ic.addelement("Oscilloscope")
        osc1_target_elec = "OSC1_target"
        ic.set("name", osc1_target_elec)
        ic.set("x position", 700 + x_target)
        ic.set("y position", opw_y + y_target + 100)


        # Optical power meter for the output of the BPF_target
        ic.addelement("Power Meter")
        pwm_target_elec_bpf = "PWM_target_BPF"
        ic.set("name", pwm_target_elec_bpf)
        ic.set("x position", 900 + x_target)
        ic.set("y position", opw_y + y_target)

        # Add the ESA for target
        ic.addelement("Spectrum Analyzer")
        rf_sa_target = "ESA_output_target"
        ic.set("name", rf_sa_target)
        ic.set("x position", 1250 + x_target)
        ic.set("y position", osa_y + y_target)
        ic.set("limit frequency range", False)
        ic.set("remove dc", 0)
        ic.set("limit time range", 0)
        ic.set("resolution", "rectangular function")
        ic.set("bandwidth", 1e6)
        # ic.set("sensitivity", -130)

        # Power meter
        ic.addelement("Power Meter")
        pw_out_target = "PWM_out_target"
        ic.set("name", pw_out_target)
        ic.set("x position", 1250 + x_target)
        ic.set("y position", opw_y + y_target)
        ic.set("input kind", "voltage")
        ic.set("impedance", 1)
        ic.set("power unit", "dBm")
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        # Power meter - RF_signal
        ic.addelement("Power Meter")
        pw1_rx = "PWM1_rx"
        ic.set("name", pw1_rx)
        ic.set("x position", 2000)
        ic.set("y position", opw_y + y_target)
        ic.set("input kind", "voltage")
        ic.set("impedance", self.impedance)
        ic.set("power unit", "dBm")
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)
        # ic.rotateelement(pw1_rx)

        # Electrical Oscilloscope target
        ic.addelement("Oscilloscope")
        osc1_target_att = "OSC1_target_att"
        ic.set("name", osc1_target_att)
        ic.set("x position", 2000)
        ic.set("y position", osa_y + y_target)

        # # Fork to connect tx to target
        # ic.addelement("Fork 1xN")
        # fork_target = "Fork_target"
        # ic.set("name", fork_target)
        # ic.set("x position", x_target - 100)
        # ic.set("y position", y_target + 300)
        # ic.set("number of ports", 3)
        # ic.rotateelement(fork_target)
        # ic.rotateelement(fork_target)

        # --------------------------------------------------------------------
        # RECEIVER -----------------------------------------------------------

        x_ref_pos = 0
        y_ref_pos = 1200
        # Add the laser
        # ic.addelement("CW Laser")
        # laser_rx = "CW_Laser_rx"
        # ic.set("name", laser_rx)
        # ic.set("x position", -200 + x_ref_pos)
        # ic.set("y position", 0 + y_ref_pos)
        # ic.set("frequency", self.c / self.lambda_central)
        # ic.set("power", self.power_laser)  # dBm for it to give me 10 dBm
        # ic.set("enable RIN", True)
        # ic.set("RIN", self.RIN)
        # # ic.set("linewidth, 5 ") #MHz
        # # ic.set("phase", 0) #rad
        # ic.set("reference power", self.power_laser)

        # MMI
        ic.addelement("Waveguide Splitter")
        mmi1x2_2 = "MMI_2"
        ic.set("name", mmi1x2_2)
        ic.set("x position",  -200 + x_ref_pos)
        ic.set("y position", 0 + y_ref_pos - 200)
        ic.set("insertion loss", 0)
        ic.set("coupling coefficient 1", 0.5)

        # After the laser ref branch from transmitter
        # Add waveguide
        ic.addelement("Straight Waveguide Unidirectional")
        wg1_rx = 'wg1_rx'
        ic.set("name", wg1_rx)
        ic.set("x position", 0 + x_ref_pos)
        ic.set("y position", 0 + y_ref_pos)
        ic.set("length", self.wg_length)
        ic.set("loss 1", self.wg_loss)
        ic.set("effective index 1", self.neff_te)
        ic.set("group index 1", self.ngroup_te)
        ic.set("enabled", wg_enabled)  # 1 or 0

        # Add the Modulator for the 2 RF signals
        ic.addelement("Mach-Zehnder Modulator")
        mzm1_rx = "MZM_RF_rx"
        ic.set("name", mzm1_rx)
        ic.set("x position", 150 + x_ref_pos)
        ic.set("y position", 0 + y_ref_pos)
        ic.set("modulator type", "balanced single drive")
        # ic.set("modulator type", "dual drive")
        # ic.set("modulator type", "unbalanced single drive")
        ic.set("dc bias source", "internal")  # internal because I do not need extra components
        ic.set("bias voltage 1", self.v_bias_rx)  # Why is it 1? Because is quadrature and then it divides by 2, and then
        # Lumerical counts it as you would have a bias voltage in 2 arms, and that is why
        # you have to divide it by 2 again
        ic.set("pi dc voltage", self.v_pi_dc_rx)
        ic.set("pi rf voltage", self.v_pi_rf_rx)
        ic.set("extinction ratio", 100)  # dB
        ic.set("insertion loss", 5)  # dB
        ic.set("phase shift", 0)  # rad

        # Add waveguide
        ic.addelement("Straight Waveguide Unidirectional")
        wg2_rx = 'wg2_rx'
        ic.set("name", wg2_rx)
        ic.set("x position", 325 + x_ref_pos)
        ic.set("y position", 0 + y_ref_pos)
        ic.set("length", self.wg_length)
        ic.set("loss 1", self.wg_loss)
        ic.set("effective index 1", self.neff_te)
        ic.set("group index 1", self.ngroup_te)
        ic.set("enabled", wg_enabled)  # 1 or 0

        # Optical Amplifier
        ic.addelement("Optical Amplifier")
        gain1_rx = "AMP_1_rx"
        ic.set("name", gain1_rx)
        ic.set("x position", 500 + x_ref_pos)
        ic.set("y position", 0 + y_ref_pos)
        ic.set("gain", self.SOA_gain_rx)
        ic.set("enable noise", True)
        ic.set("noise figure", 5)
        ic.set("enabled", True)

        # Add waveguide
        ic.addelement("Straight Waveguide Unidirectional")
        wg3_rx = 'wg3_rx'
        ic.set("name", wg3_rx)
        ic.set("x position", 700 + x_ref_pos)
        ic.set("y position", 0 + y_ref_pos)
        ic.set("length", self.wg_length)
        ic.set("loss 1", self.wg_loss)
        ic.set("effective index 1", self.neff_te)
        ic.set("group index 1", self.ngroup_te)
        ic.set("enabled", wg_enabled)  # 1 or 0

        # Add fiber
        # ic.addelement("Optical Linear Fiber")
        # fiber1 = "FIB_1"
        # ic.set("name", fiber1)
        # ic.set("x position", 900 + x_ref_pos)
        # ic.set("y position", 0 + y_ref_pos)
        # ic.set("configuration", "unidirectional")
        # ic.set("reference frequency", 1550)

        # Second part of the receiver

        # Mach Zehnder interferometer
        # ic.addelement("Mach Zehnder Interferometer")
        # # for mode 1
        # mzi1_rx = 'MZI1_rx'
        # ic.set("name", mzi1_rx)
        # ic.set("x position", 900 + x_ref_pos)
        # ic.set("y position", 0 + y_ref_pos)
        # ic.set("loss 1", self.mmi_loss)
        # ic.set("effective index 1", self.neff_te)
        # ic.set("coupling coefficient 1 1", 0.5)
        # ic.set("frequency", self.c / self.lambda_central)
        # ic.set("length 1", 50e-6 + self.delta_l)
        # ic.set("length 2", 50e-6)
        # ic.set("enabled", False)

        # Add waveguide
        ic.addelement("Straight Waveguide Unidirectional")
        wg4_rx = 'wg4_rx'
        ic.set("name", wg4_rx)
        ic.set("x position", 1100 + x_ref_pos)
        ic.set("y position", 0 + y_ref_pos)
        ic.set("length", self.wg_length)
        ic.set("loss 1", self.wg_loss)
        ic.set("effective index 1", self.neff_te)
        ic.set("group index 1", self.ngroup_te)
        ic.set("enabled", wg_enabled)  # 1 or 0

        # Add the Attenuator
        ic.addelement("Optical Attenuator")
        att_pd1_rx = "ATT_PD_rx"
        ic.set("name", att_pd1_rx)
        ic.set("x position", 1300 + x_ref_pos)
        ic.set("y position", 0 + y_ref_pos)
        ic.set("attenuation", 21)  # dB
        ic.set("configuration", 'bidirectional')
        ic.set("enabled", False)

        # Add waveguide
        ic.addelement("Straight Waveguide Unidirectional")
        wg5_rx = 'wg5_rx'
        ic.set("name", wg5_rx)
        ic.set("x position", 1500 + x_ref_pos)
        ic.set("y position", 0 + y_ref_pos)
        ic.set("length", self.wg_length)
        ic.set("loss 1", self.wg_loss)
        ic.set("effective index 1", self.neff_te)
        ic.set("group index 1", self.ngroup_te)
        ic.set("enabled", wg_enabled)  # 1 or 0

        # Photodiode
        ic.addelement("PIN Photodetector")
        pd1_rx = "PIN1_rx"
        ic.set("name", pd1_rx)
        ic.set("x position", 1700 + x_ref_pos)
        ic.set("y position", 0 + y_ref_pos)
        ic.set("frequency at max power", 1)  # 0 meaning false
        # ic.set("frequency", c / lambda_central)
        ic.set("input parameter", "constant")
        ic.set("responsivity", 0.85)  # 0.85 A/W
        ic.set("dark current", 2.5e-8)  # A
        ic.set("enable power saturation", False)
        # ic.set("saturation power", self.sat_power)  # W
        ic.set("enable thermal noise", True)
        ic.set("enable shot noise", False)
        ic.set("convert noise bins", False)
        ic.set("automatic seed", True)
        ic.set("thermal noise", self.thermal_noise)

        # Add a Low-pass filter
        ic.addelement("BP Bessel Filter")
        bpf1_rx = 'BPF1_rx'
        ic.set("name", bpf1_rx)
        ic.set("x position", 1900 + x_ref_pos)
        ic.set("y position", 0 + y_ref_pos)
        ic.set("frequency", self.rf_freq1)
        ic.set("enabled", False)

        # Compound element
        comp_1_rx = "PD_OUT_1_rx"
        ic.addelement('PD_OUT_1')
        ic.set("name", comp_1_rx)
        ic.set("x position", 2100 + x_ref_pos)
        ic.set("y position", 0 + y_ref_pos)

        # ---------------------------------------------------------------------------
        # Measurement devices
        # OSA y position
        osa_y = 200
        opw_y = 300

        # OSA MZM
        ic.addelement("Optical Spectrum Analyzer")
        osa1_rx = "OSA_MZM1_rx"
        ic.set("name", osa1_rx)
        ic.set("x position", 350 + x_ref_pos)
        ic.set("y position", osa_y + y_ref_pos)
        # ic.set("sensitivity", -100) #dBm
        ic.set("limit frequency range", 0)
        ic.set("limit time range", 1)
        ic.set("start time", 0)
        ic.set("stop time", 1e-8)
        ic.set("plot kind", "wavelength")

        # Optical power meter for the MZM-RF
        ic.addelement("Optical Power Meter")
        opw_mzm1_rx = "PowerMeter_SOAin_rx"
        ic.set("name", opw_mzm1_rx)
        ic.set("x position", 350 + x_ref_pos)
        ic.set("y position", opw_y + y_ref_pos)
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        # Add the OSA for SOA
        ic.addelement("Optical Spectrum Analyzer")
        osa_soa1_rx = "OSA_AMP_rx"
        ic.set("name", osa_soa1_rx)
        ic.set("x position", 750 + x_ref_pos)
        ic.set("y position", osa_y + y_ref_pos)
        ic.set("limit frequency range", 0)
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)
        ic.set("plot kind", "wavelength")

        # Optical power meter for SOA
        ic.addelement("Optical Power Meter")
        opw_soa1_rx = "PowerMeter_AMP1_rx"
        ic.set("name", opw_soa1_rx)
        ic.set("x position", 750 + x_ref_pos)
        ic.set("y position", opw_y + y_ref_pos)

        # # OSA MZI_1
        # ic.addelement("Optical Spectrum Analyzer")
        # osa_mzi_rx = "OSA_MZI_rx"
        # ic.set("name", osa_mzi_rx)
        # ic.set("x position", 1100 + x_ref_pos)
        # ic.set("y position", osa_y + y_ref_pos)
        # # ic.set("sensitivity", -100) #dBm
        # ic.set("limit frequency range", 0)
        # ic.set("limit time range", 1)
        # ic.set("start time", 0)
        # ic.set("stop time", 1e-8)
        # ic.set("plot kind", "wavelength")
        #
        # # Optical power meter for the MZI_1
        # ic.addelement("Optical Power Meter")
        # opw_mzi_rx = "PWM_MZI_rx"
        # ic.set("name", opw_mzi_rx)
        # ic.set("x position", 1100 + x_ref_pos)
        # ic.set("y position", opw_y + y_ref_pos)
        # ic.set("limit time range", 1)
        # ic.set("start time", 1e-8)
        # ic.set("stop time", 1)

        # Optical power meter for before the PD
        ic.addelement("Optical Power Meter")
        opw_pd_rx = "OPWM_PD_rx"
        ic.set("name", opw_pd_rx)
        ic.set("x position", 1650 + x_ref_pos)
        ic.set("y position", opw_y + y_ref_pos)
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        # Add the OSA for PD
        ic.addelement("Optical Spectrum Analyzer")
        osa_pd_rx = "OSA_PD_rx"
        ic.set("name", osa_pd_rx)
        ic.set("x position", 1650 + x_ref_pos)
        ic.set("y position", osa_y + y_ref_pos)
        ic.set("limit frequency range", 0)
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)
        ic.set("plot kind", "wavelength")

        # Add the Electrical Spectrum Analyzer - PIN_1
        ic.addelement("Spectrum Analyzer")
        rf_sa1_rx = "RF_SA1_rx"
        ic.set("name", rf_sa1_rx)
        ic.set("x position", 1850 + x_ref_pos)
        ic.set("y position", osa_y + y_ref_pos)
        ic.set("limit frequency range", False)
        ic.set("remove dc", 0)
        ic.set("limit time range", 0)
        ic.set("resolution", "rectangular function")
        ic.set("bandwidth", 1e6)
        # ic.set("sensitivity", -130)

        # Electrical Power meter - PIN_1
        ic.addelement("Power Meter")
        pw_out_pd_rx = "PWM_out_pd_rx"
        ic.set("name", pw_out_pd_rx)
        ic.set("x position", 1850 + x_ref_pos)
        ic.set("y position", opw_y + y_ref_pos)
        ic.set("input kind", "voltage")
        ic.set("impedance", 1)
        ic.set("power unit", "dBm")
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        ic.addelement("Oscilloscope")
        osc_pd_elec_rx = "OSC_PD_elec_rx"
        ic.set("name", osc_pd_elec_rx)
        ic.set("x position", 1850 + x_ref_pos)
        ic.set("y position", opw_y + y_ref_pos + 100)

        # Elements for the PD matching network
        # Add the Electrical Spectrum Analyzer - After the matching PD network
        ic.addelement("Spectrum Analyzer")
        rf_out_rx = "ESA_output_rx"
        ic.set("name", rf_out_rx)
        ic.set("x position", 2300 + x_ref_pos)
        ic.set("y position", 400 + y_ref_pos)
        ic.set("limit frequency range", False)
        ic.set("remove dc", 0)
        ic.set("limit time range", 0)
        ic.set("resolution", "rectangular function")
        ic.set("bandwidth", 1e6)
        # ic.set("sensitivity", -130)

        # Power meter
        ic.addelement("Power Meter")
        pw_out_rx = "PWM_out_rx"
        ic.set("name", pw_out_rx)
        ic.set("x position", 2300 + x_ref_pos)
        ic.set("y position", 200 + y_ref_pos)
        ic.set("input kind", "voltage")
        ic.set("impedance", 1)
        ic.set("power unit", "dBm")
        ic.set("limit time range", 1)
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        ic.addelement("Spectrum Analyzer")
        esa_rf_out_dBm_Hz_rx = "ESA_RF_out_dBm/Hz_rx"
        ic.set("name", esa_rf_out_dBm_Hz_rx)
        ic.set("x position", 2300 + x_ref_pos)
        ic.set("y position", 0 + y_ref_pos)
        ic.set("power unit", "dBm/Hz")
        # ic.set("sensitivity", -180)  # dBm
        ic.set("resolution", "Gaussian function")
        ic.set("bandwidth", 200e6)
        ic.set("limit time range", 1)  # True
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)

        carriers = np.array([self.rf_freq1, self.rf_freq2])
        ic.addelement("Carrier Analyzer")
        ecn_sinc_rx = "ECN_SingleCarrier_rx"
        ic.set("name", ecn_sinc_rx)
        ic.set("x position", 2300 + x_ref_pos)
        ic.set("y position", -200 + y_ref_pos)
        # ic.set("sensitivity", -150) # dBm
        ic.set("bandwidth", 100e6)
        ic.set("interpolation offset", 200e6)
        ic.set("limit time range", 1)  # True
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)
        ic.set("frequency carriers", "user defined")
        ic.set("carriers table", carriers)

        imd3_1 = 2 * self.rf_freq1 - self.rf_freq2
        imd3_2 = 2 * self.rf_freq2 - self.rf_freq1
        imd3 = np.array([imd3_1, imd3_2])

        ic.addelement("Carrier Analyzer")
        ecn_imd3_rx = "ECN_IMD3_rx"
        ic.set("name", ecn_imd3_rx)
        ic.set("x position", 2300 + x_ref_pos)
        ic.set("y position", -400 + y_ref_pos)
        # ic.set("sensitivity", -150) # dBm
        ic.set("bandwidth", 100e6)
        ic.set("interpolation offset", 200e6)
        ic.set("limit time range", 1)  # True
        ic.set("start time", 1e-8)
        ic.set("stop time", 1)
        ic.set("frequency carriers", "user defined")
        ic.set("carriers table", imd3)

        # Connect the elements
        # Transmitter ----------------------------------------------------
        # Laser
        ic.connect(osa1_laser, "input", laser, "output")
        # RF signals & noise
        ic.connect(chirp1, "output", pw1, "input")
        ic.connect(esa1_chirp, "input", chirp1, "output")

        # ESA in
        ic.connect(fork_adder, "output 2", mzm1, "modulation 1")
        ic.connect(fork_adder, "input", sum2, "output")

        # Meas AMP
        ic.connect(osa_soa1, "input", gain1, "output")
        ic.connect(opw_soa1, "input", gain1, "output")

        # MZM1 Connections
        ic.connect(laser, "output", wg1, "input")
        ic.connect(wg1, "output", mzm1, "input")
        # ic.connect(mzm1, "output", wg2, "input")

        ic.connect(mzm1, "output", opw_mzm1, "input")
        ic.connect(mzm1, "output", osa1, "input")

        # Opt amplifier
        ic.connect(wg2, "output", gain1, "input")
        # ic.connect(gain1, "output", wg3, "input")

        # RF connections with adder
        # Rf signal
        ic.connect(noise1, "output", sum2, "input 1")
        ic.connect(chirp1, 'output', sum2, 'input 2')

        # Adder 2 signals
        ic.connect(sum2, "output", osc1, "input")
        ic.connect(gain_elec, "output", ecn_1, "input")

        # Connections pd
        ic.connect(wg4, 'output', osa_pd, "input")
        ic.connect(pd1, "output", pw_out_pd, "input")
        ic.connect(wg3, "output", att_pd1, "port 1")
        ic.connect(att_pd1, "port 2", wg4, "input")
        ic.connect(wg4, "output", pd1, "input")
        ic.connect(osc_pd_opt, "input", wg4, "output")
        ic.connect(osc_pd_elec, "input", pd1, "output")

        # PD to filter and compound
        ic.connect(pd1, "output", bpf1, "input")
        ic.connect(bpf1, "output", comp_1, "port 1")
        ic.connect(comp_1, "port 2", rf_out, "input")
        ic.connect(comp_1, "port 2", pw_out, "input")
        ic.connect(pd1, "output", rf_sa1, "input")
        ic.connect(opw_pd, "input", wg4, "output")
        ic.connect(ecn_sinc, "input", comp_1, "port 2")
        ic.connect(ecn_imd3, "input", comp_1, "port 2")
        ic.connect(esa_rf_out_dBm_Hz, "input", comp_1, "port 2")

        # Meas instruments to in rf signal
        ic.connect(gain_elec, "input", fork_adder, "output 1")
        ic.connect(gain_elec, "output", esa_rf_in_dBm_Hz, "input")
        ic.connect(gain_elec, "output", esa_rf_in, "input")

        # Connection to target emulator ---------------------------------
        # To MMI
        ic.connect(gain1, "output", mmi1x2_1, "input")
        ic.connect(mmi1x2_1, "output 1", wg3, "input")

        ic.connect(fiber1, 'input', mmi1x2_1, 'output 2')
        ic.connect(fiber1, 'output', pd_target, 'input')
        ic.connect(pd_target, 'output', bpf_target, 'input')
        ic.connect(bpf_target, 'output', gain_target, 'input')
        ic.connect(gain_target, 'output', t_delay, 'input')
        ic.connect(t_delay, 'output', att_elec, 'input')

        ic.connect(osa_target, 'input', fiber1, 'output')
        ic.connect(opw_target, 'input', fiber1, 'output')
        ic.connect(rf_sa_target, 'input', t_delay, 'output')
        ic.connect(pw_out_target, 'input', t_delay, 'output')

        ic.connect(osc1_target_opt, "input", fiber1, "output")
        ic.connect(pwm_target_elec_bpf, "input", bpf_target, "output")
        ic.connect(osc1_target_elec, "input", pd_target, "output")

        # ic.connect(att_elec, 'output', fork_target, 'input')
        # ic.connect(fork_target, 'output 2', pw1_rx, 'input')
        # ic.connect(fork_target, 'output 1', mzm1_rx, 'modulation 1')
        # ic.connect(fork_target, 'output 3', osc1_rx, 'input')
        ic.connect(att_elec, 'output', mzm1_rx, 'modulation 1')
        ic.connect(att_elec, 'output', pw1_rx, 'input')
        ic.connect(att_elec, 'output', osc1_target_att, 'input')

        # Receiver ---------------------------------------------------------
        # Meas AMP - SOA
        ic.connect(osa_soa1_rx, "input", gain1_rx, "output")
        ic.connect(opw_soa1_rx, "input", gain1_rx, "output")

        # MZM1 Connections - MMI 2
        ic.connect(mmi1x2_2, "input", mzm1, "output")
        ic.connect(mmi1x2_2, "output 1", wg2, "input")
        ic.connect(mmi1x2_2, "output 2", wg1_rx, "input")
        ic.connect(wg1_rx, "output", mzm1_rx, "input")
        ic.connect(mzm1_rx, "output", wg2_rx, "input")

        ic.connect(mzm1_rx, "output", opw_mzm1_rx, "input")
        ic.connect(mzm1_rx, "output", osa1_rx, "input")

        # Opt amplifier
        ic.connect(wg2_rx, "output", gain1_rx, "input")
        ic.connect(gain1_rx, "output", wg3_rx, "input")

        # MZI

        ic.connect(wg3_rx, "output", wg4_rx, "input")
        # ic.connect(wg3_rx, 'output', mzi1_rx, 'port 1')
        # ic.connect(wg4_rx, "input", mzi1_rx, "port 4")
        # ic.connect(mzi1_rx, 'port 4', osa1_mzi_rx, 'input')
        # ic.connect(mzi1_rx, 'port 4', opw_mzi_1_rx, 'input')

        # Connections attenuator
        ic.connect(wg4_rx, 'output', att_pd1_rx, "port 1")
        ic.connect(wg5_rx, 'input', att_pd1_rx, "port 2")

        # Connections pd
        ic.connect(pd1_rx, "output", pw_out_pd_rx, "input")
        ic.connect(wg5_rx, "output", pd1_rx, "input")
        ic.connect(opw_pd_rx, "input", wg5_rx, "output")
        ic.connect(wg5_rx, 'output', osa_pd_rx, "input")
        ic.connect(osc_pd_elec_rx, "input", pd1_rx, "output")

        # PD to filter and compound
        ic.connect(pd1_rx, "output", bpf1_rx, "input")
        ic.connect(bpf1_rx, "output", comp_1_rx, "port 1")
        ic.connect(comp_1_rx, "port 2", rf_out_rx, "input")
        ic.connect(comp_1_rx, "port 2", pw_out_rx, "input")
        ic.connect(pd1_rx, "output", rf_sa1_rx, "input")
        ic.connect(ecn_sinc_rx, "input", comp_1_rx, "port 2")
        ic.connect(ecn_imd3_rx, "input", comp_1_rx, "port 2")
        ic.connect(esa_rf_out_dBm_Hz_rx, "input", comp_1_rx, "port 2")

        # Meas instruments to in rf signal
        # ic.connect(gain_elec, "input", fork_adder, "output 1")
        # ic.connect(gain_elec, "output", esa_rf_in_dBm_Hz, "input")
        # ic.connect(gain_elec, "output", esa_rf_in, "input")
        # ic.connect(gain_elec, "output", ecn_1, "input")

        # Run the simulation
        ic.run()

        # ----------------------------------------------------------------------
        # # Transmitter
        # # Retrieve the simulation results
        # Laser
        P_laser = ic.getresult(osa1_laser, "sum/signal" )
        p1_laser = P_laser.get("power (dBm)")
        freq_laser = P_laser.get('wavelength')

        # ESA_Chirp
        P_chirp_sp = ic.getresult(esa1_chirp, "spectrogram")
        p_chirp_in = P_chirp_sp.get("power (dBm)")
        freq_rf_chirp_in = P_chirp_sp.get("frequency")

        P_chirp_sig = ic.getresult(esa1_chirp, "spectrogram")
        p_chirp_sig = P_chirp_sig.get("power (dBm)")
        freq_rf_sig = P_chirp_sig.get("frequency")

        # Oscilloscope osc_1
        osc1_amp = ic.getresult(osc1, "signal")
        osc1_amp1 = osc1_amp.get("amplitude (a.u.)")
        osc1_t = osc1_amp.get("time")

        # # Electrical Spectrum from beat of the rf frequencies
        # # ESA_RF_in
        # P_rf_in = ic.getresult(esa_rf_in, "signal")
        # p1_rf_in = P_rf_in.get("power (dBm)")
        # freq_rf_in = P_rf_in.get('frequency')
        #
        # # ESA_RF_in_dBm/Hz
        # P_rf_in_c = ic.getresult(esa_rf_in_dBm_Hz, "signal")
        # p1_rf_in_c = P_rf_in_c.get("power (dBm/Hz)")
        # freq_rf_in_c = P_rf_in_c.get('frequency')
        #
        # # ECN_SingleCarrier
        # ecn_SNR_raw_in = ic.getresult(ecn_1, "SNR")
        # ecn_SNR_in = ecn_SNR_raw_in.get("SNR (dB)")
        # ecn_SignalPower_raw_in = ic.getresult(ecn_1, "signal power")
        # ecn_SignalPower_in = ecn_SignalPower_raw_in.get("power (dBm)")
        # ecn_NoisePower_raw_in = ic.getresult(ecn_1, "noise power")
        # ecn_NoisePower_in = ecn_NoisePower_raw_in.get("power (dBm)")
        # freq_P_out_ecn_in = ic.getresult(ecn_1, 'carriers')
        #
        # # ------------------------------------------------
        # # After the MZM_RF
        # # Optical Spectrum analyzer - OSA_MZM_1
        P = ic.getresult(osa1, "sum/signal")
        p1 = P.get("power (dBm)")
        freq = P.get('wavelength')
        p1_soa_in_osa = p1  # after the modulator
        lambda_soa_in_osa = list(it.chain.from_iterable(freq))
        #
        # # Optical Power Meter
        # P_soa_in = ic.getresult(opw_mzm1, "sum/power")  # dBm
        #
        # # ------------------------------------------------------------
        # # Amplifier
        # # Optical spectrum analyzer - OSA_AMP
        # # OSA SOA1
        # P_soa1 = ic.getresult(osa_soa1, "sum/signal")
        # p1_mzi = P_soa1.get("power (dBm)")
        # # Extract the wavelength
        # freq_mzi = P_soa1.get('wavelength')
        #
        # # Optical Power Meter
        # P_pd_in = ic.getresult(opw_soa1, "sum/power")  # dBm
        #
        # # ------------------------------------------------
        # # Before the Photodiode - PIN_1
        # # Optical Spectrum analyzer - OSA_PD
        P_opt_pd = ic.getresult(osa_pd, "sum/signal")
        p1_opt_pd = P_opt_pd.get("power (dBm)")
        # Extract the wavelength
        lambda_pd = P_opt_pd.get('wavelength')
        #
        # # Optical Power Meter - PowerMeter_PD
        # P_opt_pd_pw = ic.getresult(opw_pd, "sum/power")  # dBm
        #
        # # --------------------------------------------------
        # # After the Photodiode - PIN_1
        # # Electrical Spectrum Analyzer - RF-SA1
        P_rfsa_pd = ic.getresult(rf_sa1, "signal")
        p1_rf = P_rfsa_pd.get("power (dBm)")
        # P_rfsa_pd = ic.getresult(rf_sa1,"spectrum")
        freq_rf_pd = P_rfsa_pd.get('frequency')
        # freq_rf_pd = list(it.chain.from_iterable(freq_rf_pd))
        # # print(freq_rf)
        #
        # # RF - PD output - Powermeter
        # P_elec_pd_out = ic.getresult(pw_out_pd, "total power")  # dBm
        #
        # # --------------------------------------------------
        # # Adapted network for the Photodiode
        # # Electrical Spectrum Analyzer - ESA_output
        # # Note: In here we have power units instead of amplitude
        # P_rfsa_out = ic.getresult(rf_out, "signal")
        # freq_rf_out_t = P_rfsa_out.get('frequency')
        # p1_rf_out = P_rfsa_out.get("power (dBm)")
        # freq_rf_out = list(it.chain.from_iterable(freq_rf_out_t))
        #
        # # Electrical Power meter - PWM_out
        # P_out_rf = ic.getresult(pw_out, "total power")  # dBm
        #
        # # ESA_RF_out_dBm/Hz
        # P_rfsa_out_c = ic.getresult(esa_rf_out_dBm_Hz, "signal")
        # p1_rf_out_c = P_rfsa_out_c.get("power (dBm/Hz)")
        # # P_rfsa = ic.getresult(rf_sa1,"spectrum")
        # freq_rf_out_c = P_rfsa_out_c.get('frequency')
        #
        # # ECN_IMD3
        # ecn_imd3_SNR_raw = ic.getresult(ecn_imd3, "SNR")
        # ecn_imd3_SNR = ecn_imd3_SNR_raw.get("SNR (dB)")
        # ecn_imd3_SignalPower_raw = ic.getresult(ecn_imd3, "signal power")
        # ecn_imd3_SignalPower = ecn_imd3_SignalPower_raw.get("power (dBm)")
        # ecn_imd3_NoisePower_raw = ic.getresult(ecn_imd3, "noise power")
        # ecn_imd3_NoisePower = ecn_imd3_NoisePower_raw.get("power (dBm)")
        # freq_imd3_P_out_ecn = ic.getresult(ecn_imd3, 'carriers')
        #
        # # ECN_SingleCarrier
        # ecn_SNR_raw = ic.getresult(ecn_sinc, "SNR")
        # ecn_SNR = ecn_SNR_raw.get("SNR (dB)")
        # ecn_SignalPower_raw = ic.getresult(ecn_sinc, "signal power")
        # ecn_SignalPower = ecn_SignalPower_raw.get("power (dBm)")
        # ecn_NoisePower_raw = ic.getresult(ecn_sinc, "noise power")
        # ecn_NoisePower = ecn_NoisePower_raw.get("power (dBm)")
        # freq_P_out_ecn = ic.getresult(ecn_sinc, 'carriers')

        ##----------------------------------------------------------------------
        # Plotting
        # # Plot Frequency vs Power (dBm)
        # # RF_SA ESA_IN
        # plt.figure()
        # plt.plot(freq_rf_in * 1e-9, p1_rf_in)
        # plt.xlabel('Frequency (GHz)')
        # plt.ylabel('Power (dBm)')
        # plt.title("Power$_{in}$ vs Frequency RF (ESA_RF_in )")
        # plt.grid(True)
        # plt.show()
        #
        # # RF_SA ESA_in_dBm/Hz
        # plt.figure()
        # plt.plot(freq_rf_in_c * 1e-9, p1_rf_in_c)
        # plt.xlabel('Frequency (GHz)')
        # plt.ylabel('Power (dBm/Hz)')
        # plt.title("Power$_{in}$ vs Frequency RF (ESA_RF_in_dBm/Hz)")
        # plt.grid(True)
        # plt.show()

        # Laser - OSA1_laser
        plt.figure()
        plt.plot(freq_laser, p1_laser)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Power (dBm)")
        plt.title("Power vs Wavelength (OSA laser)")
        # plt.legend()
        plt.grid(True)
        plt.show()


        # # OSA 1 - MZM_rf
        plt.figure()
        plt.plot(freq, p1, label="Input 1 Power")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Power (dBm)")
        plt.title("Power vs Wavelength (OSA 1)")
        plt.legend()
        plt.grid(True)
        plt.show()
        #
        # # OSA Attenuation
        # plt.figure()
        # plt.plot(lambda_pd, p1_opt_pd, label="Input 1 Power")
        # plt.xlabel("Wavelength (nm)")
        # plt.ylabel("Power (dBm)")
        # plt.title("Power vs Wavelength (OSA_PD)")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        #
        # # Plot Frequency vs Power (dBm)
        # # RF SA in after beating the frequencies
        # # ESA rf_pd
        plt.figure()
        plt.plot(freq_rf_pd * 1e-9, p1_rf, label="Input rf 1 Power")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Power (dBm)")
        plt.title("Power vs Frequency RF (ESA rf_pd)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # # RF ESA_output
        # plt.figure()
        # plt.plot(freq_rf_pd * 1e-9, p1_rf_out, label="Input rf 1 Power")
        # plt.xlabel("Frequency (GHz)")
        # plt.ylabel("Power (dBm)")
        # plt.title("Power vs Frequency RF (ESA_output)")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        #
        # # RF_SA ESA_OUTPUT
        # plt.figure()
        # plt.plot(freq_rf_out_c * 1e-9, p1_rf_out_c, label="Input rf 1 Power")
        # plt.xlabel("Frequency (GHz)")
        # plt.ylabel("Power spectral density (dBm/Hz)")
        # plt.title("Power vs Frequency RF (ESA_output_dBm/Hz )")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # Optical power meter
        # P_opm1 = ic.getresult(opw_mzm1, "sum/power")
        # print(f'Optical Power meter OPM_MZM1  \n', P_opm1, "dBm")
        #
        # P_opm2 = ic.getresult(opw_mzm1, "sum/power")
        # print(f'Optical Power meter OPM_PD  \n', P_opm2, "dBm")

        # SNR carrier output
        # snr_1carrier = ic.getresult(ecn_sinc, 'SNR')
        # print(f'SNR @{carriers}: {snr_1carrier}')  # list of SNR values measured @ each carrier frequency.
        # P_1carrier = ic.getresult(ecn_sinc, 'signal power')
        # print(f'Signal power @{carriers}: {P_1carrier}')
        # P_noise_carrier = ic.getresult(ecn_sinc, 'noise power')
        # print(f'Noise power @{carriers}: {P_noise_carrier}')
        # ic.switchtodesign()


        # Target
        # # After the DELAY_t1
        # # Electrical Spectrum Analyzer - RF-SA1
        P_rfsa_pd_target = ic.getresult(rf_sa_target, "signal")
        p1_rf_target = P_rfsa_pd_target.get("power (dBm)")
        freq_rf_pd_target = P_rfsa_pd_target.get('frequency')
        freq_rf_pd_target = list(it.chain.from_iterable(freq_rf_pd_target))
        # # print(freq_rf)
        #
        # # RF - PD output - Powermeter
        # P_elec_pd_out_target = ic.getresult(pw_out_pd_target, "total power")  # dBm

        # ----------------------------------------------------------------------
        # # Receiver
        # # Electrical Spectrum from beat of the target
        # # ------------------------------------------------
        # # After the MZM_RF
        # # Optical Spectrum analyzer - OSA_MZM1_rx
        P_rx = ic.getresult(osa1_rx, "sum/signal")
        p1_rx = P_rx.get("power (dBm)")
        freq_rx = P_rx.get('wavelength')
        p1_soa_in_osa_rx = p1_rx  # after the modulator
        lambda_soa_in_osa_rx = list(it.chain.from_iterable(freq_rx))
        #
        # # Optical Power Meter
        # P_soa_in_rx = ic.getresult(opw_mzm1_rx, "sum/power")  # dBm
        #
        # # ------------------------------------------------------------
        # # Amplifier
        # # Optical spectrum analyzer - OSA_AMP
        # # OSA SOA1
        # P_soa1_rx = ic.getresult(osa_soa1_rx, "sum/signal")
        # p1_mzi_rx = P_soa1_rx.get("power (dBm)")
        # # Extract the wavelength
        # freq_mzi_rx = P_soa1_rx.get('wavelength')
        #
        # # Optical Power Meter
        # P_pd_in_rx = ic.getresult(opw_soa1_rx, "sum/power")  # dBm
        #
        # # ------------------------------------------------
        # # Before the Photodiode - PIN_1
        # # Optical Spectrum analyzer - OSA_PD
        P_opt_pd_rx = ic.getresult(osa_pd_rx, "sum/signal")
        p1_opt_pd_rx = P_opt_pd_rx.get("power (dBm)")
        # Extract the wavelength
        lambda_pd_rx = P_opt_pd_rx.get('wavelength')
        #
        # # Optical Power Meter - PowerMeter_PD
        # P_opt_pd_pw = ic.getresult(opw_pd, "sum/power")  # dBm
        #
        # # --------------------------------------------------
        # # After the Photodiode - PIN_1
        # # Electrical Spectrum Analyzer - RF-SA1_rx
        P_rfsa_pd_rx = ic.getresult(rf_sa1_rx, "signal")
        p1_rf_rx = P_rfsa_pd_rx.get("power (dBm)")
        freq_rf_pd_rx = P_rfsa_pd_rx.get('frequency')
        # freq_rf_pd_rx = list(it.chain.from_iterable(freq_rf_pd_rx))
        # # print(freq_rf)
        #
        # # RF - PD output - Powermeter
        # P_elec_pd_out_rx = ic.getresult(pw_out_pd_rx, "total power")  # dBm
        #
        # # --------------------------------------------------
        # # Adapted network for the Photodiode
        # # Electrical Spectrum Analyzer - ESA_output
        # # Note: In here we have power units instead of amplitude
        P_rfsa_out_rx = ic.getresult(rf_out_rx, "signal")
        freq_rf_out_t_rx = P_rfsa_out_rx.get('frequency')
        p1_rf_out_rx = P_rfsa_out_rx.get("power (dBm)")
        freq_rf_out_rx = list(it.chain.from_iterable(freq_rf_out_t_rx))
        #
        # # Electrical Power meter - PWM_out
        # P_out_rf_rx = ic.getresult(pw_out_rx, "total power")  # dBm
        #
        # # ESA_RF_out_dBm/Hz
        # P_rfsa_out_c_rx = ic.getresult(esa_rf_out_dBm_Hz_rx, "signal")
        # p1_rf_out_c_rx = P_rfsa_out_c_rx.get("power (dBm/Hz)")
        # freq_rf_out_c_rx = P_rfsa_out_c_rx.get('frequency')
        #
        # # ECN_IMD3
        # ecn_imd3_SNR_raw_rx = ic.getresult(ecn_imd3_rx, "SNR")
        # ecn_imd3_SNR_rx = ecn_imd3_SNR_raw_rx.get("SNR (dB)")
        # ecn_imd3_SignalPower_raw_rx = ic.getresult(ecn_imd3_rx, "signal power")
        # ecn_imd3_SignalPower_rx = ecn_imd3_SignalPower_raw_rx.get("power (dBm)")
        # ecn_imd3_NoisePower_raw_rx = ic.getresult(ecn_imd3_rx, "noise power")
        # ecn_imd3_NoisePower_rx = ecn_imd3_NoisePower_raw_rx.get("power (dBm)")
        # freq_imd3_P_out_ecn_rx = ic.getresult(ecn_imd3_rx, 'carriers')
        #
        # # ECN_SingleCarrier
        # ecn_SNR_raw_rx = ic.getresult(ecn_sinc_rx, "SNR")
        # ecn_SNR_rx = ecn_SNR_raw_rx.get("SNR (dB)")
        # ecn_SignalPower_raw_rx = ic.getresult(ecn_sinc_rx, "signal power")
        # ecn_SignalPower_rx = ecn_SignalPower_raw_rx.get("power (dBm)")
        # ecn_NoisePower_raw_rx = ic.getresult(ecn_sinc_rx, "noise power")
        # ecn_NoisePower_rx = ecn_NoisePower_raw_rx.get("power (dBm)")
        # freq_P_out_ecn_rx = ic.getresult(ecn_sinc_rx, 'carriers')

        ## PLOTTING -------------------------------------------------------------------------
        # Transmitter & Receiver
        # Plot Frequency vs Power (dBm)
        # OSA 1 - MZM_rf
        plt.figure()
        plt.plot(freq, p1, label="Input 1 Power")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Power (dBm)")
        plt.title("Power vs Wavelength (OSA 1 - MZM_RF)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # OSA_MZM1_rx - MZM_RF_rx
        plt.figure()
        plt.plot(freq_rx, p1_rx, label="Input 1 Power")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Power (dBm)")
        plt.title("Power vs Wavelength (OSA 1 - MZM_RF_rx)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # RF_SA in after beating the frequencies
        # Transmitter ESA RF_SA1
        plt.figure()
        plt.plot(freq_rf_pd * 1e-9, p1_rf, label="Input rf 1 Power")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Power (dBm)")
        plt.title("Power vs Frequency RF (ESA rf_pd tx)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Receiver ESA RF_SA1_rx
        plt.figure()
        plt.plot(freq_rf_pd_rx * 1e-9, p1_rf_rx, label="Input rf 1 Power")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Power (dBm)")
        plt.title("Power vs Frequency RF (ESA rf_pd rx)")
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 1)  # Focus on 0–1 GHz
        plt.show()

        # Plot Instantaneous frequency
        # frequency at the PD vs time
        # More straightforward way, rather than differenciating and needing the phase
        # t = np.linspace(0, self.time_window)

        # osc1_t: time vector (seconds)
        # T: chirp period (seconds)
        # rf_freq1: in GHz (as your ylabel indicates)
        # k: chirp slope in GHz/s
        # tau: echo delay in seconds

        t = osc1_t

        # 1) Tx instantaneous frequency ---
        t_mod = np.mod(t, T)
        f_tx = rf_freq1 + k * t_mod

        # PD1 output (starts at 2*rf_freq1),
        # PD1 -> 2f1 + 2kT
        f_pd = 2 * rf_freq1 + 2 * k * t_mod

        # 3) Echo: delayed in time by tau ---
        # Compute chirp phase-time argument using (t - tau)
        # Echo signal -> 2*f0 +2k(t + Tau)
        t_echo = t - self.tau

        # Mask region before echo arrives (t < tau)
        mask = t >= self.tau

        # Option A: fill missing region with zeros
        f_echo = np.zeros_like(t, dtype=float)
        f_echo[mask] = 2 * rf_freq1 + 2 * k * np.mod(t_echo[mask], T)

        # Option B : use NaN so the line is not drawn before arrival
        # f_echo = np.full_like(t, np.nan, dtype=float)
        # f_echo[mask] = 2 * rf_freq1 + 2 * k * np.mod(t_echo[mask], T)

        # ---- Plot (all on one figure so they “refer” to each other) ----
        plt.figure()
        plt.plot(t, f_tx/1e9, label="Tx: f1 + k·mod(t,T)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (GHz)")
        plt.title("Instantaneous frequency (photonic radar)")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(t, f_pd / 1e9, label="PD1: 2f1 + 2k·mod(t,T)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (GHz)")
        plt.title("Instantaneous frequency (photonic radar)")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(t, f_echo / 1e9, label=f"Echo: 2f1 + 2k·mod(t-τ,T)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (GHz)")
        plt.title("Instantaneous frequency (photonic radar)")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot the chirp
        plt.figure()
        plt.plot(osc1_t, osc1_amp1)
        plt.xlabel("time (s)")
        plt.ylabel("Amplitude (a.u.)")
        plt.title("Time domain chirp waveform")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        #

        # RFSA_spectrum = getresult("RFSA_2", "signal");
        # power = RFSA_spectrum.getattribute("power (dBm)");
        # f = RFSA_spectrum.getparameter("frequency");
        # f_beat = f(find(power, max(power)));
        #
        # # Plot Range
        # # Cross-range (m) vs. Range (m) -
        # c = 3e8 # speed of light [m/s^2]
        # range = c * freq_rf_pd / (4 * k)
        # plt.figure()
        # # plt.plot(range, p1_rf)
        # plt.plot(range, p1_rf_target)
        # plt.xlabel("Range (m)")
        # plt.ylabel("Power (dBm)")

        # Cross-range (m) vs. Range (m)









tx = Transceiver()
# print(lumapi.__file__)
tx.transceiver_simple()
# tx.modulator_link(v_bias_start=0, v_bias_stop=6, v_bias_n=7)
ic.switchtodesign()

pdb.set_trace()  # Constants
