import importlib.util

# default path for current release
spec_lin = importlib.util.spec_from_file_location('lumapi', "/opt/lumerical/v241/api/python/lumapi.py")
# Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_lin)
spec_lin.loader.exec_module(lumapi)

import numpy as np
import matplotlib.pyplot as plt
import pdb
import itertools as it
import pickle


# Considering that I have an opened session
def receiver_target(self,
                    ic,
                    R_in=None,
                    voltage=1,
                    x_ref_pos=None,
                    y_ref_pos=None):
    """
    Function that sets the architecture of the transmitter. This transmitter does not include
    any specific technology.
    Waveguides included
    :param R_in:
    :param voltage:
    :param x_ref_pos: reference position x-axis
    :param y_ref_pos: reference position y-axis
    :return: Various graphs

    Args:
        x_ref_pos:
        y_ref_pos:


    """
    if R_in is None:
        R_in = self.impedance  # Default to 50 Ohms
    if x_ref_pos is None:
        x_ref_pos = self.x_pos
    if y_ref_pos is None:
        y_ref_pos = self.y_pos


    # Add the laser
    ic.addelement("CW Laser")
    laser_rx = "CW_Laser_rx"
    ic.set("name", laser_rx)
    ic.set("x position", -200 + x_ref_pos)
    ic.set("y position", 0 + y_ref_pos)
    ic.set("frequency", self.c / self.lambda_central)
    ic.set("power", self.power_laser)  # dBm for it to give me 10 dBm
    # ic.set("linewidth, 5 ") #MHz
    # ic.set("phase", 0) #rad


    # After the laser
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
    ic.set("bias voltage 1", self.v_bias)  # Why is it 1? Because is cuadrature and the it divides by 2, and then
    # Lumerical counts it as you would have a bias voltage in 2 arms, and that is why
    # you have to divide it by 2 again
    ic.set("pi dc voltage", self.v_pi_dc)
    ic.set("pi rf voltage", self.v_pi_rf)
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

    # Optical Amplifier
    ic.addelement("Optical Amplifier")
    gain1_rx = "AMP_1_rx"
    ic.set("name", gain1_rx)
    ic.set("x position", 500 + x_ref_pos)
    ic.set("y position", 0 + y_ref_pos)
    ic.set("gain", self.SOA_gain)
    # ic.set("noise figure", )
    ic.set("enable noise", False)

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

    # Add fiber
    # ic.addelement("Optical Linear Fiber")
    # fiber1 = "FIB_1"
    # ic.set("name", fiber1)
    # ic.set("x position", 900 + x_ref_pos)
    # ic.set("y position", 0 + y_ref_pos)
    # ic.set("configuration", "unidirectional")
    # ic.set("reference frequency", 1550)

    # Second part of the receiver

    # Mach Zehner interferometer
    ic.addelement("Mach Zehnder Interferometer")
    # for mode 1
    mzi1_rx = 'MZI_1_rx'
    ic.set("name", mzi1_rx)
    ic.set("x position", 900 + x_ref_pos)
    ic.set("y position", 0 + y_ref_pos)
    ic.set("loss 1", 3)
    ic.set("effective index 1", 3.47)
    ic.set("coupling coefficient 1 1", 0.5)

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

    # Photodiode
    ic.addelement("PIN Photodetector")
    pd1_rx = "PIN_1_rx"
    ic.set("name", pd1_rx)
    ic.set("x position", 1300 + x_ref_pos)
    ic.set("y position", 0 + y_ref_pos)
    ic.set("frequency at max power", 1)  # 0 meaning false
    # ic.set("frequency", c / lambda_central)
    ic.set("input parameter", "constant")
    ic.set("responsivity", 0.85)  # 0.85 A/W
    ic.set("dark current", 2.5e-8)  # A
    ic.set("enable power saturation", True)
    ic.set("saturation power", 15)  # dBm
    ic.set("enable thermal noise", False)
    ic.set("enable shot noise", False)
    ic.set("convert noise bins", False)
    ic.set("automatic seed", True)

    # Add a Low-pass filter
    ic.addelement("BP Bessel Filter")
    bpf1_rx = 'BPF_1_rx'
    ic.set("name", bpf1_rx)
    ic.set("x position", 1600 + x_ref_pos)
    ic.set("y position", 0 + y_ref_pos)
    ic.set("frequency", self.rf_freq1)

    # Compound element
    comp_1_rx = "PD_OUT_1_rx"
    ic.addelement('PD_OUT_1')
    ic.set("name", comp_1_rx)
    ic.set("x position", 1800 + x_ref_pos)
    ic.set("y position", 0 + y_ref_pos)

    # ---------------------------------------------------------------------------
    # Measurement devices
    # OSA y position
    osa_y = 200
    opw_y = 300

    # Power meter - RF_signal
    ic.addelement("Power Meter")
    pw1_rx = "PWM_1_rx"
    ic.set("name", pw1_rx)
    ic.set("x position", -150 + x_ref_pos)
    ic.set("y position", -280 + y_ref_pos)
    ic.set("input kind", "voltage")
    ic.set("impedance", self.impedance)
    ic.set("power unit", "dBm")
    ic.set("limit time range", 1)
    ic.set("start time", 1e-8)
    ic.set("stop time", 1)

    # Oscilloscope SUM_1
    ic.addelement("Oscilloscope")
    osc1_rx = "osc_1_rx"
    ic.set("name", osc1_rx)
    ic.set("x position", 400 + x_ref_pos)
    ic.set("y position", -500 + y_ref_pos)

    # OSA MZM
    ic.addelement("Optical Spectrum Analyzer")
    osa1_rx = "OSA_MZM_1_rx"
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

    # Add the Electrical Spectrum Analyzer - PIN_1
    ic.addelement("Spectrum Analyzer")
    rf_sa1_rx = "RF_SA1_rx"
    ic.set("name", rf_sa1_rx)
    ic.set("x position", 1600 + x_ref_pos)
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
    ic.set("x position", 1600 + x_ref_pos)
    ic.set("y position", opw_y + y_ref_pos)
    ic.set("input kind", "voltage")
    ic.set("impedance", 1)
    ic.set("power unit", "dBm")
    ic.set("limit time range", 1)
    ic.set("start time", 1e-8)
    ic.set("stop time", 1)

    # Optical power meter for before the PD
    ic.addelement("Optical Power Meter")
    opw_pd_rx = "PowerMeter_PD_rx"
    ic.set("name", opw_pd_rx)
    ic.set("x position", 1250 + x_ref_pos)
    ic.set("y position", opw_y + y_ref_pos)
    ic.set("limit time range", 1)
    ic.set("start time", 1e-8)
    ic.set("stop time", 1)

    # Add the OSA for PD
    ic.addelement("Optical Spectrum Analyzer")
    osa_pd_rx = "OSA_PD_rx"
    ic.set("name", osa_pd_rx)
    ic.set("x position", 1250 + x_ref_pos)
    ic.set("y position", osa_y + y_ref_pos)
    ic.set("limit frequency range", 0)
    ic.set("limit time range", 1)
    ic.set("start time", 1e-8)
    ic.set("stop time", 1)
    ic.set("plot kind", "wavelength")

    # OSA MZI_1
    ic.addelement("Optical Spectrum Analyzer")
    osa1_mzi_rx = "OSA_MZI_1_rx"
    ic.set("name", osa1_mzi_rx)
    ic.set("x position", 1100 + x_ref_pos)
    ic.set("y position", osa_y + y_ref_pos)
    # ic.set("sensitivity", -100) #dBm
    ic.set("limit frequency range", 0)
    ic.set("limit time range", 1)
    ic.set("start time", 0)
    ic.set("stop time", 1e-8)
    ic.set("plot kind", "wavelength")

    # Optical power meter for the MZI_1
    ic.addelement("Optical Power Meter")
    opw_mzi_1_rx = "PWM_MZI_1_rx"
    ic.set("name", opw_mzi_1_rx)
    ic.set("x position", 1100 + x_ref_pos)
    ic.set("y position", opw_y + y_ref_pos)
    ic.set("limit time range", 1)
    ic.set("start time", 1e-8)
    ic.set("stop time", 1)

    # Elements for the PD matching network
    # Add the Electrical Spectrum Analyzer - After the matching PD network
    ic.addelement("Spectrum Analyzer")
    rf_sa2_rx = "ESA_output_rx"
    ic.set("name", rf_sa2_rx)
    ic.set("x position", 2000 + x_ref_pos)
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
    ic.set("x position", 2000 + x_ref_pos)
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
    ic.set("x position", 2000 + x_ref_pos)
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
    ic.set("x position", 2000 + x_ref_pos)
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
    ic.set("x position", 2000 + x_ref_pos)
    ic.set("y position", -400 + y_ref_pos)
    # ic.set("sensitivity", -150) # dBm
    ic.set("bandwidth", 100e6)
    ic.set("interpolation offset", 200e6)
    ic.set("limit time range", 1)  # True
    ic.set("start time", 1e-8)
    ic.set("stop time", 1)
    ic.set("frequency carriers", "user defined")
    ic.set("carriers table", imd3)

    # Connect the elements --------------------------------------------------
    # RF signal & noise, fork 1


    # Meas AMP - SOA
    ic.connect(osa_soa1_rx, "input", gain1_rx, "output")
    ic.connect(opw_soa1_rx, "input", gain1_rx, "output")

    # MZM1 Connections
    ic.connect(laser_rx, "output", wg1_rx, "input")
    ic.connect(wg1_rx, "output", mzm1_rx, "input")
    ic.connect(mzm1_rx, "output", wg2_rx, "input")

    ic.connect(mzm1_rx, "output", opw_mzm1_rx, "input")
    ic.connect(mzm1_rx, "output", osa1_rx, "input")

    # Opt amplifier
    ic.connect(wg2_rx, "output", gain1_rx, "input")
    ic.connect(gain1_rx, "output", wg3_rx, "input")

    # Connections pd
    ic.connect(wg3_rx, 'output', mzi1_rx, 'port 1')
    ic.connect(wg4_rx, 'output', osa_pd_rx, "input")
    ic.connect(pd1_rx, "output", pw_out_pd_rx, "input")
    ic.connect(wg4_rx, "input", mzi1_rx, "port 4")
    ic.connect(wg4_rx, "output", pd1_rx, "input")
    ic.connect(mzi1_rx, 'port 4', osa1_mzi_rx, 'input')
    ic.connect(mzi1_rx, 'port 4', opw_mzi_1_rx, 'input')

    # PD to filter and compound
    ic.connect(pd1_rx, "output", bpf1_rx, "input")
    ic.connect(bpf1_rx, "output", comp_1_rx, "port 1")
    ic.connect(comp_1_rx, "port 2", rf_sa2_rx, "input")
    ic.connect(comp_1_rx, "port 2", pw_out_rx, "input")
    ic.connect(pd1_rx, "output", rf_sa1_rx, "input")
    ic.connect(opw_pd_rx, "input", wg4_rx, "output")
    ic.connect(ecn_sinc_rx, "input", comp_1_rx, "port 2")
    ic.connect(ecn_imd3_rx, "input", comp_1_rx, "port 2")
    ic.connect(esa_rf_out_dBm_Hz_rx, "input", comp_1_rx, "port 2")

    # Meas instruments to in rf signal
    # ic.connect(gain_elec, "input", fork_adder, "output 1")
    # ic.connect(gain_elec, "output", esa_rf_in_dBm_Hz, "input")
    # ic.connect(gain_elec, "output", esa_rf_in, "input")
    # ic.connect(gain_elec, "output", ecn_1, "input")


