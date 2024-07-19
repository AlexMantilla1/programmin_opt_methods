%% General Parameters

% Modulator ---------------------------------------------------------
OSR = 128;
BW = 10e3;
fs = BW*2; %Nyquist
Ts = 1/fs;
fclk = OSR*fs;
Tclk = 1/fclk;
VDD = 0.9;
VSS = 0;
Vref = VDD/2;

% Input ---------------------------------------------------------
fin = fs*509/1024;
Vin_cm = (VDD+VSS)/2;
Vin_amp_max = 0.5;
Vin_amp = Vin_amp_max; %This is differential Input Amplitude!
%Vin_amp = 0;
Vin_DC = 0;

% Simulation ---------------------------------------------------------
samples = 1024 + 2;
tsim = samples*Ts;
tsim = tsim - Tclk;

% Something for the integrator 1 ---------------------------------------------------------
q_ideal = 0.97;
q1 = q_ideal;

% Something for the integrator 2 --------------------------------------------------------- 
q2 = q_ideal; 

% Something for the integrator 2 --------------------------------------------------------- 
q3 = q_ideal; 

% Comparator ---------------------------------------------------------
comp_noise_rms = 10e-6; % comp rms input-refered noise.
comp_offset = normrnd(0,5e-3/3); % comp input offset 
%comp_offset = 0,5e-3; % comp input offset 

% Filter Window ---------------------------------------------------------
Ar = 1;
Wr = Ar*ones(1,OSR); % Rectangular

n = 0:OSR/2;
At = 1;
Wt = At*2*n/OSR;
Wt = [Wt flip(Wt(2:end-1))]; % Triangular

Ah = 1;
Wh = Ah*hamming(OSR)'; % Hamming

W = Wh; % Select the one you want!