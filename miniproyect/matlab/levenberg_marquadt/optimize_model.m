%% Clean workspace
clc; clear all; close all;

%& Calculate the parameter to model de modulator
config_model
run_all_sims = false;

%% Call the main function if this script is run 
if run_all_sims
    % Define an arbitrary initial point for training 
    initial_model = [0.35; % ps1 
                     0.35; % ps2 
                     0.8; % ps3 
                     ];  
    
    % Define the objective function to optimize
    obj_function = @run_sim_and_get_SNDR;
    
    % Calculate the model
    solution = levenberg_marquadt(0.01, initial_model, obj_function); 
    if ~solution.converged
        disp('WARNING!!! method did not converge');
    end  
    solution.value 
    save('levenberg_marquadt.mat',"solution");
else
    load('levenberg_marquadt_best.mat')
end

%% Test the model
model = solution.value;
ps1 = model(1);
pf1 = ps1;  
ps2 = model(2);
pf2 = ps2; 
ps3 = model(3);
pf3 = ps3; 


% Run the sim
out = sim('../DS3or');

% Get Dout_diff output
[~,Dout_diff] = getData(out.Dout_diff); 
[X,f,~] = calculate_fft(Dout_diff,fclk); 
X = 20*log10(X);

BW_index = find(f == (fs/2));
[amp_dB,index] = max(X(1:BW_index));
X(index) = X(1);
[noise,~] = max(X(1:BW_index));
X(index) = amp_dB;
noise_floor = noise; % Output noise floor
amps = amp_dB; % Output signal amplitude
SNR = 20*log10(Vin_amp_max) - noise_floor;
SNDR = amps - noise_floor;

% Plot
fig = figure;
canvas = subplot(1,1,1);
semilogx(f,X); 
hold on
% get the axis limits
ylim = canvas.YLim;
xlim = canvas.XLim;
semilogx([xlim(1) BW],[noise_floor noise_floor],'--','LineWidth',2)
semilogx([xlim(1) BW],[amp_dB amp_dB],'g--','LineWidth',2)
semilogx([BW/10 BW/10],[noise_floor amp_dB],'--','LineWidth',2,'color',[0.5 0.5 0.5])
t = text(BW/60,(amp_dB + noise_floor)/2,string(round(SNDR,1))+" dB", 'FontSize', 15);
title("Bit Stream FFT")
fig.Position = [680 276 806 602];
axis([10^(1.1) 10^(6.1) canvas.YLim])
ylabel("|X(w)| [dB]"); xlabel("Frequency [Hz]");
grid on; grid minor
set(canvas,"FontSize",15) 


exportgraphics(fig,'../../some_figures/levenber_marquadt.pdf','Resolution',300)
exportgraphics(fig,'../../some_figures/levenber_marquadt.png','Resolution',300)

