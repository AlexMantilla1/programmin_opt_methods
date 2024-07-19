function SNDR = run_sim_and_get_SNDR(model)
    % Calculate the parameter to model de modulator
    config_model 
 
    ps1 = model(1);
    pf1 = ps1;  
    ps2 = model(2);
    pf2 = ps2; 
    ps3 = model(3);
    pf3 = ps3;
    assignin('base','ps1',ps1)
    assignin('base','pf1',pf1)
    assignin('base','ps2',ps2)
    assignin('base','pf2',pf2)
    assignin('base','ps3',ps3)
    assignin('base','pf3',pf3)

    % Run the sim
    fprintf(".")
    out = sim('../DS3or');

    % Get the SNDR
    [~,Dout_diff] = getData(out.Dout_diff); 
    [X,f,~] = calculate_fft(Dout_diff,fclk); 
    X = 20*log10(X);
    
    
    BW_index = find(f == (fs/2));
    % Get only interest BW 
    X = X(1:BW_index);
    % Output signal amplitude
    [amp_dB,index] = max(X); 
    % Remove main signal from spectrum to see noise floor
    X(index) = X(1);
    % Output noise floor 
    [noise_floor,~] = max(X); 
    % Calculate Measured SNDR
    SNDR = amp_dB - noise_floor;
    SNDR = -SNDR;
 
end

 