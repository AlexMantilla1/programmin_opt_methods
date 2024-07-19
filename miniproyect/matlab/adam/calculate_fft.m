function [X,f,L] = calculate_fft(x,fs)
    % Calculate and Shift FFT
    L = length(x);
    X = fftshift(fft(x)/L);
    X = abs(X);
    % Get Frequency Domain
    f = fs*(0:L-1)/L;
    % Shift Frequency
    f = f - fs/2;
    % Return only Possitive Side
    f = f((end/2) + 1:end);
    X = X((end/2) + 1:end);
end