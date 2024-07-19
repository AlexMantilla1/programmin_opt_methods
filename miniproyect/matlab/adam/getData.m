function [t,x] = getData(obj)
    % Get the data, with exact L = 2^N, with larger N possible. 
    L = length(obj.Time);
    di = L -  2 ^ (floor((log2(L)))); 
    x = obj.Data(di+1:end);
    t = obj.Time(di+1:end); 
end