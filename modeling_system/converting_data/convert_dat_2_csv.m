clear, clc; close all
load dryer.dat

input = dryer(:,1);
output = dryer(:,2);

plot(input)
figure
plot(output)

Data = table(input,output);
writetable(Data,'data.csv')