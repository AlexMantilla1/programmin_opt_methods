clear, clc; close all
load dryer.dat

input = dryer(:,1);
output = dryer(:,2);

input = input - mean(input);
output = output - mean(output);
figure
plot(input)
hold on
plot(output)

Data = table(input,output);
writetable(Data,'data.csv')