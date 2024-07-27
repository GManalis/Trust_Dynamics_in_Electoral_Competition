clear all
close all
clc

%% Figure 5: CASE 1 HIGH COST = 6
nb0 = 1; % How many different initial beliefs do you want to consider. 
T = 60; % Essentially how many periods you want. 
b0 = 0.5; %linspace(0,1,nb0); % different initial values of belief 
b = zeros(1,T,nb0); % current b
br = zeros(1, T, nb0); % current gamma
b(:,1,:) = b0'; % initial belief
alpha = 0.5; % alpha is the weight on the mistake
c = 6;  % fixed cost for lying
br(:,1,:) = (c*(1-b(:,1,:)).^2)./(c*(1-b(:,1,:)).^2+1); % Best response for given b (19 page 13)
eshock = zeros(1,T,nb0);
eshock(1,30,nb0) = .8;

for j = 1:nb0;
for t= 2:T
    b(1,t,j) = b(1,t-1,j) + alpha*(1-br(1,t-1,j) - b(1,t-1,j))% apo to provlima III
    br(1,t,j) = (c*(1-b(1,t,j)).^2)./(c*(1-b(1,t,j)).^2+1)- eshock(1,t-1,j);
end
end
% figure(2)
% subplot(1,2,2)
% for i = 1:nb0
% plot(1-b(:,:,i), '-*')
% hold on 
% plot(br(:,:,i), '-o')
% end

% Check the unstable case where the initial belief should be exactly that
% in order for the system to stay there. 
b_unst = zeros(1,T,1); % current b
br_unst = zeros(1, T, 1);
b_unst(1,1,1) = (c + (c*(c - 4))^(1/2))/(2*c);
br_unst(1,1,1) = (c*(1-b_unst(1,1,1)).^2)./(c*(1-b_unst(1,1,1)).^2+1); % Best response for given b
for t=2:T
    b_unst(1,t,1) = b_unst(1,t-1,1) + alpha*(1-br_unst(1,t-1,1) - b_unst(1,t-1,1));
    br_unst(1,t,1) = (c*(1-b_unst(1,t,1)).^2)./(c*(1-b_unst(1,t,1)).^2+1)
end


figure('Units', 'inches', 'Position', [1, 1, 9, 4]); % Set the figure size (6 inches by 4 inches))
subplot(1,2,1)
for i = 1:nb0
plot(1-b(:,:,i), '-*')
hold on 
plot(br(:,:,i), '-o')
end
hold on
plot(1-b_unst(1,:,1), '-*','Color', [1, 0.8, 0])
hold on 
plot(br_unst(1,:,1), '-o','Color', [1, 0.8, 0])
hold off
ylim([0 1])
title('\textbf{Dynamic equilibria (Country A)}', 'Interpreter', 'Latex')
legend('$1-b_t$', '$1-\gamma(b_t)$','Interpreter', 'Latex')
xlabel('\textbf{time (t)}', 'Interpreter', 'Latex')
ylabel('\textbf{trust, honesty}', 'Interpreter', 'Latex')
saveas(gcf,'Fig_HC.fig')

%% Figure 5 CASE 2 LOW COST = 4.5
nb0 = 1; % How many different initial beliefs do you want to consider. 
T = 60; % Essentially how many periods you want. 
b0 = 0.5; %linspace(0,1,nb0); % different initial values of belief 
b = zeros(1,T,nb0); % current b
br = zeros(1, T, nb0); % current gamma
b(:,1,:) = b0'; % initial belief
alpha = 0.5; % alpha is the weight on the mistake
c = 4.5;  % fixed cost for lying
br(:,1,:) = (c*(1-b(:,1,:)).^2)./(c*(1-b(:,1,:)).^2+1); % Best response for given b (19 page 13)
eshock = zeros(1,T,nb0);
eshock2 = zeros(1,T,nb0);
eshock3 = zeros(1,T,nb0);
eshock(1,30,nb0) = 0.8;
eshock2(1,70,nb0) = 0;
eshock3(1,10,nb0) = 0;
for j = 1:nb0;
for t= 2:T
    b(1,t,j) = b(1,t-1,j) + alpha*(1-br(1,t-1,j) - b(1,t-1,j));% apo to provlima III
    br(1,t,j) = (c*(1-b(1,t,j)).^2)./(c*(1-b(1,t,j)).^2+1)+ eshock2(1,t-1,j)- eshock(1,t-1,j)- eshock3(1,t-1,j)
end
end
% figure(2)
% subplot(1,2,2)
% for i = 1:nb0
% plot(1-b(:,:,i), '-*')
% hold on 
% plot(br(:,:,i), '-o')
% end

% Check the unstable case where the initial belief should be exactly that
% in order for the system to stay there. 
b_unst = zeros(1,T,1); % current b
br_unst = zeros(1, T, 1);
b_unst(1,1,1) = (c + (c*(c - 4))^(1/2))/(2*c);
br_unst(1,1,1) = (c*(1-b_unst(1,1,1)).^2)./(c*(1-b_unst(1,1,1)).^2+1); % Best response for given b
for t=2:T
    b_unst(1,t,1) = b_unst(1,t-1,1) + alpha*(1-br_unst(1,t-1,1) - b_unst(1,t-1,1));
    br_unst(1,t,1) = (c*(1-b_unst(1,t,1)).^2)./(c*(1-b_unst(1,t,1)).^2+1)
end


subplot(1,2,2)
for i = 1:nb0
plot(1-b(:,:,i), '-*')
hold on 
plot(br(:,:,i), '-o')
end
hold on
plot(1-b_unst(1,:,1), '-*', 'Color', [1, 0.8, 0])
hold on 
plot(br_unst(1,:,1), '-o', 'Color', [1, 0.8, 0])
hold off
ylim([0 1])
title('\textbf{Dynamic equilibria (Country B)}', 'Interpreter', 'Latex')
legend('$1-b_t$', '$1-\gamma(b_t)$','Interpreter', 'Latex')
xlabel('\textbf{time (t)}', 'Interpreter', 'Latex')
ylabel('\textbf{trust, honesty}', 'Interpreter', 'Latex')
% saveas(gcf,'Fig_LC.fig')

% Adjust the paper size
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, 9, 4]); % Set the paper size to match the figure size


% % Export the plot as a PNG
exportgraphics(gcf, 'figure5_new.png', 'Resolution', 300); % Save as PNG with 300 DPI resolution


%% Figure 6: Country C
clear all
close all
clc

nb0 = 1; % How many different initial beliefs do you want to consider. 
T = 100; % Essentially how many periods you want. 
b0 = 0.5; %linspace(0,1,nb0); % different initial values of belief 
b = zeros(1,T,nb0); % current b
br = zeros(1, T, nb0); % current gamma
b(:,1,:) = b0'; % initial belief
alpha = 0.5; % alpha is the weight on the mistake
c = 4.5;  % fixed cost for lying
br(:,1,:) = (c*(1-b(:,1,:)).^2)./(c*(1-b(:,1,:)).^2+1); % Best response for given b (19 page 13)
eshock = zeros(1,T,nb0);
eshock2 = zeros(1,T,nb0);
eshock3 = zeros(1,T,nb0);
eshock4 = zeros(1,T,nb0);
eshock(1,30,nb0) = 0.7;
eshock2(1,70,nb0) = 0.5;
eshock3(1,10,nb0) = 0.5;
eshock4(1,50,nb0) = 0.5;
for j = 1:nb0;
for t= 2:T
    b(1,t,j) = b(1,t-1,j) + alpha*(1-br(1,t-1,j) - b(1,t-1,j));% apo to provlima III
    br(1,t,j) = (c*(1-b(1,t,j)).^2)./(c*(1-b(1,t,j)).^2+1)+ eshock2(1,t-1,j)- eshock(1,t-1,j)- eshock3(1,t-1,j)- eshock4(1,t-1,j)
end
end

b_unst = zeros(1,T,1); % current b
br_unst = zeros(1, T, 1);
b_unst(1,1,1) = (c + (c*(c - 4))^(1/2))/(2*c);
br_unst(1,1,1) = (c*(1-b_unst(1,1,1)).^2)./(c*(1-b_unst(1,1,1)).^2+1); % Best response for given b
for t=2:T
    b_unst(1,t,1) = b_unst(1,t-1,1) + alpha*(1-br_unst(1,t-1,1) - b_unst(1,t-1,1));
    br_unst(1,t,1) = (c*(1-b_unst(1,t,1)).^2)./(c*(1-b_unst(1,t,1)).^2+1)
end


figure('Units', 'inches', 'Position', [1, 1, 9, 4]); % Set the figure size (6 inches by 4 inches))
subplot(1,2,1)
for i = 1:nb0
plot(1-b(:,:,i), '-*')
hold on 
plot(br(:,:,i), '-o')
end
hold on
plot(1-b_unst(1,:,1), '-*','Color', [1, 0.8, 0])
hold on 
plot(br_unst(1,:,1), '-o','Color', [1, 0.8, 0])
hold off
ylim([0 1])
xticks(linspace(0,100,11))
title('\textbf{Dynamic equilibria (Country C)}', 'Interpreter', 'Latex')
legend('$1-b_t$', '$1-\gamma(b_t)$','Interpreter', 'Latex')
xlabel('\textbf{time (t)}', 'Interpreter', 'Latex')
ylabel('\textbf{trust, honesty}', 'Interpreter', 'Latex')

%% Figure 6: Country D
nb0 = 1; % How many different initial beliefs do you want to consider. 
T = 100; % Essentially how many periods you want. 
b0 = 0.5; %linspace(0,1,nb0); % different initial values of belief 
b = zeros(1,T,nb0); % current b
br = zeros(1, T, nb0); % current gamma
b(:,1,:) = b0'; % initial belief
alpha = 0.8; % alpha is the weight on the mistake
c = 4.5;  % fixed cost for lying
br(:,1,:) = (c*(1-b(:,1,:)).^2)./(c*(1-b(:,1,:)).^2+1); % Best response for given b (19 page 13)
eshock = zeros(1,T,nb0);
eshock2 = zeros(1,T,nb0);
eshock3 = zeros(1,T,nb0);
eshock4 = zeros(1,T,nb0);
eshock(1,30,nb0) = 0.7;
eshock2(1,70,nb0) = 0.5;
eshock3(1,10,nb0) = 0.5;
eshock4(1,50,nb0) = 0.5;
for j = 1:nb0;
for t= 2:T
    b(1,t,j) = b(1,t-1,j) + alpha*(1-br(1,t-1,j) - b(1,t-1,j));% apo to provlima III
    br(1,t,j) = (c*(1-b(1,t,j)).^2)./(c*(1-b(1,t,j)).^2+1)+ eshock2(1,t-1,j)- eshock(1,t-1,j)- eshock3(1,t-1,j)- eshock4(1,t-1,j)
end
end

b_unst = zeros(1,T,1); % current b
br_unst = zeros(1, T, 1);
b_unst(1,1,1) = (c + (c*(c - 4))^(1/2))/(2*c);
br_unst(1,1,1) = (c*(1-b_unst(1,1,1)).^2)./(c*(1-b_unst(1,1,1)).^2+1); % Best response for given b
for t=2:T
    b_unst(1,t,1) = b_unst(1,t-1,1) + alpha*(1-br_unst(1,t-1,1) - b_unst(1,t-1,1));
    br_unst(1,t,1) = (c*(1-b_unst(1,t,1)).^2)./(c*(1-b_unst(1,t,1)).^2+1)
end


subplot(1,2,2)
for i = 1:nb0
plot(1-b(:,:,i), '-*')
hold on 
plot(br(:,:,i), '-o')
end
hold on
plot(1-b_unst(1,:,1), '-*','Color', [1, 0.8, 0])
hold on 
plot(br_unst(1,:,1), '-o','Color', [1, 0.8, 0])
hold off
ylim([0 1])
xticks(linspace(0,100,11))
title('\textbf{Dynamic equilibria (Country D)}', 'Interpreter', 'Latex')
legend('$1-b_t$', '$1-\gamma(b_t)$','Interpreter', 'Latex')
xlabel('\textbf{time (t)}', 'Interpreter', 'Latex')
ylabel('\textbf{trust, honesty}', 'Interpreter', 'Latex')

% Adjust the paper size
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, 9, 4]); % Set the paper size to match the figure size


% % Export the plot as a PNG
exportgraphics(gcf, 'figure6_new.png', 'Resolution', 300); % Save as PNG with 300 DPI resolution
