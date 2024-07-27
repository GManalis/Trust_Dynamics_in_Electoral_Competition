clear all
close all
clc

%% Proposition 2
syms c b 
% below is the equilibrium condition
eq = ((c*(1-b)^2)/(c*(1-b)^2+1))-(1-b) ==0;
sol = solve(eq, b)
sol(2,1)
sol(3,1)


%% Dynamic Game w/ Adaptive Beliefs. Proposition 3 case (a), c>c*
nb0 = 20; % How many different initial beliefs do you want to consider. 
T = 60; % Essentially how many periods you want. 
b0 = linspace(0,1,nb0); % different initial values of belief 
b = zeros(1,T,nb0); % current b
br = zeros(1, T, nb0); % current gamma
b(:,1,:) = b0'; % initial belief
alpha = 0.5; % alpha is the weight on the mistake
c = 6;  % fixed cost for lying c>c*=4
br(:,1,:) = (c*(1-b(:,1,:)).^2)./(c*(1-b(:,1,:)).^2+1); % Best response for given b

for j = 1:nb0;
for t= 2:T
    b(1,t,j) = b(1,t-1,j) + alpha*(1-br(1,t-1,j) - b(1,t-1,j));
    br(1,t,j) = (c*(1-b(1,t,j)).^2)./(c*(1-b(1,t,j)).^2+1)
end
end


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
subplot(1,2,2)
for i = 1:nb0
plot(1-b(:,:,i), '-*')
hold on 
plot(br(:,:,i), '-o')
end
hold on
plot(1-b_unst(1,:,1), '-*')
hold on 
plot(br_unst(1,:,1), '-o')
hold off
title('\textbf{Dynamic equilibria}', 'Interpreter', 'Latex')
legend('$1-b_t$', '$1-\gamma(b_t)$','Interpreter', 'Latex')
xlabel('\textbf{time (t)}', 'Interpreter', 'Latex')
ylabel('\textbf{trust, honesty}', 'Interpreter', 'Latex')

% Figure with the equilibrium condition 
c = 6;
b = linspace(0, 1, 50);
y = (c*(1-b).^2)./(1+c*(1-b).^2);
x = 1-b;
subplot(1,2,1)
plot(x, y, 'Linewidth', 2)
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';
hline.HandleVisibility = 'off';
hold on 
plot(0.2113, 0.2113, 'r*', 'MarkerSize',12)
plot(0.7887, 0.7887, 'r*', 'MarkerSize',12)
xlim([0 1])
ylim([0 1])
title('\textbf{Static game equilibria}', 'Interpreter', 'latex')
xlabel('\textbf{trust:} $\mathbf{1-b}$', 'Interpreter', 'Latex')
ylabel('\textbf{honesty:} \boldmath$1-\gamma$', 'Interpreter', 'Latex')
hold off

% Adjust the paper size
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, 9, 4]); % Set the paper size to match the figure size

% Export the plot as a PNG
exportgraphics(gcf, 'plot.png', 'Resolution', 300); % Save as PNG with 300 DPI resolution


%% Figure 4 out of Proposition 3: Left panel: Dynamic equilibria for c=c* (case (b)), 

nb0 = 20; % How many different initial beliefs do you want to consider. 
T = 60; % Essentially how many periods you want. 
b0 = linspace(0,1,nb0); % different initial values of belief 
b = zeros(1,T,nb0); % current b
br = zeros(1, T, nb0); % current gamma
b(:,1,:) = b0'; % initial belief
alpha = 0.5; % alpha is the weight on the mistake
c = 4;  % fixed cost for lying case (b) c=c*
br(:,1,:) = (c*(1-b(:,1,:)).^2)./(c*(1-b(:,1,:)).^2+1); % Best response for given b

for j = 1:nb0;
for t= 2:T
    b(1,t,j) = b(1,t-1,j) + alpha*(1-br(1,t-1,j) - b(1,t-1,j));
    br(1,t,j) = (c*(1-b(1,t,j)).^2)./(c*(1-b(1,t,j)).^2+1)
end
end


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
plot(1-b_unst(1,:,1), '-*')
hold on 
plot(br_unst(1,:,1), '-o')
hold off
title('\textbf{Dynamic equilibria ($c=c^*$)}', 'Interpreter', 'Latex')
legend('$1-b_t$', '$1-\gamma(b_t)$','Interpreter', 'Latex')
xlabel('\textbf{time (t)}', 'Interpreter', 'Latex')
ylabel('\textbf{trust, honesty}', 'Interpreter', 'Latex')


%  Right panel: Dynamic equilibria for c<c* (case (c))

c = 3;  % fixed cost for lying case (c) c<c*
br(:,1,:) = (c*(1-b(:,1,:)).^2)./(c*(1-b(:,1,:)).^2+1); % Best response for given b

for j = 1:nb0;
for t= 2:T
    b(1,t,j) = b(1,t-1,j) + alpha*(1-br(1,t-1,j) - b(1,t-1,j));
    br(1,t,j) = (c*(1-b(1,t,j)).^2)./(c*(1-b(1,t,j)).^2+1)
end
end


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


%figure('Units', 'inches', 'Position', [1, 1, 9, 4]); % Set the figure size (6 inches by 4 inches))
subplot(1,2,2)
for i = 1:nb0
plot(1-b(:,:,i), '-*')
hold on 
plot(br(:,:,i), '-o')
end
hold on
plot(1-b_unst(1,:,1), '-*')
hold on 
plot(br_unst(1,:,1), '-o')
hold off
ylim([0 1])
title('\textbf{Dynamic equilibria ($c<c^*$)}', 'Interpreter', 'Latex')
legend('$1-b_t$', '$1-\gamma(b_t)$','Interpreter', 'Latex')
xlabel('\textbf{time (t)}', 'Interpreter', 'Latex')
ylabel('\textbf{trust, honesty}', 'Interpreter', 'Latex')

% Adjust the paper size
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, 9, 4]); % Set the paper size to match the figure size

% Export the plot as a PNG
exportgraphics(gcf, 'fig4.png', 'Resolution', 300); % Save as PNG with 300 DPI resolution