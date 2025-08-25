%% Static game
close all 
clear all 
clc

%% Solution of ODE -- Proposition 1
syms f(y) c x
ode = diff(f,y,1) == c - c*y/f;
ySol(y) = dsolve(ode)
ySol = simplify(ySol)
ySol = formula(ySol)
sol_1 = ySol(1) % These are the solutions in proposition 1
sol_2 = ySol(2) % These are the solutions in proposition 1
% now take the f'(y) which is the coefficient, notice that these 
% are the corresponding (1-b)s
dif_sol_1 = diff(sol_1, y,1)
dif_sol_2 = diff(sol_2,y,1)
% Now in order to create the y = g(x) you get the following: 
announce_1 = x./dif_sol_1; 
announce_2 = x./dif_sol_2;
% Render them as functions to plot them
announce_1 = matlabFunction(announce_1);
announce_2 = matlabFunction(announce_2);
%% Figure 2
% Plot the announcement-type figure 
x = linspace(-10,10) % the types
y_45 = x;            % 45-degree line
c=5                  % fixed cost of lying
figure('Units', 'inches', 'Position', [1, 1, 9, 4]); % Set the figure size (6 inches by 4 inches))
subplot(1,2,1)
xline(0, 'k', 'LineWidth', 0.5, 'HandleVisibility', 'off'); 
yline(0, 'k', 'LineWidth', 0.5, 'HandleVisibility', 'off'); 
hold on
plot(x, announce_1(c,x),'Color', [0, 0.4470, 0.7410], 'Linewidth',2)
hold on 
plot(x, announce_2(c,x),'Color', [1, 0.8, 0], 'Linewidth',2)
hold on 
plot(x, y_45, '--', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2); % 45-degree line (Grayish)
% hline = refline([1 0]);
% hline.Color = 'k';
% hline.LineStyle = ':';
ylabel('\textbf{candidate''s announcement:} $\mathbf{y_j}$', 'Interpreter', 'Latex')
xlabel('\textbf{candidate''s type:} $\mathbf{x_j}$', 'Interpreter', 'Latex')
legend("Low trust",  "High trust", '45^\circ','Location', 'southeast')
title("\textbf{Equilibrium announcement by the candidate}", 'Interpreter', 'Latex')
numTicks = 5;        % just for the figure
yLimits = ylim;
yTicks = linspace(yLimits(1), yLimits(2), numTicks);
set(gca, 'YTick', yTicks);

% Voter side - Trust levels 
trust_l = matlabFunction(sol_1) % its the equivalent of eq. 6.1
trust_h = matlabFunction(sol_2) % its the equivalent of eq. 6.2

subplot(1,2,2)
xline(0, 'k', 'LineWidth', 0.5, 'HandleVisibility', 'off'); 
yline(0, 'k', 'LineWidth', 0.5, 'HandleVisibility', 'off'); 
hold on
plot(announce_1(c,x),trust_l(c,announce_1(c,x)),'Color', [0, 0.4470, 0.7410], 'Linewidth',2)
hold on 
plot(announce_2(c,x), trust_h(c,announce_2(c,x)),'Color', [1, 0.8, 0], 'Linewidth',2)
hold on 
plot(x, y_45, '--', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2); % 45-degree line (Grayish)
% hline = refline([1 0]);
% hline.Color = 'k';
% hline.LineStyle = ':';
ylabel('\textbf{inferred type:} $\mathbf{\hat{x}_j}$', 'Interpreter', 'Latex')
xlabel('\textbf{candidate''s announcement:} $\mathbf{y_j}$', 'Interpreter', 'Latex')
legend("Low trust",  "High trust", '45^\circ','Location', 'southeast')
title("\textbf{Equilibrium inferred type by the voter}", 'Interpreter', 'Latex')
numTicks = 5;        % just for the figure
yLimits = ylim;
yTicks = linspace(yLimits(1), yLimits(2), numTicks);
set(gca, 'YTick', yTicks);

% Adjust the paper size
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, 9, 4]); % Set the paper size to match the figure size


% % Export the plot as a PNG
exportgraphics(gcf, 'Figure_2.png', 'Resolution', 300); % Save as PNG with 300 DPI resolution

%% Figure B2
slope_l = matlabFunction(dif_sol_1) 
slope_h = matlabFunction(dif_sol_2) 
c = linspace(6,10)

figure('Units', 'inches', 'Position', [1, 1, 9, 4]); % Set the figure size (6 inches by 4 inches))
subplot(1,2,2)
plot(c, slope_l(c),'Color', [0, 0.4470, 0.7410],'Linewidth', 2)
hold on 
plot(c, slope_h(c),'Color', [1, 0.8, 0],'Linewidth', 2)
ylabel('\textbf{slope of eq. belief:} $\mathbf{\frac{\partial^2f(y_j)}{\partial y_j\partial c}}$', 'Interpreter', 'Latex')
xlabel('\textbf{fixed cost:} $\mathbf{c}$', 'Interpreter', 'Latex')
title('\textbf{Cost effect on the slope of equilibrium beliefs}', 'Interpreter', 'Latex')
legend('Low trust','High trust', 'Location', 'northwest', 'Interpreter', 'Latex')
ylim([0 10])

syms x c
announcement_l = x/(1 + (1/c)*(dif_sol_1)^2)
announcement_h = x/(1 + (1/c)*(dif_sol_2)^2)
dif_announcement_l = diff(announcement_l,x)
dif_announcement_h = diff(announcement_h,x)
slope_ann_l = matlabFunction(dif_announcement_l)
slope_ann_h = matlabFunction(dif_announcement_h)

c = linspace(6,10)

subplot(1,2,1)
plot(c, slope_ann_l(c),'Color', [0, 0.4470, 0.7410],'Linewidth', 2)
hold on 
plot(c, slope_ann_h(c),'Color', [1, 0.8, 0],'Linewidth', 2)
ylabel('\textbf{slope of eq. announcement:} $\mathbf{\frac{\partial^2g(x_j)}{\partial x_j\partial c}}$', 'Interpreter', 'Latex')
xlabel('\textbf{fixed cost:} $\mathbf{c}$', 'Interpreter', 'Latex')
title('\textbf{Cost effect on the slope of equilibrium announcements}', 'Interpreter', 'Latex')
legend('Low trust','High trust', 'Location', 'northwest', 'Interpreter', 'Latex')
ylim([0 1])

% Adjust the paper size
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, 9, 4]); % Set the paper size to match the figure size


% % Export the plot as a PNG
exportgraphics(gcf, 'Figure_C2.png', 'Resolution', 300); % Save as PNG with 300 DPI resolution

























