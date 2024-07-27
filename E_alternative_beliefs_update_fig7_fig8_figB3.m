%% Bayesian Inference w/ the two cost distributions. 
%  (Extension B: Alternative beliefs update)

clear all 
clc 
close all 

%% Figure 7 - equilibrium conditions
% Parametrization 
cl = 5;                          % Low cost of lying
ch =15;                          % High cost of lying
p0=0.5;                          % prior belief 
p = 0.2;                         % probability of low cost when state is theta_high
q = 0.8;                         % probability of low cost when state is theta_low
p_cl = p0*q+(1-p0)*p;            % Equation 19 in the draft.
p_ch = p0*(1-q)+(1-p0)*(1-p);    % Equation 20 in the draft. 

% Discount / beliefs
N = 17;
bl = linspace(0,1,N);      % note that b^low>b^high. bl: discount when i believe that challenger is low cost
bh = linspace(0,01,N);      % discount when I believe that challenger is high cost
%b = linspace(0.01, 0.99, 50)

% Best responses 
br_low = (p_cl./((1-bl).^2*cl)+p_ch./((1-bh).^2*cl)+1).^(-1);  % Equation 21.2 in the draft
br_high = (p_cl./((1-bl).^2*ch)+p_ch./((1-bh).^2*ch)+1).^(-1); % Equation 21.1 in the draft


% Note that you have the system of equilibrium conditions (system 22)  on
% the draft. Above the br_low and the br_high have been expressed as
% vectors, while now you want them as matrices. In this way you will be
% able to generate the full set of combos between bl and bh and then work
% w/ the part of the matrix that satisfies the bl>bh condition derived also
% by the model and the 

% components at the berst response of the low cost. 
A_l = p_cl./((1-bl).^2*cl)
B_l = p_ch./((1-bh).^2*cl)+1

% components of the best response of the high cost
A_h = p_cl./((1-bl).^2*ch)
B_h = p_ch./((1-bh).^2*ch)+1


% Turning the components into matrices
% low cost
A_l_mat = repmat(A_l, N,1)
B_l_mat = repmat(B_l',1,N)
% high cost
A_h_mat = repmat(A_h,N,1)
B_h_mat = repmat(B_h',1,N)

% Best responses as matrices 
br_low_mat = (A_l_mat + B_l_mat).^(-1) % each row different bh each column different bl 
br_high_mat = (A_h_mat+B_h_mat).^(-1)  % each row different bh each column different bl


figure('Units', 'inches', 'Position', [1, 1, 9, 4]); % Set the figure size (6 inches by 4 inches))
subplot(1,2,1)
for i = 1:N
 p(i) = plot(1-bl(:), br_low_mat(i,:), 'Linewidth', 2)  
 legendInfo{i} = ['b_h = ' num2str(round(bh(i),2))];
%  legend(legendInfo, 'Location','northwest');
 hold on
end
legend([p(1) p(N)],{'max $1-b^{low}$','min $1-b^{low}$'}, 'Location', 'northwest','Interpreter', 'Latex')
% hold on
% plot(1-sol1_bl,br_low_star1,'r*')
% hold on  
% plot(1-sol2_bl, br_low_star2, 'r*')
hold on
xlim([0 1])
ylabel('\textbf{honesty of low cost candidate}', 'Interpreter', 'Latex')
xlabel('\textbf{trust to low cost candidate: }$\mathbf{1-b^{low}}$', 'Interpreter', 'Latex')
hold off 
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';
hline.HandleVisibility = 'off';

subplot(1,2,2)
for i = 1:N
 q(i) = plot(1-bh(:), br_high_mat(:,i), 'Linewidth', 2)  
 legendInfo{i} = ['b_l = ' num2str(round(bl(i),2))];
%  legend(legendInfo, 'Location','northwest');
 hold on
end
legend([q(1) q(N)],{'max $1-b^{high}$','min $1-b^{high}$'}, 'Location', 'northwest','Interpreter', 'Latex')
xlim([0 1])
ylabel('\textbf{honesty of high cost candidate}', 'Interpreter', 'Latex')
xlabel('\textbf{trust to high cost candidate: }$\mathbf{1-b^{high}}$', 'Interpreter', 'Latex')
hold on 
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';
hline.HandleVisibility = 'off';
% han=axes(figure(7),'visible','off'); 
% han.Title.Visible='on';
% han.XLabel.Visible='on';
% han.YLabel.Visible='on';
% %ylabel(han,'yourYLabel');
% %xlabel(han,'$(1-b^{low})P(c^{low})+(1-b^{high})P(c^{high})$', 'Interpreter', 'Latex');
sgtitle('\textbf{Equilibrium conditions}', 'Interpreter', 'Latex');

% Adjust the paper size
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, 9, 4]); % Set the paper size to match the figure size


% % Export the plot as a PNG
exportgraphics(gcf, 'figure7_new.png', 'Resolution', 300); % Save as PNG with 300 DPI resolution

%% Figure 8:  
%Solve symbolically for the bl bh, the system of equilibrium conditions.
syms cl ch p_cl p0 q p bl bh
eq1 = ((p0*q+(1-p0)*p)./((1-bl).^2*cl)+(p0*(1-q)+(1-p0)*(1-p))./((1-bh).^2*cl)+1).^(-1) == 1-bl
eq2 = ((p0*q+(1-p0)*p)./((1-bl).^2*ch)+(p0*(1-q)+(1-p0)*(1-p))./((1-bh).^2*ch)+1).^(-1) == 1-bh
S = solve(eq1, eq2, bl, bh);
sol_bl =  simplify(S.bl)
sol_bh = simplify(S.bh)

% Assign each solution to distinct symbolic variable
sol1_bl = sol_bl(1,1)
sol2_bl = sol_bl(2,1)
sol1_bh = sol_bh(1,1)
sol2_bh = sol_bh(2,1)

diff(sol1_bl, 1)

N=17
p0=linspace(0,1,N)
% Parametrization 
cl = 5;                          % Low cost of lying
ch =15;                          % High cost of lying
p = 0.2;                         % probability of low cost when state is theta_high
q = 0.8;                       
%bl1
sol1_bl_fnc = matlabFunction(sol1_bl)
bl1 = feval(sol1_bl_fnc,ch,cl,p,p0,q)

%bl2
sol2_bl_fnc = matlabFunction(sol2_bl)
bl2 = feval(sol2_bl_fnc,ch,cl,p,p0,q)

%bh1
sol1_bh_fnc = matlabFunction(sol1_bh)
bh1 = feval(sol1_bh_fnc,ch,cl,p,p0,q)

%bh2
sol2_bh_fnc = matlabFunction(sol2_bh)
bh2 = feval(sol2_bh_fnc,ch,cl,p,p0,q)

figure('Units', 'inches', 'Position', [1, 1, 6, 4]); % Set the figure size (6 inches by 4 inches))
bl1 = plot(p0,1-bl1, 'Linewidth',2);
hold on;
bl2 = plot(p0,1-bl2, 'Linewidth',2);
hold on;
bh1 = plot(p0,1-bh1, 'Linewidth',2);
hold on;
bh2 = plot(p0,1-bh2, 'Linewidth',2);
hold off;
xlabel('\textbf{prior belief:} $\mathbf{p_0}$', 'Interpreter', 'Latex');
ylabel('\textbf{trust levels}', 'Interpreter', 'Latex');
lgd = legend([bl1 bl2 bh1 bh2],...
             {'$1-b_{low,1}^*$','$1-b_{low,2}^*$','$1-b_{high,1}^*$','$1-b_{high,2}^*$'},...
              'Interpreter', 'Latex');
lgd.Position = [0.15, 0.5 - lgd.Position(4)/2, lgd.Position(3), lgd.Position(4)];

% Adjust the paper size
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, 6, 4]); % Set the paper size to match the figure size


% % Export the plot as a PNG
exportgraphics(gcf, 'figure8_new.png', 'Resolution', 300); % Save as PNG with 300 DPI resolution

%% Figure B3: Contour sets
cl = 5;                          % Low cost of lying
ch =15;                          % High cost of lying
p0=0.5;                          % prior belief 
p = 0.2;                         % probability of low cost when state is theta_high
q = 0.8;                         % probability of low cost when state is theta_low
N = 17;
bl = linspace(0,1,N);      % note that b^low>b^high. bl: discount when i believe that challenger is low cost
bh = linspace(0,01,N);      % discount when I believe that challenger is high cost
[BL,BH] = meshgrid(1-bl,1-bh);
Z1 = ((p0*q+(1-p0)*p)./((BL).^2*cl)+(p0*(1-q)+(1-p0)*(1-p))./((BH).^2*cl)+1).^(-1) -(BL)
Z2 = ((p0*q+(1-p0)*p)./((BL).^2*ch)+(p0*(1-q)+(1-p0)*(1-p))./((BH).^2*ch)+1).^(-1) -(BH)

% Find also here the equilibrium b's for p0 = 0.5
blow1 = feval(sol1_bl_fnc,ch,cl,p,p0,q)
blow2 = feval(sol2_bl_fnc,ch,cl,p,p0,q)
bhigh1 = feval(sol1_bh_fnc,ch,cl,p,p0,q)
bhigh2 = feval(sol2_bh_fnc,ch,cl,p,p0,q)

figure(3);
subplot(2,1,1)
[c,h] = contour(BL,BH,Z1,'ShowText','on', 'Color', [0, 0.4470, 0.7410])
hold on 
[n m] = contour(BL,BH,Z2,'ShowText','on', 'Color', [1, 0.8, 0])
hold on 
[c1,h1] = contour(BL,BH,Z1,[0 0], 'Color', [0, 0.4470, 0.7410], 'Linewidth', 2)
hold on 
[n1 m1] = contour(BL,BH,Z2,[0 0],'Color', [1, 0.8, 0], 'Linewidth', 2)
hold on
plot(1-blow1, 1-bhigh1, 'r*','MarkerSize',20)
hold on 
plot(1-blow2, 1-bhigh2, 'r*','MarkerSize',20)
legend("c_{low}", "c_{high}")
% hline = refline([0 1-bhigh1]);
% plot([0,1-blow1], [1-blow1, 1-bhigh1], 'r')
% hline.Color = 'r';
% dim = [.13 0.2 .3 .3];
% str = 'P(\theta^{low})=0.5';
% annotation('textbox',dim,'String',str,'FitBoxToText','on');
% xlabel('Trust level: $1-b_{low}$', 'Interpreter', 'Latex')
% ylabel('Trust level: $1-b_{high}$', 'Interpreter', 'Latex')
% title('for $P(\theta^{low}) = 0.5$', 'Interpreter', 'Latex')
title('for $P(\theta^{low}) = 0.5$', 'Interpreter', 'Latex', 'Units', 'normalized', 'Position', [0.88, -0.3, 0.3]);

% Evalueat the same thing for p0=0.8
p0=0.8;                          % prior belief 
% p = 0.2;                         % probability of low cost when state is theta_high
% q = 0.8;                         % probability of low cost when state is theta_low
% N = 17
% bl = linspace(0,1,N);      % note that b^low>b^high. bl: discount when i believe that challenger is low cost
% bh = linspace(0,01,N);      % discount when I believe that challenger is high cost
% [BL,BH] = meshgrid(bl,bh);
Z1 = ((p0*q+(1-p0)*p)./((BL).^2*cl)+(p0*(1-q)+(1-p0)*(1-p))./((BH).^2*cl)+1).^(-1) -(BL)
Z2 = ((p0*q+(1-p0)*p)./((BL).^2*ch)+(p0*(1-q)+(1-p0)*(1-p))./((BH).^2*ch)+1).^(-1) -(BH)

% Evaluate the equilibrium b's 
blow1 = feval(sol1_bl_fnc,ch,cl,p,p0,q)
blow2 = feval(sol2_bl_fnc,ch,cl,p,p0,q)
bhigh1 = feval(sol1_bh_fnc,ch,cl,p,p0,q)
bhigh2 = feval(sol2_bh_fnc,ch,cl,p,p0,q)


subplot(2,1,2)
[c,h] = contour(BL,BH,Z1,'ShowText','on', 'Color', [0, 0.4470, 0.7410])
hold on 
[n m] = contour(BL,BH,Z2,'ShowText','on', 'Color', [1, 0.8, 0])
hold on 
[c1,h1] = contour(BL,BH,Z1,[0 0], 'Color', [0, 0.4470, 0.7410], 'Linewidth', 2)
hold on 
[n1 m1] = contour(BL,BH,Z2,[0 0],'Color', [1, 0.8, 0], 'Linewidth', 2)
hold on
plot(1-blow1, 1-bhigh1, 'r*','MarkerSize',20)
hold on 
plot(1-blow2, 1-bhigh2, 'r*','MarkerSize',20)
legend("c_{low}", "c_{high}")
% xlabel('Trust level: $1-b_{low}$', 'Interpreter', 'Latex')
% ylabel('Trust level: $1-b_{high}$', 'Interpreter', 'Latex')
title('for $P(\theta^{low}) = 0.8$', 'Interpreter', 'Latex', 'Units', 'normalized', 'Position', [0.88, -0.3, 0.3]);
han=axes(figure(3),'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel('\textbf{trust level:} $\mathbf{1-b_{high}}$', 'Interpreter', 'Latex');
xlabel('\textbf{trust level:} $\mathbf{1-b_{low}}$', 'Interpreter', 'Latex')
title(han,'Isolines for different priors');

fig = figure(3)
fig.Units = 'inches'
fig.Position = [1, 1, 7, 4]

% Adjust the paper size
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, 7, 4]); % Set the paper size to match the figure size

exportgraphics(gcf, 'figureB3_new.png', 'Resolution', 300); % Save as PNG with 300 DPI resolution
