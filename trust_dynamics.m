clc
clear all 
close all 
%% Static game

syms c b 
% below is the equilibrium condition
eq = ((c*(1-b)^2)/(c*(1-b)^2+1))-(1-b) ==0;
sol = solve(eq, b)
sol(1,1)
sol(2,1)
sol(3,1)

% trying dif. things: 
syms c b 
eq1 = b^3 + c*b -c == 0;
sol = solve(eq1,b)

f1 = 2*c/(c+(c*(c-4))^(1/2))
f2 = (c-(c*(c-4))^(1/2))/2
f11 = matlabFunction(f1)
f22 = matlabFunction(f2)
f11(5)
f22(5)
% If I open and manipulate eq then I read the eq1 which is a 2nd degree 
% polynomial in b which has only two of the solutions of the eq. 
eq1 = c*b^2 -c*b +1 ==0
sol1 = solve(eq1,b)

b = linspace(0, 1, 50);
c = linspace(0.1,10,10);
sol_2 = sol(2);
sol_3 = sol(3);


figure(1)
subplot(1,2,2)
for i = 1:length(c)
q(i)= plot(1-b(:), (c(i).*(1-b).^2)./(c(i).*(1-b).^2+1), 'Linewidth', 2);
title('Best response for different $c$`s', 'Interpreter', 'latex')
hold on 
%xlabel('Voter`s belief ($1-b$)', 'Interpreter', 'Latex')
%ylabel('$br(b;c) = \frac{c\cdot (1-b)^2}{c\cdot(1-b)^2+1}$', 'Interpreter', 'Latex')
% q(i) = plot(1-bh(:), br_high_mat(:,i), 'Linewidth', 2)  
%  legendInfo{i} = ['b_l = ' num2str(round(bl(i),2))];
%  legend(legendInfo, 'Location','northwest');
 hold on
end
legend([q(1) q(length(c))],{'min c','max c'}, 'Location', 'northwest')


% Figure with the equilibrium condition 
c = 6;
y = (c*(1-b).^2)./(1+c*(1-b).^2);
x = 1-b;
figure(1)
%subplot(1,2,1)
plot(x, y, 'Linewidth', 2)
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';
hline.HandleVisibility = 'off';
hold on 
plot(0.2113, 0.2113, 'r*') % these are the sol_2 for c = 6
plot(0.7887, 0.7887, 'r*') % this is the sol_3 for c=6
xlim([0 1])
ylim([0 1])
hold on
% iz=linspace(0.2113,1,40);
% yz=(c*(iz).^2)./(1+c*(iz).^2);
% l= area(iz,yz)
% l(2).FaceColor = [0.2 0.6 0.5];
% l(2).EdgeColor = [0.63 0.08 0.18];
% l(2).LineWidth = 2;
title('Equilibrium condition ($1-b = 1-\gamma)$ for fixed $c$''s', 'Interpreter', 'latex')
% xlabel('Trust level: $(1-b)$', 'Interpreter', 'Latex')
% ylabel('$1-\gamma$', 'Interpreter', 'Latex')
han=axes(figure(1),'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
xlabel('Trust level: $1-b$', 'Interpreter', 'Latex');
ylabel('Honesty level: $1-\gamma$', 'Interpreter', 'Latex')
% title(han,'Equilibria of the Static game');
hold off

%sol2 = matlabFunction(sol_2)
%sol3 = matlabFunction(sol_3)
%%plot of equilibrium type on announcement 
%y = linspace(1,10); 
%x1 = y./(sol2(6)); 
%x2 = y./(sol3(6))
%figure(14)
%plot(y,x1)
%hold on
%plot(y,x2)
%xlabel('y - announcement')
%ylabel('x - type')

%% Dynamic Game w/ Adaptive Beliefs
nb0 = 20; % How many different initial beliefs do you want to consider. 
T = 60; % Essentially how many periods you want. 
b0 = linspace(0,1,nb0); % different initial values of belief 
b = zeros(1,T,nb0); % current b
br = zeros(1, T, nb0); % current gamma
b(:,1,:) = b0'; % initial belief
alpha = 0.5; % alpha is the weight on the mistake
c = 6;  % fixed cost for lying
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


figure(2)
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
title('Dynamic equilibria', 'Interpreter', 'Latex')
legend('$1-b_t$', '$\gamma(b_t)$','Interpreter', 'Latex')
xlabel('time (t)', 'Interpreter', 'Latex')
ylabel('trust, honesty', 'Interpreter', 'Latex')

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
plot(0.2113, 0.2113, 'r*')
plot(0.7887, 0.7887, 'r*')
xlim([0 1])
ylim([0 1])
title('Static game equilibria', 'Interpreter', 'latex')
xlabel('$1-b$: trust', 'Interpreter', 'Latex')
ylabel('honesty', 'Interpreter', 'Latex')
hold off

%% Bayesian Inference - w/ reference to the Gaussian model. 

T = 20
c = linspace(0,10,20) % known state space
% Prior Belief about the state
m0 = 8
sigma0 = 5
y0 = normpdf(c, m0, sigma0)
plot(c, y0)
rho0 = 1/sigma0  % precision
% signal (let c(10) be the selected type)
s0 = c(10) + randn % initial signal 
rho_e = 1 % as the error in the signal specification is stand. normal. ie sigma =1 hence precision =1 

% Initialize the vectors to be used: 
s = zeros(1,T)
s(1) = s0
% mean of the distribution 
m = zeros(1,T)
m(1) = m0

% variance of the distribution 
sigma = zeros(1,T)
sigma(1) = sigma0

% precision of the distribution
rho = zeros(1,T)
rho(1) = rho0


for i = 2:T; 
    s(i) = c(10)+randn
    sigma(i) = (sigma(i-1)*1)/(sigma(i-1)+1)
    rho(i) = 1/sigma(i)
    m(i) = rho_e/rho(i)*s(i) + (1-rho_e/rho(i))*m(i-1)
    y = normpdf(c, m(i), sigma(i))
    figure(6)
    plot(c, y, 'DisplayName', ['t=' num2str(i)])
    hold on
    legend('show');
end 

%% Bayesian Inference w/ reference to the Gaussian applied on the static game




%% Bayesian Inference w/ the two cost distributions. 

% clear all 
% clc 
% close all 
% First I define two outcomes, namely c_low, c_high 
c_low = 6; 
c_high = 10; 
c = [c_low, c_high];  % x: set of outcomes

% Theta_High: this is going to be the true model. 
p = 0.2; % probability under theta_high that the cost is low. (1-p) the cost 
% is high. 
p_high = [p, 1-p]; % p is the probability that cost is low under theta_high

% Next i define the low cost distribution: 
q = 0.8; % probability under theta_low that the cost is low. with (1-p) the
% cost is high. 
q_low = [q, 1-q];

% Next I define the signals. Note that the signals will be the cost of the 
% incumbent - who is going to have the same cost values. 
S = [c_low, c_high]; 
% however, I need to have a conditional probability of signals given the
% models, which would tell me what is the probability of the model being
% the low-cost in case I observe c_low of the incumbent.
phi = 0.75; % Probability of model theta_low when signal is c_low
pi = 0.25; % Probability of model theta_high when signal is c_low
prob_thetal = [phi , 1-phi]; % Note phi corresponds to c_l, 1-phi to c_h under the theta_l model
prob_thetah = [pi , 1-pi]; % Note pi corresponds to c_l, 1-pi to c_h under the theta_h model

% Note that phi and pi need not be complementary. 

%% Incumbent's costs. 
% Generate the sequence of costs for the incumbent which will be drawn from
% the theta_high distribution. 
T = 20; % time periods you want to put
inc_cost = zeros(1,T)
for i = 1:T
    r = rand;
    if r <= p 
        inc_cost(i) = S(1); 
    else 
        inc_cost(i) = S(2); 
    end 
end

%% Updating Rule: 


% Finding the posterior on the STATES/MODELS of the world eq. 10, 11, 12, 13

p0 = 0.5; 
post_prob = zeros(1,T+1)
post_prob(1) = p0
for i=2:T 
    if inc_cost(i) == S(1) 
        % Calculate the posteriors
        post_thetalow_clow = p0*phi/(p0*phi+(1-p0)*pi)
        post_thetahigh_clow = 1-post_thetalow_clow
        % Update the rule
        p0 = post_thetalow_clow; 
    else 
        % Calculate the posteriors
        post_thetalow_chigh = p0*(1-phi)/(p0*(1-phi)+(1-p0)*(1-pi))
        post_thetahigh_chigh = 1- post_thetalow_chigh
        % Update the priors
        p0 = post_thetalow_chigh
    end
        post_prob(i) = p0
end

figure(3)
yyaxis left
plot(post_prob(1:20), 'Linewidth', 2)
ylim([0 0.5])
ylabel('$P(\theta^{low}|c_{inc})$', 'Interpreter', 'Latex')
yyaxis right
plot(inc_cost(:,1:20),'-*')
ylabel('Cost of Incumbent', 'Interpreter', 'Latex')
ylim([6 10])
hold off
title('Signals drawn from $\theta^{high}$ \& posterior $P(\theta^{low}|c_{inc})$', 'Interpreter', 'latex')
xlabel('$t$: Time', 'Interpreter', 'Latex')

% Finding the posterior on the OUTCOMES of the world eq. 14, 15, 16, 17

post_prob_cl = zeros(1,T+1)
post_prob_cl(1) = p0*q + (1-p0)*p
for i=2:T 
    if inc_cost(i) == S(1) 
        % Calculate the posteriors
        post_thetalow_clow = p0*phi/(p0*phi+(1-p0)*pi)
        post_thetahigh_clow = 1-post_thetalow_clow
        % Update the rule
        p0 = post_thetalow_clow; 
        % Now update on the outcomes: 
        post_clow_clow = post_thetalow_clow*q + post_thetahigh_clow*pi
        post_chigh_clow = post_thetalow_clow*(1-q)+post_thetahigh_clow*(1-pi)
        % keep it for a plot
        post_prob_cl(i) = post_clow_clow
    else 
        % Calculate the posteriors
        post_thetalow_chigh = p0*(1-phi)/(p0*(1-phi)+(1-p0)*(1-pi))
        post_thetahigh_chigh = 1- post_thetalow_chigh
        % Update the priors
        p0 = post_thetalow_chigh
        % Now update on the outcomes: 
        post_clow_chigh = post_thetalow_chigh*q + post_thetahigh_chigh*pi
        post_chigh_chigh = post_thetalow_chigh*(1-q)+post_thetahigh_chigh*(1-pi)
        % keep it for a plot
        post_prob_cl(i) = post_clow_chigh
    end
end

figure(4)
yyaxis left
plot(post_prob_cl(1:20), 'Linewidth', 2)
ylim([0 1])
ylabel('$P(c^{low}|c_{inc})$', 'Interpreter', 'Latex')
yyaxis right
plot(inc_cost(:,1:20),'-*')
ylabel('Cost of Incumbent', 'Interpreter', 'Latex')
ylim([6 10])
hold off
title('Signals drawn from $\theta^{high}$ \& posterior $P(c^{low}|c_{inc})$', 'Interpreter', 'latex')
xlabel('$t$: Time', 'Interpreter', 'Latex')






% %% Different Cases of low/high cost incumbent w/ different updates. 
% % Assume now that the voter observes a LOW cost incumbent. Then she would
% % update the probabilities of the challenger being low or high. What I am
% % interested in is to find the updated p_cl and p_ch. i use equations 23, 24 
% % from the draft
% p_cl_upd_low = (p0*q^2+(1-p0)*p^2)/(p0*q+(1-p0)*p);
% p_ch_upd_low = (p0*q*(1-q)+(1-p0)*p*(1-p))/(p0*q+(1-p0)*p)
% % Best responses 
% br_low_upd_low = (p_cl_upd_low./((1-bl).^2*cl)+p_ch_upd_low./((1-bh).^2*cl)+1).^(-1);
% br_high_upd_low = (p_cl_upd_low./((1-bl).^2*ch)+p_ch_upd_low./((1-bh).^2*ch)+1).^(-1);
%  
%  
% % Assume now that the voter observes a HIGH cost incumbent. Then she would
% % update the probabilities of the challenger being low or high. What I am
% % interested in is to find the updated p_cl and p_ch. i use equations 25, 26 
% % from the draft
% p_cl_upd_high = (p0*(1-q)*q+(1-p0)*(1-p)*p)/(p0*(1-q)+(1-p0)*(1-p));
% p_ch_upd_high = (p0*(1-q)^2+(1-p0)*(1-p)^2)/(p0*(1-q)+(1-p0)*(1-p)); 
%  
% % Best responses 
% br_low_upd_high = (p_cl_upd_high./((1-bl).^2*cl)+p_ch_upd_high./((1-bh).^2*cl)+1).^(-1);
% br_high_upd_high = (p_cl_upd_high./((1-bl).^2*ch)+p_ch_upd_high./((1-bh).^2*ch)+1).^(-1);
% % 
%  
%  
%  
% figure(6)
% subplot(1,2,1)
% plot(((1-bl)*p_cl) + ((1-bh)*p_ch), br_low, 'b', 'Linewidth', 1) % priors
% hold on 
% plot(((1-bl)*p_cl_upd_low) + ((1-bh)*p_ch_upd_low), br_low_upd_low,'r-o','Linewidth', 1) % posterior w/ low
% hold on 
% plot(((1-bl)*p_cl_upd_high) + ((1-bh)*p_ch_upd_high), br_low_upd_high,'r*','Linewidth', 1) % posterior w/ high
% ylabel('$BR^{low}$: Best response of low cost challenger', 'Interpreter', 'Latex')
% xlim([0 1])
% legend('w/ prior', 'w/ posterior/low inc', 'w/ posterior/high inc','Location','northwest')
% hold on 
% hline = refline([1 0]);
% hline.Color = 'k';
% hline.LineStyle = ':';
% hline.HandleVisibility = 'off';
%  
% subplot(1,2,2)
% plot(((1-bl)*p_cl) + ((1-bh)*p_ch), br_high,'r', 'Linewidth', 1) % priors
% hold on 
% plot(((1-bl)*p_cl_upd_low) + ((1-bh)*p_ch_upd_low), br_high_upd_low, 'r-o', 'Linewidth', 1) % posterior w/ low
% hold on 
% plot(((1-bl)*p_cl_upd_high) + ((1-bh)*p_ch_upd_high), br_high_upd_high,'r*','Linewidth', 1) % posterior w/ high
% ylabel('$BR^{high}$: Best response of high cost challenger', 'Interpreter', 'Latex')
% xlim([0 1])
% legend('w/ prior', 'w/ posterior/low inc', 'w/ posterior/high inc', 'Location','northwest')
% hold on
% hline = refline([1 0]);
% hline.Color = 'k';
% hline.LineStyle = ':';
% hline.HandleVisibility = 'off';
% %Give common xlabel, ylabel and title to your figure
% % % han=axes(figure(6),'visible','off'); 
% % % han.Title.Visible='on';
% % % han.XLabel.Visible='on';
% % % %han.YLabel.Visible='on';
% % % %ylabel(han,'yourYLabel');
% xlabel(han,'$(1-b^{low})P(c^{low})+(1-b^{high})P(c^{high})$', 'Interpreter', 'Latex');
% title(han,'Equilibrium Conditions');
%  
 
 
 


%% Treating the eq. conditions w/ all possible combos. 


% Parametrization 
cl = 5;                          % Low cost of lying
ch =15;                          % High cost of lying
p0=0.5;                          % prior belief 
p = 0.2;                         % probability of low cost when state is theta_high
q = 0.8;                         % probability of low cost when state is theta_low
p_cl = p0*q+(1-p0)*p;            % Equation 19 in the draft.
p_ch = p0*(1-q)+(1-p0)*(1-p);    % Equation 20 in the draft. 

% Discount / beliefs
N = 17
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


figure(7)
subplot(2,2,3)
for i = 1:N
 p(i) = plot(1-bl(:), br_low_mat(i,:), 'Linewidth', 2)  
 legendInfo{i} = ['b_h = ' num2str(round(bh(i),2))];
%  legend(legendInfo, 'Location','northwest');
 hold on
end
legend([p(1) p(N)],{'min b_h','max b_h'}, 'Location', 'northwest')
% hold on
% plot(1-sol1_bl,br_low_star1,'r*')
% hold on  
% plot(1-sol2_bl, br_low_star2, 'r*')
hold on
xlim([0 1])
ylabel('$BR^{low}$: Best response of low cost challenger', 'Interpreter', 'Latex')
xlabel('$1-b^{low}$', 'Interpreter', 'Latex')
hold off 
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';
hline.HandleVisibility = 'off';

subplot(2,2,4)
for i = 1:N
 q(i) = plot(1-bh(:), br_high_mat(:,i), 'Linewidth', 2)  
 legendInfo{i} = ['b_l = ' num2str(round(bl(i),2))];
%  legend(legendInfo, 'Location','northwest');
 hold on
end
legend([q(1) q(N)],{'min b_h','max b_h'}, 'Location', 'northwest')
xlim([0 1])
ylabel('$BR^{high}$: Best response of high cost challenger', 'Interpreter', 'Latex')
xlabel('$1-b^{high}$', 'Interpreter', 'Latex')
hold off 
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';
hline.HandleVisibility = 'off';
han=axes(figure(7),'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
%ylabel(han,'yourYLabel');
%xlabel(han,'$(1-b^{low})P(c^{low})+(1-b^{high})P(c^{high})$', 'Interpreter', 'Latex');
title(han,'Equilibrium Conditions');


subplot(2,2,1)
surf(1-bl,1-bh,br_low_mat)
xlabel("$1-b^{low}$", "Interpreter", "Latex")
ylabel("$1-b^{high}$", "Interpreter", "Latex")
zlabel('$BR^{low}$: Best response of low cost challenger', 'Interpreter', 'Latex')


subplot(2,2,2)
surf(1-bl,1-bh,br_high_mat)
xlabel("$1-b^{low}$", "Interpreter", "Latex")
ylabel("$1-b^{high}$", "Interpreter", "Latex")
zlabel('$BR^{high}$: Best response of high cost challenger', 'Interpreter', 'Latex')




%% Solve symbolically for the bl bh, the system of equilibrium conditions.


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


figure(10)
plot(p0,1-bl1, 'Linewidth',2)
hold on 
plot(p0,1-bl2, 'Linewidth',2)
hold on
plot(p0,1-bh1, 'Linewidth',2)
hold on
plot(p0,1-bh2, 'Linewidth',2)
legend('1-b_{low,1}^*','1-b_{low,2}^*','1-b_{high,1}^*','1-b_{high,2}^*', 'Interpreter ', 'Latex')
xlabel('p0: prior')
ylabel('$1-b_{low,1}^*$, $1-b_{low,2}^*$, $1-b_{high,1}^*$, $1-b_{high,2}^*$', 'Interpreter', 'Latex')
hold off
         

p0 = 0.5; % I need this to take the value 0.5 for the remaining of the script

% %% Below I repeat the two sections above but now I am considering also the
% %  updating belief of the voter about the state of the world, given the
% %  information derived from observing the incumbent. 
% % Parametrization 
% cl = 5;                          % Low cost of lying
% ch =15;                          % High cost of lying
% p0=0.5;                          % prior belief 
% p = 0.2;                         % probability of low cost when state is theta_high
% q = 0.8;                         % probability of low cost when state is theta_low
% p_cl = p0*q+(1-p0)*p;            % Equation 19 in the draft.
% p_ch = p0*(1-q)+(1-p0)*(1-p);    % Equation 20 in the draft. ------ Nothing changes in the parametrisation 
% 
% % Discount / beliefs
% N = 17
% bl = linspace(0,1,N);      % note that b^low>b^high. bl: discount when i believe that challenger is low cost
% bh = linspace(0,01,N);      % discount when I believe that challenger is high cost
% %b = linspace(0.01, 0.99, 50)
% 
% %..% Case 0: Using only the priors
% br_low = (p_cl./((1-bl).^2*cl)+p_ch./((1-bh).^2*cl)+1).^(-1);  % Equation 21.2 in the draft
% br_high = (p_cl./((1-bl).^2*ch)+p_ch./((1-bh).^2*ch)+1).^(-1); % Equation 21.1 in the draft
% 
% 
% %..% Case I: Assume that the incumbent is of low cost, then the probabilities 
% % P(c_low) and P(c_high) about the challenger would be updated according to
% % eq. 23 and eq. 24 respectively. 
% p_cl_upd_low = (p0*q^2+(1-p0)*p^2)/(p0*q+(1-p0)*p);          % Equation 23
% p_ch_upd_low = (p0*q*(1-q)+(1-p0)*p*(1-p))/(p0*q+(1-p0)*p)   % Equation 24
% 
% % Best responses 
% br_low_upd_low = (p_cl_upd_low./((1-bl).^2*cl)+p_ch_upd_low./((1-bh).^2*cl)+1).^(-1);
% br_high_upd_low = (p_cl_upd_low./((1-bl).^2*ch)+p_ch_upd_low./((1-bh).^2*ch)+1).^(-1);
% 
% %. In matrix form: 
% % components at the berst response of the low cost. 
% A_l_upd_low = p_cl_upd_low./((1-bl).^2*cl)
% B_l_upd_low = p_ch_upd_low./((1-bh).^2*cl)+1
%  
% % % components of the best response of the high cost
% A_h_upd_low = p_cl_upd_low./((1-bl).^2*ch)
% B_h_upd_low = p_ch_upd_low./((1-bh).^2*ch)+1
%  
% % Turning the components into matrices
% % low cost
% A_l_mat_upd_low = repmat(A_l_upd_low, N,1)
% B_l_mat_upd_low = repmat(B_l_upd_low',1,N)
% % high cost
% A_h_mat_upd_low = repmat(A_h_upd_low,N,1)
% B_h_mat_upd_low = repmat(B_h_upd_low',1,N)
% % 
% % % Best responses as matrices 
% br_low_mat_upd_low = (A_l_mat_upd_low + B_l_mat_upd_low).^(-1) % each row different bh each column different bl 
% br_high_mat_upd_low = (A_h_mat_upd_low+B_h_mat_upd_low).^(-1)  % each row different bh each column different bl
% 
% 
% %..% Case II: Assume now that the voter observes a HIGH cost incumbent. Then she would
% % update the probabilities of the challenger being low or high. What I am
% % interested in is to find the updated p_cl and p_ch. i use equations 25, 26 
% % from the draft
% 
% p_cl_upd_high = (p0*(1-q)*q+(1-p0)*(1-p)*p)/(p0*(1-q)+(1-p0)*(1-p));   % Equation 25
% p_ch_upd_high = (p0*(1-q)^2+(1-p0)*(1-p)^2)/(p0*(1-q)+(1-p0)*(1-p));   % Equation 26
%  
% % Best responses 
% br_low_upd_high = (p_cl_upd_high./((1-bl).^2*cl)+p_ch_upd_high./((1-bh).^2*cl)+1).^(-1);
% br_high_upd_high = (p_cl_upd_high./((1-bl).^2*ch)+p_ch_upd_high./((1-bh).^2*ch)+1).^(-1);
% 
% %. In matrix form: 
% % components at the berst response of the low cost. 
% A_l_upd_high = p_cl_upd_high./((1-bl).^2*cl)
% B_l_upd_high = p_ch_upd_high./((1-bh).^2*cl)+1
%  
% % % components of the best response of the high cost
% A_h_upd_high = p_cl_upd_high./((1-bl).^2*ch)
% B_h_upd_high = p_ch_upd_high./((1-bh).^2*ch)+1
%  
% % Turning the components into matrices
% % low cost
% A_l_mat_upd_high = repmat(A_l_upd_high, N,1)
% B_l_mat_upd_high = repmat(B_l_upd_high',1,N)
% % high cost
% A_h_mat_upd_high = repmat(A_h_upd_high,N,1)
% B_h_mat_upd_high = repmat(B_h_upd_high',1,N)
% % 
% % % Best responses as matrices 
% br_low_mat_upd_high = (A_l_mat_upd_high + B_l_mat_upd_high).^(-1) % each row different bh each column different bl 
% br_high_mat_upd_high = (A_h_mat_upd_high+B_h_mat_upd_high).^(-1)  % each row different bh each column different bl
% 
% 
% figure(7)
% subplot(1,2,1)
% plot(1-bl(:), br_low_mat(:,10), 'Linewidth', 2)  
%  hold on
% plot(1-bl(:), br_low_mat_upd_low(:,10),'-o' ,'Linewidth', 1) 
%  hold on
% plot(1-bl(:), br_low_mat_upd_high(:,10), '-*', 'Linewidth', 1)
% legend('prior', 'posterior w/ low inc', 'posterior w/ high inc', 'Location', 'northwest')
% % hold on
% % plot(1-sol1_bl,br_low_star1,'r*')
% % hold on  
% % plot(1-sol2_bl, br_low_star2, 'r*')
% hold on
% xlim([0 1])
% ylabel('$BR^{low}$: Best response of low cost challenger', 'Interpreter', 'Latex')
% xlabel('$1-b^{low}$', 'Interpreter', 'Latex')
% hold off 
% hline = refline([1 0]);
% hline.Color = 'k';
% hline.LineStyle = ':';
% hline.HandleVisibility = 'off';
% 
% subplot(1,2,2)
% plot(1-bh(:), br_high_mat(:,10), 'Linewidth', 2)  
%  hold on
% plot(1-bh(:), br_high_mat_upd_low(:,10),'-o' ,'Linewidth', 1) 
%  hold on
% plot(1-bh(:), br_high_mat_upd_high(:,10), '-*', 'Linewidth', 1)
% legend('prior', 'posterior w/ low inc', 'posterior w/ high inc', 'Location', 'northwest')
% hold on
% xlim([0 1])
% ylabel('$BR^{high}$: Best response of high cost challenger', 'Interpreter', 'Latex')
% xlabel('$1-b^{high}$', 'Interpreter', 'Latex')
% hold off 
% hline = refline([1 0]);
% hline.Color = 'k';
% hline.LineStyle = ':';
% hline.HandleVisibility = 'off';
% han=axes(figure(7),'visible','off'); 
% han.Title.Visible='on';
% han.XLabel.Visible='on';
% han.YLabel.Visible='on';
% %ylabel(han,'yourYLabel');
% %xlabel(han,'$(1-b^{low})P(c^{low})+(1-b^{high})P(c^{high})$', 'Interpreter', 'Latex');
% title(han,'Equilibrium Conditions');


%% Contour sets
cl = 5;                          % Low cost of lying
ch =15;                          % High cost of lying
p0=0.5;                          % prior belief 
p = 0.2;                         % probability of low cost when state is theta_high
q = 0.8;                         % probability of low cost when state is theta_low
N = 17
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


figure(11)
subplot(2,1,1)
[c,h] = contour(BL,BH,Z1,'ShowText','on', 'Color', [0 1 0.6])
hold on 
[n m] = contour(BL,BH,Z2,'ShowText','on', 'Color', [0 0.3 1])
hold on 
[c1,h1] = contour(BL,BH,Z1,[0 0], 'Color', [0 1 0.6], 'Linewidth', 2)
hold on 
[n1 m1] = contour(BL,BH,Z2,[0 0],'Color', [0 0.3 1], 'Linewidth', 2)
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
[c,h] = contour(BL,BH,Z1,'ShowText','on', 'Color', [0 1 0.6])
hold on 
[n m] = contour(BL,BH,Z2,'ShowText','on', 'Color', [0 0.3 1])
hold on 
[c1,h1] = contour(BL,BH,Z1,[0 0], 'Color', [0 1 0.6], 'Linewidth', 2)
hold on 
[n1 m1] = contour(BL,BH,Z2,[0 0],'Color', [0 0.3 1], 'Linewidth', 2)
hold on
plot(1-blow1, 1-bhigh1, 'r*','MarkerSize',20)
hold on 
plot(1-blow2, 1-bhigh2, 'r*','MarkerSize',20)
legend("c_{low}", "c_{high}")
% xlabel('Trust level: $1-b_{low}$', 'Interpreter', 'Latex')
% ylabel('Trust level: $1-b_{high}$', 'Interpreter', 'Latex')
title('for $P(\theta^{low}) = 0.8$', 'Interpreter', 'Latex', 'Units', 'normalized', 'Position', [0.88, -0.3, 0.3]);
han=axes(figure(11),'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel('Trust level: $1-b_{high}$', 'Interpreter', 'Latex');
xlabel('Trust level: $1-b_{low}$', 'Interpreter', 'Latex')
title(han,'Isolines for Different priors');

%% Unknown Function of voter's beliefs. 
% So far we have assumed that the voter discounts linearly the announcment
% of the voter- now we do not take a stance on that and we proceed by
% taking a function with the only condition of being increasing. 


%%  example 
close all 
clear all 
clc
syms f(y) c x
% c = 6
ode = diff(f,y,1) == c - c*y/f;
ySol(y) = dsolve(ode)
ySol = simplify(ySol)
ySol = formula(ySol)
sol_1 = ySol(1)
sol_2 = ySol(2)
%sol_3 = ySol(3)
% now take the f'(y) which is the coefficient, notice that these 
% are the corresponding (1-b)s
dif_sol_1 = diff(sol_1, y,1)
dif_sol_2 = diff(sol_2,y,1)
% 
% Now in order to create the y = g(x) you get the following: 
announce_1 = x./dif_sol_1; 
announce_2 = x./dif_sol_2

% Render them as functions to plot them
announce_1 = matlabFunction(announce_1)
announce_2 = matlabFunction(announce_2)

% Plot the announcement-type figure 
x = linspace(1,10) % the types
c=6                % fixed cost of lying

figure(12)
subplot(1,2,1)
plot(x, announce_1(c,x), 'Linewidth',2)
hold on 
plot(x, announce_2(c,x), 'Linewidth',2)
hold on 
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';
ylabel('y = g(x) - announcement', 'Interpreter', 'Latex')
xlabel('x type', 'Interpreter', 'Latex')
legend("big lies",  "little lies")
title("Equilibrium w/ voter's beliefs: $\hat{x} = f(y)$", 'Interpreter', 'Latex')


% Below I have the previous system where I have assumed linear beliefs.
syms c b ;
eq = (c*(1-b)^2)/(c*(1-b)^2+1)-(1-b) ==0;
sol = solve(eq, b);
solution_1 = matlabFunction(sol(2,1))
solution_1(6)
solution_2 = matlabFunction(sol(3,1))
solution_2(6)
y = linspace(1,10) 
y1 = y*solution_1(6);
y2 = y*solution_2(6)
subplot(1,2,2)
plot(y,y1, 'Linewidth',2)
hold on 
plot(y, y2, 'Linewidth',2)
hold on 
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';
ylabel('y - announcement', 'Interpreter', 'Latex')
xlabel('x type', 'Interpreter', 'Latex')
legend("big lies",  "little lies")
title("Equilibrium w/ voter's beliefs: $\hat{x} = \frac{y}{1-b}$", 'Interpreter', 'Latex')
hold off

%% 
close all 

% Voter side - Trust levels 
trust_l = matlabFunction(sol_1) % its the equivalent of eq. 6.1
trust_h = matlabFunction(sol_2) % its the equivalent of eq. 6.2
trust_l(5,4)
trust_h(5,4)


% Candidate's side - Honesty levels solving the br for y 
syms c x
announcement_l = x/(1 + (1/c)*(dif_sol_1)^2)
announcement_h = x/(1 + (1/c)*(dif_sol_2)^2)
announcement_l = matlabFunction(announcement_l)
announcement_h = matlabFunction(announcement_h)


% Parameters
c=6
y = linspace(-10,10)  % announcements - only used for the voters' side. 
x = linspace(-10,10)  % types - only used for the candidate's side


%  Figure 1 on the draft. 
figure(13)
subplot(2,1,2)
plot(y,trust_l(c,y), 'Linewidth', 2)
hold on 
plot(y,trust_h(c,y), 'Linewidth', 2)
hold on 
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';
ylabel('$\hat{x}_j$: inferred type ', 'Interpreter', 'Latex')
xlabel('$y_j$: announcement', 'Interpreter', 'Latex')
title('Equilibrium beliefs of the voter', 'Interpreter', 'Latex')
legend('low trust', 'high trust', '45^o', 'Location', 'northwest')

subplot(2,1,1)
plot(x, announcement_l(c,x), 'Linewidth', 2)
hold on 
plot(x, announcement_h(c,x), 'Linewidth', 2)
hold on 
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';
ylabel('$y_j$: announcement ', 'Interpreter', 'Latex')
xlabel('$x_j$: type', 'Interpreter', 'Latex')
title('Equilibrium announcement by the candidate', 'Interpreter', 'Latex')
legend('low trust', 'high trust', '45^o', 'Location', 'northwest')

%%
close all 
c = linspace(6,12,4)
%  Figure 1 on the draft. 
figure(14)
subplot(1,2,2)
for i = 1:length(c)
plot(y,trust_l(c(i),y),'-o', 'Linewidth', 1)
hold on 
end
hold on 
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';
ylabel('$\hat{x}_j$: inferred type ', 'Interpreter', 'Latex')
xlabel('$y_j$: announcement', 'Interpreter', 'Latex')
title('Equilibrium inferred type by the voter under low trust', 'Interpreter', 'Latex')
legend('c=6', 'c=8', 'c=10', 'c=12', 'Location', 'northwest')

subplot(1,2,1)
for i = 1:length(c)
plot(y,trust_h(c(i),y), '--', 'Linewidth', 1)
hold on
end
hold on
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';
ylabel('$\hat{x}_j$: inferred type ', 'Interpreter', 'Latex')
xlabel('$y_j$: announcement', 'Interpreter', 'Latex')
title('Equilibrium inferred type by the voter under high trust', 'Interpreter', 'Latex')
legend('c=6', 'c=8', 'c=10', 'c=12', 'Location', 'northwest')

% Or equivalently, 
slope_l = matlabFunction(dif_sol_1) % its the slope of eq. 6.1
slope_h = matlabFunction(dif_sol_2) % its the slope of eq. 6.2

c = linspace(6,8)
figure(15)
subplot(1,2,2)
plot(c, slope_l(c),'Linewidth', 2)
hold on 
plot(c, slope_h(c),'Linewidth', 2)
ylabel('$\frac{\partial^2f(y)}{\partial y\partial c}$', 'Interpreter', 'Latex')
xlabel('$c$: fixed cost', 'Interpreter', 'Latex')
title('Effect of the fixed cost on the slope of equilibrium beliefs', 'Interpreter', 'Latex')
legend('$\frac{\partial^2\ddot{f}(y)}{\partial y\partial c}$: low trust',...
    '$\frac{\partial^2\dot{f}(y)}{\partial y\partial c}$: high trust', 'Location', 'northwest', 'Interpreter', 'Latex')

syms x c
announcement_l = x/(1 + (1/c)*(dif_sol_1)^2)
announcement_h = x/(1 + (1/c)*(dif_sol_2)^2)
dif_announcement_l = diff(announcement_l,x)
dif_announcement_h = diff(announcement_h,x)
slope_ann_l = matlabFunction(dif_announcement_l)
slope_ann_h = matlabFunction(dif_announcement_h)

c = linspace(6,8)

subplot(1,2,1)
plot(c, slope_ann_l(c),'Linewidth', 2)
hold on 
plot(c, slope_ann_h(c),'Linewidth', 2)
ylabel('$\frac{\partial^2f(y)}{\partial y\partial c}$', 'Interpreter', 'Latex')
xlabel('$c$: fixed cost', 'Interpreter', 'Latex')
title('Effect of the fixed cost on the slope of equilibrium announcements', 'Interpreter', 'Latex')
legend('$\frac{\partial^2\ddot{g}(x)}{\partial x\partial c}$: low trust',...
    '$\frac{\partial^2\dot{g}(x)}{\partial x\partial c}$: high trust', 'Location', 'northwest', 'Interpreter', 'Latex')



%% Unknown Function in both the voter's beliefs and how it enters candidate's objective. 
close all 
clear all 
clc
syms fpr c hpr 
eq1 = hpr*(fpr)^2-c*fpr +1 == 0; 
sol = solve(eq1,fpr)

