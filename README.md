# Trust_Dynamics_in_Electoral_Competition
Includes the R & Matlab code for the numerical analysis of the paper "Trust Dynamics in Electoral Competition" by Nectaria Glynia, Georgios Manalis and Dimitrios Xefteris.

## Eurobarometer data - Figure 1
Figure_1_Eurobarometer_data.R file receives the Eurobarometer.csv file as input and creates the annual mean of share of people that tend to trust national governments, for selected European countries. 
This gives as output: Eurobarometer_final.csv
## A_figure1.m 
This is the matlab file that creates Figure 1 of the draft. It receives as input the Euorbarometer_final.csv, generated above. 
## B_static_game_prop1_fig2_figB2.m
This is the matlab file that solves the first order differential equation of the static game and creates Figure 2 and Figure B2. 
## C_dynamic_game_prop2_prop3_fig3_fig4.m
This is the matlab file that gives the solution of the dynamic game presented in Proposition 2 and creates Figure 3 and Figure 4. 
## D_example_fig5_fig6.m
This is the matlab file that creates Figure 5 and Figure 6 of the example with countries differing at c and alpha. 
## E_alternative_beliefs_update_fig7_fig8_figB3.m
This is the matlab file that solves the extension of alternative beliefs with Bayesian updating and creates Figure 7, Figure 8 and Figure B3. 
