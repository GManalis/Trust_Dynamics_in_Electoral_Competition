%% Figure 1: Eurobarometer data
clear all
close all
clc
% Read thE CSV files with the Eurobarometer data
% Use your directory
data = readtable('MATLAB Drive/ARIA/Eurobarometer.csv');
% countries = unique(data.Country);

% Rank them for the same colouring across the two panels
% Define the desired order of countries
desiredOrder = {'Cyprus', 'Greece', 'Ireland', 'Portugal', 'Spain', 'Denmark', 'Latvia',...
                'Netherlands'};

% Create a ranking map based on the desired order
countryRankMap = containers.Map(desiredOrder, 1:length(desiredOrder));

% Add the rank column to the data table
data.rank = zeros(height(data), 1);
for i = 1:height(data)
    data.rank(i) = countryRankMap(data.Country{i});
end

% Sort the data based on the 'rank' column
data = sortrows(data, 'rank');

% Sort the data based on the 'rank' column
data = sortrows(data, 'rank');

% Extract unique countries in the new order
countries = unique(data.Country, 'stable');

% Insert the left panel. 
data_med = readtable('./Data_figure1/Eurobarometer_pigs.csv');
countries_med = unique(data_med.Country);

% Define line styles and colors
lineStyles = {'-', '--', ':', '-.', '-', '--', ':', '-.'};
lineColors = lines(length(countries));  % Generate distinct colors for each line

% Create the plot
figure('Units', 'inches', 'Position', [1, 1, 12, 5]); % Set the figure size (6 inches by 4 inches))
subplot(1,2,1);
hold on;

for i = 1:length(countries)
    % Extract data for each country
    countryData = data(strcmp(data.Country, countries{i}), :);
    
    % Plot data for the current country
    plot(countryData.Year, countryData.mean_trust, 'LineStyle', lineStyles{i}, ...
         'Color', lineColors(i, :), 'LineWidth', 2, 'DisplayName', countries{i});
end
ylim([0, 0.7])
% Add title and labels
xlabel('\textbf{year}', 'Interpreter', 'Latex');
ylabel('\textbf{trust level}', 'Interpreter', 'Latex');
legend('show', 'Location', 'northwest', 'FontSize', 5, 'FontWeight', 'bold');
grid on;
hold off;


eventYears = [2008, 2010, 2011, 2012]; % Example event years
eventLabels = {'Financial crisis', 'GRC & IRL enter IMF-EC-ECB programs', 'PRT enters IMF-EC-ECB program', 'ESP enters IMF-EC-ECB program'}; % Example event labels
eventLineStyles = {'--', ':', '-.', '-'}; % Example line styles for events
eventColors = {'[0.4940 0.1840 0.5560]', 'k', '[0.6350 0.0780 0.1840]', '[0.8500 0.3250 0.0980]'}; % Example colors for events

lineStyles = {'-', '--', ':', '-.', '-'};
% lineColors = lines(length(countries_med));
% Initialize arrays for legend entries
legendEntries = cell(length(countries_med) + length(eventYears), 1);
legendLabels = cell(length(countries_med) + length(eventYears), 1);

subplot(1,2,2)
hold on;

% Plot data for each country
for i = 1:length(countries_med)
    % Extract data for each country
    countryData = data(strcmp(data.Country, countries_med{i}), :);
    
    % Plot data for the current country
    h = plot(countryData.Year, countryData.mean_trust, 'LineStyle', lineStyles{i}, ...
             'Color', lineColors(i, :), 'LineWidth', 2);
    
    % Store legend entry
    legendEntries{i} = h;
    legendLabels{i} = countries_med{i};
end

% Plot vertical lines for specific events and add them to the legend
for j = 1:length(eventYears)
    h = xline(eventYears(j), eventLineStyles{j}, 'Color', eventColors{j}, 'LineWidth', 2);
    
    % Add a hidden line for legend
    eventHandle = plot(nan, nan, eventLineStyles{j}, 'Color', eventColors{j}, 'LineWidth', 2);
    
    % Store legend entry
    legendEntries{length(countries_med) + j} = eventHandle;
    legendLabels{length(countries_med) + j} = eventLabels{j};
end

ylim([0, 0.7])

% Add title and labels
xlabel('\textbf{year}', 'Interpreter', 'Latex');
ylabel('\textbf{trust level}', 'Interpreter', 'Latex');

% Add legend with specified entries
lgd = legend([legendEntries{:}], legendLabels, 'Location', 'northwest');
lgd.FontSize = 5; 
lgd.FontWeight = 'bold';
% [left, bottom, width, height]
% lgd.Position = [0.7, 0.5, 0.2, 0.4];
% Add grid
grid on;

% Finish plot
hold off;
sgtitle('\textbf{Trust levels in European countries}', 'Interpreter', 'Latex')

% Adjust the paper size
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [1, 1, 12, 5]); % Set the paper size to match the figure size


% % Export the plot as a PNG
exportgraphics(gcf, 'Figure_1.png', 'Resolution', 300); % Save as PNG with 300 DPI resolution
