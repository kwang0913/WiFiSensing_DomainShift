% Sample data for the bar chart
userGroups = 1:5; % This represents the user variations on the x-axis
baselineData = [0.6, 0.7, 0.8, 0.5, 0.9]; % Replace with your actual data
profileBasedData = [0.7, 0.8, 0.9, 0.6, 1.0]; % Replace with your actual data
conformityBasedData = [0.8, 0.9, 0.95, 0.65, 0.95]; % Replace with your actual data

% Combine the data into a matrix where each column is a set of data
dataMatrix = [baselineData; profileBasedData; conformityBasedData]';

% Create the bar chart
bar(userGroups, dataMatrix, 'grouped');

% Add labels and title
xlabel('User Variations');
ylabel('Activity Recognition Accuracy');
title('Comparison of Recognition Accuracy');

% Add a legend
legend('Baseline', 'Profile-based fusion', 'Conformity-based fusion', 'Location', 'NorthEastOutside');

% Optionally, set the x-axis tick labels to custom names
set(gca, 'XTickLabel', {'User 1', 'User 2', 'User 3', 'User 4', 'User 5'});

% Adjust the y-axis limits if necessary
ylim([0 1]);

% Apply grid lines for better readability
grid on;
