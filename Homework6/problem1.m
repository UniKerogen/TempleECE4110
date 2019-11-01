clear; clc;
syms x a
% Set up corresponding PDF function
function1 = x + 1;
function2 = (6/a^2) * x + (6/a);

% Case 1 where a > 6
p1 = 1/4 * int(function1, x, -1, (a^2-6*a) / (6-a^2)) + ...
    1/4 * int(function2, x, (a^2-6*a) / (6-a^2), 0);
simplify(p1);

% Case 2 where 1 < a < 6
p2 = 1/4 * int(function1, x, -1, 0);
simplify(p2);

% Case 3 where 0 < a < 1
p3 = 1/4 * int(function2, x, -a, (a^2-6*a)/(6-a^2)) ...
    + 1/4 * int(function1, x, (a^2-6*a)/(6-a^2), 0);
simplify(p3);

%% Analytical Plot
clear; clc; close all;
% Creating a value for each region
a1 = linspace(6, 20, 10000);
a2 = linspace(1, 6, 1000);
a3 = linspace(0, 1, 1000);

% Apply a values to corresponding region for analytical error rate
p1 = (3.*(2.*a1 - 7))./(4.*(a1.^2 - 6));
p2 = 1/8 + 0 * a2;
p3 = (a3.*(7.*a3 - 12))./(8.*(a3.^2 - 6));

% Generate the plot for Error Rate vs. a Value
plot(a1, p1, 'b', 'LineWidth', 2);
hold on
plot(a2, p2, 'g', 'LineWidth', 2);
plot(a3, p3, 'r', 'LineWidth', 2);
xlabel('a value');
ylabel('Error Rate');
title('Error Rate vs. a Value');

%% Data Generation for Experimental Error Rate Calculation
% clear; clc;

% case 1 - a > 6
fprintf("For Case 1 where a > 6\n");
for a = 30:(-2):8
    error = errorCal(a);
    fprintf("When a = %.f, error rate is %.4f\n", a, error);
end
fprintf("\n");

% Case 2 - 1 < a < 6
fprintf("For Case 2 where 1 < a < 6\n");
for a = 6:(-1):1
    error = errorCal(a);
    fprintf("When a = %.f, error rate is %.4f\n", a, error);
end
fprintf("\n");

% Case 3 - 0 < a < 1
fprintf("For Case 3 where 0 < a < 1\n");
for a = 0.5:(-0.08):0.1
    error = errorCal(a);
    fprintf("When a = %.2f, error rate is %.4f\n", a, error);
end

%%
index = 0; error_list = []; a_list = [];
for a = 0.01 : 0.05 : 30
    index = index + 1;
    error = errorCal(a);
    if a < 1
        error = error * 2;
    end
    error_list(index) = error;
    a_list(index) = a;
end
plot(a_list, error_list);