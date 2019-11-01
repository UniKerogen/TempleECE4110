function error_rate=errorCal(a_value)
%
% FUNCTION error_rate=errorCal(a_value)
%
% This function returns the calculated error rate for a specific case of
% two classes with given PDF function of each class by generating points
% based on the PDF and classify all data using the PDF.
%
% ### WARNING: THIS IS A CASE SPECIFIC FUNCTION ###
%
% Input: a_value
% Output: error_rate
%
% Example:
% a = 6;
% error = errorCal(a);
% fprintf("When a = %.2f, error rate is %.4f\n", a, error);
%
% Created and Edited by: Brandon B. Casey B. and Kuang J.
% 
    syms x
    % Set Value for a
    a = a_value;
    % Generate CDF function for given PDF
    pdf1 = @(x) (2./(a.^2)) .*x + (2./a);
    CDF1 = int(pdf1, x);
    
    CDFi1 = finverse(CDF1);
    CDFi1 = matlabFunction(CDFi1);

    pdf = @(x) (x + 1);
    CDF = int(pdf, x);
    CDFi = finverse(CDF);
    CDFi = matlabFunction(CDFi);

    % Assign Points for each Class
    point1 = 7000;
    point2 = point1*2*3; % *FULL_CLASS*PRIOR
    % Generate points based on the CDF function for each Class
    points = rand(point1, 1).' * -1;
    data = feval(CDFi, points).';
    one = ones(point1);
    one = one(:, 1);
    data = [one, data];

    points = rand(point2, 1).' .* -1 .* (a);
    data2 = feval(CDFi1, points).';
    zero = zeros(point2);
    zero = zero(:, 1);
    data2 = [zero data2];
    
    % Pool all data together
    all = [data; data2];
    
    % Set up for Classify
    classify = zeros(point1+point2);
    classify = classify(:, 1);
    % Classify based on the PDF of the function
    for i = 1:(point1 + point2)
       if feval(pdf, all(i, 2)) > feval(pdf1, all(i, 2)) * 3
          classify(i) = 1;
       else
           classify(i) = 0;
       end
    end
    
    % Calculate the error 
    error = 0;
    for i = 1:(point1 + point2)
        if all(i) ~= classify(i)
            error = error + 1;
        end
    end
    
    % Calculate and return error rate
    error_rate = error / (2*point1 + point2);
end