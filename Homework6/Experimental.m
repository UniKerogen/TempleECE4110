clear;
syms x

a = 0.06;

pdf1 = @(x) (2./(a.^2)) .*x + (2./a);
CDF1 = int(pdf1, x);

CDFi1 = finverse(CDF1);
CDFi1 = matlabFunction(CDFi1);

pdf = @(x) (x + 1);
CDF = int(pdf, x);
CDFi = finverse(CDF);
CDFi = matlabFunction(CDFi);


point1 = 7000;
point2 = point1*6;

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

all = [data;data2];

classify = zeros(point1+point2);
classify = classify(:, 1);

for i=1:point1 + point2
    
   if feval(pdf, all(i, 2)) > feval(pdf1, all(i, 2)) * 3
      classify(i) = 1;
   else
       classify(i) = 0;
   end
end

error = 0;

for i=1:point1 + point2
    if all(i) ~= classify(i)
        error = error + 1;
    end
end

error_rate = error/(2*point1 + point2)





