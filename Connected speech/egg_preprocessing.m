function [Y, dydt] = egg_preprocessing(data,time)
wname = 'bior3.7';

[c, l] = wavedec(data, 9, wname);

d9 = wrcoef('d',c,l,wname,9);
d8 = wrcoef('d',c,l,wname,8);
d7 = wrcoef('d',c,l,wname,7);
d6 = wrcoef('d',c,l,wname,6);
d5 = wrcoef('d',c,l,wname,5);
d4 = wrcoef('d',c,l,wname,4);
d3 = wrcoef('d',c,l,wname,3);
d2 = wrcoef('d',c,l,wname,2);
d1 = wrcoef('d',c,l,wname,1);

y3 =d7+d6+d5+d4+d3+d2;
y3 = movmean(y3,10);
y3 = smoothdata(y3,'gaussian',20);

Y = normalize(y3,'range',[-1 1]);
dydt = gradient(y3)./gradient(time);

dydt = normalize(dydt,'range',[-1 1]);

end