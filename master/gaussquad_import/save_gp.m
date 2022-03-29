for i=0:16
    [gpts2{i+1, 1}, gweights2{i+1, 2}] = gaussquad2d(i); 
end

for i=0:15
    [gpts3{i+1, 1}, gweights3{i+1, 2}] = gaussquad3d(i); 
end

save 'gptsweights.mat' gpts2 gpts3 gweights2 gweights3