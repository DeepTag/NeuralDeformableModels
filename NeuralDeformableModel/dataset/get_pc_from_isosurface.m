system('rm *~');
clear all
clc
close all


data_path = '/ailab/data/Cardiac super-resolution label maps/smoothed/';

file_name = ["HR_ED_MYO", "HR_ED_RV", "HR_ES_MYO", "HR_ES_RV"];

for i = 0:1330
    for k=1:4
        file = file_name(k);
        file_name_1 = file+'.mat';
        file_path = fullfile(data_path, num2str(i), file_name_1);
        vol = load(file_path);
        MyFieldNames = fieldnames(vol);
        p = isosurface(getfield(vol, MyFieldNames{1}));
        pc = p.vertices;
        file_name_temp = file+'_pc.mat';
        save(fullfile(data_path, num2str(i), file_name_temp),'pc')
    end
end
