folders = {'rockit/external/acados/examples','examples'};

for i=1:numel(folders)
    folder = folders{i};
    list    = dir(fullfile(folder, '*.m'));
    nFile   = length(list);
    for k = 1:nFile
      file = list(k).name;
      run(fullfile(folder, file));
    end
end