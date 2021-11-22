% Run with 'runtests'
classdef TestExamples < matlab.unittest.TestCase
    
    properties(TestParameter)
        file = getListOfFiles;
    end
    methods(Test)
        function testErrorFree(testCase, file)
           run(file) 
        end
    end
end

function [files] = getListOfFiles
    list=dir(fullfile('examples','*.m'));
    files={};
    for i=1:length(list)
        files{end+1} = [list(i).folder filesep list(i).name];
    end
end