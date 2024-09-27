classdef DiffSimulinkModel < handle
    properties
        io
        op
        mdl
        opts

        N

        dims_in
        dims_out

        nx
        nu
        np
        ny

        x_subsel
        u_subsel

        with_diff

        sllin

        varargin
    end
  methods
      function self = DiffSimulinkModel(varargin)
        % DiffSimulinkModel
        % 
        %   This class exposes derivatives of continuous or discrete
        %   Simulink blocks.
        %   This class is used to make a bridge to CasADi/rockit/impact.
        %   In fact, CasADi will itself create instances of this class.
        %   For debugging and obtaining meta-data, you may instantiate this class manually.
        %
        %
        % Arguments:
        %   mdl (char): 
        %       The name of the Simulink model to be loaded. This is a required argument.
        %
        % Optional Parameters (specified as name-value pairs):
        %   'path' (char, optional): 
        %       The matlab path to add for model loading. If not specified or empty, no path will be added. Default is '' (empty string).
        %       Note that CasADi will instantiate this class in a fresh
        %       matlab context, so any non-permanent matlab path changes
        %       will not be present.
        %       You may use ';' as separator to add multiple paths.
        %       A '*' prefix may be used to signify 'with subfolders'.
        %   'block' (char, optional): 
        %       The specific block within the model to be used. If not specified or empty, the entire model is used. Default is '' (empty string).
        %   'init_script' (char, optional): 
        %       The initialization script to be executed before loading the model. If not provided, the function defaults to using the script named after the model followed by '_init'.
        %       Typically, such script would compute values of block
        %       parameters.
        %   'exclude_inputs' (char, optional): 
        %       A list (seperated by ';') of input names to exclude from the model. Default is '' (empty string).
        %   'exclude_outputs' (char, optional): 
        %       A list (seperated by ';') of output names to exclude from the model. Default is '' (empty string).
        %   'with_diff' (char, optional)
        %       'true' or 'false' indicating if we want derivatives or not.
        %   'cwd'
        %       Current working directory. Default is '' (empty string).

        %-  'N' Number of requested evaluations

        p = inputParser;
        addRequired(p,'mdl',@ischar);
        addParameter(p,'path','',@ischar);
        addParameter(p,'block','',@ischar);
        addParameter(p,'init_script','',@ischar);
        addParameter(p,'exclude_inputs','',@ischar);
        addParameter(p,'exclude_outputs','',@ischar);
        addParameter(p,'with_diff','true',@ischar);
        addParameter(p,'cwd','',@ischar);
        addParameter(p,'N',1,@isnumeric);

        parse(p,varargin{:});

        args = p.Results;

        mdl = args.mdl;

        self.N = args.N;
        self.with_diff = strcmp(args.with_diff,'true');
        
        
        if ~isempty(args.cwd)
            cd(args.cwd);
        end

        if ~isempty(args.path)
            path_split = strsplit(args.path,';');
            for i=1:numel(path_split)
                path = path_split{i};
                if strcmp(path(1),'*')
                    addpath(genpath(path(2:end)));
                else
                    addpath(path);
                end
            end
        end
        if isempty(args.block)
            block = mdl;
        else
            block_name = args.block;
            block = [mdl '/' block_name];
        end
        if isempty(args.init_script)
            init_script = [mdl '_init'];
        else
            init_script = args.init_script;
        end
        evalin('base', init_script);
        load_system(mdl);

        if isempty(args.exclude_outputs)
            exclude_outputs = {};
        else
            exclude_outputs = strsplit(args.exclude_outputs,';');
        end

        if isempty(args.exclude_inputs)
            exclude_inputs = {};
        else
            exclude_inputs = strsplit(args.exclude_inputs,';');
        end
        
        op = operpoint(mdl);

        set_param(mdl,'SimulationCommand','start')
        set_param(mdl,'SimulationCommand','pause')
        opts = linearizeOptions;
        
        opts.BlockReduction = 'off';
        %sllin.Options.AreParamsTunable = true;
        opts.LinearizationAlgorithm = 'blockbyblock';
        opts.StoreOffsets = true;

        self.opts = opts;

        io = linearize.IOPoint();
        io(1) = [];
        
        inputs = find_system(block,'SearchDepth',1,'BlockType','Inport');
        dims_in = zeros(0,1);

        for i=1:numel(inputs)
            input = inputs{i};
            name = getfullname(input);
            name = name(length(block)+2:end);

            if any(strcmp(exclude_inputs,name))
                continue
            end

            ph = get_param(input,'PortHandles');
            dims_in(end+1) = prod(get_param(ph.Outport,'CompiledPortDimensions'));

            io(end+1) = linio(input,1,'input');
        end

        outputs = find_system(block,'SearchDepth',1,'BlockType','Outport');

        dims_out = zeros(0,1);

        for i=1:numel(outputs)
            output = outputs{i};
            name = getfullname(output);
            name = name(length(block)+2:end);

            if any(strcmp(exclude_outputs,name))
                continue
            end

            ph = get_param(output,'PortHandles');
            dims_out(end+1) = prod(get_param(ph.Inport,'CompiledPortDimensions'));

            % which block is the output outputting?
            res = get_param(output,'PortConnectivity');
            output = getfullname(res.SrcBlock);

            io(end+1) = linio(output,1,'output');
        end
        

        set_param(mdl,'SimulationCommand','stop')

        self.io = io;
        
        op = {};
        for i=1:self.N
            op{end+1} = operpoint(mdl);
        end
        op = [op{:}];
        self.op = op;

        self.dims_in = dims_in;
        self.dims_out = dims_out;

        self.x_subsel = 1:sum(dims_out);
        self.u_subsel = 1:sum(dims_in);
        
        self.nx = length(self.x_subsel);
        self.nu = length(self.u_subsel);
        self.np = 0; % get_param('model2/Gain4', 'DialogParameters')
        self.ny = 0;

        self.mdl = mdl;

        sllin = slLinearizer(self.mdl,self.io, self.op, self.opts);
        self.sllin = sllin;
        [sys,info] = getIOTransfer(sllin,self.io);
        fastRestartForLinearAnalysis(mdl,'on');
        self.nx = size(sys.B,1);
        self.nu = size(sys.B,2);
        self.ny = size(sys.C,1);

        %simIn = Simulink.SimulationInput(mdl);
        %setInitialState(simIn,op)

        %getstatestruct(op)
        %xInitial = getstatestruct(op);
        %uInitial = getinputstruct(op);

        %simIn = setInitialState(simIn,xInitial);



        %set_param(mdl, 'LoadInitialState', 'on', 'InitialState', 'xInitial');
        %set_param(mdl, 'LoadInitialState', 'on', 'InitialState', 'xInitial');
        %3+3
        self.varargin = varargin;
    end
    function [] = export(self, name)
        fileID = fopen(name, 'w');
        fprintf(fileID, 'equations:\n');
        fprintf(fileID, '  type: simulink\n');
        fprintf(fileID, '  config:\n');
        for i=1:numel(self.varargin)
            fprintf(fileID, '    -- %s\n',self.varargin{i});
        end
        fprintf(fileID, '    equations:\n');
        fclose(fileID);
    end
    function [sys,info] = debug(self)
        sllin = slLinearizer(self.mdl,self.io, self.op, self.opts);  
        [sys,info] = getIOTransfer(sllin,self.io);
    end

    function out = get_n_in(self)
      out = 3;
    end
    function out = get_n_out(self)
      if self.with_diff
          out = 6;
      else
          out = 2;
      end
    end
    function out = get_sparsity_in(self,i)
      switch(i)
          case 0
              out = int64([self.nx,self.N]);
          case 1
              out = int64([self.nu,self.N]);
          case 2
              out = int64([self.np,1]);
          otherwise
              error('Internal error');
      end
    end
    function out = get_sparsity_out(self,i)
      switch(i)
          case 0
              out = int64([self.nx,self.N]);
          case 1
              out = int64([self.ny,self.N]);
          case 2
              out = int64([self.nx,self.N*self.nx]);
          case 3
              out = int64([self.nx,self.N*self.nu]);
          case 4
              out = int64([self.ny,self.N*self.nx]);
          case 5
              out = int64([self.ny,self.N*self.nu]);
          otherwise
              error('Internal error');
      end
    end
    function out = get_name_in(self,i)
      names = {'x','u','p'};
      out = names{i+1};
    end
    function out = get_name_out(self,i)
      if self.with_diff
          names = {'dx','y','jac_dx_x', 'jac_dx_u','jac_y_x', 'jac_y_u'};
      else
          names = {'dx','y'};
      end
      out = names{i+1};
    end
    function delete(self)
        fileID = fopen('lookatyou.txt','w');
        fprintf(fileID, 'we\n');
        fastRestartForLinearAnalysis(self.mdl,'off');
        fprintf(fileID, 'rule\n');
        fclose(fileID);
    end
    function [res] = eval(self, arg)
        offset = 0;
        x = arg{1}
        for i=1:numel(self.op.States)
            n = numel(self.op.States(i).x)
            offset
            offset+1:offset+n

            self.op.States(i).x = x(offset+1:offset+n);
            offset = offset + n;
        end
        offset = 0;
        u = arg{2}
        for i=1:numel(self.op.Inputs)
            n = numel(self.op.Inputs(i).u);
            self.op.Inputs(i).u = u(offset+1:offset+n);
            offset = offset + n;
        end

        self.sllin.OperatingPoints = self.op;

        

        tic
        [sys,info] = getIOTransfer(self.sllin,self.io);
        r=toc;

        %fileID = fopen('log.txt','a');
        %fprintf(fileID, 'getIOTransfer: %f\n',r);
        %fclose(fileID);

        dx = info.Offsets.dx;
        y = info.Offsets.y;
        if self.with_diff
            res = {dx, y, reshape(sys.A,self.nx,self.nx*self.N), reshape(sys.B,self.nx,self.nu*self.N), reshape(sys.C,self.ny,self.N*self.nx), reshape(sys.D,self.ny,self.N*self.nu)};
        else
            res = {dx, y};
        end
    end

  end
end


