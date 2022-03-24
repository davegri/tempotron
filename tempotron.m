% simulation of a tempotron learning

clear; close all; clc;

%% Declare simulation parameters

R           = 2e4;              %[Ohm] resistance
C           = 1e-6;             %[F] capacitance
tau_m       = R*C;  %TODO 1     %[sec] membrane time constant
tau_s       = tau_m/4;          %[sec] synaptic time constant
TH_in_mV    = 30;               %[mV] firing threshold
theta       = TH_in_mV/1000;    %[V] firing threshold
tau_R       = 0.002;            %[sec] refractory period

dt      = 0.0001;           %[sec] time step for numerical integration
t_final = 0.5;              %[sec] duration of numerical simulation
t       = 0:dt:t_final;     %[sec] time vector
ref_per = round(tau_R/dt);  %[steps] refractory period

%% Define the (normalized) kernel function

alpha   = tau_m/tau_s;
kappa   = alpha^(alpha/(alpha-1))/(alpha-1); % TODO 2: set kappa s.t. the maximal value of K will be exactly 1V
K       = @(T) (T > 0).*kappa.*(exp(-T/tau_m) - exp(-T/tau_s));

%% Load inputs
% Load two variables: Samples and N. 
% - Samples is a struct array containing three fields pre sample:
%   - times is a vector containing the sorted spike times of all
%           presynaptic neurons.
%   - neurons is a vector containing the input neuron's index for each
%           spike in times.
%   - y0 is the teacher's output (0 for "do not fire" or 1 for "fire"). 
% - N is the number of input neurons. 

load X_2SDIW;   % Samples, N
% load X_2SDIF;   % Samples, N
% load X_2PGN;    % Samples, N
% load X_2PVTG;   % Samples, N
% load X_PRND;    % Samples, N

%% Declare the learning parameters

eta      	= 10e-3;	% Learning rate
max_epochs  = 100;	% Maximal number of learning epochs
n_samples   = length(Samples);

%% Initialize the tempotron's weights

W   = theta.*0.5.*rand(N, 1);	% tempotron's weights
W0  = W;                        % initial weights (for future reference)

%% Learn

% Loop over learning epochs
for ep = 1:max_epochs
    
    % random order of samples
    samp_order = randperm(n_samples);
    
    % errors counter
    ep_errors = 0;
    
    % Loop over all samples
    for samp = 1:n_samples
    
        % Choose a random sample
        s = Samples(samp_order(samp));
        n_spk = length(s.times);

        % Initialize the voltage
        V = zeros(size(t));

        % Initialize the maximum voltage and its corresponding time
        V_max = 0;  % The global maximum of the voltage
        t_max = 0;  % The time at which the voltage attaines its global 
                    % maximum
        k_max = 0;  % The index of the input spike that caused the voltage 
                    % to reach its global maximum

        % Initialize spike occurence indicator
        spk = false;

        % Loop over presynaptic spikes
        for k = 1:n_spk

            % Extract the presynaptic spike data
            i       = s.neurons(k);
            t_i_j   = s.times(k);

            % Define the change in the postsynaptic voltage
            DeltaV = W(i)*K(t-t_i_j); % TODO 4: set the PSP caused by the current input 
                                      %spike

            % Update the voltage
            V = V + DeltaV;

            % Find the maximum voltage and its corresponding time
            % find the voltage global maximum, and update the
            % variables V_max, t_max and k_max if necessary
            if max(V) > V_max
                V_max = max(V);
                k_max = k;
                t_max = t_i_j;
            end
            
            % Check for a spike occurence
            if V_max >= theta
                spk = true;
                % Shunting
                % TODO 6: simulate the shunting process
                break;
            end

        end

        % Get the gradient of the maximal voltage w.r.t. the synaptic weights
        GradW = zeros(size(W));
        for k = 1:k_max
            % Extract the presynaptic spike data
            i       = s.neurons(k);
            t_i_j   = s.times(k);

            % Update the weights
            GradW(i) = GradW(i) + K(t_max-t_i_j); % TODO 7: update the gradient
        end

        % Train
        sign = (-1)^(s.y0+1) *(s.y0 ~= spk);  % if y0=1 then sign=1, if y0=0 then sign=-1
        W = W + sign*eta*GradW; % TODO 8: use the gradient to update the synaptic weights
        
        % Count errors
        ep_errors = ep_errors + (s.y0 ~= spk);
    
    end
    
    if ep_errors == 0
        break;
    end
    
end

%% Plots

% Set the subplots grid
n_plots	= 3*length(Samples);
n_rows 	= 3*ceil(length(Samples)/4);
n_cols 	= ceil(n_plots/n_rows);

% Create the figure
figure('Name', 'Tempotron''s results', ...
    'Units', 'normalized', ...
    'Position', [0, 0, 1, 1]);

% For each sample, draw 3 different plots
for s = 1:length(Samples)
    
    % Define the sample's plot color according to the teacher's signal
    if Samples(s).y0
        sample_color = 'r';
    else
        sample_color = 'b';
    end
    
    % Input's raster plot
    subplot(n_rows, n_cols, ...
        3*n_cols*floor((s-1)/n_cols)+rem((s-1),n_cols)+1);
    plot_input_spikes(Samples(s).times, Samples(s).neurons, ...
        t_final, N, sample_color);
    title(['Sample #' num2str(s) ': Input']);
    
    % Tempotron's response - before learning
    subplot(n_rows, n_cols, ...
        3*n_cols*floor((s-1)/n_cols)+rem((s-1),n_cols)+1+n_cols);
    plot_postsynaptic_voltage(t, Samples(s).times, Samples(s).neurons, ...
        W0, theta, tau_m, tau_s, R, ref_per, sample_color);
    title(['Sample #' num2str(s) ': Voltage before learning']);
    
    % Tempotron's response - after learning
    subplot(n_rows, n_cols, ...
        3*n_cols*floor((s-1)/n_cols)+rem((s-1),n_cols)+1+2*n_cols);
    plot_postsynaptic_voltage(t, Samples(s).times, Samples(s).neurons, ...
        W, theta, tau_m, tau_s, R, ref_per, sample_color);
    title(['Sample #' num2str(s) ': Voltage after learning']);
    
end
