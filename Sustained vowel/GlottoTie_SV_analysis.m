%% GlottoTie_SV_analysis.m
% Data Preprocessing and FFT Plot for Sustained Vowel (SV) Analysis
% Supports Segmented ('s') or Full Signal ('f') analysis modes.

clear; clc; close all;

%% 1. Configuration & Mode Selection
analysisMode = input('Enter analysis mode ("s" for segmented, "f" for full signal): ', 's');

% Define the path to the sample data
filename = 'SV_Normal_date_time_subject1_dob.csv';

fprintf('Loading sample data: %s\n', filename);

if ~isfile(filename)
    error('File not found. Please check the sample data directory.');
end

%% 2. Read CSV File and Extract Data
dataTable = readtable(filename);
dataArray = table2array(dataTable);

eggData = dataArray(:,1)';  % EGG data
micData = dataArray(:,2)';  % Mic data
samplingFreq = 11000;       % Sampling frequency (Hz)
timeResolution = 1/samplingFreq;
time = 0:timeResolution:(length(eggData)-1)/samplingFreq;

%% 3. Define Valid Segments (for 's' mode)
% Example valid segments for the sample subject (in seconds)
validSegments = [4	    6.2
                 11	    13
                 18	    20
                 25.8	28.3
                 33.3	36.5];

%% 4. Analysis Based on User Selection
if strcmpi(analysisMode, 's')
    %% Segmented Analysis
    fundamentalFrequencies = zeros(size(validSegments, 1), 1);
    cycleInitLocations = cell(size(validSegments, 1), 1);
    allEggCycles = [];
    allDeggCycles = [];
    allParameters = [];
    
    for segIndex = 1:size(validSegments, 1)
        segStart = validSegments(segIndex, 1);
        segEnd   = validSegments(segIndex, 2);
        segRange = floor(segStart * samplingFreq + 1) : floor(segEnd * samplingFreq);
        
        segEggData = eggData(segRange);
        segMicData = micData(segRange);
        segTime    = time(segRange);
        
        % Preprocessing
        [filtered_egg, filtered_degg] = egg_preprocessing(segEggData, segTime);
        
        % FFT for Filtered EGG Data to find fundamental frequency (f0)
        L = length(filtered_egg);
        Y = fft(filtered_egg);
        P1_fft = abs(Y/L);
        P1_fft = P1_fft(1:floor(L/2)+1);
        P1_fft(2:end-1) = 2*P1_fft(2:end-1);
        f_axis = samplingFreq*(0:(L/2))/L;
        
        f_min = 50; f_max = 350;
        rangeIdx = find(f_axis >= f_min & f_axis <= f_max);
        
        if ~isempty(rangeIdx)
            [~, idx_max] = max(P1_fft(rangeIdx));
            f0 = f_axis(rangeIdx(idx_max));
        else
            f0 = NaN;
        end
        fundamentalFrequencies(segIndex) = f0;
        
        % Cycle Detection
        if ~isnan(f0)
            [cycle_init_locs, ~, ~] = cycle_init_detection(filtered_egg, filtered_degg, samplingFreq, f0);
        else
            cycle_init_locs = [];
        end
        cycleInitLocations{segIndex} = cycle_init_locs;
        
        % Feature Extraction & Aggregation
        if length(cycle_init_locs) >= 2
            normalize_sample = 0:1/1000:1;
            [egg_cycle_acc, degg_cycle_acc] = cycle_separation(filtered_egg, filtered_degg, samplingFreq, f0, cycle_init_locs);
            
            aggregatedEggCycleMean  = mean(egg_cycle_acc, 1);
            aggregatedEggCycleStd   = std(egg_cycle_acc, 0, 1);
            aggregatedDeggCycleMean = mean(degg_cycle_acc, 1);
            aggregatedDeggCycleStd  = std(degg_cycle_acc, 0, 1);
            
            egg_std_pos = aggregatedEggCycleMean + aggregatedEggCycleStd;
            egg_std_neg = aggregatedEggCycleMean - aggregatedEggCycleStd;
            degg_std_pos = aggregatedDeggCycleMean + aggregatedDeggCycleStd;
            degg_std_neg = aggregatedDeggCycleMean - aggregatedDeggCycleStd;
            
            [cycle_std, V_tot, V1, V2, V3, V4, cycle_std_in, cycle_std_out, x_centre, y_centre] = ...
                synchronized_egg_degg_plot(egg_cycle_acc, degg_cycle_acc, aggregatedEggCycleMean, aggregatedDeggCycleMean);
            
            parameter = [V_tot, V1, V2, V3, V4, V1/V_tot, V2/V_tot, V3/V_tot, V4/V_tot];
            allParameters = [allParameters; parameter];
            allEggCycles = [allEggCycles; egg_cycle_acc];
            allDeggCycles = [allDeggCycles; degg_cycle_acc];
        else
            % Empty placeholders if not enough cycles
            aggregatedEggCycleMean = []; aggregatedDeggCycleMean = [];
            x_centre = []; y_centre = [];
        end
        
        % --- Plotting Segment Results ---
        figure('Name', sprintf('Segment %d [%.1fs - %.1fs]', segIndex, segStart, segEnd), 'NumberTitle', 'off', 'Color', 'w');
        
        subplot(3,3,[1,3]);
        plot(segTime, filtered_egg, 'LineWidth', 1.5, 'Color', "#DE5F61"); hold on;
        if ~isempty(cycle_init_locs)
            plot(segTime(cycle_init_locs), filtered_egg(cycle_init_locs), 'ko');
        end
        title(sprintf('Segment %d: Filtered EGG', segIndex)); xlabel('Time (s)'); hold off;
        
        subplot(3,3,[4,6]);
        plot(segTime, segMicData, 'LineWidth', 1.5, 'Color', "#6B9ECF");
        title(sprintf('Segment %d: Mic Data', segIndex)); xlabel('Time (s)');
        
        subplot(3,3,7);
        if ~isempty(aggregatedEggCycleMean)
            plot(normalize_sample, aggregatedEggCycleMean, 'Color',[0.85 0.33 0.10], 'LineWidth', 2); hold on;
            fill([normalize_sample, fliplr(normalize_sample)], [egg_std_neg, fliplr(egg_std_pos)], [0.85 0.33 0.10], 'FaceAlpha', 0.5, 'LineStyle', 'none');
            title('Egg Cycle Mean \pm STD'); xlabel('Normalized Time'); hold off;
        end
        
        subplot(3,3,8);
        if ~isempty(aggregatedDeggCycleMean)
            plot(normalize_sample, aggregatedDeggCycleMean, 'Color',[0 0.45 0.74], 'LineWidth', 2); hold on;
            fill([normalize_sample, fliplr(normalize_sample)], [degg_std_neg, fliplr(degg_std_pos)], [0 0.45 0.74], 'FaceAlpha', 0.5, 'LineStyle', 'none');
            title('dEGG Cycle Mean \pm STD'); xlabel('Normalized Time'); hold off;
        end
        
        subplot(3,3,9);
        if ~isempty(x_centre)
            plot(x_centre, y_centre, 'o'); hold on;
            for i = 1:length(cycle_std)
                plot([cycle_std_in(1,i) cycle_std_out(1,i)], [cycle_std_in(2,i) cycle_std_out(2,i)], 'Color','#808080');
            end
            plot(aggregatedEggCycleMean, aggregatedDeggCycleMean, 'k', 'LineWidth',2);
            title('Synchronized Egg-dEGG'); hold off;
        end
        drawnow;
    end
    
    disp('--- Analysis Complete ---');
    disp('Fundamental Frequencies for each segment (Hz):'); disp(fundamentalFrequencies);
    if ~isempty(allEggCycles)
        disp('Extracted Parameters [V_tot, V1~V4, Ratios]:'); disp(allParameters);
    end

elseif strcmpi(analysisMode, 'f')
    %% Full Signal Analysis
    [filtered_egg, ~] = egg_preprocessing_Tmp(eggData, time);
    
    % Mic FFT (Original)
    L_mic = length(micData);
    Y_mic = fft(micData);
    P1_mic = abs(Y_mic/L_mic);
    P1_mic = P1_mic(1:floor(L_mic/2)+1);
    P1_mic(2:end-1) = 2*P1_mic(2:end-1);
    f_axis_mic = samplingFreq*(0:(L_mic/2))/L_mic;
    
    % Notch Filtering of micData for 275Hz Harmonics
    f_base = 275;
    N_harmonics = floor((samplingFreq/2)/f_base);
    Q = 150;
    micData_filtered = micData;
    for k = 1:N_harmonics-1
        f0 = k * f_base;
        wo = f0/(samplingFreq/2);
        bw = wo/Q;
        [b, a] = iirnotch(wo, bw);
        micData_filtered = filtfilt(b, a, micData_filtered);
    end
    
    % Mic FFT (Filtered)
    Y_mic_filt = fft(micData_filtered);
    P1_mic_filt = abs(Y_mic_filt/L_mic);
    P1_mic_filt = P1_mic_filt(1:floor(L_mic/2)+1);
    P1_mic_filt(2:end-1) = 2*P1_mic_filt(2:end-1);
    
    % Plotting Full Signal Results
    figure('Color', 'w');
    subplot(5,1,1); plot(time, eggData, 'b'); title('Full Signal: EGG Data'); xlabel('Time (s)');
    subplot(5,1,2); plot(time, filtered_egg, 'r'); title('Full Signal: Filtered EGG'); xlabel('Time (s)');
    subplot(5,1,3); plot(time, micData, 'm'); title('Full Signal: Original Mic Data'); xlabel('Time (s)');
    subplot(5,1,4); plot(f_axis_mic, P1_mic, 'LineWidth', 1.5); xlim([20 5500]); title('Mic FFT (Original)'); xlabel('Freq (Hz)');
    subplot(5,1,5); plot(f_axis_mic, P1_mic_filt, 'LineWidth', 1.5); xlim([20 5500]); title('Mic FFT (Notch Filtered)'); xlabel('Freq (Hz)');
else
    error('Invalid input. Please enter "s" or "f".');
end