%% GlottoTie_CS_analysis.m
% =========================================================================
% Description:
%   Data Preprocessing, FFT, and Mel-Spectrogram Analysis for Connected Speech (CS).
%   Optimized for GitHub public release to ensure reproducibility.
%   Processes a single public sample file: 'CS_Normal_date_time_subject1_dob.csv'
%
% Analysis Modes:
%   's' : Save All Learning Data - Splits the signal into 3-second segments 
%         and exports them as .mat files for classification/model training.
%   'f' : Full Signal Analysis - Comprehensive visualization including Time-domain,
%         FFT (Original vs Filtered), PSD, and Mel-Spectrogram with Spectral Centroid.
% =========================================================================

clear all; clc; close all;

%% 1. Choose Analysis Mode
disp('=== GlottoTie Connected Speech Analysis (GitHub Release) ===');
analysisMode = input('Enter analysis mode ("s" for save all learning data, "f" for full signal): ', 's');

%% 2. Data Configuration (Single Sample File)
filename = 'CS_Normal_date_time_subject1_dob.csv';
fprintf('\nLoading sample data: %s\n', filename);

if ~isfile(filename)
    error('File not found. Please ensure "%s" is in the working directory.', filename);
end

% Parse subject information from the sample filename
parts = split(filename, '_');
if length(parts) >= 5
    personName = parts{5}; % Extracts 'subject1'
else
    personName = 'SampleSubject';
end
fprintf('Subject Name Extracted: %s\n', personName);

%% 3. Read CSV File and Extract Data
dataTable = readtable(filename);
dataArray = table2array(dataTable);

eggData = dataArray(:,1)';  % EGG Data (Row vector)
micData = dataArray(:,2)';  % Microphone Data (Row vector)

% Preprocess EGG: Noise removal (values > 2 set to 0)
eggData(eggData > 2) = 0;

samplingFreq = 11000;       % Sampling frequency (Hz)
timeResolution = 1/samplingFreq;
time = 0:timeResolution:(length(eggData)-1)/samplingFreq;

%% 4. Analysis Execution Branch
if strcmpi(analysisMode, 's')
    %% --- Mode S: Segmented Analysis & Learning Data Export ---
    disp('Running Segmented Data Preparation...');
    
    % Preprocess EGG signal before segmentation
    [filtered_egg, ~] = egg_preprocessing(eggData, time);
    
    % Apply Notch Filtering to Mic Data (Remove 275Hz harmonics)
    f_base = 275;
    N_harmonics = floor((samplingFreq/2)/f_base);
    Q = 300;
    filtered_mic = micData;
    for k = 1:N_harmonics-1
        f0 = k * f_base;
        if f0 < samplingFreq/2
            wo = f0/(samplingFreq/2);
            bw = wo/Q;
            [b, a] = iirnotch(wo, bw);
            filtered_mic = filtfilt(b, a, filtered_mic);
        end
    end
    
    % Define segment length for classification model (e.g., 3 seconds)
    segment_duration = 3; 
    segment_length = round(samplingFreq * segment_duration);
    num_segments = floor(length(filtered_mic) / segment_length);
    
    fprintf('Exporting %d segments (%ds each) for model training...\n', num_segments, segment_duration);
    
    % Create an output directory for reproduction
    outputDir = 'extracted_segments';
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    % Save each segment to show reviewers the preprocessing pipeline for classification
    for segIndex = 1:num_segments
        start_idx = (segIndex-1)*segment_length + 1;
        end_idx   = segIndex*segment_length;
        
        seg_mic = filtered_mic(start_idx:end_idx);
        seg_egg = filtered_egg(start_idx:end_idx);
        seg_time = time(start_idx:end_idx);
        
        % Save as a .mat file (used as input for downstream machine learning models)
        outFilename = fullfile(outputDir, sprintf('%s_seg_%02d.mat', personName, segIndex));
        save(outFilename, 'seg_mic', 'seg_egg', 'seg_time', 'samplingFreq');
    end
    
    fprintf('Successfully saved all learning data segments to: "%s/"\n', outputDir);
    disp('--- Segmented Data Preparation Complete ---');
    
elseif strcmpi(analysisMode, 'f')
    %% --- Mode F: Full Signal Analysis & Visualization ---
    disp('Running Full Signal Analysis...');
    
    % 4.1 EGG Preprocessing & FFT
    [filtered_egg, filtered_degg] = egg_preprocessing(eggData, time);
    
    L = length(filtered_egg);
    Y = fft(filtered_egg);
    P2 = abs(Y/L);
    P1_fft = P2(1:floor(L/2)+1);
    P1_fft(2:end-1) = 2*P1_fft(2:end-1);
    f_axis = samplingFreq*(0:(L/2))/L;
    
    % 4.2 Microphone Preprocessing (Notch Filtering for 275Hz Harmonics)
    f_base = 275;
    N_harmonics = floor((samplingFreq/2)/f_base);  
    Q = 300;  
    filtered_mic = micData;
    
    for k = 1:N_harmonics-1  
        f0 = k * f_base;
        if f0 < samplingFreq/2
            wo = f0/(samplingFreq/2);
            bw = wo/Q;
            [b, a] = iirnotch(wo, bw);
            filtered_mic = filtfilt(b, a, filtered_mic);
        end
    end
    
    % FFT for Original Mic
    L_mic = length(micData);
    Y_mic = fft(micData);
    P2_mic = abs(Y_mic/L_mic);
    P1_mic = P2_mic(1:floor(L_mic/2)+1);
    P1_mic(2:end-1) = 2*P1_mic(2:end-1);
    f_axis_mic = samplingFreq*(0:(L_mic/2))/L_mic;
    
    % FFT for Filtered Mic
    L_mic_filt = length(filtered_mic);
    Y_mic_filt = fft(filtered_mic);
    P2_mic_filt = abs(Y_mic_filt/L_mic_filt);
    P1_mic_filt = P2_mic_filt(1:floor(L_mic_filt/2)+1);
    P1_mic_filt(2:end-1) = 2*P1_mic_filt(2:end-1);

    %% 4.3 Plot 1: Time Domain & FFT Comparison
    figure('Name', sprintf('%s: Time & Frequency Analysis', personName), 'Color', 'w');
    
    subplot(5,1,1);
    plot(time, eggData, 'b'); xlim([0 max(time)]);
    title(sprintf('%s: Original EGG (Time Domain)', personName)); xlabel('Time (s)'); ylabel('Amplitude');
    
    subplot(5,1,2);
    plot(time, filtered_egg, 'r'); xlim([0 max(time)]);
    title(sprintf('%s: Filtered EGG (Time Domain)', personName)); xlabel('Time (s)'); ylabel('Amplitude');
    
    subplot(5,1,3);
    plot(time, micData, 'r'); xlim([0 max(time)]); ylim([0.5 1.8]);
    title(sprintf('%s: Original Mic (Time Domain)', personName)); xlabel('Time (s)'); ylabel('Amplitude');
    
    subplot(5,1,4);
    plot(f_axis_mic, P1_mic, 'm', 'LineWidth', 1.5); xlim([20 5500]);
    title(sprintf('%s: Mic FFT (Original)', personName)); xlabel('Frequency (Hz)'); ylabel('Magnitude');
    
    subplot(5,1,5);
    plot(f_axis_mic, P1_mic_filt, 'm', 'LineWidth', 1.5); xlim([20 5500]);
    title(sprintf('%s: Mic FFT (Filtered)', personName)); xlabel('Frequency (Hz)'); ylabel('Magnitude');

    %% 4.4 Plot 2: Power Spectrum Density (PSD)
    figure('Name', sprintf('%s: Power Spectrum Density', personName), 'Color', 'w');
    
    subplot(1,2,1);
    pspectrum(filtered_egg, samplingFreq, 'persistence'); xlim([0.02 5.5]);
    title(sprintf('%s: Filtered EGG PSD', personName)); xlabel('Frequency (Hz)'); ylabel('Power Spectrum');
    
    subplot(1,2,2);
    pspectrum(eggData, samplingFreq, 'persistence'); xlim([0.02 5.5]);
    title(sprintf('%s: Original EGG PSD', personName)); xlabel('Frequency (Hz)'); ylabel('Power Spectrum');

    %% 4.5 Plot 3: Mel-Spectrogram & Spectral Centroid
    window_length = 2^nextpow2(samplingFreq * 0.025);  % 25ms window
    step_length   = 2^nextpow2(samplingFreq * 0.010);  % 10ms hop
    num_mels      = 512;
    fftLength     = 4096;
    freqRange     = [62.5, samplingFreq/2];

    % Mel-Spectrogram (EGG)
    [S_egg, f_egg, t_egg] = melSpectrogram(filtered_egg', samplingFreq, ...
        'Window', hamming(window_length, 'periodic'), 'OverlapLength', step_length, ...
        'FFTLength', fftLength, 'NumBands', num_mels, 'FrequencyRange', freqRange);
    S_egg_dB = 10*log10(S_egg);
    
    % Mel-Spectrogram (Mic)
    [S_mic, f_mic, t_mic] = melSpectrogram(filtered_mic', samplingFreq, ...
        'Window', hamming(window_length, 'periodic'), 'OverlapLength', step_length, ...
        'FFTLength', fftLength, 'NumBands', num_mels, 'FrequencyRange', freqRange);
    S_mic_dB = 10*log10(S_mic);

    % Calculate Spectral Centroid for Mic
    f_mic_col = f_mic(:);
    spectralCentroid = zeros(1, length(t_mic));
    for i = 1:length(t_mic)
        spectralCentroid(i) = sum(f_mic_col .* S_mic(:, i)) / (sum(S_mic(:, i)) + eps);
    end

    figure('Name', sprintf('%s: Mel-Spectrogram Analysis', personName), 'Color', 'w');
    
    subplot(3,1,1);
    imagesc(t_egg, f_egg, S_egg_dB); axis xy; colorbar; xlim([0 max(time)]);
    title(sprintf('%s: Full Signal EGG Mel-Spectrogram', personName)); xlabel('Time (s)'); ylabel('Frequency (Hz)');
    
    subplot(3,1,2);
    imagesc(t_mic, f_mic, S_mic_dB); axis xy; colorbar; xlim([0 max(time)]);
    title(sprintf('%s: Full Signal Mic Mel-Spectrogram', personName)); xlabel('Time (s)'); ylabel('Frequency (Hz)');
    
    subplot(3,1,3);
    plot(t_mic, spectralCentroid, 'LineWidth', 1, 'Color', 'k'); xlim([0 max(time)]);
    title(sprintf('%s: Mic Spectral Centroid Overlay', personName)); xlabel('Time (s)'); ylabel('Centroid (Hz)');

    %% 4.6 Plot 4: 3-Second Segment Grouped Visualizations
    segment_length = round(samplingFreq * 3); 
    num_segments = floor(length(micData) / segment_length);
    groupSize = 5; % Number of segment plots per window
    num_groups = ceil(num_segments / groupSize);
    
    for g = 1:num_groups
        segStart = (g-1)*groupSize + 1;
        segEnd = min(g*groupSize, num_segments);
        currentGroup = segStart:segEnd;
        num_plots = length(currentGroup);
        
        figure('Name', sprintf('%s: Segment Analysis Group %d', personName, g), 'Color', 'w', 'Position', [100, 100, 1200, 800]);
        
        for idx = 1:num_plots
            segIndex = currentGroup(idx);
            start_idx = (segIndex-1)*segment_length + 1;
            end_idx   = segIndex*segment_length;
            
            segment_mic_filt = filtered_mic(start_idx:end_idx);
            segment_egg = normalize(filtered_egg(start_idx:end_idx), 'range', [-1 1]);
            
            % Segment FFT
            L_seg = length(segment_mic_filt);
            Y_seg = fft(segment_mic_filt);
            P2_seg = abs(Y_seg / L_seg);
            P1_seg = P2_seg(1:floor(L_seg/2)+1);
            P1_seg(2:end-1) = 2*P1_seg(2:end-1);
            f_seg = samplingFreq*(0:(L_seg/2))/L_seg;
            
            % Segment Mel-Spectrograms
            [melspec_mic_filt, f_mic_filt, t_mic_filt] = melSpectrogram(segment_mic_filt', samplingFreq, ...
                'Window', hamming(window_length, 'periodic'), 'OverlapLength', step_length, ...
                'FFTLength', fftLength, 'NumBands', num_mels, 'FrequencyRange', freqRange);
            melspec_mic_filt = 10*log10(melspec_mic_filt);
            
            [melspec_egg, f_egg_seg, t_egg_seg] = melSpectrogram(segment_egg', samplingFreq, ...
                'Window', hamming(window_length, 'periodic'), 'OverlapLength', step_length, ...
                'FFTLength', fftLength, 'NumBands', num_mels, 'FrequencyRange', freqRange);
            melspec_egg = 10*log10(melspec_egg);
            
            % Row 1: Filtered Mic Frequency Response
            subplot(3, num_plots, idx);
            plot(f_seg, P1_seg, 'LineWidth', 1.5, 'Color', '#0072BD'); xlim([20 samplingFreq/2]);
            title(sprintf('Mic Filt FFT (Seg %d)', segIndex)); xlabel('Frequency (Hz)'); ylabel('Magnitude');
            
            % Row 2: Filtered Mic Mel-Spectrogram
            subplot(3, num_plots, idx + num_plots);
            imagesc(t_mic_filt, f_mic_filt, melspec_mic_filt); axis xy; colorbar;
            title(sprintf('Mic Filt Mel (Seg %d)', segIndex)); xlabel('Time (s)'); ylabel('Freq (Hz)');
            
            % Row 3: Filtered EGG Mel-Spectrogram
            subplot(3, num_plots, idx + 2*num_plots);
            imagesc(t_egg_seg, f_egg_seg, melspec_egg); axis xy; colorbar;
            title(sprintf('EGG Mel (Seg %d)', segIndex)); xlabel('Time (s)'); ylabel('Freq (Hz)');
        end
    end
    
    disp('--- Full Signal Analysis Complete ---');
else
    error('Invalid input. Please enter "s" for segmented or "f" for full signal.');
end