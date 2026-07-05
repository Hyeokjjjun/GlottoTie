%%commercial EGG
function [cycle_init_locs, peak_pos, locs] = cycle_init_detection_Tmp(egg, degg, Fs, f0)

%%%%%%%%%%%%%%%%%%%%Peak detection%%%%%%%%%%%%%%%%%%%%%%%%%%
%백분위수(quantile)를 이용한 트리밍(위아래20%씩 자른 후 std계산, outlier 제거 목적)
lowerBound = prctile(degg, 20);
upperBound = prctile(degg, 80);
degg_trim = degg(degg >= lowerBound & degg <= upperBound);
amp_thrs = std(degg_trim);
% amp_thrs = std(degg)
time_thrs = (Fs/f0)/2; 
[pks, locs] = findpeaks(degg,'MinPeakHeight',amp_thrs, 'MinPeakDistance',time_thrs);
dist = diff(locs);

%%250% time threshold 
[major_peaks, index] = find(dist < 2.5*time_thrs);
peak_pos = locs(index);

cycle_init_locs = [];

ratio = 1;


for i = 1:1:length(peak_pos)
    if peak_pos(1,i)-round(time_thrs/ratio) > 0
        egg_cycle_temp  = egg(1,peak_pos(1,i)-round(time_thrs/ratio) : peak_pos(1,i));
        degg_cycle_temp = degg(1,peak_pos(1,i)-round(time_thrs/ratio) : peak_pos(1,i));
        [E_min, t_min] = min(egg_cycle_temp);
        % [E_max, t_max] = max(egg_cycle_temp);
        E_max = egg_cycle_temp(end);
        t_max = length(egg_cycle_temp);
        
        if(i >= 2)
            [E1_temp,index_t1] = min(abs(egg_cycle_temp(1,t_min:t_max)-(E_max + E_min)/2));
            index_t1 = t_min+index_t1;
            if(index_t1 > length(egg_cycle_temp))
                E1 = egg_cycle_temp(1,end);
                index_t1 = length(egg_cycle_temp(1,end));
            else
                E1 = egg_cycle_temp(1,index_t1);
            end
    
            [dE_max, idx_dE_max] = find(degg_cycle_temp == degg(1,peak_pos(1,i)));
            index_t2 = idx_dE_max - round(time_thrs/ratio);
            E2 = egg_cycle_temp(1,index_t2);
    
            diff_seg_to_egg_acc = [];
    
            for i2 = index_t2:1:index_t1 
                diff_seg_to_egg = ((E1-E2)/(index_t1-index_t2))*i2 + ((index_t2*E1 + index_t1*E2)/(index_t1-index_t2)) - egg_cycle_temp(1,i2);
                diff_seg_to_egg_acc = [diff_seg_to_egg_acc diff_seg_to_egg];
            end
            
            % 새로 추가된 알고리즘: 증가하다가 처음으로 감소하는 지점 찾기
            first_decrease_idx = -1;
            for j = 2:length(diff_seg_to_egg_acc)
                if diff_seg_to_egg_acc(j) < diff_seg_to_egg_acc(j-1)
                    first_decrease_idx = j-1; % 바로 직전 값이 최고점
                    break;
                end
            end

            if first_decrease_idx ~= -1
                cycle_min = diff_seg_to_egg_acc(first_decrease_idx);
                idx_EGG_cycle_init = first_decrease_idx;
            else
                % fallback: 기존 방식 사용
                [cycle_min, idx_EGG_cycle_init] = max(diff_seg_to_egg_acc);
            end

            cycle_init_locs = [cycle_init_locs peak_pos(1,i) - idx_dE_max + idx_EGG_cycle_init];  
    
        end

    end

    % 조건: egg 값이 양수인 시작점만 남기기
    valid_idx = egg(cycle_init_locs) < 0;
    cycle_init_locs = cycle_init_locs(valid_idx);

end


%% 시각적 검증을 위한 플롯 추가
% t = (1:length(egg)) / Fs;  % 시간축 (초)
% 
% figure;
% subplot(2,1,1);
% plot(t, egg, 'b-', 'LineWidth',1.2); hold on;
% plot(t(locs), egg(locs), 'ko', 'MarkerFaceColor','k');
% plot(t(peak_pos), egg(peak_pos), 'rx', 'MarkerSize',8, 'LineWidth',1.5);
% if ~isempty(cycle_init_locs)
%     plot(t(cycle_init_locs), egg(cycle_init_locs), 'g*', 'MarkerSize',10);
% end
% legend('egg', 'locs', 'peak pos', 'cycle init locs');
% title('EGG Signal with Detected Peaks and Cycle Initiation Locations');
% xlabel('Time (s)'); ylabel('Amplitude');
% grid on;
% 
% subplot(2,1,2);
% plot(t, degg, 'b-', 'LineWidth',1.2); hold on;
% plot(t(locs), degg(locs), 'ko', 'MarkerFaceColor','k');
% plot(t(peak_pos), degg(peak_pos), 'rx', 'MarkerSize',8, 'LineWidth',1.5);
% if ~isempty(cycle_init_locs)
%     plot(t(cycle_init_locs), degg(cycle_init_locs), 'g*', 'MarkerSize',10);
% end
% legend('degg', 'locs', 'peak pos', 'cycle init locs');
% title('degg Signal with Detected Peaks and Cycle Initiation Locations');
% xlabel('Time (s)'); ylabel('Amplitude');
% grid on;
% 
% end