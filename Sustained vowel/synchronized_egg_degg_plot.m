function [cycle_std, V_tot, V1, V2, V3, V4, cycle_std_in, cycle_std_out, x_centre, y_centre] = synchronized_egg_degg_plot(eggCycles, deggCycles,egg_cycle_mean, degg_cycle_mean)
    %% EGG-dEGG plot for synchronized EGG cycle and corresponding standard deviation
    
    sync_mean = [egg_cycle_mean; degg_cycle_mean];
    dist_sum = 0;

    for i = 1:1:length(eggCycles(:,1))
        point = [eggCycles(i,:) ; deggCycles(i,:)];
        dist_diff = (point-sync_mean).^2;
        dist = dist_diff(1,:) + dist_diff(2,:);
        dist_sum = dist_sum + dist;
    end
    cycle_std = sqrt(dist_sum/length(eggCycles(:,1)));

    cycle_std_in = [];
    cycle_std_out = [];

    std_x_acc = [];
    std_y_acc = [];
    
    [degg_max, degg_max_idx] = max(degg_cycle_mean);
    x_centre = mean(egg_cycle_mean);
    y_centre = 0;

    V1 = 0;
    V2 = 0;
    V3 = 0;
    V4 = 0;

    V1_num = 0;
    V2_num = 0;
    V3_num = 0;
    V4_num = 0;
    
    for i = 1:1:length(egg_cycle_mean)
        x_egg = egg_cycle_mean(1,i);
        y_degg = degg_cycle_mean(1,i);
    
        x_std = cycle_std(1,i)*cos(atan(abs(-y_degg)/abs(x_centre-x_egg)));
        y_std = cycle_std(1,i)*sin(atan(abs(-y_degg)/abs(x_centre-x_egg)));


        std_x_acc = [std_x_acc ; x_std];
        std_y_acc = [std_y_acc ; y_std];
    
        if(x_egg < x_centre && y_degg >= 0)
            cycle_std_in(1,i)  = x_egg  + x_std;
            cycle_std_in(2,i)  = y_degg - y_std;
            cycle_std_out(1,i) = x_egg  - x_std;
            cycle_std_out(2,i) = y_degg + y_std;
            V1 = V1 + cycle_std(1,i);
            V1_num = V1_num + 1;
        elseif(x_egg >= x_centre && y_degg >= 0)
            cycle_std_in(1,i)  = x_egg  - x_std;
            cycle_std_in(2,i)  = y_degg - y_std;
            cycle_std_out(1,i) = x_egg  + x_std;
            cycle_std_out(2,i) = y_degg + y_std;
            V2 = V2 + cycle_std(1,i);
            V2_num = V2_num + 1;
        elseif(x_egg >= x_centre && y_degg < 0)
            cycle_std_in(1,i)  = x_egg  - x_std;
            cycle_std_in(2,i)  = y_degg + y_std;
            cycle_std_out(1,i) = x_egg  + x_std;
            cycle_std_out(2,i) = y_degg - y_std;
            V3 = V3 + cycle_std(1,i);
            V3_num = V3_num + 1;
        elseif(x_egg < x_centre && y_degg < 0)
            cycle_std_in(1,i)  = x_egg  + x_std;
            cycle_std_in(2,i)  = y_degg + y_std;
            cycle_std_out(1,i) = x_egg  - x_std;
            cycle_std_out(2,i) = y_degg - y_std;
            V4 = V4 + cycle_std(1,i);
            V4_num = V4_num + 1;
        end
    end

    V1 = V1/V1_num;
    V2 = V2/V2_num;
    V3 = V3/V3_num;
    V4 = V4/V4_num;
    V_tot = V1 + V2 + V3 + V4;

end