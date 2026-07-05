function [egg_cycle_acc, degg_cycle_acc] = cycle_separation(egg, degg, Fs, f0, cycle_init_locs)
    egg_cycle_acc  = [];
    degg_cycle_acc = [];
    egg_cycle = [];
    
    time = diff(cycle_init_locs);
    normalize_sample = 0:1/1000:1;


    for i = 1:1:length(time)
%         110%
        if(time(1,i) <= (Fs/f0)*1.1)
            egg_cycle_temp = egg(1,cycle_init_locs(1,i):cycle_init_locs(1,i+1));
            original_sample = 0:1/length(egg_cycle_temp):1-1/length(egg_cycle_temp);
            egg_cycle = interp1(original_sample,egg_cycle_temp,normalize_sample,'linear','extrap');
            egg_cycle = egg_cycle - egg_cycle(1,1);
            
            degg_cycle_temp = degg(1,cycle_init_locs(1,i):cycle_init_locs(1,i+1));
            original_sample = 0:1/length(degg_cycle_temp):1-1/length(degg_cycle_temp);
            degg_cycle = interp1(original_sample,degg_cycle_temp,normalize_sample,'linear','extrap');
            degg_cycle = degg_cycle - degg_cycle(1,1);

            egg_cycle_acc = [egg_cycle_acc ; egg_cycle];
            degg_cycle_acc = [degg_cycle_acc ; degg_cycle];

        end

    end




end