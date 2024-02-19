table = data64_5mice;

for j = 1:5
    data_table = table(table.mouse_No == j,:);
    [sig, run, speed] = preprocess(data_table);
    eval(['signal',num2str(j),' = sig;'])
    eval(['runrest',num2str(j),' = run;'])
    eval(['speed',num2str(j),' = speed;'])
end

function [sig, run, speed] = preprocess(table)
    sigpath = table.Signal_Path;
    sig = {};
    for j = 1:length(sigpath)
        load(sigpath(j), 'dat_roi')
        signal = dat_roi;
        base = movmean(signal,1000,2); % calcurate baseline
        signal2 = signal - base; % substract baseline
        sig{j} = signal2;
    end
    birpath = table.Behavior_Path;
    run = {};
    speed = {};
    for j = 1:length(birpath)
        myVars = {'runrest','speed_rs'};
        load(birpath(j), myVars{:})
        run{j} = runrest;
        speed{j} = movmean(speed_rs,30, 2);  % speed preprocessing to delete noize
    end
end

% Save signal1-5, runrest1-5, speed1-5 to matfile
