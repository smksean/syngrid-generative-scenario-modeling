% run_case30_baseline.m
% ---------------------
% Minimal IEEE 30-bus OPF baseline using MATPOWER.

clear; clc;

mpopt = mpoption('verbose', 0, 'out.all', 0);

mpc = loadcase('case30');
results = runopf(mpc, mpopt);

success = results.success;
objective_cost = results.f;
total_load_mw = sum(results.bus(:, 3));
total_generation_mw = sum(results.gen(:, 2));

fprintf('IEEE 30-bus OPF baseline\n');
fprintf('Success              : %d\n', success);
fprintf('Objective cost       : %.4f\n', objective_cost);
fprintf('Total load           : %.4f MW\n', total_load_mw);
fprintf('Total generation     : %.4f MW\n', total_generation_mw);

out = table(success, objective_cost, total_load_mw, total_generation_mw);
writetable(out, 'case30_baseline_results.csv');

gen = array2table( ...
    results.gen(:, [1 2 3 9 10]), ...
    'VariableNames', {'bus_id', 'pg_mw', 'qg_mvar', 'pmax_mw', 'pmin_mw'} ...
);
writetable(gen, 'case30_baseline_generators.csv');

bus = array2table( ...
    results.bus(:, [1 8 9 3 4]), ...
    'VariableNames', {'bus_id', 'vm_pu', 'va_deg', 'pd_mw', 'qd_mvar'} ...
);
writetable(bus, 'case30_baseline_buses.csv');

fprintf('\nSaved: case30_baseline_results.csv\n');
fprintf('Saved: case30_baseline_generators.csv\n');
fprintf('Saved: case30_baseline_buses.csv\n');
