% run_case30_scenario_load_scales.m
% ---------------------------------
% Runs IEEE 30-bus OPF for scenario-derived single-hour load scales.

clear; clc;

in_file = 'case30_test_sample0_load_scales.csv';
out_file = 'case30_test_sample0_dispatch_results.csv';

scales = readtable(in_file);
mpopt = mpoption('verbose', 0, 'out.all', 0);

n = height(scales);

scenario_id = scales.scenario_id;
hour = scales.hour;
load_scale = scales.load_scale;
success = zeros(n, 1);
objective_cost = zeros(n, 1);
base_total_load_mw = zeros(n, 1);
scaled_total_load_mw = zeros(n, 1);
total_generation_mw = zeros(n, 1);

for i = 1:n
    mpc = loadcase('case30');

    base_total_load_mw(i) = sum(mpc.bus(:, 3));

    mpc.bus(:, 3) = load_scale(i) * mpc.bus(:, 3);  % Pd
    mpc.bus(:, 4) = load_scale(i) * mpc.bus(:, 4);  % Qd

    results = runopf(mpc, mpopt);

    success(i) = results.success;
    objective_cost(i) = results.f;
    scaled_total_load_mw(i) = sum(results.bus(:, 3));
    total_generation_mw(i) = sum(results.gen(:, 2));

    fprintf( ...
        'scenario %02d | scale %.4f | success %d | cost %.2f\n', ...
        scenario_id(i), load_scale(i), success(i), objective_cost(i) ...
    );
end

out = table( ...
    scenario_id, hour, load_scale, success, objective_cost, ...
    base_total_load_mw, scaled_total_load_mw, total_generation_mw ...
);
writetable(out, out_file);

fprintf('\nSaved: %s\n', out_file);
