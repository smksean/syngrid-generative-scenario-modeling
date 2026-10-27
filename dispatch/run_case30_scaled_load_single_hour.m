% run_case30_scaled_load_single_hour.m
% ------------------------------------
% Single-hour IEEE 30-bus OPF with uniformly scaled load.

clear; clc;

load_scale = 1.10;

mpopt = mpoption('verbose', 0, 'out.all', 0);

mpc = loadcase('case30');
base_total_load_mw = sum(mpc.bus(:, 3));

mpc.bus(:, 3) = load_scale * mpc.bus(:, 3);  % Pd
mpc.bus(:, 4) = load_scale * mpc.bus(:, 4);  % Qd

results = runopf(mpc, mpopt);

success = results.success;
objective_cost = results.f;
total_load_mw = sum(results.bus(:, 3));
total_generation_mw = sum(results.gen(:, 2));

fprintf('IEEE 30-bus scaled-load OPF\n');
fprintf('Load scale           : %.4f\n', load_scale);
fprintf('Success              : %d\n', success);
fprintf('Objective cost       : %.4f\n', objective_cost);
fprintf('Base total load      : %.4f MW\n', base_total_load_mw);
fprintf('Scaled total load    : %.4f MW\n', total_load_mw);
fprintf('Total generation     : %.4f MW\n', total_generation_mw);

out = table( ...
    load_scale, success, objective_cost, ...
    base_total_load_mw, total_load_mw, total_generation_mw ...
);
writetable(out, 'case30_scaled_load_single_hour_results.csv');

fprintf('\nSaved: case30_scaled_load_single_hour_results.csv\n');
