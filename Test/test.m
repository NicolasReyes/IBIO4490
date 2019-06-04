% ------------------------------------------------------------------------ 
% Jordi Pont-Tuset - http://jponttuset.github.io/
% April 2016
% ------------------------------------------------------------------------ 
% This file is part of the DAVIS package presented in:
%   Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams,
%   Luc Van Gool, Markus Gross, Alexander Sorkine-Hornung
%   A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
%   CVPR 2016
% Please consider citing the paper if you use this code.
% ------------------------------------------------------------------------
% This file shows the comparison table between all techniques and on all
% measures (Table 2 in the paper)
% ------------------------------------------------------------------------

% Get the parameters
clear;
experiments_params();

% Which set of the ground truth use
year = 2016;
db_set_properties(year,0,'480p');
gt_set = 'val';
[db_seq_list, stab_seqs]= db_seqs(gt_set);

%% Evaluate them or load pre-computed evaluation
F = cell(1,length(techniques));
J = cell(1,length(techniques));
T = cell(1,length(techniques));


for ii=1:length(techniques)
    eval = eval_result(techniques{ii},{'F','J','T'},gt_set);
    F{ii} = eval.F;
    J{ii} = eval.J;
    T{ii} = eval.T;
end
eval = eval_result('gt','T',gt_set);
Tgt  = eval.T;

%% Put them in a single matrix
all_F.mean   = zeros(length(techniques),length(F{1}.mean));
all_F.recall = zeros(length(techniques),length(F{1}.mean));
all_F.decay  = zeros(length(techniques),length(F{1}.mean));
all_J = all_F;
all_T.mean   = zeros(length(techniques),length(F{1}.mean));

for ii=1:length(techniques)
    all_F.mean(ii,:)   = F{ii}.mean;
    all_F.recall(ii,:) = F{ii}.recall;
    all_F.decay(ii,:)  = F{ii}.decay;

    all_J.mean(ii,:)   = J{ii}.mean;
    all_J.recall(ii,:) = J{ii}.recall;
    all_J.decay(ii,:)  = J{ii}.decay;  
    
    all_T.mean(ii,:)   = T{ii}.mean;
end

%% Display evaluation table

disp(repmat('=',[1,165]))
fprintf('\t\t');
for ii=1:length(techniques), fprintf('%s\t',techniques{ii}), end; fprintf('\n');
disp(repmat('-',[1,165]))
fprintf('J mean  \t');fprintf('%0.3f\t',mean(all_J.mean,2)'); fprintf('\n');
fprintf('J recall\t');fprintf('%0.3f\t',mean(all_J.recall,2)'); fprintf('\n');
fprintf('J decay \t');fprintf('%0.3f\t',mean(all_J.decay,2)'); fprintf('\n');
disp(repmat('-',[1,165]))
fprintf('F mean  \t');fprintf('%0.3f\t',mean(all_F.mean,2)'); fprintf('\n');
fprintf('F recall\t');fprintf('%0.3f\t',mean(all_F.recall,2)'); fprintf('\n');
fprintf('F decay \t');fprintf('%0.3f\t',mean(all_F.decay,2)'); fprintf('\n');
disp(repmat('-',[1,165]))
%fprintf('T (GT %0.3f)\t',mean(Tgt.mean(stab_seqs)));fprintf('%0.3f\t',mean(all_T.mean(:,stab_seqs),2)'); fprintf('\n');
disp(repmat('=',[1,165]))







%% Show all means
% figure;
% plot(all_T.mean')
% hold on
% plot(Tgt.mean,'k--');
% legend([techniques(:); 'GT'])


