addpath code
addpath images
run('code/vlfeat-0.9.21/toolbox/vl_setup')

feature_params = struct('template_size', 36, 'hog_cell_size', 6);
data_path = '../data/'; 
test_scn_path = 'data/demo/';
%test_scn_path = fullfile(data_path,'demo');
load('w.mat')
load('b.mat')
[bboxes, confidences, image_ids,random_number] = run_detector_demo(test_scn_path, w, b, feature_params);
%[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
 %   evaluate_detections(bboxes, confidences, image_ids, label_path);

visualize_detections_by_image_no_gt_demo(bboxes, confidences, image_ids, test_scn_path,random_number)