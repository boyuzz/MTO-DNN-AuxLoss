function measure(architecture, prefix)
    sets = {'Set5','Set14','BSDS100'}; % , 
    for s = 1:length(sets)
        disp(s)
        [model_1_psnr, model_1_ssim, psnr_1_list, ssim_1_list, gt_files] = measure_result(1.0, sets{s}, architecture, prefix);
        [model_0_psnr, model_0_ssim, psnr_0_list, ssim_0_list] = measure_result(0.0, sets{s}, architecture, prefix);
        disp(['mean psnr comparison:', num2str(mean(model_1_psnr)), ' ', num2str(mean(model_0_psnr))]);
        disp(['mean ssim comparison:', num2str(mean(model_1_ssim)), ' ', num2str(mean(model_0_ssim))]);
        [p,h] = signrank(model_1_psnr, model_0_psnr);
        disp(h);
        [p,h] = signrank(model_1_ssim, model_0_ssim);
        disp(h);
    end
    % diff_psnr = psnr_1_list-psnr_0_list;
    % diff_ssim = ssim_1_list-ssim_0_list;
    % [value_psnr, idx_psnr] = sort(diff_psnr);
    % [value_ssim, idx_ssim] = sort(diff_ssim);
end

function [model_psnr, model_ssim, psnr_list, ssim_list, gt_files] = measure_result(alpha, sets, architecture, prefix)
    scale = 2;
    gt_folder = ['../data/', sets];
    total_run = 10;
    model_psnr = zeros([1,total_run]);
    model_ssim = zeros([1,total_run]);
    gt_files = [];
    gt_files = [gt_files; dir(fullfile(gt_folder, ['*', num2str(scale), '.mat']))];
    for run = 1:total_run
        if alpha ~= 0
            out_psnr_folder = ['./result/',architecture,'/', prefix,'/', num2str(run-1),'/psnr/',sets, '/', num2str(alpha,'%1.1f')];
            out_ssim_folder = ['./result/',architecture,'/', prefix,'/', num2str(run-1),'/ssim/',sets, '/', num2str(alpha,'%1.1f')];
        else
            out_psnr_folder = ['./result/',architecture,'/', prefix,'/', num2str(run-1),'/psnr/',sets, '/', num2str(alpha,'%1.1f')];
            out_ssim_folder = ['./result/',architecture,'/', prefix,'/', num2str(run-1),'/ssim/',sets, '/', num2str(alpha,'%1.1f')];
        end
        % out_folder = '/home/boyu/Dropbox/work/SmartAI/SR/super_resolution/result';
        % out_folder = './Set5_bic';
        out_psnr_files = [];
        out_psnr_files = [out_psnr_files; dir(fullfile(out_psnr_folder, '*.mat'))];

        out_ssim_files = [];
        out_ssim_files = [out_ssim_files; dir(fullfile(out_ssim_folder, '*.mat'))];

        bic_psnr = 0;
        bic_ssim = 0;

        psnr_list = [];
        ssim_list = [];
        for i = 1:numel(gt_files)
            f_info = gt_files(i);
            f_path = fullfile(f_info.folder, f_info.name);
            load(f_path);
            img_raw = im_gt_y/255.;

    %         [height, width] = size(img_raw);
    %         bic_img = imresize(imresize(img_raw,1/scale,'bicubic'), [height, width], 'bicubic');
    %         bic_psnr = bic_psnr + psnr(bic_img, img_raw);
    %         bic_ssim = bic_ssim + ssim(bic_img, img_raw);

            f_psnr_info = out_psnr_files(i);
            f_psnr_path = fullfile(f_psnr_info.folder, f_psnr_info.name);
            load(f_psnr_path);
            img_psnr = double(hr);

            f_ssim_info = out_ssim_files(i);
            f_ssim_path = fullfile(f_ssim_info.folder, f_ssim_info.name);
            load(f_ssim_path);
            img_ssim = double(hr);
            %(11-4) *3 - (3-1)
            if strcmp(architecture, 'srcnn')
                edge = 8;
                [height, width] = size(img_raw);
                img_raw = img_raw(edge+1:height-edge, edge+1:width-edge);
            elseif strcmp(architecture, 'fsrcnn')
                [h, w] = size(img_psnr);
                [height, width] = size(img_raw);
                hgap = (height-h)/2;
                wgap = (width-w)/2;
                img_raw = img_raw(hgap+1:height-hgap, wgap+1:width-wgap);
            else
                img_raw = modcrop(img_raw, scale);
                img_psnr = modcrop(img_psnr, scale);
                img_ssim = modcrop(img_ssim, scale);
                img_raw = shave(img_raw, [1,1] * scale);
                img_psnr = shave(img_psnr, [1,1] * scale);
                img_ssim = shave(img_ssim, [1,1] * scale);
            end

            psnr_list = [psnr_list, psnr(img_psnr, img_raw)];
            ssim_list = [ssim_list, ssim(img_ssim, img_raw)];
%             model_psnr(run) = model_psnr(run) + psnr(img_psnr, img_raw);
%             model_ssim(run) = model_ssim(run) + ssim(img_ssim, img_raw);
        end

        model_psnr(run) = sum(psnr_list)/numel(gt_files);
        model_ssim(run) = sum(ssim_list)/numel(gt_files);
    %     disp(['bicubic psnr:', string(bic_psnr/numel(gt_files)), 'bicubic ssim:', string(bic_ssim/numel(gt_files))])
%         disp(['model psnr:', string(model_psnr(run)), 'model ssim:', string(model_ssim(run))])
    end
end
