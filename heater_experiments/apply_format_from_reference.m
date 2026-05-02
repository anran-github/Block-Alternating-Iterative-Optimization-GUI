function apply_format_from_reference(ref_fig_path, target_fig_paths)

    % Load reference figure
    ref_fig = openfig(ref_fig_path, 'invisible');
    ref_ax_all = findall(ref_fig, 'type', 'axes');

    for i = 1:length(target_fig_paths)

        % Load target figure
        tgt_fig = openfig(target_fig_paths{i}, 'invisible');
        tgt_ax_all = findall(tgt_fig, 'type', 'axes');

        % Sort axes to ensure consistent order (important!)
        ref_ax_all = flipud(ref_ax_all);
        tgt_ax_all = flipud(tgt_ax_all);

        n = min(length(ref_ax_all), length(tgt_ax_all));

        for j = 1:n
            copy_axes_format(ref_ax_all(j), tgt_ax_all(j));
        end

        % Save updated figure
        savefig(tgt_fig, target_fig_paths{i});
        close(tgt_fig);
    end

    close(ref_fig);
end


function copy_axes_format(ref_ax, tgt_ax)

    % Copy axes-level properties
    props = { ...
        'FontName', ...
        'FontSize', ...
        'LineWidth', ...
        'XColor', ...
        'YColor', ...
        'Box', ...
        'GridLineStyle', ...
        'XGrid', ...
        'YGrid'};

    for k = 1:length(props)
        try
            val = get(ref_ax, props{k});
            set(tgt_ax, props{k}, val);
        catch
            % Skip incompatible properties safely
        end
    end

    % ---- Copy line styles (but NOT data) ----
    ref_lines = findall(ref_ax, 'Type', 'Line');
    tgt_lines = findall(tgt_ax, 'Type', 'Line');

    m = min(length(ref_lines), length(tgt_lines));

    for k = 1:m
        try
            set(tgt_lines(k), ...
                'LineWidth', get(ref_lines(k), 'LineWidth'), ...
                'LineStyle', get(ref_lines(k), 'LineStyle'), ...
                'Marker', get(ref_lines(k), 'Marker'), ...
                'MarkerSize', get(ref_lines(k), 'MarkerSize') ...
            );
        catch
        end
    end

    % ---- Copy legend style if exists ----
    ref_leg = findall(ancestor(ref_ax, 'figure'), 'Type', 'Legend');
    tgt_leg = findall(ancestor(tgt_ax, 'figure'), 'Type', 'Legend');

    if ~isempty(ref_leg) && ~isempty(tgt_leg)
        try
            set(tgt_leg(1), ...
                'FontSize', get(ref_leg(1), 'FontSize'), ...
                'Box', get(ref_leg(1), 'Box'));
        catch
        end
    end
end