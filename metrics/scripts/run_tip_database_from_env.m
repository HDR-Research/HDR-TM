function run_tip_database_from_env()
hdrRoot = getenv_required('METRICS_HDR_ROOT');
ldrRoot = getenv_required('METRICS_LDR_DIR');
preset = getenv_required('METRICS_PRESET');
outputCsv = getenv_required('METRICS_OUTPUT_CSV');
summaryTxt = getenv_required('METRICS_SUMMARY_TXT');
resizeSize = str2double_default(getenv('METRICS_RESIZE_SIZE'), 512);
maxImages = str2double_default(getenv('METRICS_MAX_IMAGES'), 0);
matchMode = lower(string(getenv('METRICS_MATCH_MODE')));
if strlength(matchMode) == 0
    matchMode = "auto";
end

setup_metric_paths();
hdrFiles = gather_hdr_files(hdrRoot, preset);
if maxImages > 0
    firstIndex = max(1, numel(hdrFiles) - maxImages + 1);
    hdrFiles = hdrFiles(firstIndex:end);
end
ldrIndex = build_ldr_index(ldrRoot);

records = struct([]);
missing = strings(0, 1);
errors = strings(0, 1);
fprintf('[*] HDR dir: %s\n', hdrRoot);
fprintf('[*] LDR dir: %s\n', ldrRoot);
fprintf('[*] Preset: %s\n', preset);
fprintf('[*] Found HDR images: %d\n', numel(hdrFiles));
fprintf('[*] MATLAB native metric implementation\n');

for index = 1:numel(hdrFiles)
    hdrPath = hdrFiles{index};
    name = safe_name(hdrPath, hdrRoot);
    ldrPath = find_ldr(hdrPath, hdrRoot, ldrIndex, matchMode);
    if strlength(ldrPath) == 0
        missing(end + 1) = string(hdrPath); %#ok<AGROW>
        fprintf('[%d/%d] %s: missing LDR\n', index, numel(hdrFiles), name);
        continue;
    end

    try
        result = evaluate_metric_pair(hdrPath, char(ldrPath), preset, resizeSize);
        result.name = string(name);
        result = rmfield(result, {'status', 'error'});
        if isempty(records)
            records = result;
        else
            records(end + 1) = result; %#ok<AGROW>
        end
        fprintf(['[%d/%d] %s: NLPD=%.4f, TMQI=(%.4f,%.4f,%.4f), ' ...
            'TMQI2=(%.4f,%.4f,%.4f)\n'], index, numel(hdrFiles), name, ...
            result.nlpd, result.tmqi_q, result.tmqi_s, result.tmqi_n, ...
            result.tmqi2_q, result.tmqi2_s, result.tmqi2_n);
    catch exception
        errors(end + 1) = string(sprintf('%s | %s | %s', ...
            hdrPath, ldrPath, exception.message)); %#ok<AGROW>
        fprintf(2, '[%d/%d] %s: error: %s\n', ...
            index, numel(hdrFiles), name, exception.message);
    end
end

write_metrics_csv(records, outputCsv);
summary = write_summary(records, numel(hdrFiles), missing, errors, summaryTxt);
fprintf('\n%s\n', summary);
end

function files = gather_hdr_files(rootDir, preset)
switch lower(preset)
    case 'hdtv1k_pq_rec2020'
        extensions = {'.png'};
        listing = dir(fullfile(rootDir, '*'));
    case 'hdrps_linear100_rec709'
        extensions = {'.exr', '.hdr'};
        listing = dir(fullfile(rootDir, '*'));
    case 'lvzhdr_linear100_rec709'
        extensions = {'.hdr', '.exr'};
        listing = dir(fullfile(rootDir, '**', '*'));
    otherwise
        error('metrics:Preset', 'Unsupported preset: %s', preset);
end

files = {};
for index = 1:numel(listing)
    if listing(index).isdir
        continue;
    end
    [~, ~, extension] = fileparts(listing(index).name);
    if any(strcmpi(extension, extensions))
        files{end + 1} = fullfile(listing(index).folder, listing(index).name); %#ok<AGROW>
    end
end
files = sort(files);
if isempty(files)
    error('metrics:NoHDR', 'No HDR files found under %s', rootDir);
end
end

function index = build_ldr_index(rootDir)
index.byRel = containers.Map('KeyType', 'char', 'ValueType', 'char');
index.byName = containers.Map('KeyType', 'char', 'ValueType', 'char');
index.byStem = containers.Map('KeyType', 'char', 'ValueType', 'char');
index.bySafeStem = containers.Map('KeyType', 'char', 'ValueType', 'char');
index.byId = containers.Map('KeyType', 'char', 'ValueType', 'char');
extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'};
listing = dir(fullfile(rootDir, '**', '*'));

for item = 1:numel(listing)
    if listing(item).isdir
        continue;
    end
    [~, stem, extension] = fileparts(listing(item).name);
    if ~any(strcmpi(extension, extensions))
        continue;
    end
    path = fullfile(listing(item).folder, listing(item).name);
    relative = erase(path, [char(java.io.File(rootDir).getCanonicalPath()) filesep]);
    relative = strrep(relative, filesep, '/');
    relativeStem = strip_extension(relative);
    safeStem = regexprep(relativeStem, '[^A-Za-z0-9_.-]+', '_');
    put_if_missing(index.byRel, relative, path);
    put_if_missing(index.byName, listing(item).name, path);
    put_if_missing(index.byStem, stem, path);
    put_if_missing(index.bySafeStem, safeStem, path);
    imageId = extract_image_id(stem);
    if ~isempty(imageId)
        put_if_missing(index.byId, imageId, path);
    end
end
end

function path = find_ldr(hdrPath, hdrRoot, index, mode)
relative = erase(hdrPath, [char(java.io.File(hdrRoot).getCanonicalPath()) filesep]);
relative = strrep(relative, filesep, '/');
relativeStem = strip_extension(relative);
[~, stem] = fileparts(hdrPath);
safeStem = regexprep(relativeStem, '[^A-Za-z0-9_.-]+', '_');
extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'};
path = "";

imageId = extract_image_id(stem);
if any(mode == ["auto", "id_prefix"]) && ~isempty(imageId) && isKey(index.byId, imageId)
    path = string(index.byId(imageId));
end
if strlength(path) == 0 && any(mode == ["auto", "safe_name"])
    path = lookup_extensions(index.byName, safeStem, extensions);
    if strlength(path) == 0 && isKey(index.bySafeStem, safeStem)
        path = string(index.bySafeStem(safeStem));
    end
end
if strlength(path) == 0 && any(mode == ["auto", "relative"])
    path = lookup_extensions(index.byRel, relativeStem, extensions);
end
if strlength(path) == 0 && any(mode == ["auto", "stem"])
    path = lookup_extensions(index.byName, stem, extensions);
    if strlength(path) == 0 && isKey(index.byStem, stem)
        path = string(index.byStem(stem));
    end
end
end

function imageId = extract_image_id(stem)
token = regexp(stem, '^(a\d+)', 'tokens', 'once', 'ignorecase');
if isempty(token)
    imageId = '';
else
    imageId = lower(token{1});
end
end

function path = lookup_extensions(mapping, stem, extensions)
path = "";
for index = 1:numel(extensions)
    key = [stem extensions{index}];
    if isKey(mapping, key)
        path = string(mapping(key));
        return;
    end
end
end

function name = safe_name(path, rootDir)
relative = erase(path, [char(java.io.File(rootDir).getCanonicalPath()) filesep]);
name = regexprep(strip_extension(relative), '[^A-Za-z0-9_.-]+', '_');
end

function value = strip_extension(path)
[folder, name] = fileparts(path);
if isempty(folder)
    value = name;
else
    value = strrep(fullfile(folder, name), filesep, '/');
end
end

function put_if_missing(mapping, key, value)
if ~isKey(mapping, key)
    mapping(key) = value;
end
end

function write_metrics_csv(records, outputCsv)
ensure_parent(outputCsv);
fields = {'name', 'hdr_path', 'ldr_path', 'height', 'width', ...
    'tmqi_q', 'tmqi_s', 'tmqi_n', ...
    'tmqi2_q', 'tmqi2_s', 'tmqi2_n', 'nlpd'};
if isempty(records)
    tableResult = cell2table(cell(0, numel(fields)), 'VariableNames', fields);
else
    tableResult = struct2table(orderfields(records, fields));
end
writetable(tableResult, outputCsv);
end

function text = write_summary(records, numHdr, missing, errors, outputPath)
lines = [
    ""
    "===== Metric Summary ====="
    "num_hdr: " + numHdr
    "num_valid: " + numel(records)
    "num_missing: " + numel(missing)
    "num_errors: " + numel(errors)
    metric_line("mean_tmqi_q", records, "tmqi_q")
    metric_line("mean_tmqi_s", records, "tmqi_s")
    metric_line("mean_tmqi_n", records, "tmqi_n")
    metric_line("mean_tmqi2_q", records, "tmqi2_q")
    metric_line("mean_tmqi2_s", records, "tmqi2_s")
    metric_line("mean_tmqi2_n", records, "tmqi2_n")
    metric_line("mean_nlpd", records, "nlpd")
];
if ~isempty(missing)
    lines = [lines; ""; "Missing LDR files:"; missing(1:min(50, end))]; %#ok<AGROW>
end
if ~isempty(errors)
    lines = [lines; ""; "Errors:"; errors(1:min(50, end))]; %#ok<AGROW>
end
text = strjoin(lines, newline);
ensure_parent(outputPath);
file = fopen(outputPath, 'w');
cleanup = onCleanup(@() fclose(file)); %#ok<NASGU>
fprintf(file, '%s\n', text);
end

function line = metric_line(label, records, field)
if isempty(records)
    value = NaN;
else
    values = arrayfun(@(record) double(record.(field)), records);
    value = mean(values(isfinite(values)));
end
line = label + ": " + sprintf('%.6f', value);
end

function ensure_parent(path)
folder = fileparts(path);
if ~isempty(folder) && ~isfolder(folder)
    mkdir(folder);
end
end

function value = getenv_required(name)
value = getenv(name);
if isempty(value)
    error('metrics:Environment', '%s must be set.', name);
end
end

function value = str2double_default(text, fallback)
value = str2double(text);
if isempty(text) || isnan(value)
    value = fallback;
end
end
