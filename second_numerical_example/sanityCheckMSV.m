function sanityCheckMSV(stage, model, samples, mcmcoptions)
% sanityCheckMSV(stage, model, samples, mcmcoptions)
% stage: 'preMCMC' or 'postMCMC' or 'preForecast'

assert(isfield(model,'horizon') && isnumeric(model.horizon) && isscalar(model.horizon) && model.horizon>=1, ...
    'model.horizon must exist and be a positive scalar.');

% --- Basic model dimensions ---
reqFields = {'N','T','K','tildeK','Y','L','Weights','Ft','Givset'};
for i=1:numel(reqFields)
    assert(isfield(model,reqFields{i}), 'model.%s is missing.', reqFields{i});
end

[N,T] = size(model.Y);
assert(model.N == N, 'model.N mismatch: model.N=%d, size(model.Y,1)=%d', model.N, N);
assert(model.T == T, 'model.T mismatch: model.T=%d, size(model.Y,2)=%d', model.T, T);

assert(isequal(size(model.L), [model.N, model.K]), ...
    'model.L must be [N,K]. Got [%d,%d].', size(model.L,1), size(model.L,2));
assert(isequal(size(model.Weights), [model.N, model.K]), ...
    'model.Weights must be [N,K]. Got [%d,%d].', size(model.Weights,1), size(model.Weights,2));
assert(isequal(size(model.Ft), [model.K, model.T]), ...
    'model.Ft must be [K,T]. Got [%d,%d].', size(model.Ft,1), size(model.Ft,2));

% --- Givens / tildeK consistency ---
assert(size(model.Givset,2)==2, 'model.Givset must have 2 columns (i,j pairs).');
assert(model.tildeK == size(model.Givset,1), ...
    'model.tildeK mismatch: model.tildeK=%d but size(Givset,1)=%d', model.tildeK, size(model.Givset,1));

% --- Forecast holdout consistency ---
if isfield(model,'actual') && ~isempty(model.actual)
    assert(isequal(size(model.actual), [model.N, model.horizon]), ...
        'model.actual must be [N,horizon]. Got [%d,%d].', size(model.actual,1), size(model.actual,2));
end

% --- MCMC option sanity ---
if nargin >= 4 && ~isempty(mcmcoptions) && isfield(mcmcoptions,'train')
    assert(isfield(mcmcoptions.train,'T') && isfield(mcmcoptions.train,'StoreEvery'), ...
        'mcmcoptions.train must have fields T and StoreEvery.');
    assert(mod(mcmcoptions.train.StoreEvery,1)==0 && mcmcoptions.train.StoreEvery>=1, ...
        'StoreEvery must be a positive integer.');
end

% --- Post-MCMC / sample checks ---
if nargin >= 3 && ~isempty(samples) && ~strcmpi(stage,'preMCMC')
    reqS = {'F','Phi_h','Phi_delta','h_0','delta_0','sigma2_h','sigma2_delta','Weights','Ft','sigma2'};
    for i=1:numel(reqS)
        assert(isfield(samples,reqS{i}), 'samples.%s is missing.', reqS{i});
    end

    S = size(samples.F,1);
    n = model.K*model.T + model.tildeK*model.T;
    assert(size(samples.F,2)==n, ...
        'samples.F must be [S, n] with n=(K+tildeK)*T. Got n=%d expected %d.', size(samples.F,2), n);

    assert(isequal(size(samples.Ft), [model.K, model.T, S]), ...
        'samples.Ft must be [K,T,S]. Got [%d,%d,%d].', size(samples.Ft,1), size(samples.Ft,2), size(samples.Ft,3));

    assert(isequal(size(samples.Weights), [model.N, model.K, S]), ...
        'samples.Weights must be [N,K,S]. Got [%d,%d,%d].', size(samples.Weights,1), size(samples.Weights,2), size(samples.Weights,3));

    assert(isequal(size(samples.Phi_h), [S, model.K]), ...
        'samples.Phi_h must be [S,K]. Got [%d,%d].', size(samples.Phi_h,1), size(samples.Phi_h,2));
    assert(isequal(size(samples.Phi_delta), [S, model.tildeK]), ...
        'samples.Phi_delta must be [S,tildeK]. Got [%d,%d].', size(samples.Phi_delta,1), size(samples.Phi_delta,2));

    assert(isequal(size(samples.h_0), [S, model.K]), ...
        'samples.h_0 must be [S,K].');
    assert(isequal(size(samples.delta_0), [S, model.tildeK]), ...
        'samples.delta_0 must be [S,tildeK].');

    assert(isequal(size(samples.sigma2_h), [S, model.K]), ...
        'samples.sigma2_h must be [S,K].');
    assert(isequal(size(samples.sigma2_delta), [S, model.tildeK]), ...
        'samples.sigma2_delta must be [S,tildeK].');

    % sigma2 in mcmcTrain.m is scalar per draw: [1,S]
    assert(isequal(size(samples.sigma2), [1, S]) || isequal(size(samples.sigma2), [S, 1]), ...
        'samples.sigma2 expected [1,S] (scalar noise per draw). Got [%d,%d].', size(samples.sigma2,1), size(samples.sigma2,2));
end

fprintf('[sanityCheckMSV] %s passed. N=%d, T=%d, K=%d, tildeK=%d\n', stage, model.N, model.T, model.K, model.tildeK);
end