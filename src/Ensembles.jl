module Ensembles

using Random
using MLJ
using MLJBase
import MLJModelInterface
using CategoricalArrays
using ..Output: printConfusionMatrix
using MLJ
import LIBSVM
using ..Training: evaluate_regression
using Printf;


const SVMPlassifier = @load ProbabilisticSVC pkg = LIBSVM verbosity = 0
const EpsilonSVRRegressor = @load EpsilonSVR pkg=LIBSVM verbosity=0


# For metadata
const TableType   = MLJModelInterface.Table
const ContinuousT = MLJModelInterface.Continuous
const FiniteT     = MLJModelInterface.Finite


export VotingClassifier,
       evaluate_voting_classifier,
       run_voting_ensembles,
       run_regression_ensembles,
       VotingRegressor,
       run_stacking_ensemble,
       summarize_regression_ensembles,
       plotEnsembleConfusionMatrix,
       plotConfusionEnsemble,
       run_regression_stacking_ensembles


# ===================================================
# DEFINITION OF VOTINGCLASSIFIER COMPATIBLE WITH MLJ
# ===================================================

"""
    VotingClassifier <: MLJModelInterface.Probabilistic

An ensemble classifier that combines predictions from multiple base models using voting strategies.

# Fields
- `models::Vector{MLJModelInterface.Probabilistic}`: Base probabilistic models
- `voting::Symbol`: `:hard` (majority vote) or `:soft` (probability averaging)
- `weights::Union{Nothing, Vector{Float64}}`: Optional weights (normalized internally)
"""
mutable struct VotingClassifier <: MLJModelInterface.Probabilistic
    models::Vector{MLJModelInterface.Probabilistic}
    voting::Symbol  # :hard or :soft
    weights::Union{Nothing, Vector{Float64}}
end

"""
    VotingClassifier(; models=MLJModelInterface.Probabilistic[], voting=:hard, weights=nothing)

Constructor with sanity checks and automatic weight normalization.
"""
function VotingClassifier(; models=MLJModelInterface.Probabilistic[], voting=:hard, weights=nothing)
    @assert voting in [:hard, :soft] "The only possible labels are :hard or :soft"
    
    normalized_weights = nothing
    if weights !== nothing
        @assert length(weights) == length(models) "Number of weights must match number of models"
        @assert all(w >= 0 for w in weights) "All weights must be non-negative"
        
        # Normalize weights to sum to 1.0
        normalized_weights = Float64.(weights) ./ sum(weights)
    end
    
    return VotingClassifier(models, voting, normalized_weights)
end

# -----------------------------
# FIT: train all base models
# -----------------------------
function MLJModelInterface.fit(model::VotingClassifier, verbosity::Int, X, y)
    # Train each base model on the same data
    machs = [begin
        mm = machine(deepcopy(m), X, y)
        MLJ.fit!(mm, verbosity=0)
        mm
    end for m in model.models]

    fitresults = (
        machines = machs,
        class_levels = collect(levels(y)),           # fixed order of levels
        class_pool   = CategoricalArrays.pool(y)     # pool for UnivariateFinite
    )
    
    cache  = nothing
    report = (n_models = length(model.models),
              voting   = model.voting,
              weights  = model.weights)
    
    return fitresults, cache, report
end

# -----------------------------
# HARD VOTING: predict_mode
# -----------------------------
function MLJModelInterface.predict_mode(model::VotingClassifier, fitresult, Xnew)
    machines     = fitresult.machines
    class_levels = fitresult.class_levels
    
    # Get mode predictions from all base models
    predictions = [MLJModelInterface.predict_mode(mach, Xnew) for mach in machines]

    n_samples = length(predictions[1])
    n_models  = length(machines)

    # Determine weights (equal if not specified)
    weights = model.weights === nothing ? fill(1.0/n_models, n_models) : model.weights

    # Output categorical vector with same levels
    ensemble_pred = similar(predictions[1])

    for i in 1:n_samples
        vote_counts = Dict{eltype(predictions[1][1]), Float64}()
        
        for (j, prediction) in enumerate(predictions)
            lbl = prediction[i]
            vote_counts[lbl] = get(vote_counts, lbl, 0.0) + weights[j]
        end
        
        best_label = nothing
        best_score = -Inf
        for (lbl, sc) in vote_counts
            if sc > best_score
                best_score = sc
                best_label = lbl
            end
        end

        ensemble_pred[i] = best_label
    end

    return ensemble_pred
end

# -----------------------------
# PREDICT (hard or soft)
# -----------------------------
function MLJModelInterface.predict(model::VotingClassifier, fitresult, Xnew)
    machines     = fitresult.machines
    class_levels = fitresult.class_levels
    class_pool   = fitresult.class_pool

    if model.voting == :hard
        # Hard voting: deterministic prediction via predict_mode
        yhat = MLJModelInterface.predict_mode(model, fitresult, Xnew)
        yhat = categorical(yhat; levels=class_levels)

        # Wrap as one-hot UnivariateFinite
        return [MLJBase.UnivariateFinite(
                    class_levels,
                    [lvl == yhat[i] ? 1.0 : 0.0 for lvl in class_levels];
                    pool=class_pool
                ) for i in 1:length(yhat)]
    end

    # Soft voting: average probabilities
    all_predictions = [MLJ.predict(mach, Xnew) for mach in machines]

    n_samples = length(all_predictions[1])
    n_models  = length(machines)
    n_classes = length(class_levels)
    weights   = model.weights === nothing ? fill(1.0/n_models, n_models) : model.weights

    avg_probs = zeros(n_samples, n_classes)
    for (w, prediction) in zip(weights, all_predictions)
        for i in 1:n_samples
            p_i = prediction[i]
            if p_i isa MLJBase.UnivariateFinite
                for (j, level) in enumerate(class_levels)
                    avg_probs[i, j] += w * pdf(p_i, level)
                end
            else
                # Deterministic → one-hot
                for (j, level) in enumerate(class_levels)
                    avg_probs[i, j] += w * (p_i == level ? 1.0 : 0.0)
                end
            end
        end
    end

    # Normalize each row
    for i in 1:n_samples
        s = sum(@view avg_probs[i, :])
        if s > 0
            @. avg_probs[i, :] = avg_probs[i, :] / s
        end
    end

    return [MLJBase.UnivariateFinite(class_levels, @view avg_probs[i, :]; pool=class_pool)
            for i in 1:n_samples]
end

# -----------------------------
# MLJ metadata
# -----------------------------
MLJModelInterface.metadata_model(
    VotingClassifier,
    input_scitype   = TableType(ContinuousT),
    target_scitype  = AbstractVector{<:FiniteT},
    supports_weights = false,
    load_path       = "VotingClassifier"
)

# ------------------------------------------------------
# Convenience: evaluate voting classifier on test set
# ------------------------------------------------------
"""
    evaluate_voting_classifier(mach, X_test, y_test; title, weighted=false)

Helper to:
1. Get predictions via `predict_mode`.
2. Convert to strings.
3. Print macro metrics + confusion matrix via `printConfusionMatrix`.
"""
function evaluate_voting_classifier(mach,
                                    X_test,
                                    y_test;
                                    title::String = "Hard Voting Ensemble",
                                    weighted::Bool = false)

    yhat_cat = MLJModelInterface.predict_mode(mach, MLJ.table(X_test))
    outputs  = string.(yhat_cat)
    targets  = string.(y_test)

    classes = unique(vcat(outputs, targets))

    println()
    println(title * " – Test Results:")
    printConfusionMatrix(outputs, targets, classes; weighted=weighted)

    return outputs
end

# (add this to the export list if you have one at the top)

"""
    run_voting_ensembles(ensemble_specs, y_train, y_test;
                         voting=:hard, weights=nothing,
                         title_prefix="Ensemble",
                         cm_plotter=nothing)

Train a VotingClassifier ensemble for each entry in `ensemble_specs`.

- `ensemble_specs` :: Dict{String, NamedTuple}
    Each entry: name => (X_train, X_test, models::Vector)

- `y_train` :: AbstractVector
    Training labels (categorical for classification).

- `y_test` :: AbstractVector
    Test labels (same for all representations).

Keyword args:
- `voting` :: Symbol        – `:hard` or `:soft`
- `weights` :: Union{Nothing, Vector{<:Real}} – optional model weights
- `title_prefix` :: String  – used in plot titles
- `cm_plotter` :: Union{Nothing, Function}
    If provided, it is called as:
    ``

Returns:
- `ensemble_results` :: Dict{String, Tuple{Vector{String}, Vector{String}}}
    Mapping dataset name => (predicted_labels, true_labels_as_string)
"""
function run_voting_ensembles(
    ensemble_specs::Dict{String, <:NamedTuple},
    y_train,
    y_test;
    voting::Symbol = :hard,
    weights = nothing,
    title_prefix::String = "Ensemble",
    rng_seed::Union{Nothing,Int} = 1234,
    weighted_metrics::Bool = true
)
    if rng_seed !== nothing
        Random.seed!(rng_seed)
    end

    y_train_cat = categorical(y_train)
    y_test_str  = string.(y_test)

    ensemble_results = Dict{String, Tuple{Vector{String}, Vector{String}}}()

    println("\nVOTING ENSEMBLES – CLASSIFICATION")
    println("=================================\n")

    for (ds_name, spec) in ensemble_specs
        X_train     = spec.X_train
        X_test      = spec.X_test
        base_models = spec.models

        println(">>> [$title_prefix] Dataset: $ds_name (voting=$(voting))")

        voting_model = VotingClassifier(
            models  = base_models,
            voting  = voting,
            weights = weights
        )

        mach = machine(voting_model, MLJ.table(X_train), y_train_cat)
        MLJ.fit!(mach, rows = 1:size(X_train, 1), verbosity = 0)

        yhat    = MLJ.predict_mode(mach, MLJ.table(X_test))
        outputs = string.(yhat)

        # Guardamos para gráficos u otros análisis posteriores
        ensemble_results[ds_name] = (outputs, y_test_str)

        # Métricas + matriz de confusión con el formato del profesor
        classes = unique(vcat(outputs, y_test_str))
        printConfusionMatrix(outputs, y_test_str, classes; weighted=weighted_metrics)
        println()  # línea en blanco para legibilidad
    end

    return ensemble_results
end


"""
    run_stacking_ensembles(specs, y_train, y_test;
                           base_names = (:ANN, :SVM, :DT, :KNN),
                           metalearner = SVMPlassifier(),
                           nfolds = 5,
                           rng_seed = 1234,
                           title_prefix = "Stacking Ensemble",
                           plot_cm = nothing)

Build and evaluate a stacking ensemble for each representation in `specs`.

- `specs` is a Dict like:
    Dict(
        "Raw Attributes" => (X_train = ..., X_test = ..., models = [m_ann, m_svm, m_dt, m_knn]),
        "PCA (95% Var)"  => (X_train = ..., X_test = ..., models = [...]),
        ...
    )

- `y_train`, `y_test` are the target vectors (same encoding as in the rest of the notebook).
- `base_names` are the names used for the base models inside the `Stack`.
- `metalearner` is the meta-model (default: probabilistic SVM).
- If `plot_cm` is not `nothing`, it will be called as
    plot_cm(outputs, targets, classes, title).

Returns a Dict{String, NamedTuple} with metrics and confusion matrix per representation.
"""
function run_stacking_ensembles(
    specs::Dict{String, <:NamedTuple},
    y_train,
    y_test;
    base_names::NTuple = (:ANN, :SVM, :DT, :KNN),
    metalearner = SVMPlassifier(),
    nfolds::Int = 5,
    rng_seed::Int = 1234,
    title_prefix::String = "Stacking",
    weighted_metrics::Bool = true
)
    Random.seed!(rng_seed)

    ensemble_results = Dict{String, Tuple{Vector{String}, Vector{String}}}()

    println("\nSTACKING ENSEMBLES – CLASSIFICATION")
    println("===================================\n")

    for (repr_name, spec) in specs
        Xtr         = spec.X_train
        Xte         = spec.X_test
        base_models = spec.models

        @assert length(base_models) == length(base_names) "Number of base models must match base_names."

        println(">>> [$title_prefix] Representation: $repr_name")

        base_tuple = NamedTuple{base_names}(base_models)

        stack_model = Stack(
            metalearner = metalearner,
            resampling  = CV(nfolds = nfolds, shuffle = true, rng = rng_seed),
            measures    = [log_loss];
            base_tuple...
        )

        stack_mach = machine(stack_model, MLJ.table(Xtr), categorical(y_train))
        MLJ.fit!(stack_mach, verbosity = 0)

        yhat_mode = MLJ.predict_mode(stack_mach, MLJ.table(Xte))

        outputs = string.(yhat_mode)
        targets = string.(y_test)

        ensemble_results[repr_name] = (outputs, targets)

        classes = unique(vcat(outputs, targets))
        printConfusionMatrix(outputs, targets, classes; weighted=weighted_metrics)
        println()
    end

    return ensemble_results
end


# =========================================
# REGRESSION ENSEMBLE (VOTING REGRESSOR)
# =========================================

mutable struct VotingRegressor <: Deterministic
    models::Vector{Deterministic}
    weights::Union{Nothing, Vector{Float64}}
end

function VotingRegressor(; models=Deterministic[], weights=nothing)
    normalized_weights = nothing
    if weights !== nothing
        @assert length(weights) == length(models) "Number of weights must match number of models"
        @assert all(w >= 0 for w in weights) "All weights must be non-negative"
        normalized_weights = Float64.(weights)
        s = sum(normalized_weights)
        @assert s > 0 "Sum of weights must be > 0"
        normalized_weights ./= s
    end
    VotingRegressor(models, normalized_weights)
end

function MLJModelInterface.fit(model::VotingRegressor, verbosity::Int, X, y)
    machs = [begin
        mm = machine(deepcopy(m), X, y)
        MLJ.fit!(mm, verbosity=0)
        mm
    end for m in model.models]

    fitresults = (machines = machs,)
    cache = nothing
    report = (n_models = length(model.models), weights = model.weights)

    return fitresults, cache, report
end

function MLJModelInterface.predict(model::VotingRegressor, fitresult, Xnew)
    machines = fitresult.machines
    preds    = [MLJ.predict(m, Xnew) for m in machines]

    n_samples = length(preds[1])
    n_models  = length(machines)
    weights   = model.weights === nothing ? fill(1.0 / n_models, n_models) : model.weights

    yhat = zeros(Float64, n_samples)
    for i in 1:n_samples
        s = 0.0
        for (j, p) in enumerate(preds)
            s += weights[j] * float(p[i])
        end
        yhat[i] = s
    end

    return yhat
end

MLJModelInterface.metadata_model(VotingRegressor,
    input_scitype  = Table(Continuous),
    target_scitype = AbstractVector{Continuous},
    supports_weights = false,
    load_path = "VotingRegressor"
)

# =========================================
# HELPER TO RUN REGRESSION ENSEMBLES
# =========================================

function run_regression_ensembles(
    ensemble_specs::Dict{String,<:NamedTuple},
    y_train,
    y_test;
    weights = nothing
)
    results = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()

    println("\nVOTING ENSEMBLES – REGRESSION")
    println("================================\n")

    for (ds_name, spec) in ensemble_specs
        Xtr         = spec.X_train
        Xte         = spec.X_test
        base_models = spec.models

        println(">>> Regression ensemble on dataset: $ds_name")

        model = VotingRegressor(models = base_models, weights = weights)
        mach  = machine(model, MLJ.table(Xtr), Float32.(y_train))
        MLJ.fit!(mach, rows = 1:size(Xtr, 1), verbosity = 0)

        yhat = MLJ.predict(mach, MLJ.table(Xte))

        yhat_f  = Float64.(yhat)
        ytest_f = Float64.(y_test)

        results[ds_name] = (yhat_f, ytest_f)

        # Métricas oficiales de regresión
        m = evaluate_regression(yhat_f, ytest_f)
        println("  RMSE = $(round(m.RMSE, digits=4))")
        println("  R²   = $(round(m.R2,   digits=4))\n")
    end

    return results
end

"""
    summarize_regression_ensembles(results, ordered_datasets; metric_fn)

Utility to summarize regression ensembles over multiple representations.

Arguments
---------
- `results` :: Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}
    Output of `run_regression_ensembles`, mapping
    dataset name => (predictions, ground_truth).

- `ordered_datasets` :: Vector{String}
    Order in which you want the results.

Keyword arguments
-----------------
- `metric_fn` :: Function
    A function `metric_fn(preds, truth)` returning `(value, extra...)`,
    e.g. your `evaluate_regression`.

Returns
-------
Named tuple:

- `representations` :: Vector{String}
- `metrics`         :: Vector{Float64}
"""
function summarize_regression_ensembles(
    results::Dict{String, Tuple{Vector{Float64}, Vector{Float64}}},
    ordered_datasets::Vector{String};
    metric_fn
)
    reps    = String[]
    metrics = Float64[]

    for ds_name in ordered_datasets
        preds, truth = results[ds_name]
        val, _ = metric_fn(Float32.(preds), Float32.(truth))
        push!(reps, ds_name)
        push!(metrics, val)
    end

    return (representations = reps, metrics = metrics)
end

function run_regression_stacking_ensembles(
    specs::Dict{String, <:NamedTuple},
    y_train,
    y_test;
    base_names::NTuple = (:ANN, :SVM, :DT, :KNN),
    meta_regressor = nothing,
    nfolds::Int = 5,
    rng_seed::Int = 1234,
    title_prefix::String = "Stacking Regression"
)
    Random.seed!(rng_seed)

    # Default meta-regressor if not provided
    if meta_regressor === nothing
        meta_regressor = EpsilonSVRRegressor(
            kernel = LIBSVM.Kernel.RadialBasis,
            cost   = 10.0,
            gamma  = 0.1
        )
    end

    reg_stacking_results = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()

    println("\nREGRESSION ENSEMBLES – STACKING")
    println("================================")

    for (name, spec) in specs
        Xtr         = spec.X_train
        Xte         = spec.X_test
        base_models = spec.models   # [ann, svm, dt, knn] in the order of base_names

        @assert length(base_models) == length(base_names) "Number of base models must match base_names."

        println("\n>>> Building stacking regression ensemble for: $name")

        # (ANN = m1, SVM = m2, DT = m3, KNN = m4)
        base_tuple = NamedTuple{base_names}(base_models)

        stack_model = Stack(
            metalearner = meta_regressor,
            resampling  = CV(nfolds = nfolds, shuffle = true, rng = rng_seed),
            measures    = [rms];   # internal CV metric (RMSE)
            base_tuple...
        )

        # Train meta-model
        stack_mach = machine(stack_model, MLJ.table(Xtr), Float32.(y_train))
        MLJ.fit!(stack_mach, verbosity = 0)

        # Predict on test data
        yhat  = MLJ.predict(stack_mach, MLJ.table(Xte))
        preds = Float64.(yhat)
        truth = Float64.(y_test)

        reg_stacking_results[name] = (preds, truth)

        # Optional: print RMSE per representation using your evaluate_regression
        rmse, _ = evaluate_regression(Float32.(preds), Float32.(truth))
        @printf("%s – %s: RMSE = %.4f\n", title_prefix, name, rmse)
    end

    return reg_stacking_results
end


end # module
