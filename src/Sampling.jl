module Sampling

using Random
using DelimitedFiles: readdlm
using ..Preprocessing: oneHotEncoding

export holdOut,
       crossvalidation,
       loadIrisDataset

# -----------------------------------------------------------------------------
# Hold-out splitting routines: random train/val/test partition helpers.
# -----------------------------------------------------------------------------
"""
    holdOut(N, P)

Randomly divide indices into training and test partitions.

# Arguments:
- `N::Int`: Total number of samples to split.
- `P::Real`: Fraction of samples assigned to the test split.

# Returns:
- `Tuple{Vector{Int}, Vector{Int}}`: Train indices followed by test indices.
"""
function holdOut(N::Int, P::Real)
    @assert 0.0 <= P <= 1.0 "P must be between 0 and 1"
    @assert N > 1 "N must be greater than 1"
    # Generar una permutación aleatoria de índices
    indices = randperm(N)
    
    # Calcular cuántos patrones van al test
    nTest = Int(round(N * P))
    
    # Dividir los índices
    testIndices = indices[1:nTest]
    trainIndices = indices[nTest+1:end]

    return (trainIndices, testIndices)
end

"""
    holdOut(N, Pval, Ptest)

Create train, validation, and test partitions while preserving proportions.

# Arguments:
- `N::Int`: Number of available samples.
- `Pval::Real`: Fraction of the original dataset dedicated to validation.
- `Ptest::Real`: Fraction of the original dataset dedicated to testing.

# Returns:
- `Tuple{Vector{Int}, Vector{Int}, Vector{Int}}`: Train, validation, and test index arrays.
"""
function holdOut(N::Int, Pval::Real, Ptest::Real)
    # Separar el conjunto de test usando la anterior función
    trainValIndices, testIndices = holdOut(N, Ptest)
    
    # Ajustar porcentaje de validación respecto al tamaño restante
    NRemaining = length(trainValIndices)
    # PvalAdjusted = Pval / (1 - Ptest) # viene de hacer Pval * N / ((1 - Ptest) * N) para sacar el porcentaje de ese conjunto nuevo
    PvalAdjusted = Pval * N / NRemaining # Alternativa equivalente
    trainIndices, valIndices = holdOut(NRemaining, PvalAdjusted)

    # Mapear índices locales a originales
    valIndices = trainValIndices[valIndices]
    trainIndices = trainValIndices[trainIndices]

    return (trainIndices, valIndices, testIndices)
end

# -----------------------------------------------------------------------------
# Stratified cross-validation helpers: maintain label balance per fold.
# -----------------------------------------------------------------------------
"""
    crossvalidation(N, k)

Distribute samples uniformly at random across `k` folds.

# Arguments:
- `N::Int64`: Total number of samples to assign.
- `k::Int64`: Desired number of folds (`k > 1`).

# Returns:
- `Vector{Int}`: Fold identifier for each sample.
"""
function crossvalidation(N::Int64, k::Int64)
    @assert k ≤ N "k must be less than or equal to N"
    @assert k > 1 "k must be greater than 1"

    sorted = collect(1:k)
    repeated = repeat(sorted, ceil(Int, N / k))
    idx = repeated[1:N]
    shuffle!(idx)
    return idx
end

"""
    crossvalidation(targets::AbstractArray{Bool,1}, k)

Generate stratified folds for binary targets.

# Arguments:
- `targets::AbstractArray{Bool,1}`: Boolean labels with true/false outcomes.
- `k::Int64`: Number of folds to create.

# Returns:
- `Vector{Int}`: Fold identifier per sample with balanced class representation.
"""
function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    idx = zeros(Int, length(targets))

    pos_idx = findall(targets) # positive targets
    idx[pos_idx] = crossvalidation(length(pos_idx), k)

    neg_idx = findall(.!targets) # negative targets
    idx[neg_idx] = crossvalidation(length(neg_idx), k)

    return idx
end

"""
    crossvalidation(targets::AbstractArray{Bool,2}, k)

Stratify folds for multi-class one-hot targets while balancing each class.

# Arguments:
- `targets::AbstractArray{Bool,2}`: One-hot encoded labels `(samples, classes)`.
- `k::Int64`: Number of folds to produce.

# Returns:
- `Vector{Int}`: Fold identifiers preserving class proportions.
"""
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    N, C = size(targets)
    idx = zeros(Int, N)

    for i in 1:C
        idx[targets[:, i]] = crossvalidation(sum(targets[:, i]), k)
    end

    return idx
end

"""
    crossvalidation(targets::AbstractArray{<:Any,1}, k)

Wrap categorical labels into a stratified cross-validation assignment.

# Arguments:
- `targets::AbstractArray{<:Any,1}`: Arbitrary labels that will be one-hot encoded internally.
- `k::Int64`: Number of folds to create.

# Returns:
- `Vector{Int}`: Fold identifiers respecting class balance.
"""
function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    return crossvalidation(oneHotEncoding(targets), k)
end

end
