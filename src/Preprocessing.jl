module Preprocessing

using Statistics

export oneHotEncoding,
       vectorizerCategorialFeature,
       calculateMinMaxNormalizationParameters,
       normalizeMinMax!,
       calculateZeroMeanNormalizationParameters,
       normalizeZeroMean!,
       bin_to_multiclass

# -----------------------------------------------------------------------------
# Feature encoding utilities: convert categorical labels to/from one-hot format.
# -----------------------------------------------------------------------------
"""
    oneHotEncoding(feature, classes)

Create a one-hot encoding matrix for a categorical feature vector.

# Arguments:
- `feature::AbstractArray{<:Any,1}`: Input labels to encode.
- `classes::AbstractArray{<:Any,1}`: Ordered list of class labels defining the column mapping.

# Returns:
- `BitArray{2}`: Boolean matrix with one true per row, sized `(length(feature), length(classes))`.
"""
function oneHotEncoding(feature::AbstractArray{<:Any,1},      
        classes::AbstractArray{<:Any,1})
    # First we are going to set a line as defensive to check values
    @assert(all([in(value, classes) for value in feature]));
    
    # Second defensive statement, check the number of classes
    num_classes = length(classes);
    @assert(num_classes>1)
    
    if (num_classes==2)
        # Case with only two classes
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        #Case with more than two clases
        oneHot =  BitArray{2}(undef, length(feature), num_classes);
        for numClass = 1:num_classes
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;

"""
    oneHotEncoding(feature)

Infer the unique classes from the feature vector before one-hot encoding.

# Arguments:
- `feature::AbstractArray{<:Any,1}`: Labels that will be encoded; their unique values define the class order.

# Returns:
- `BitArray{2}`: Boolean matrix containing the inferred one-hot representation.
"""
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

"""
    vectorizerCategorialFeature(feature, classes)

Convert one-hot encoded samples back into their categorical labels.

# Arguments:
- `feature::AbstractArray{Bool,2}`: Boolean matrix where each row is a one-hot vector.
- `classes::AbstractArray{<:Any,1}`: Labels associated with each column of the one-hot matrix.

# Returns:
- `Vector{<:Any}`: Decoded labels restoring the original categorical values.
"""
function vectorizerCategorialFeature(feature::AbstractArray{Bool,2},
    classes::AbstractArray{<:Any,1})
    numClasses = size(feature, 2)
    @assert numClasses == length(classes)

    vectorizedFeature = Array{eltype(classes),1}(undef, size(feature, 1))
    for i in 1:size(feature, 1)
        idx = findfirst(feature[i, :])
        @assert !isnothing(idx) "No true value found in one-hot encoding row $i"
        vectorizedFeature[i] = classes[idx]
    end
    return vectorizedFeature
end

"""
    vectorizerCategorialFeature(feature)

Decode a one-hot matrix using sequential integers as the class catalogue.

# Arguments:
- `feature::AbstractArray{Bool,2}`: Boolean matrix representing one-hot labels.

# Returns:
- `Vector{Int}`: Class indices corresponding to the column position of each one-hot row.
"""
vectorizerCategorialFeature(feature::AbstractArray{Bool,2}) =
    vectorizerCategorialFeature(feature, collect(1:size(feature, 2)))

# -----------------------------------------------------------------------------
# Normalization helpers: compute and apply min-max or z-score scalings.
# -----------------------------------------------------------------------------
"""
    calculateMinMaxNormalizationParameters(dataset)

Compute per-column minima and maxima that can be reused for min-max scaling.

# Arguments:
- `dataset::AbstractArray{<:Real,2}`: Numeric features arranged as `(samples, features)`.

# Returns:
- `NTuple{2, AbstractArray{<:Real,2}}`: Tuple containing row vectors with column-wise minima and maxima.
"""
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end

"""
    normalizeMinMax!(dataset, normalizationParameters)

Apply min-max scaling to a dataset in-place using precomputed statistics.

# Arguments:
- `dataset::AbstractArray{<:Real,2}`: Features to scale.
- `normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}`: Tuple with precomputed minima and maxima.

# Returns:
- `AbstractArray{<:Real,2}`: The same dataset reference after in-place scaling.
"""
function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    
    if isempty(dataset)
        @warn "Dataset provided for normalization is empty (0x0 or Nx0). Skipping normalization."
        return dataset
    end

    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # eliminate any atribute that do not add information
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;

"""
    normalizeMinMax!(dataset)

Scale a dataset to `[0, 1]` by first computing the required min-max statistics.

# Arguments:
- `dataset::AbstractArray{<:Real,2}`: Features to scale.

# Returns:
- `AbstractArray{<:Real,2}`: The scaled dataset, identical to the input reference.
"""
function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
end

"""
    calculateZeroMeanNormalizationParameters(dataset)

Compute per-column statistics needed for zero-mean normalization.

# Arguments:
- `dataset::AbstractArray{<:Real,2}`: Numeric matrix whose columns will be normalized.

# Returns:
- `NTuple{2, AbstractArray{<:Real,2}}`: Tuple containing the column-wise means and standard deviations.
"""
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims=1), std(dataset, dims=1)
end

"""
    normalizeZeroMean!(dataset, normalizationParameters)

Normalize a dataset to zero mean and unit variance using precomputed statistics.

# Arguments:
- `dataset::AbstractArray{<:Real,2}`: Matrix to be normalized in-place.
- `normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}`: Tuple with column-wise means and standard deviations.

# Returns:
- `AbstractArray{<:Real,2}`: The normalized dataset reference supplied as input.
"""
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},
                        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    if isempty(dataset)
        @warn "Dataset provided for normalization is empty (0x0 or Nx0). Skipping normalization."
        return dataset
    end

    avgValues = normalizationParameters[1]
    stdValues = normalizationParameters[2]
    dataset .-= avgValues
    dataset ./= stdValues

    dataset[:, vec(stdValues .== 0)] .= 0

    return dataset
end

"""
    normalizeZeroMean!(dataset)

Normalize a dataset by computing and applying z-score statistics on the fly.

# Arguments:
- `dataset::AbstractArray{<:Real,2}`: Matrix whose columns will be standardized.

# Returns:
- `AbstractArray{<:Real,2}`: The same dataset reference after normalization.
"""
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset))
end

# -----------------------------------------------------------------------------
# Target binning helpers: discretize continuous values into ordinal classes.
# -----------------------------------------------------------------------------
"""
    bin_to_multiclass(data_vector, bin_edges)

Assign each scalar to the interval defined by successive bin edges.

# Arguments:
- `data_vector::AbstractVector{<:Real}`: Numeric values that will be discretized.
- `bin_edges::AbstractVector{<:Real}`: Monotonic edges defining the interval boundaries.

# Returns:
- `Vector{Int}`: One-based bin indices suitable for downstream multi-class labeling.
"""
function bin_to_multiclass(data_vector::AbstractVector{<:Real}, bin_edges::AbstractVector{<:Real})
    
    # 1. Error checking for valid input
    if length(bin_edges) < 2
        throw(ArgumentError("bin_edges must contain at least 2 elements."))
    end
    
    # 2. Map each data point to its corresponding bin index
    # We use broadcasting (`.`) and an anonymous function/comprehension.
    # searchsortedlast(A, x) returns the index of the last element in A <= x.
    # We adjust the index for values outside the defined edges:
    #   - For value x < bin_edges[1], searchsortedlast returns 0. We map it to 1 (first bin).
    #   - For value x >= bin_edges[end], searchsortedlast returns length(bin_edges) - 1. We map it to length(bin_edges) - 1 (last bin).
    #
    # To simplify: we treat the bins as [edge_i, edge_{i+1})
    # The bin index will be 1 to length(bin_edges) - 1.
    
    num_bins = length(bin_edges) - 1
    
    # Use a list comprehension for efficiency and clarity
    return [
        max(1, min(num_bins, searchsortedlast(bin_edges, x)))
        for x in data_vector
    ]
end

end
