module Output

using Plots
using ..Preprocessing: oneHotEncoding
using ..Training: confusionMatrix, classifyOutputs

export printConfusionMatrix,
       draw_results

"""
    printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

Print the binary confusion matrix and derived metrics.

# Arguments:
- `outputs::AbstractArray{Bool,1}`: Boolean predictions for each sample.
- `targets::AbstractArray{Bool,1}`: Ground-truth boolean labels of identical length.

# Returns:
- `Nothing`: Metrics are displayed to stdout.
"""
function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    accuracy_val, errorRate, sensitivity, specificity, PPV, NPV, Fscore, confMatrix = confusionMatrix(outputs, targets)

    println("Confusion Matrix:")
    println("   PN   PP")
    println("RN ", confMatrix[1, 1], "    ", confMatrix[1, 2])
    println("RP ", confMatrix[2, 1], "    ", confMatrix[2, 2])
    println("\nMetrics:")
    println("  Accuracy: ", accuracy_val)
    println("  Error Rate: ", errorRate)
    println("  Sensitivity (Recall): ", sensitivity)
    println("  Specificity: ", specificity)
    println("  Positive Predictive Value (Precision): ", PPV)
    println("  Negative Predictive Value (Precision): ", NPV)
    println("  F-score: ", Fscore)
end

"""
    printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted=true)

Print multi-class confusion metrics for one-hot predictions and targets.

# Arguments:
- `outputs::AbstractArray{Bool,2}`: One-hot encoded predictions.
- `targets::AbstractArray{Bool,2}`: One-hot encoded targets.
- `weighted::Bool`: When true, metrics are averaged using class weights; otherwise macro averages are used.

# Returns:
- `Nothing`: Metrics and confusion matrix are printed.
"""
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    metrics = confusionMatrix(outputs, targets; weighted=weighted)
    strategy = weighted ? "Weighted" : "Macro"
    println("$strategy Metrics:")
    println("Accuracy: $(metrics[1])")
    println("Error Rate: $(metrics[2])")
    println("Sensitivity (Recall): $(metrics[3])")
    println("Specificity: $(metrics[4])")
    println("Positive Predictive Value (Precision): $(metrics[5])")
    println("Negative Predictive Value (Precision): $(metrics[6])")
    println("F-score: $(metrics[7])")
    println("Confusion Matrix:\n$(metrics[8])")
end

"""
    printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold=0.5, weighted=true)

Classify numeric predictions (if needed) before printing multi-class metrics.

# Arguments:
- `outputs::AbstractArray{<:Real,2}`: Continuous predictions per class.
- `targets::AbstractArray{Bool,2}`: One-hot encoded targets.
- `threshold::Real`: Threshold for binary problems when `size(targets, 2) == 1`.
- `weighted::Bool`: Metric aggregation strategy forwarded to the boolean overload.

# Returns:
- `Nothing`: Results are printed to stdout.
"""
function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)

    @assert(all(size(outputs).==size(targets)));

    if (size(targets,2)==1)
        # This is a binary problem.
        # Call the print function for binary numeric (Real,1) inputs.
        printConfusionMatrix(outputs[:,1], targets[:,1]; threshold=threshold)
    else
        # This is a multi-class problem.
        # Classify first, then call the print function for multi-class boolean (Bool,2) inputs.
        classifiedOutputs = classifyOutputs(outputs; threshold=threshold) # threshold is ignored by findmax path
        printConfusionMatrix(classifiedOutputs, targets; weighted=weighted)
    end
end

"""
    printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold=0.5)

Threshold numeric scores before printing binary confusion metrics.

# Arguments:
- `outputs::AbstractArray{<:Real,1}`: Continuous predictions per sample.
- `targets::AbstractArray{Bool,1}`: Ground-truth boolean labels.
- `threshold::Real`: Decision boundary used to convert scores into booleans.

# Returns:
- `Nothing`: Delegates to the boolean overload, printing metrics.
"""
function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    boolOutputs = outputs .>= threshold
    printConfusionMatrix(boolOutputs, targets)
end

"""
    printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted=true)

Convert categorical labels into one-hot form using an explicit class order before printing metrics.

# Arguments:
- `outputs::AbstractArray{<:Any,1}`: Predicted categorical labels.
- `targets::AbstractArray{<:Any,1}`: Ground-truth categorical labels.
- `classes::AbstractArray{<:Any,1}`: Ordered class list used for one-hot encoding.
- `weighted::Bool`: Whether to compute weighted or macro averages.

# Returns:
- `Nothing`: Prints the confusion metrics derived from the encoded labels.
"""
function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Use one-hot encoding for both
    oneHotOutputs = oneHotEncoding(outputs, classes)
    oneHotTargets = oneHotEncoding(targets, classes)
    
    # oneHotEncoding returns (N,1) for 2 classes.
    # The printConfusionMatrix(Bool,2) function will correctly
    # dispatch this to the printConfusionMatrix(Bool,1) function.
    printConfusionMatrix(oneHotOutputs, oneHotTargets; weighted=weighted)
end

"""
    printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted=true)

Infer the class catalogue automatically before printing confusion metrics.

# Arguments:
- `outputs::AbstractArray{<:Any,1}`: Predicted categorical labels.
- `targets::AbstractArray{<:Any,1}`: Ground-truth categorical labels.
- `weighted::Bool`: Metric aggregation mode.

# Returns:
- `Nothing`: Metrics are printed after delegating to the explicit-class overload.
"""
function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    uniqueClasses = unique(vcat(outputs, targets))
    sort!(uniqueClasses)
    printConfusionMatrix(outputs, targets, uniqueClasses; weighted=weighted)
end

"""
    draw_results(x, y; colors, target_names=nothing)

Scatter plot the first two features, coloring samples by their class assignments.

# Arguments:
- `x`: Feature matrix whose first two columns will be plotted.
- `y`: One-hot encoded labels indicating class membership.
- `colors`: Collection of color identifiers, one per class.
- `target_names`: Optional labels for the legend entries.

# Returns:
- `Nothing`: The plot is rendered via Plots.jl side effects.
"""
function draw_results(x, y; colors, target_names=nothing)
    num_classes = length(unique(colors))

    if !isnothing(target_names)
        @assert num_classes == length(target_names)
        label = target_names
    else
        label = [string(i) for i in 1:num_classes]
    end

    fig = plot()
    if num_classes == 2
        possitive_class = y[:, 1] .== 1
        scatter!(fig, x[possitive_class, 1], x[possitive_class, 2], markercolor=colors[1], label=label[1])
        scatter!(fig, x[.!possitive_class, 1], x[.!possitive_class, 2], markercolor=colors[2], label=label[2])
    else
        for i in 1:num_classes
            index_class = y[:, i] .== 1
            scatter!(fig, x[index_class, 1], x[index_class, 2], markercolor=colors[i], label=label[i])
        end
    end
end

end
