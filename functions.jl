using Flux
using Flux.Losses
using Random
using Statistics
using DelimitedFiles
import Pkg;
Pkg.add("MLJ")
using MLJ;
const SVMClassifier = @load SVC pkg=LIBSVM verbosity=0
const SVMPlassifier = @load ProbabilisticSVC pkg=LIBSVM

const DTClassifier = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
const kNNClassifier = @load KNNClassifier pkg=NearestNeighborModels verbosity=0
import Pkg; Pkg.add("MLJLIBSVMInterface")
import Pkg; Pkg.add("MLJDecisionTreeInterface")

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
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

function vectorizerCategorialFeature(feature::AbstractArray{Bool,2},      
        classes::AbstractArray{<:Any,1})
    numClasses = size(feature, 2);
    @assert(numClasses==length(classes));
    
    vectorizedFeature = Array{eltype(classes),1}(undef, size(feature,1));
    for i in 1:size(feature,1)
        idx = findfirst(feature[i,:]);
        @assert !isnothing(idx) "No true value found in one-hot encoding row $i"
        vectorizedFeature[i] = classes[idx];
    end;
    return vectorizedFeature;
end;

vectorizerCategorialFeature(feature::AbstractArray{Bool,2}) = 
    vectorizerCategorialFeature(feature, collect(1:size(feature,2)));

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

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims=1), std(dataset, dims=1)
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
                        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    # CHANGED For empty dataset (validation in cross validation is empty sometimes)
    if isempty(dataset)
        @warn "Dataset provided for normalization is empty (0x0 or Nx0). Skipping normalization."
        return dataset
    end

    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    
    dataset[:, vec(stdValues.==0)] .= 0;
    
    return dataset; 
end;  

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    @assert numInputs > 0 "Number of inputs must be greater than 0"
    @assert numOutputs > 0 "Number of outputs must be greater than 0"
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;   

function trainClassANN(topology::AbstractArray{<:Int,1},  
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
            maxEpochsVal::Int=20, showText::Bool=false) 

    # Extract the inputs and targets
    (trainInputs, trainTargets) = trainingDataset
    (valInputs, valTargets) = validationDataset
    (testInputs, testTargets) = testDataset

    # This function assumes that each sample is in a row
    # we are going to check the number of samples to have same inputs and targets
    @assert(size(trainInputs,1)==size(trainTargets,1));
    @assert(size(valInputs,1)==size(valTargets,1));
    @assert(size(testInputs,1)==size(testTargets,1));
    @assert(length(transferFunctions)==length(topology));

    # We define the ANN
    ann = buildClassANN(size(trainInputs, 2), topology, size(trainTargets, 2); transferFunctions=transferFunctions)

    # Setting up the loss funtion to reduce the error
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    # These vectors are going to contain the losses and precission
    trainingLosses = Float32[];
    validationLosses = Float32[];
    testLosses = Float32[];

    # Variables for early stopping
    useValidation = !isempty(valInputs)
    useTest = !isempty(testInputs)
    bestValLoss = Inf
    bestANN = nothing
    epochsSinceImprovement = 0
    bestEpoch = 0

    # Inicialize the counter to 0
    numEpoch = 0;
    # Calcualte the loss without training
    trainingLoss = loss(ann, trainInputs', trainTargets');
    # Store this one for checking the evolution.
    push!(trainingLosses, trainingLoss);

    if useValidation
        valLoss = loss(ann, valInputs', valTargets')
        push!(validationLosses, valLoss)
        bestValLoss = valLoss
        bestANN = deepcopy(ann)
    end

    if useTest
        testLoss = loss(ann, testInputs', testTargets')
        push!(testLosses, testLoss)
    end

    # Give feedback
    if showText
        println("Epoch 0: Training loss: ", trainingLoss);
        if useValidation
            println("           Validation loss: ", valLoss)
        end
        if useTest
            println("           Test loss: ", testLoss)
        end
    end

    # Define the optimazer for the network
    opt_state = Flux.setup(Adam(learningRate), ann);

    # Start the training until it reaches one of the stop critteria
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && 
            (!useValidation || (epochsSinceImprovement<maxEpochsVal))

        # For each epoch, we habve to train and consequently traspose the pattern to have then in columns
        Flux.train!(loss, ann, [(trainInputs', trainTargets')], opt_state);

        numEpoch += 1;
        # calculate the loss for this epoch
        trainingLoss = loss(ann, trainInputs', trainTargets');
        # store it
        push!(trainingLosses, trainingLoss);

        # Calculate validation loss
        if useValidation
            valLoss = loss(ann, valInputs', valTargets')
            push!(validationLosses, valLoss)
            
            if valLoss < bestValLoss
                bestValLoss = valLoss
                bestANN = deepcopy(ann)
                epochsSinceImprovement = 0
                bestEpoch = numEpoch
            else
                epochsSinceImprovement += 1
            end
        end
        
        # Calculate test loss
        if useTest
            testLoss = loss(ann, testInputs', testTargets')
            push!(testLosses, testLoss)
        end

        # Give feedback
        if showText
            println("Epoch ", numEpoch, ": Training loss: ", trainingLoss);
            if useValidation
                println("           Validation loss: ", valLoss)
            end
            if useTest
                println("           Test loss: ", testLoss)
            end
        end

        # Check early stopping condition
        if useValidation && (epochsSinceImprovement >= maxEpochsVal)
            if showText
                println("Early stopping at epoch ", numEpoch, ". Best validation loss: ", bestValLoss, " at epoch ", bestEpoch);
            end
            break
        end

    end;

    # Return the best ANN if validation was used, otherwise the last one
    if useValidation && showText
        println("Best validation loss: ", bestValLoss, " at epoch ", bestEpoch);
    end
    finalANN = useValidation ? bestANN : ann
    
    # Return the ANN and the losses
    return (finalANN, trainingLosses, validationLosses, testLosses)
end

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    mean(outputs.==targets);
end;
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;
function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
                threshold::Real=0.5)
    accuracy(outputs.>=threshold, targets);
end;
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
                threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets) "Outputs and targets must have the same length"
    @assert length(outputs) > 0 "Outputs and targets must be non-empty"

    # Calculate confusion matrix values
    TP = sum(outputs .& targets)
    TN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)

    # All are true negatives (TN) or true positives (TP)
    if ((TP == 0) && (FN == 0) && (FP == 0)) || ( (TN == 0) && (FN == 0) && (FP == 0))
        sensitivity = 1.0
        PPV = 1.0
        specificity = 1.0
        NPV = 1.0
    else
        sensitivity = TP + FN == 0 ? 0.0 : TP / (TP + FN)
        specificity = TN + FP == 0 ? 0.0 : TN / (TN + FP)
        PPV = TP + FP == 0 ? 0.0 : TP / (TP + FP)
        NPV = TN + FN == 0 ? 0.0 : TN / (TN + FN)
    end

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    errorRate = (FP + FN) / (TP + TN + FP + FN)

    if (sensitivity == 0.0) && (PPV == 0.0) # If both are 0, F-score is 0
        Fscore = 0.0
    elseif (TP == 0) && (FN == 0) && (FP == 0) # If all are TN, F-score is 1
        Fscore = 1.0
    else
        Fscore = (2 * TP) / (2 * TP + FP + FN)
    end

    confMatrix = [TN FP; FN TP]

    return (accuracy, errorRate, sensitivity, specificity, PPV, NPV, Fscore, confMatrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Convert outputs to boolean based on the threshold
    boolOutputs = outputs .>= threshold

    return confusionMatrix(boolOutputs, targets)
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert size(outputs) == size(targets)

    numClasses = size(outputs, 2)
    if numClasses == 1
        return confusionMatrix(outputs[:, 1], targets[:, 1])
    end

    @assert numClasses > 2 "Outputs must have more than 2 columns for multiclass classification."

    confMatrix = zeros(Int, numClasses, numClasses)
    for i in 1:numClasses
        for j in 1:numClasses
            confMatrix[i, j] = sum(targets[:, i] .& outputs[:, j])
        end
    end

    sensitivity = zeros(Float64, numClasses)
    specificity = zeros(Float64, numClasses)
    PPV = zeros(Float64, numClasses)
    NPV = zeros(Float64, numClasses)
    Fscore = zeros(Float64, numClasses)

    for i in 1:numClasses
        metrics = confusionMatrix(outputs[:, i], targets[:, i])
        sensitivity[i] = metrics[3]
        specificity[i] = metrics[4]
        PPV[i] = metrics[5]
        NPV[i] = metrics[6]
        Fscore[i] = metrics[7]
    end

    acc = accuracy(outputs, targets)
    errorRate = 1.0 - acc

    if weighted
        classCounts = vec(sum(targets, dims=1))
        weights = classCounts ./ sum(classCounts)
        
        sensitivity_avg = sum(weights .* sensitivity)
        specificity_avg = sum(weights .* specificity)
        PPV_avg = sum(weights .* PPV)
        NPV_avg = sum(weights .* NPV)
        Fscore_avg = sum(weights .* Fscore)
    else
        sensitivity_avg = sum(sensitivity) / numClasses
        specificity_avg = sum(specificity) / numClasses
        PPV_avg = sum(PPV) / numClasses
        NPV_avg = sum(NPV) / numClasses
        Fscore_avg = sum(Fscore) / numClasses
    end

    return (acc, errorRate, sensitivity_avg, specificity_avg, 
        PPV_avg, NPV_avg, Fscore_avg, confMatrix)
end
function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
                        threshold::Real=0.5) 
   numOutputs = size(outputs, 2);
    # @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Look for the maximum value using the findmax funtion
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Set up then boolean matrix to everything false while max values aretrue.
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Defensive check if all patterns are in a single class
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2},targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    classifiedOutputs = classifyOutputs(outputs; threshold=threshold)
    return confusionMatrix(classifiedOutputs, targets; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert length(outputs) == length(targets)
    numClasses = length(classes)
    @assert numClasses > 1 "There must be at least two classes."
    
    if numClasses == 2
        boolOutputs = outputs .== classes[2]
        boolTargets = targets .== classes[2]
        return confusionMatrix(boolOutputs, boolTargets)
    else
        oneHotOutputs = oneHotEncoding(outputs, classes)
        oneHotTargets = oneHotEncoding(targets, classes)
        return confusionMatrix(oneHotOutputs, oneHotTargets; weighted=weighted)
    end
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    uniqueClasses = unique(vcat(outputs, targets))
    return confusionMatrix(outputs, targets, uniqueClasses; weighted=weighted)
end
function printConfusionMatrix(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1})
    accuracy, errorRate, sensitivity, specificity, PPV, NPV, Fscore, confMatrix = confusionMatrix(outputs, targets)
    
    println("Confusion Matrix:")
    println("   PN   PP")
    println("RN ", confMatrix[1,1], "    ", confMatrix[1,2])
    println("RP ", confMatrix[2,1], "    ", confMatrix[2,2])
    println("\nMetrics:")
    println("  Accuracy: ", accuracy)
    println("  Error Rate: ", errorRate)
    println("  Sensitivity (Recall): ", sensitivity)
    println("  Specificity: ", specificity)
    println("  Positive Predictive Value (Precision): ", PPV)
    println("  Negative Predictive Value: ", NPV)
    println("  F-score: ", Fscore)
end
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    metrics = confusionMatrix(outputs, targets; weighted=weighted)
    strategy = weighted ? "Weighted" : "Macro"
    println("$strategy Metrics:")
    println("Accuracy: $(metrics[1])")
    println("Error Rate: $(metrics[2])")
    println("Sensitivity (Recall): $(metrics[3])")
    println("Specificity: $(metrics[4])")
    println("Positive Predictive Value (Precision): $(metrics[5])")
    println("Negative Predictive Value: $(metrics[6])")
    println("F-score: $(metrics[7])")
    println("Confusion Matrix:\n$(metrics[8])")
end
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
function printConfusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    boolOutputs = outputs .>= threshold
    printConfusionMatrix(boolOutputs, targets)
end
function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Use one-hot encoding for both
    oneHotOutputs = oneHotEncoding(outputs, classes)
    oneHotTargets = oneHotEncoding(targets, classes)
    
    # oneHotEncoding returns (N,1) for 2 classes.
    # The printConfusionMatrix(Bool,2) function will correctly
    # dispatch this to the printConfusionMatrix(Bool,1) function.
    printConfusionMatrix(oneHotOutputs, oneHotTargets; weighted=weighted)
end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    uniqueClasses = unique(vcat(outputs, targets))
    sort!(uniqueClasses)
    printConfusionMatrix(outputs, targets, uniqueClasses; weighted=weighted)
end

function loadIrisDataset(;iris_path::String="work/iris.data", testSplit::Real=0.2, valSplit::Real=0.2, oneHot::Bool=true, numericTargets::Bool=false)
    dataset = readdlm(iris_path, ',')

    inputs = Float32.(dataset[:,1:4]);
    if oneHot
        targets = oneHotEncoding(dataset[:,5]);
    else
        targets_ = dataset[:,5];
        if numericTargets
            uniqueClasses = unique(targets_);
            # convert to numeric
            targets = [findfirst(==(targets_[i]), uniqueClasses) for i in 1:length(targets_)];
        else
            targets = targets_;
        end
    end

    # Split the dataset into training, validation, and test sets
    N = size(inputs, 1)
    trainIdx, valIdx, testIdx = holdOut(N, valSplit, testSplit)

    trainInputs = inputs[trainIdx, :]
    trainTargets = targets[trainIdx, :]
    valInputs = inputs[valIdx, :]
    valTargets = targets[valIdx, :]
    testInputs = inputs[testIdx, :]
    testTargets = targets[testIdx, :]
    # Return named tuples
    if !oneHot
        to_return = (
            train=(trainInputs, vec(trainTargets)),
            val=(valInputs, vec(valTargets)),
            test=(testInputs, vec(testTargets))
        )
        if numericTargets
            to_return = (
                to_return...,
                classesToLabels=unique(dataset[:,5]))
        end
        return to_return
    end
    return (
        train=(trainInputs, trainTargets),
        val=(valInputs, valTargets),
        test=(testInputs, testTargets))
end
# Unit 5 
function crossvalidation(N::Int64, k::Int64)
    @assert k ≤ N "k must be less than or equal to N"
    @assert k > 1 "k must be greater than 1"

    sorted = collect(1:k)
    repeated = repeat(sorted, ceil(Int, N/k))
    idx = repeated[1:N]
    shuffle!(idx)
    return idx
end
function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    idx = zeros(Int, length(targets))

    pos_idx = findall(targets) # positive targets
    idx[pos_idx] = crossvalidation(length(pos_idx), k)

    neg_idx = findall(.!targets) # negative targets
    idx[neg_idx] = crossvalidation(length(neg_idx), k)

    return idx
end
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    println("HOLA")

    N, C = size(targets)
    idx = zeros(Int, N)

    for i in 1:C
        idx[targets[:, i]]  = crossvalidation(sum(targets[:, i]), k)
    end

    return idx
end
function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    return crossvalidation(oneHotEncoding(targets), k)
end
function ANNCrossValidation(
    topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20,
    showText::Bool=false
)
    inputs, targets = dataset

    # Extract class labels
    classes = unique(targets)
    num_classes = length(classes)

    # OneHotEncoding
    one_hot_targets = oneHotEncoding(targets, classes)

    # Determine the number of folds
    num_folds = maximum(crossValidationIndices)

    # Create one vector per metric to store fold results
    acc_folds = zeros(Float32, num_folds)
    err_folds = zeros(Float32, num_folds)
    sens_folds = zeros(Float32, num_folds)
    spec_folds = zeros(Float32, num_folds)
    ppv_folds = zeros(Float32, num_folds)
    npv_folds = zeros(Float32, num_folds)
    f1_folds = zeros(Float32, num_folds)

    # Initialize a confusion matrix accumulator
    accumulator_confusion = zeros(Float32, num_classes, num_classes)

    for fold in 1:num_folds
        # Select train and test indices
        train_idx = crossValidationIndices .!= fold
        test_idx = crossValidationIndices .== fold

        input_train = inputs[train_idx, :]
        target_train = one_hot_targets[train_idx, :]
        input_test = inputs[test_idx, :]
        target_test = one_hot_targets[test_idx, :]

        # Métricas por ejecución
        acc_exec = zeros(Float32, numExecutions)
        err_exec = zeros(Float32, numExecutions)
        sens_exec = zeros(Float32, numExecutions)
        spec_exec = zeros(Float32, numExecutions)
        ppv_exec = zeros(Float32, numExecutions)
        npv_exec = zeros(Float32, numExecutions)
        f1_exec = zeros(Float32, numExecutions)

        # Create a 3D array to store test confusion matrices from each execution
        confusion_exec = zeros(Float32, num_classes, num_classes, numExecutions)
        if validationRatio > 0
            valRatioAdjusted = validationRatio * (size(inputs, 1) / size(input_train, 1))
            train_idx, val_idx = holdOut(size(input_train, 1), valRatioAdjusted)
    
            input_train_exec = input_train[train_idx, :]
            target_train_exec = target_train[train_idx, :]
            input_val_exec = input_train[val_idx, :]
            target_val_exec = target_train[val_idx, :]
        
        else
            input_train_exec = input_train
            target_train_exec = target_train
            input_val_exec = Array{eltype(input_train),2}(undef,0,0)
            target_val_exec = falses(0, 0)
        end

        input_test_exec = input_test
        target_test_exec = target_test
        parameters = calculateZeroMeanNormalizationParameters(input_train_exec)
        normalizeZeroMean!(input_train_exec ,parameters)
        normalizeZeroMean!(input_val_exec ,parameters)
        normalizeZeroMean!(input_test_exec ,parameters)
        for e in 1:numExecutions
            
            # Train the network
            ann, _, _, _ = trainClassANN(
                topology,
                (input_train_exec, target_train_exec);
                validationDataset=(input_val_exec, target_val_exec),
                testDataset=(input_test_exec, target_test_exec),
                transferFunctions=transferFunctions,
                maxEpochs=maxEpochs,
                minLoss=minLoss,
                learningRate=learningRate,
                maxEpochsVal=maxEpochsVal,
                showText=showText
            )
            
            # Evaluación
            raw_outputs = ann(input_test_exec')
            outputs_test = Array(raw_outputs')

            acc, err, sens, spec, ppv, npv, f1, conf = confusionMatrix(outputs_test, target_test_exec; weighted=true)
            
            acc_exec[e]  = acc
            err_exec[e]  = err
            sens_exec[e] = sens
            spec_exec[e] = spec
            ppv_exec[e]  = ppv
            npv_exec[e]  = npv
            f1_exec[e]   = f1
            confusion_exec[:, :, e] = conf
        end

        # Promedio por fold
        acc_folds[fold] = mean(acc_exec)
        err_folds[fold] = mean(err_exec)
        sens_folds[fold] = mean(sens_exec)
        spec_folds[fold] = mean(spec_exec)
        ppv_folds[fold] = mean(ppv_exec)
        npv_folds[fold] = mean(npv_exec)
        f1_folds[fold] = mean(f1_exec)

        fold_conf_mean = dropdims(mean(confusion_exec, dims=3), dims=3)
        accumulator_confusion .+= fold_conf_mean
    end

    # Promedio y desviación final
    results = (
        accuracy = (mean(acc_folds), std(acc_folds)),
        error = (mean(err_folds), std(err_folds)),
        sensitivity = (mean(sens_folds), std(sens_folds)),
        specificity = (mean(spec_folds), std(spec_folds)),
        PPV = (mean(ppv_folds), std(ppv_folds)),
        NPV = (mean(npv_folds), std(npv_folds)),
        F1 = (mean(f1_folds), std(f1_folds)),
        conf_matrix=accumulator_confusion
    )

    return results
end

getparam(hp::Dict, key, default=nothing) = get(hp, key, get(hp, string(key), default))

function createModel(modelType::Symbol, modelHyperparameters::Dict)
    """Create an MLJ model of type `modelType` with the given `modelHyperparameters`.
    # Arguments:
    - `modelType::Symbol`: The type of model to create. Supported types are `:SVC`, `:DecisionTreeClassifier`, `:KNeighborsClassifier`, `:SVPC`, and `:ANN`.
    - `modelHyperparameters::Dict`: A dictionary containing the hyperparameters for the model.
    # Returns:
    - `AbstractClassifier`: An instance of the specified model type with the provided hyperparameters.
    # Raises:
    - `AssertionError`: If required hyperparameters are missing for the specified model type.
    - `Error`: If the specified `modelType` is not supported.
    """
    if modelType == :SVC || modelType == :SVPC
        # Map the string kernel name from `modelHyperparameters` to the correct `LIBSVM.Kernel` constant
        kernel_name = getparam(modelHyperparameters, :kernel, nothing)
        cost = getparam(modelHyperparameters, :C, nothing)
        @assert kernel_name != nothing "For SVM model, 'kernel' must be specified in modelHyperparameters"
        @assert cost != nothing "For SVM model, 'cost' must be specified in modelHyperparameters"
        kernel_name = uppercase(string(kernel_name))
        kernel_map = Dict(
            "LINEAR" => LIBSVM.Kernel.Linear,
            "POLY" => LIBSVM.Kernel.Polynomial,
            "RBF" => LIBSVM.Kernel.RadialBasis,
            "SIGMOID" => LIBSVM.Kernel.Sigmoid
        )
        if !haskey(kernel_map, kernel_name)
            @warn "Unknown kernel type '$kernel_name'. Using RBF as default."
        end
        kernel_symbol = get(kernel_map, kernel_name, LIBSVM.Kernel.RadialBasis)

        # Extract optional hyperparameters
        gamma_val = getparam(modelHyperparameters, :gamma, nothing)
        degree_val = getparam(modelHyperparameters, :degree, nothing)
        coef0_val = getparam(modelHyperparameters, :coef0, nothing)
        
        input_dict = Dict(
            :kernel => kernel_symbol,
            :cost => Float64(cost)
        )
        for (param, value) in [(:gamma, gamma_val), (:degree, degree_val), (:coef0, coef0_val)]
            if value !== nothing
                input_dict[param] = value
            end
        end
        if modelType == :SVC
            model = SVMClassifier(; input_dict...)
        else
            model = SVMPlassifier(; input_dict...)
        end

    elseif modelType == :DecisionTreeClassifier
        max_depth = getparam(modelHyperparameters, :max_depth, nothing)
        rng = getparam(modelHyperparameters, :rng, Random.MersenneTwister(1))
        @assert max_depth != nothing "For Decision Tree model, 'max_depth' must be specified in modelHyperparameters"

        model = DTClassifier(max_depth=max_depth, rng=rng)
    elseif modelType == :KNeighborsClassifier
        k_val = getparam(modelHyperparameters, :K, nothing)
        @assert k_val != nothing "For kNN model, 'K' must be specified in modelHyperparameters"
        model = kNNClassifier(K=k_val)
    elseif modelType == :ANN
        numInputs = getparam(modelHyperparameters, :numInputs, nothing)
        @assert numInputs !== nothing "In ANN numInputs must be defined"
        topology = getparam(modelHyperparameters, :topology, nothing)
        @assert topology !== nothing "In ANN topology must be defined"
        numOutputs = getparam(modelHyperparameters, :numOutputs, nothing)
        @assert numOutputs !== nothing "In ANN numOutputs must be defined"
        transferFunctions = getparam(modelHyperparameters, :transferFunctions, fill(σ, length(topology)))
        model = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)
    else
        error("The model type $(modelType) is not allowed")
    end
    return model
end

function modelCrossValidation(
    modelType::Symbol, modelHyperparameters::Dict,
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1})
    """Perform cross-validation for a given model type and hyperparameters on the provided dataset.
    # Arguments:
    - `modelType::Symbol`: The type of model to use for cross-validation. Supported types are `:ANN`, `:SVC`, `:DecisionTreeClassifier`, and `:KNeighborsClassifier`.
    - `modelHyperparameters::Dict`: A dictionary containing the hyperparameters for the model.
    - `dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}`: A tuple containing the input features and target labels.
    - `crossValidationIndices::Array{Int64,1}`: An array indicating the fold assignment for each sample in the dataset.
    # Returns:
    - `NamedTuple`: A named tuple containing the mean and standard deviation of various performance metrics across the folds, as well as the average confusion matrix.
    """
    
    if modelType == :ANN
        topology = getparam(modelHyperparameters, :topology, nothing)
        @assert topology != nothing "For ANN model, 'topology' must be specified in modelHyperparameters"
        
        # Remove topology from the dict to avoid passing it twice
        filtered_params = filter(p -> p.first != :topology && p.first != "topology", modelHyperparameters)
        
        return ANNCrossValidation(
            topology,
            dataset,
            crossValidationIndices;
            filtered_params...
        )
    end
    
    inputs, targets = dataset
    targets = string.(targets)
    # Extract class labels
    classes = unique(targets)
    num_classes = length(classes)

    # Determine the number of folds
    num_folds = maximum(crossValidationIndices)

    acc_folds = zeros(Float32, num_folds)
    err_folds = zeros(Float32, num_folds)
    sens_folds = zeros(Float32, num_folds)
    spec_folds = zeros(Float32, num_folds)
    ppv_folds = zeros(Float32, num_folds)
    npv_folds = zeros(Float32, num_folds)
    f1_folds = zeros(Float32, num_folds)
    accumulator_confusion = zeros(Float32, num_classes, num_classes, num_folds)

    for fold in 1:num_folds
        # Select train and test indices
        train_idx = crossValidationIndices .!= fold
        test_idx = crossValidationIndices .== fold

        input_train = inputs[train_idx, :]
        target_train = targets[train_idx]
        input_test = inputs[test_idx, :]
        target_test = targets[test_idx]


        parameters = calculateZeroMeanNormalizationParameters(input_train)
        normalizeZeroMean!(input_train ,parameters)
        normalizeZeroMean!(input_test ,parameters)

        # Train the network
        model = createModel(modelType, modelHyperparameters)
        mach = machine(model, MLJ.table(input_train), categorical(target_train))
        MLJ.fit!(mach, verbosity=0)

        # Evaluación
        outputs_test = MLJ.predict(mach, MLJ.table(input_test))
        
        if modelType == :DecisionTreeClassifier || modelType == :KNeighborsClassifier
            outputs_test = mode.(outputs_test)
        end

        # Convert CategoricalArray to string vector for confusionMatrix
        outputs_test = string.(outputs_test)
        acc, err, sens, spec, ppv, npv, f1, conf = confusionMatrix(outputs_test, target_test, classes; weighted=false)

        
        acc_folds[fold]  = acc
        err_folds[fold]  = err
        sens_folds[fold] = sens
        spec_folds[fold] = spec
        ppv_folds[fold]  = ppv
        npv_folds[fold]  = npv
        f1_folds[fold]   = f1
        accumulator_confusion[:, :, fold] = conf
    end

    results = (
        accuracy = (mean(acc_folds), std(acc_folds)),
        error = (mean(err_folds), std(err_folds)),
        sensitivity = (mean(sens_folds), std(sens_folds)),
        specificity = (mean(spec_folds), std(spec_folds)),
        PPV = (mean(ppv_folds), std(ppv_folds)),
        NPV = (mean(npv_folds), std(npv_folds)),
        F1 = (mean(f1_folds), std(f1_folds)),
        conf_matrix=dropdims(mean(accumulator_confusion, dims=3), dims=3)
    ) 
    return results
end