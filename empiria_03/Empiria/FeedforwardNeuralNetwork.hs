-- Version: 0.3.4 (2018 November 27)

module Empiria.FeedforwardNeuralNetwork (createNetwork,outputs,train) where

import Numeric.LinearAlgebra -- requires the hmatrix package
                ( R
                , Vector
                , Matrix
                , scalar
                , vector
                , toList
                , dropColumns
                , outer
                , (#>)
                , (<.>)
                , tr
                )

-- The learning rate 
learningRate :: Double
learningRate = 1.2

-- The learning error tolerance
tolerance :: Double
tolerance = 0.0001

data InputLayer = InputLayer 
                { ilActivationFunction :: Double -> Double
                , ilBias :: Double
                }

data HiddenLayer = HiddenLayerDiff -- for neurons with differentiable activation function
                { hlActivationFunction :: Double -> Double
                , hlBias :: Double
                , hlWeights :: Matrix Double
                , hlActivationFunctionDerivative :: Double -> Double
                }
                | HiddenLayerNotDiff -- for neurons with non-differentiable activation function
                { hlActivationFunction :: Double -> Double
                , hlBias :: Double
                , hlWeights :: Matrix Double
                }

data OutputLayer = OutputLayerDiff -- for neurons with differentiable activation function
                { olActivationFunction :: Double -> Double
                , olWeights :: Matrix Double
                , olActivationFunctionDerivative :: Double -> Double
                }
                | OutputLayerNotDiff -- for neurons with non-differentiable activation function
                { olActivationFunction :: Double -> Double
                , olWeights :: Matrix Double
                }

data Network = Network { inputLayer :: InputLayer 
                       , hiddenLayers :: [HiddenLayer]
                       , outputLayer :: OutputLayer
                       }

createNetwork :: [Matrix Double] -> Network
createNetwork networkBasis = Network il hls ol where
                             olw = last networkBasis
                             hlsw = init networkBasis
                             il = createInputLayer
                             hls = map createHiddenLayer hlsw
                             ol = createOutputLayer olw

instance Outputs Network where
         outputs net inputSignals = outputsOfOutputLayer ns where
                                    ns = networkSignals net inputSignals 

train :: Network -> [([Double],[Double])] -> Network
train net trainingSet = let
        in if networkMeanSquareError net trainingSet <= tolerance
           then net
           else train net' trainingSet where
                net' = trainingEpoch net trainingSet

networkWeights :: Network -> [Matrix Double]
networkWeights net = hlw ++ [olw] where
                hlw = map hlWeights (hiddenLayers net)
                olw = olWeights (outputLayer net)

biasNeuron :: Double
biasNeuron = 1.0

createInputLayer :: InputLayer
createInputLayer = InputLayer id biasNeuron -- layer of linear neurons

createHiddenLayer :: Matrix Double -> HiddenLayer
createHiddenLayer weights = HiddenLayerDiff sigmoid biasNeuron weights sigmoid'

createOutputLayer :: Matrix Double -> OutputLayer
createOutputLayer weights = OutputLayerDiff sigmoid weights sigmoid'

-- Binary threshold function
heavisideFunction :: Double -> Double
heavisideFunction x = if x >= 0.0
                      then 1.0
                      else 0.0

-- Sigmoid function
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1 + exp (-x))

-- The first derivative of the sigmoid function
sigmoid' :: Double -> Double
sigmoid' x = sigmoid x * (1 - sigmoid x)

class Activations a where
      activations :: a -> [Double] -> [Double]

instance Activations HiddenLayer where
         activations hidLayer inputSignals = toList ((hlWeights hidLayer) #> (vector inputSignals))

instance Activations OutputLayer where
         activations outLayer inputSignals = toList ((olWeights outLayer) #> (vector inputSignals))

class Outputs a where
      outputs :: a -> [Double] -> [Double]

instance Outputs InputLayer where
        outputs inLayer inputSignals = [biasNeuron] ++ ino where
                ino = map (ilActivationFunction inLayer) inputSignals  

instance Outputs HiddenLayer where
         outputs hidLayer activationLevels = [biasNeuron] ++ hno where
                hno = map (hlActivationFunction hidLayer) activationLevels

instance Outputs OutputLayer where
         outputs outLayer activationLevels = map (olActivationFunction outLayer) activationLevels

data NetworkSignals = NetworkSignals
                { outputsOfInputLayer :: [Double]
                , activationsOfHiddenLayers :: [[Double]]
                , outputsOfHiddenLayers :: [[Double]]
                , activationsOfOutputLayer :: [Double]
                , outputsOfOutputLayer :: [Double]
                }

networkSignals :: Network -> [Double] -> NetworkSignals
networkSignals net inputSignals = NetworkSignals oil ahl ohl aol ool where
                il = inputLayer net
                hls = hiddenLayers net
                ol = outputLayer net
                oil = outputs il inputSignals
                aohls (inp,(h:hs),act,out) = aohls (o,hs,act++[a],out++[o])
                        where 
                        a = activations h inp
                        o = outputs h a
                aohls (inp,[],act,out) = (inp,[],act,out)
                (_,_,ahl,ohl) = aohls (oil,hls,[],[])
                aol = activations ol (last ohl)
                ool = outputs ol aol

trainingEpoch :: Network -> [([Double],[Double])] -> Network
trainingEpoch net [] = net
trainingEpoch net trainingSet = let
        (trainingPair:trainingSet') = trainingSet
        c = trainingPairPresentation net trainingPair
        wu ((w:ws),(dw:dws),w') = wu (ws,dws,w'++[w+dw]) -- weights update
        wu ([],[],w') = ([],[],w')
        nw = networkWeights net
        (_,_,nw') = wu (nw,c,[])
        net' = createNetwork nw'
        in trainingEpoch net' trainingSet'

trainingPairPresentation :: Network -> ([Double],[Double]) -> [Matrix Double]
trainingPairPresentation net (trainingInputs,desiredOutputs) = let
        s = networkSignals net trainingInputs
        ahl = activationsOfHiddenLayers s
        aol = activationsOfOutputLayer s
        oil = outputsOfInputLayer s
        ohl = outputsOfHiddenLayers s
        ool = outputsOfOutputLayer s
        inputsToHiddenLayers = [oil] ++ (init ohl)
        ol = outputLayer net
        daol = map (olActivationFunctionDerivative ol) aol
        olDelta = (vector desiredOutputs - vector ool) * vector daol
        olCorr = scalar learningRate * olDelta `outer` vector (last ohl)
        (_,_,_,_,_,hlCorrs) = hlc (olDelta,(olWeights ol),ahl,inputsToHiddenLayers,(hiddenLayers net),[])
        hlc (nld,nlw,ah,ihl,[],corr) = (nld,nlw,ah,ihl,[],corr)
        hlc (nextLayerDelta,nextLayerWeights,activationsHidLayers,inputsToHidLayers,hidLayers,corr) =
                hlc (nextLayerDelta',nextLayerWeights',activationsHidLayers',inputsToHidLayers',hs,corr')
                        where
                        h = last hidLayers
                        hs = init hidLayers
                        nextLayerWeights' = hlWeights h
                        a = last activationsHidLayers
                        activationsHidLayers' = init activationsHidLayers
                        inputSignals = last inputsToHidLayers
                        inputsToHidLayers' = init inputsToHidLayers
                        dahl = map (hlActivationFunctionDerivative h) a
                        nextLayerDelta' = tr (dropColumns 1 nextLayerWeights) #> nextLayerDelta * vector dahl
                        c = scalar learningRate * nextLayerDelta' `outer` vector inputSignals
                        corr' = [c] ++ corr
        in hlCorrs ++ [olCorr]

networkMeanSquareError :: Network -> [([Double],[Double])] -> Double
networkMeanSquareError net pairsOfInputsAndDesiredOutputs = let
        se pair = 0.5 * oe <.> oe where
                inp = fst pair
                desout = snd pair
                out = outputs net inp
                oe = (vector desout - vector out)
        sel = map se pairsOfInputsAndDesiredOutputs
        l = length sel
        in (foldr (+) 0 sel) / (fromIntegral l)
