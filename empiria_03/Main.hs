-- Version: 0.3.3 (2018 November 26)

module Main (main) where

import Numeric.LinearAlgebra -- requires the hmatrix package
                ( Matrix
                , matrix
                , rand
                )
import Empiria.FeedforwardNeuralNetwork (createNetwork,train,outputs)

-- Creates an initial basis for the neural network from a list of numbers of non-bias neurons in consecutive layers (starting from the input layer), e.g. [3,4,2]
-- The second argument i.e. "nb" should be given as the empty list i.e. []
createFeedforwardNetworkBasis :: [Int] -> [Matrix Double] -> IO [Matrix Double]
createFeedforwardNetworkBasis arch nb = do
                        if length arch < 2
                        then return nb
                        else do
                                let a = last arch
                                let as = init arch
                                let nw = (last as) + 1 -- +1 is for the bias neuron
                                wm <- rand a nw
                                let nb' = [wm] ++ nb
                                createFeedforwardNetworkBasis as nb'

testRandom = do
       nb <- createFeedforwardNetworkBasis [3,4,2] []
       let n = createNetwork nb
       let input = [1.0,0.5,0.75]
       let output = outputs n input
       print output

testKumarP177 = do
        let hl = matrix 3 [0.01,0.1,-0.2,-0.02,0.3,0.55]
        let ol = matrix 3 [0.31,0.37,0.9,0.27,-0.22,-0.12]
        let n = createNetwork [hl,ol]
        let input1 = [0.5,-0.5]
        let input2 = [-0.5,0.5]
        let output1 = outputs n input1
        let n' = train n [(input1,[0.9,0.1]),(input2,[0.1,0.9])]
        let output2 = outputs n' input2
        putStrLn "Outputs of the network 1:"
        print output1
        putStrLn "Outputs of the network 2:"
        print output2

testLogFun = do
       nb <- createFeedforwardNetworkBasis [2,3,4,1] []
       let n = createNetwork nb
       let n' = train n [([0.0,0.0],[0.0]),([1.0,0.0],[1.0]),([0.0,1.0],[1.0]),([1.0,1.0],[0.0])] -- teaches the XOR function
       print (outputs n' [0.0,0.0])
       print (outputs n' [1.0,0.0])
       print (outputs n' [0.0,1.0])
       print (outputs n' [1.0,1.0])

main :: IO () 
main = do
       putStrLn "OK"
       testLogFun 
