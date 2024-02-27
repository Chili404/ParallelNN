package Network

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork struct {
	inputNodes  int
	hiddenNodes int
	outputNodes int
	lr          float64
	weights_ih  *mat.Dense
	weights_ho  *mat.Dense
	hiddenData  *mat.VecDense
	bias_h      *mat.VecDense
	bias_o      *mat.VecDense
}

func randomMatrix(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.Float64()
	}
	return mat.NewDense(rows, cols, data)
}

func randomVector(rows int) *mat.VecDense {
	data := make([]float64, rows)
	for i := range data {
		data[i] = rand.Float64()
	}
	return mat.NewVecDense(rows, data)
}

func (n *NeuralNetwork) Initialize(numInput, numHidden, numOutput int, lr float64) {
	n.inputNodes = numInput
	n.hiddenNodes = numHidden
	n.outputNodes = numOutput
	n.lr = lr
	n.weights_ih = randomMatrix(numHidden, numInput)
	n.weights_ho = randomMatrix(numOutput, numHidden)
	n.bias_h = randomVector(numHidden, 1)
	n.bias_o = randomVector(numOutput, 1)
}

func (n *NeuralNetwork) FeedForward(input *mat.VecDense) *mat.VecDense {
	//Feed data forward to hidden layer
	n.hiddenData = mat.NewVecDense(n.hiddenNodes, nil)
	n.hiddenData.MulVec(n.weights_ih, input)
	n.hiddenData.AddVec(n.hiddenData, n.bias_h)

	activate(n.hiddenData)

	//Feed from hidden to Ouput
	output := mat.NewVecDense(n.outputNodes, nil)
	output.MulVec(n.weights_ho, n.hiddenData)
	output.AddVec(output, n.bias_o)
	activate(output)

	return output
}

func activate(vector *mat.VecDense) {
	for i := 0; i < vector.Len(); i++ {
		vector.SetVec(i, 1.0/(1+math.Exp(-vector.At(i, 0))))
	}
}

func derivative(vector *mat.VecDense) *mat.VecDense {
	result := mat.NewVecDense(vector.Len(), nil)
	for i := 0; i < vector.Len(); i++ {
		val := vector.At(i, 0)
		result.SetVec(i, val*(1-val))
	}
	return result
}

func (n *NeuralNetwork) Train(input *mat.VecDense, targets *mat.VecDense) {
	output := n.FeedForward(input)

	//
	outError := mat.NewVecDense(n.outputNodes, nil)
	outError.SubVec(targets, output)

	gradient := derivative(output)
	gradient.MulVec(gradient, outError)
	gradient.ScaleVec(n.lr, gradient)

	weights_ho_deltas := mat.NewDense(n.outputNodes, n.hiddenNodes, nil)
	weights_ho_deltas.Mul(gradient, n.hiddenData.T())
	n.bias_h.AddVec(n.bias_h, gradient)
	n.weights_ho.Add(n.weights_ho, weights_ho_deltas)

	hiddenError := mat.NewVecDense(n.hiddenNodes, nil)
	hiddenError.MulVec(n.weights_ho.T(), outError)
	hiddenGradient := derivative(hiddenError)
	hiddenGradient.ScaleVec(n.lr, hiddenError)

	n.bias_h.AddVec(n.bias_h, hiddenGradient)
	weights_ih_deltas := mat.NewDense(n.hiddenNodes, n.inputNodes, nil)
	weights_ih_deltas.Mul(hiddenGradient, input.T())
	n.weights_ih.Add(n.weights_ih, weights_ih_deltas)
}

//Leaky ReLu Activation
/*
   private void activate(Matrix hidden) {
       for (int i = 0; i < hidden.data.length; i++) {
           if (hidden.data[i][0] < 0) {
               hidden.data[i][0] = 0.1 * hidden.data[i][0];
           }
       }
   }

   private double[] derivative(Matrix m) {
       double[] result= new double[m.data.length];
       for (int i = 0; i < m.data.length; i++) {
           if (m.data[i][0] < 0) {
               result[i] = 0.1;
           }else {
               result[i] = 1;
           }
       }
       return result;
   }
*/
