package Network

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork struct {
	inputNodes  int
	hiddenNodes int
	outputNodes int
	lr          float64
	weights_ih  *mat.Dense
	weights_ho  *mat.Dense
	hiddenData  *mat.Dense
	bias_h      *mat.Dense
	bias_o      *mat.Dense
}

func randomMatrix(rows, cols int) *mat.Dense {
	randSource := rand.NewSource(time.Now().Local().Unix())
	randGen := rand.New(randSource)

	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = randGen.Float64()
	}
	return mat.NewDense(rows, cols, data)
}

func randomVector(cols int) *mat.Dense {
	randSource := rand.NewSource(time.Now().Local().Unix())
	randGen := rand.New(randSource)
	data := make([]float64, cols)
	for i := range data {
		data[i] = randGen.Float64()
	}
	return mat.NewDense(1, cols, data)
}

func Initialize(numInput, numHidden, numOutput int, lr float64) *NeuralNetwork {
	n := &NeuralNetwork{}
	n.inputNodes = numInput
	n.hiddenNodes = numHidden
	n.outputNodes = numOutput
	n.lr = lr
	n.weights_ih = randomMatrix(numInput, numHidden)
	n.weights_ho = randomMatrix(numHidden, numOutput)
	n.bias_h = randomVector(numHidden)
	n.bias_o = randomVector(numOutput)

	return n
}

func (nn *NeuralNetwork) FeedForward(input *mat.Dense) *mat.Dense {
	//Feed data forward to hidden layer
	nn.hiddenData = new(mat.Dense)
	nn.hiddenData.Mul(input, nn.weights_ih)
	nn.hiddenData.Add(nn.hiddenData, nn.bias_h)

	nn.hiddenData.Apply(activate, nn.hiddenData)

	//Feed from hidden to Ouput
	output := new(mat.Dense)
	output.Mul(nn.hiddenData, nn.weights_ho)
	output.Add(output, nn.bias_o)
	output.Apply(activate, output)

	return output
}

func activate(i, j int, x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func derivative(i, j int, x float64) float64 {
	return activate(i, j, x) * (1.0 - activate(i, j, x))
}

func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func (nn *NeuralNetwork) Train(x, y *mat.Dense) {
	output := new(mat.Dense)

	wHidden := mat.DenseCopyOf(nn.weights_ih)
	bHidden := mat.DenseCopyOf(nn.bias_h)
	wOut := mat.DenseCopyOf(nn.weights_ho)
	bOut := mat.DenseCopyOf(nn.bias_o)

	// Use backpropagation to adjust the weights and biases.
	nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output)

	// Define our trained neural network.
	nn.weights_ih = wHidden
	nn.bias_h = bHidden
	nn.weights_ho = wOut
	nn.bias_o = bOut
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}
func (nn *NeuralNetwork) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, wOut)
	addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)
	// Complete the backpropagation.
	networkError := new(mat.Dense)
	networkError.Sub(y, output)

	slopeOutputLayer := new(mat.Dense)
	applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
	slopeOutputLayer.Apply(applySigmoidPrime, output)
	slopeHiddenLayer := new(mat.Dense)
	slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

	dOutput := new(mat.Dense)
	dOutput.MulElem(networkError, slopeOutputLayer)
	errorAtHiddenLayer := new(mat.Dense)
	errorAtHiddenLayer.Mul(dOutput, wOut.T())

	dHiddenLayer := new(mat.Dense)
	dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

	// Adjust the parameters.
	wOutAdj := new(mat.Dense)
	wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
	wOutAdj.Scale(nn.lr, wOutAdj)
	wOut.Add(wOut, wOutAdj)

	bOutAdj, err := sumAlongAxis(0, dOutput)
	if err != nil {
		return err
	}
	bOutAdj.Scale(nn.lr, bOutAdj)
	bOut.Add(bOut, bOutAdj)

	wHiddenAdj := new(mat.Dense)
	wHiddenAdj.Mul(x.T(), dHiddenLayer)
	wHiddenAdj.Scale(nn.lr, wHiddenAdj)
	wHidden.Add(wHidden, wHiddenAdj)

	bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
	if err != nil {
		return err
	}
	bHiddenAdj.Scale(nn.lr, bHiddenAdj)
	bHidden.Add(bHidden, bHiddenAdj)

	return nil
}

func (nn *NeuralNetwork) BatchTest(data *mat.Dense, labels *mat.Dense) float64 {
	n := data.RawMatrix().Rows
	correct := 0
	for i := 0; i < n; i++ {
		row := mat.DenseCopyOf(data.RowView(i).T())
		output := nn.FeedForward(row)

		correct += prediction(output, labels.At(i, 0))

	}
	fmt.Printf("Got %d correct out of %d\n", correct, n)
	return float64(correct) / float64(n)
}

func (nn *NeuralNetwork) BatchTrain(data *mat.Dense, labels *mat.Dense) {
	n := data.RawMatrix().Rows
	for i := 0; i < n; i++ {
		row := mat.DenseCopyOf(data.RowView(i).T())
		nn.Train(row, mat.DenseCopyOf(labels.RowView(0)))
	}
}
func prediction(output *mat.Dense, label float64) int {
	if output.At(0, 0) > 0.6 && label == 1 {
		return 1
	} else if output.At(0, 0) < 0.4 && label == 0 {
		return 1
	}
	return 0
}
