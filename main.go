package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ParallelNN/Network"
	"gonum.org/v1/gonum/mat"
)

func Normalize(inputs [][]float64) *mat.Dense {
	// Convert records to a matrix
	data := mat.NewDense(len(inputs), len(inputs[0]), nil)
	for i, d := range inputs {
		data.SetRow(i, d)
	}

	// Compute the mean and standard deviation for each column
	rows, cols := data.Dims()
	means := make([]float64, cols)
	stdDevs := make([]float64, cols)
	for j := 0; j < cols; j++ {
		// Extract column data into a vector
		col := make([]float64, rows)
		for i := 0; i < rows; i++ {
			col[i] = data.At(i, j)
		}
		// Compute mean
		var sum float64
		for _, v := range col {
			sum += v
		}
		mean := sum / float64(rows)
		means[j] = mean

		// Compute standard deviation
		var variance float64
		for _, v := range col {
			diff := v - mean
			variance += diff * diff
		}
		stdDev := math.Sqrt(variance / float64(rows))
		stdDevs[j] = stdDev
	}

	// Normalize each column
	normalizedData := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			value := data.At(i, j)
			normalizedValue := (value - means[j]) / stdDevs[j]
			normalizedData.Set(i, j, normalizedValue)
		}
	}
	return normalizedData
}

func shuffleData(trainingData *mat.Dense, testingLabel *mat.Dense) (*mat.Dense, *mat.Dense) {
	// Get the number of rows in the data
	numRows, _ := trainingData.Dims()

	// Initialize a slice to hold row indices
	indices := make([]int, numRows)
	for i := range indices {
		indices[i] = i
	}

	// Shuffle the indices
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(numRows, func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	// Apply the same shuffle to both trainingData and testingLabel
	shuffledTrainingData := mat.NewDense(numRows, trainingData.RawMatrix().Cols, nil)
	shuffledTestingLabel := mat.NewDense(numRows, 1, nil)
	for i, idx := range indices {
		row := make([]float64, trainingData.RawMatrix().Cols)
		for j := 0; j < len(row); j++ {
			row[j] = trainingData.At(i, j)
		}
		shuffledTrainingData.SetRow(i, row)
		shuffledTestingLabel.Set(i, 0, testingLabel.At(idx, 0))
	}

	return shuffledTrainingData, shuffledTestingLabel
}
func parseEntry(stuff []string) []float64 {
	var output = make([]float64, len(stuff)-1)

	for i := 1; i < len(stuff); i++ {
		var err error
		if i == len(stuff)-1 {
			switch stuff[i] {
			case "2":
				output[i-1] = 0

			default:
				output[i-1] = 1
			}
		} else {
			output[i-1], err = strconv.ParseFloat(strings.TrimSpace(stuff[i]), 64)
			if err != nil { //Port location
				output[i-1] = 0
			}
			output[i-1] /= 10
		}

	}

	return output

}

func main() {
	epochs := 5000
	file, err := os.Open("data/data.csv")

	if err != nil {
		fmt.Println("Couldn't read data")
		return
	}

	reader := csv.NewReader(file)

	record, err := reader.ReadAll()

	if err != nil {
		fmt.Println("Error reading records")
		return
	}
	fields := 9

	//Parse Data into data and label
	var inputs = []float64{}
	var label = make([]float64, len(record))
	numfields := reader.FieldsPerRecord
	for i := 0; i < len(record); i++ {
		val := parseEntry(record[i])
		inputs = append(inputs, val[0:numfields-2]...)
		label[i] = val[numfields-2]
	}

	//Normalize and Vectorize
	//normalizedData := Normalize(inputs)
	normalizedData := mat.NewDense(len(record), fields, inputs)
	rows, cols := normalizedData.Dims()
	labelMat := mat.NewVecDense(len(label), label)

	//Split 75% of data to train
	trainingCutoff := int(0.75 * float64(normalizedData.RawMatrix().Rows))
	trainingData := mat.DenseCopyOf(normalizedData.Slice(0, trainingCutoff, 0, cols))
	trainingLabel := mat.DenseCopyOf(labelMat.SliceVec(0, trainingCutoff))
	testingData := mat.DenseCopyOf(normalizedData.Slice(trainingCutoff, rows, 0, cols))
	testingLabel := mat.DenseCopyOf(labelMat.SliceVec(trainingCutoff, rows))

	NN := Network.Initialize(fields, int(float64(numfields)*0.666), 1, 0.3)

	//Pre-training accuracy
	rate := NN.BatchTest(testingData, testingLabel)

	//fmt.Printf("Pre-Correct rate is %.4f\n", rate)

	rate = NN.BatchTest(trainingData, trainingLabel)

	fmt.Printf("Pre-Correct training rate is %.4f\n\n", rate)

	//Train
	for i := 0; i < epochs; i++ {
		randData, ranLabel := shuffleData(trainingData, trainingLabel)
		NN.BatchTrain(randData, ranLabel)
	}

	//Post-training accuracy
	rate = NN.BatchTest(testingData, testingLabel)

	//fmt.Printf("Post-Correct rate is %.4f\n", rate)

	rate = NN.BatchTest(trainingData, trainingLabel)

	fmt.Printf("Post-Correct training rate is %.4f\n", rate)
}
