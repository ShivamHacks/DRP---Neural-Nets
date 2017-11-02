package version1;

import java.util.Map;
import java.util.TreeMap;

public class Network {
	
	/*
	 * Main network class:
	 * - underlying data structure is doubly linked list
	 */
	
	class Layer {
		
		int numNeurons;
		int numWeights;
		double weights[][]; // first index is neuron in layer, second one is num weights in each neuron
		Layer prev;
		Layer next;
		
		Layer(int numNeurons, int numWeights) {
			this.numNeurons = numNeurons;
			this.numWeights = numWeights;
			weights = new double[numNeurons][numWeights];
			
			for (int i = 0; i < numNeurons; i++) {
				for (int j = 0; j < numWeights; j++) {
					weights[i][j] = 1;
				}
			}
			
			prev = null;
			next = null;
		}
		
	}
	
	Layer head;
	Layer tail;
	int numLayers;
	int[] struct;
	
	public Network(int inputSize, int[] struct) {
		head = null;
		tail = null;
		numLayers = struct.length;
		this.struct = struct;
		
		/*
		 * struct.length = numLayers
		 * numNeuronsForLayerI = struct[i]
		 */
		
		
		/* Initialize network */
		head = new Layer(struct[0], inputSize);
		Layer prev = head;
		for (int i = 1; i < struct.length - 1; i++) {
			Layer nextLayer = new Layer(struct[i], prev.numNeurons);
			prev.next = nextLayer;
			nextLayer.prev = prev;
			prev = nextLayer;
		}
		tail = new Layer(struct[struct.length - 1], prev.numNeurons);
		prev.next = tail;
		
	}
	
	public double[] evaluate(double[] x) {
		
		Layer currentLayer = head;
		double[] input = x;
		double[] output = new double[0];
		while (currentLayer != null) {
			output = new double[currentLayer.numNeurons];
			for (int i = 0; i < output.length; i++) {
				double result = 0;
				for (int j = 0; j < input.length; j++) { // within neuron now
					result += currentLayer.weights[i][j] * input[j];
				}
				output[i] = result; //sigmoid(result);
			}
			input = output;
			currentLayer = currentLayer.next;
		}
		
		return output;
	}
	
	private void trainOne(double[] x, double[] y, int epochs, int learningRate) {
		
		/*
		 * stochastic gradient descent
		 * using backpropogation algorithm
		 */
		
		/*** Use feedforward to calculate activations at each layer ***/
		
		Map<Integer, double[]> activations = new TreeMap<Integer, double[]>();
		
		Layer currentLayer = head;
		double[] input = x; // TODO: fix with epochs
		double[] output = new double[0];
		int l = 1;
		while (currentLayer != null) {
			output = new double[currentLayer.numNeurons];
			for (int i = 0; i < output.length; i++) {
				double result = 0;
				for (int j = 0; j < input.length; j++) { // within neuron now
					result += currentLayer.weights[i][j] * input[j];
				}
				output[i] = result; //sigmoid(result);
			}
			activations.put(l++, output); // put in activation map
			input = output;
			currentLayer = currentLayer.next;
		}
		
		/*** Compute output error ***/
		
		Map<Integer, double[]> errors = new TreeMap<Integer, double[]>();
		
		double[] outputError = new double[struct.length - 1];
		for (int i = 0; i < y.length; i++) { // TODO: replace 0 with the one for iteration
			outputError[i] = Math.abs(y[i] - activations.get(numLayers)[i]); // TODO: remove abs val
		}
		errors.put(numLayers, hadmard(outputError, activations.get(numLayers))); // TODO: wrap activations in sigmoid

		/*** Backpropagate the error ***/
		
		Layer cLayer = tail.prev;
		int layerIndex = numLayers - 1;
		while (cLayer.prev != null) {
			
			/** calculate error for current layer **/
			double[] nextError = errors.get(layerIndex + 1);
			double[][] nextWeightsT = transpose(cLayer.next.weights);
			double[] error = hadmard(multiply(nextWeightsT, nextError), 
					sigmoidPrime(activations.get(layerIndex--)));
			errors.put(layerIndex + 1, error);
			/** backpropagate **/
			nextError = error;
			cLayer = cLayer.prev;
			
		}
		
		
		
		/*
		//double[] prevError = new double[struct.length - 1];
		for (int t = 0; t < prevError.length; t++) prevError[t] = 1;
		
		//Layer cLayer = tail;
		//int layerIndex = numLayers - 1;
		while (cLayer.prev != null) { // when true, we are at 1st hidden layer
			
			double[] error = new double[struct[layerIndex]];
			//for (int i = 0; i < y[0])
			
		}
		
		//Layer cLayer = tail.prev;
		//int layerIndex = numLayers - 2;
		while (cLayer.prev != null) {
			
			double[] errorNext = errors.get(layerIndex + 1);
			double[][] weightsNext = cLayer.next.weights; // need to take transpose
			
			
			
			// need to multiply error next by weights next
			// need to take hadmard product of sigmoid prime of current layer and result of previous line
			
		}
		
		//double outputError = //Math.abs(y[0][0]
		
		
		//double[][] activations = new double[numLayers]*/
		
		
	}
	
	/***** Static Helper Methods *****/
	
	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}
	
	public static double[] sigmoidPrime(double[] x) { // TODO
		double[] result = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			double sigmoidX = sigmoid(x[i]);
			result[i] = sigmoidX / (1 - sigmoidX); 
		}
		return result;
	}
	
	public static double[][] transpose(double[][] matrix) { // WORKS!!!
		
		double[][] newMatrix = new double[matrix[0].length][matrix.length];
		for(int i = 0; i < matrix[0].length; i++) {
			for(int j = 0; j < matrix.length; j++) {
				newMatrix[i][j] = matrix[j][i];
			}
			
		}
		return newMatrix;
	}
	
	public static double[] multiply(double[][] A, double[] B) { // WORKS!
		
		double[] newMatrix = new double[A.length];
		
		int aRows = A.length;
        int aColumns = A[0].length;
        int bRows = B.length;
        int bColumns = 1; // error vector
		
		for (int i = 0; i < aRows; i++) { // aRow
            for (int j = 0; j < bColumns; j++) { // bColumn
                for (int k = 0; k < aColumns; k++) { // aColumn
                    newMatrix[i] += A[i][k] * B[k];
                }
            }
        }
		
		return newMatrix;

	}
	
	/*
	 * Requirements: A and B should be the same length
	 */
	public static double[] hadmard(double[] A, double[] B) {
		
		double[] newVector = new double[A.length];
		for (int i = 0; i < A.length; i++) {
			newVector[i] = A[i] * B[i];
		}
		
		return newVector;
		
	}

}
