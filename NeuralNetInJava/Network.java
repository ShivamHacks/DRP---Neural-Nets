package version1;

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
	
	public Network(int inputSize, int[] struct) {
		head = null;
		tail = null;
		numLayers = struct.length;
		
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
	
	private double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

}
