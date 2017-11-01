package version1;

import java.util.Arrays;

public class Driver {

	public static void main(String[] args) {
		
		Network nn = new Network(2, new int[] { 3, 3, 2 }); // network with 2 hidden layers
		System.out.println(Arrays.toString(nn.evaluate(new double[] { 1, 1 })));
		
		
	}
	
}
