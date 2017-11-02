package version1;

import java.util.Arrays;

import org.junit.Test;

public class Tests {

	@Test
	public void testHelperFunctions() {
		
		Network nn = new Network(2, new int[] { 3, 3, 2 }); // network with 2 hidden layers
		
		double[][] A = nn.transpose(new double[][] {
			{ 1, 2, 3 } , { 4, 5, 6 }
		});
		double[] B = new double[] { 7, 8 };
		//System.out.println(Arrays.toString(nn.multiply(A, B)));
		
		
	}
	
	@Test
	public void testBackprop() {
		
		Network nn = new Network(2, new int[] { 3, 3, 2 }); // network with 2 hidden layers
		
		double[] x = new double[] { 0, 1 };
		double[] y = new double[] { 1, 0 };
		
		nn.trainOne(x, y, 0, 0);
		
		
	}
	
}
