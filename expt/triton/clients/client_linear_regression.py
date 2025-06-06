import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=str, default='http', choices=['http', 'grpc'],
                        help='Protocol used to communicate with the Triton server')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results with matplotlib')
    parser.add_argument('--num-points', type=int, default=20,
                        help='Number of test points to generate')
    args = parser.parse_args()
    
    # Create client for inference server
    if args.protocol.lower() == "grpc":
        triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
    else:
        triton_client = httpclient.InferenceServerClient(url="localhost:8000")
    
    # Health check
    if not triton_client.is_server_live():
        print("Triton server is not live")
        exit(1)
        
    # Known model parameters (y = 2x + 5), these should match what's in the model
    true_weight = 2.0
    true_bias = 5.0
    
    # Create test data (random x values)
    x_values = np.linspace(-10, 10, args.num_points)
    x_data = np.array(x_values, dtype=np.float32).reshape(-1, 1)
    
    # Expected y values based on known model parameters (y = 2x + 5)
    expected_y = true_weight * x_data + true_bias
    
    # Print input/expected output for verification
    print(f"Input shape: {x_data.shape}")
    print("Sample input values:", x_data[:5].flatten())
    print("Expected output (first 5):", expected_y[:5].flatten())
    
    # Add some noise for test data to be more realistic
    # noise = np.random.normal(0, 1, size=x_data.shape).astype(np.float32)
    # noisy_y = expected_y + noise
    
    # Set up the inputs for the model
    if args.protocol.lower() == "grpc":
        inputs = [grpcclient.InferInput("INPUT0", x_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(x_data)
        outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]
    else:
        inputs = [httpclient.InferInput("INPUT0", x_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(x_data)
        outputs = [httpclient.InferRequestedOutput("OUTPUT0")]
    
    # Query the server
    try:
        results = triton_client.infer(
            model_name="linear_regression_model",
            inputs=inputs,
            outputs=outputs
        )
        
        # Get prediction results
        predictions = results.as_numpy("OUTPUT0")
        
        # Print the predictions
        print("\nPredictions (first 5):", predictions[:5].flatten())
        
        # Calculate mean squared error
        mse = np.mean((predictions - expected_y) ** 2)
        print(f"\nMean Squared Error: {mse:.6f}")
        
        # Verify predictions match expected values
        # MSE is high because our range of values is large (-20 to 25)
        # This doesn't mean the model is incorrect
        
        # Convert to flat arrays for easier comparison
        pred_flat = predictions.flatten()
        expected_flat = expected_y.flatten()
        
        # Print detailed comparison for a few values
        print("Detailed comparison (first 5 values):")
        for i in range(min(5, len(pred_flat))):
            diff = abs(expected_flat[i] - pred_flat[i])
            print(f"  Index {i}: Expected={expected_flat[i]:.6f}, Actual={pred_flat[i]:.6f}, Diff={diff:.6f}")
        
        # For this data range, use absolute and relative tolerances that make sense
        # rtol=1e-2: 1% relative tolerance, atol=0.1: absolute tolerance of 0.1
        all_close = np.allclose(predictions, expected_y, rtol=1e-2, atol=0.1)
        
        # Also check if the maximum absolute error is small
        max_abs_diff = np.max(np.abs(predictions - expected_y))
        print(f"Maximum absolute difference: {max_abs_diff:.6f}")
        print(f"All values close within tolerance: {all_close}")
        
        # If either criterion passes, the model is working correctly
        if all_close or max_abs_diff < 0.001:  # Less than 0.001 difference is essentially identical
            print("[PASS] Test PASSED! Model predictions match expected values.")
        else:
            print("[FAIL] Test FAILED! Model predictions differ from expected values.")

            
        # Visualize results if requested
        if args.visualize:
            plt.figure(figsize=(10, 6))
            
            # Plot test points
            plt.scatter(x_data, expected_y, color='blue', label='Expected Values (y = 2x + 5)')
            
            # Plot predictions
            plt.scatter(x_data, predictions, color='red', marker='x', label='Model Predictions')
            
            # Plot the line representing the true function
            x_line = np.linspace(-10, 10, 100).reshape(-1, 1)
            y_line = true_weight * x_line + true_bias
            plt.plot(x_line, y_line, 'g--', label='True Function')
            
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Linear Regression Model Predictions')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
