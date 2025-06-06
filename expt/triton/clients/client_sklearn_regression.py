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
    parser.add_argument('--num-points', type=int, default=50,
                        help='Number of test points to generate')
    args = parser.parse_args()
    
    # Create client for inference server
    if args.protocol.lower() == "grpc":
        triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
    else:
        triton_client = httpclient.InferenceServerClient(url="localhost:8000")
    
    # Health check
    try:
        if not triton_client.is_server_live():
            print("❌ Triton server is not live")
            exit(1)
        else:
            print("✅ Connected to Triton server")
            
        # Check if model is ready
        if not triton_client.is_model_ready("sklearn_regression"):
            print("❌ Model 'sklearn_regression' is not ready")
            exit(1)
        else:
            print("✅ Model 'sklearn_regression' is ready")
    except Exception as e:
        print(f"❌ Error connecting to Triton server: {e}")
        exit(1)
        
    # Create test data (random x values in a range)
    x_min, x_max = -10, 10
    x_test = np.linspace(x_min, x_max, args.num_points).reshape(-1, 1).astype(np.float32)
    
    print(f"Input shape: {x_test.shape}")
    print("Sample input values:", x_test[:5].flatten())
    
    # Set up the inputs for the model
    if args.protocol.lower() == "grpc":
        inputs = [grpcclient.InferInput("INPUT0", x_test.shape, "FP32")]
        inputs[0].set_data_from_numpy(x_test)
        outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]
    else:
        inputs = [httpclient.InferInput("INPUT0", x_test.shape, "FP32")]
        inputs[0].set_data_from_numpy(x_test)
        outputs = [httpclient.InferRequestedOutput("OUTPUT0")]
    
    # Query the server
    try:
        print("\n🔄 Sending inference request...")
        results = triton_client.infer(
            model_name="sklearn_regression",
            inputs=inputs,
            outputs=outputs
        )
        
        # Get prediction results
        predictions = results.as_numpy("OUTPUT0")
        
        print("✅ Received predictions from model")
        print("Predictions shape:", predictions.shape)
        print("First 5 predictions:", predictions[:5].flatten())
        
        # Estimate model parameters from predictions
        # For a linear model y = wx + b, we can estimate w and b
        # by solving the line equation using two points
        x1, y1 = x_test[0][0], predictions[0][0]
        x2, y2 = x_test[-1][0], predictions[-1][0]
        
        estimated_w = (y2 - y1) / (x2 - x1)
        estimated_b = y1 - estimated_w * x1
        
        print(f"\nEstimated model parameters: weight ≈ {estimated_w:.4f}, bias ≈ {estimated_b:.4f}")
        print("This is likely close to y = 3x + 2 with some noise as defined in the model")
            
        # Visualize results if requested
        if args.visualize:
            plt.figure(figsize=(12, 8))
            
            # Plot test points and predictions
            plt.scatter(x_test, predictions, color='blue', label='Model Predictions')
            
            # Plot the estimated line
            x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
            y_estimated = estimated_w * x_line + estimated_b
            plt.plot(x_line, y_estimated, 'r--', label=f'Estimated Model: y = {estimated_w:.4f}x + {estimated_b:.4f}')
            
            # Reference line for y = 3x + 2
            y_reference = 3 * x_line + 2
            plt.plot(x_line, y_reference, 'g-', label='Reference: y = 3x + 2')
            
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Scikit-learn Regression Model Predictions')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    except Exception as e:
        print(f"❌ Error during inference: {e}")

if __name__ == "__main__":
    main()
