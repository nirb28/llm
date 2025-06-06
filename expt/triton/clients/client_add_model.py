import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=str, default='http', choices=['http', 'grpc'],
                        help='Protocol used to communicate with the Triton server')
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
    
    # Create random input data - adding an explicit batch dimension [1, 16]
    input0_data = np.random.rand(1, 16).astype(np.float32)
    input1_data = np.random.rand(1, 16).astype(np.float32)
    
    # Expected output (for verification)
    expected_output = input0_data + input1_data
    
    # Print original input shapes
    print(f"Input shapes: {input0_data.shape}")
    
    # Print inputs for verification
    print("Input 0:", input0_data)
    print("Input 1:", input1_data)
    print("Expected output:", expected_output)
    
    # Set up the inputs for the model
    if args.protocol.lower() == "grpc":
        inputs = [
            grpcclient.InferInput("INPUT0", input0_data.shape, "FP32"),
            grpcclient.InferInput("INPUT1", input1_data.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(input0_data)
        inputs[1].set_data_from_numpy(input1_data)
        outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]
    else:
        inputs = [
            httpclient.InferInput("INPUT0", input0_data.shape, "FP32"),
            httpclient.InferInput("INPUT1", input1_data.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(input0_data)
        inputs[1].set_data_from_numpy(input1_data)
        outputs = [httpclient.InferRequestedOutput("OUTPUT0")]
    
    # Query the server
    try:
        results = triton_client.infer(
            model_name="add_model",
            inputs=inputs,
            outputs=outputs
        )
        output0_data = results.as_numpy("OUTPUT0")
        
        # Print the output data
        print("Received output:", output0_data)
        
        # Check if the output matches expected output
        if np.allclose(output0_data, expected_output):
            print("✅ Test PASSED! The model correctly added the inputs.")
        else:
            print("❌ Test FAILED! The model output does not match the expected output.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
