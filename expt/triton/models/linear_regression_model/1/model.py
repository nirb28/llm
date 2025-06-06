import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Linear regression model implemented as a Triton Python model.
    """

    def initialize(self, args):
        """Initialize the model parameters.
        
        In a real-world scenario, these parameters would typically be loaded
        from a saved model file.
        """
        # Initialize model parameters
        # For the linear function: y = w * x + b
        # Using y = 2x + 5 for this example
        self.weights = np.array([2.0], dtype=np.float32)
        self.bias = np.array([5.0], dtype=np.float32)
        print('Linear Regression Model Initialized...')
        print(f'Model parameters: w={self.weights}, b={self.bias}')

    def execute(self, requests):
        """Execute the linear regression model on incoming requests.
        """
        responses = []
        
        for request in requests:
            # Get input features X
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input_data = input_tensor.as_numpy()
            
            # Apply linear regression: y = w * x + b
            output_data = np.dot(input_data, self.weights) + self.bias
            
            # Create output tensor
            output_tensor = pb_utils.Tensor("OUTPUT0", output_data.astype(np.float32))
            
            # Create InferenceResponse
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
            
        return responses

    def finalize(self):
        """Clean up when the model is unloaded.
        """
        print('Linear Regression Model Finalized...')
