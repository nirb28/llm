import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        """
        print('Initialized...')

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.
        """
        responses = []
        
        # Every Python backend must iterate through list of requests and create a
        # pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            input0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input0_data = input0.as_numpy()
            
            # Get INPUT1
            input1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")
            input1_data = input1.as_numpy()
            
            # Add the inputs and create the output tensor
            output0_data = input0_data + input1_data
            output0 = pb_utils.Tensor("OUTPUT0", output0_data)
            
            # Create InferenceResponse
            inference_response = pb_utils.InferenceResponse(output_tensors=[output0])
            responses.append(inference_response)
            
        # Return the list of responses for all requests
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to clean up any state associated with it.
        """
        print('Finalizing...')
