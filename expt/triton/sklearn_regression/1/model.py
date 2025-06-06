import numpy as np
import triton_python_backend_utils as pb_utils
import joblib
import os


class TritonPythonModel:
    """Sklearn-trained linear regression model for Triton."""

    def initialize(self, args):
        """Load the trained model from the artifacts directory."""
        # In a real scenario, the model would be pre-trained and saved
        # Here we'll create and save a model during initialization for demonstration
        
        # Path to the model directory
        model_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(model_dir, "sklearn_linear_model.joblib")
        
        # Check if model already exists
        if os.path.exists(model_path):
            # Load the pre-trained model
            self.model = joblib.load(model_path)
            print(f"Loaded pre-trained model from {model_path}")
        else:
            # If model doesn't exist, create and save a simple one
            # In a real scenario, this would be done offline during model preparation
            from sklearn.linear_model import LinearRegression
            
            # Generate synthetic training data (X: features, y: target)
            X_train = np.array([[-5], [-2], [0], [2], [5], [7], [9]]).astype(np.float32)
            # Using y = 3x + 2 with some noise
            y_train = (3 * X_train + 2 + np.random.randn(*X_train.shape) * 0.5).astype(np.float32)
            
            # Train the model
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            
            # Save the trained model
            joblib.dump(self.model, model_path)
            print(f"Trained and saved new model to {model_path}")
            
        # Print model coefficients
        print(f"Model coefficients: weight={self.model.coef_[0]:.4f}, bias={self.model.intercept_:.4f}")

    def execute(self, requests):
        """Perform inference using the sklearn model."""
        responses = []
        
        for request in requests:
            # Get input tensor
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input_data = input_tensor.as_numpy()
            
            # Get prediction from sklearn model
            predictions = self.model.predict(input_data).astype(np.float32)
            
            # Create output tensor
            output_tensor = pb_utils.Tensor("OUTPUT0", predictions.reshape(-1, 1))
            
            # Create and append the inference response
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
            
        return responses

    def finalize(self):
        """Clean up resources."""
        print('Sklearn Regression Model Finalized...')
