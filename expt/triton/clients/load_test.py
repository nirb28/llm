#!/usr/bin/env python
import argparse
import numpy as np
import time
import threading
import random
from functools import partial
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

def send_requests(protocol, model_name, batch_size, num_threads, duration, qps):
    """Send inference requests to the Triton server."""
    
    if protocol == "http":
        client = httpclient.InferenceServerClient("localhost:8000")
        client_fn = httpclient.InferInput
        request_fn = partial(client.infer, model_name=model_name)
    else:
        client = grpcclient.InferenceServerClient("localhost:8001")
        client_fn = grpcclient.InferInput
        request_fn = partial(client.infer, model_name=model_name)
    
    # Set up inputs based on the model
    if model_name == "add_model":
        input_fn = lambda: (
            client_fn("INPUT0", [batch_size, 16], "FP32"),
            client_fn("INPUT1", [batch_size, 16], "FP32")
        )
        prepare_fn = lambda inputs: [
            inputs[0].set_data_from_numpy(np.random.rand(batch_size, 16).astype(np.float32)),
            inputs[1].set_data_from_numpy(np.random.rand(batch_size, 16).astype(np.float32))
        ]
    elif model_name == "linear_regression_model":
        input_fn = lambda: client_fn("INPUT0", [batch_size, 1], "FP32")
        prepare_fn = lambda inputs: inputs.set_data_from_numpy(np.random.rand(batch_size, 1).astype(np.float32))
    elif model_name == "sklearn_regression":
        input_fn = lambda: client_fn("INPUT0", [batch_size, 1], "FP32")
        prepare_fn = lambda inputs: inputs.set_data_from_numpy(np.random.rand(batch_size, 1).astype(np.float32))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    def worker():
        end_time = time.time() + duration
        sleep_time = 1.0 / qps if qps > 0 else 0
        
        success_count = 0
        fail_count = 0
        latencies = []
        
        while time.time() < end_time:
            try:
                start_time = time.time()
                
                # Create inputs and prepare data
                inputs = input_fn()
                if isinstance(inputs, tuple):
                    for inp in inputs:
                        prepare_fn(inputs)
                    request_fn(inputs=inputs)
                else:
                    prepare_fn(inputs)
                    request_fn(inputs=[inputs])
                
                latency = (time.time() - start_time) * 1000  # ms
                latencies.append(latency)
                success_count += 1
                
                # Sleep to maintain QPS
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"Error: {e}")
                fail_count += 1
        
        return success_count, fail_count, latencies
    
    # Start worker threads
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    print(f"Load test complete for model {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Load test for Triton Server")
    parser.add_argument("--protocol", type=str, choices=["http", "grpc"], default="http",
                       help="Protocol to use (http or grpc)")
    parser.add_argument("--model", type=str, default="linear_regression_model",
                       choices=["add_model", "linear_regression_model", "sklearn_regression"], 
                       help="Model to test")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--threads", type=int, default=4, help="Number of concurrent threads")
    parser.add_argument("--duration", type=int, default=60, help="Duration of test in seconds")
    parser.add_argument("--qps", type=int, default=10, help="Queries per second per thread (0 for max speed)")
    
    args = parser.parse_args()
    
    print(f"Starting load test for {args.model} using {args.protocol}...")
    print(f"Running {args.threads} threads for {args.duration} seconds with batch size {args.batch_size}")
    
    send_requests(
        args.protocol, 
        args.model, 
        args.batch_size, 
        args.threads, 
        args.duration, 
        args.qps
    )

if __name__ == "__main__":
    main()
