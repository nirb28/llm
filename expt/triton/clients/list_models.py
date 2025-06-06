import argparse
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="List all models served by Triton Inference Server")
    parser.add_argument('--protocol', type=str, default='http', choices=['http', 'grpc'],
                        help='Protocol used to communicate with the Triton server (http or grpc)')
    parser.add_argument('--url', type=str, default='localhost:8000',
                        help='URL of Triton server (default: localhost:8000 for HTTP, use localhost:8001 for gRPC)')
    parser.add_argument('--verbose', action='store_true',
                        help='Display detailed information about each model')
    args = parser.parse_args()
    
    # Set default URL based on protocol if not explicitly provided
    if args.url == 'localhost:8000' and args.protocol.lower() == 'grpc':
        args.url = 'localhost:8001'
    
    # Create client for inference server
    is_grpc = args.protocol.lower() == "grpc"
    if is_grpc:
        triton_client = grpcclient.InferenceServerClient(url=args.url)
    else:
        triton_client = httpclient.InferenceServerClient(url=args.url)
    
    # Health check
    if not triton_client.is_server_live():
        print("❌ Triton server is not live")
        exit(1)
        
    print(f"✅ Connected to Triton server at {args.url} using {args.protocol.upper()} protocol\n")
    
    # Get server metadata
    server_metadata = triton_client.get_server_metadata()
    
    # Handle different response types for HTTP vs gRPC
    if is_grpc:
        server_name = server_metadata.name
        server_version = server_metadata.version
        server_extensions = ', '.join(server_metadata.extensions)
    else:
        server_name = server_metadata['name']
        server_version = server_metadata['version']
        server_extensions = ', '.join(server_metadata['extensions'])
    
    print(f"Server: {server_name}")
    print(f"Version: {server_version}")
    print(f"Extensions: {server_extensions}")
    print()
    
    # Get model repository index
    model_repository_index = triton_client.get_model_repository_index()
    
    # Add timestamp for the scan
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Scan time: {current_time}")
    print()
    
    # Handle different response types for HTTP vs gRPC
    if is_grpc:
        model_count = len(model_repository_index)
        models = model_repository_index
    else:
        model_count = len(model_repository_index)
        models = model_repository_index
    
    # Display information about available models
    print(f"Found {model_count} model(s):")
    print("-" * 50)
    
    for model_info in models:
        # Extract model information based on protocol
        if is_grpc:
            model_name = model_info.name
            model_state = getattr(model_info, 'state', 'UNKNOWN')
            model_version = getattr(model_info, 'version', 'UNKNOWN')
        else:
            model_name = model_info.get('name', '')
            model_state = model_info.get('state', 'UNKNOWN')
            model_version = model_info.get('version', 'UNKNOWN')
            
        # Skip entries that aren't actual models (like 'clients' directory)
        if model_state == 'UNKNOWN' and not args.verbose:
            continue
            
        # Print model header with state indication
        state_icon = '✅' if model_state == 'READY' else '⏳'
        print(f"{state_icon} Model: {model_name}")
        
        # Show basic model information
        if model_state != 'UNKNOWN':
            print(f"   State: {model_state}")
        if model_version != 'UNKNOWN':
            print(f"   Version: {model_version}")
        
        # Get detailed model metadata if verbose flag is set
        if args.verbose:
            try:
                model_metadata = triton_client.get_model_metadata(model_name)
                
                # Print model metadata - handle different formats based on protocol
                if is_grpc:
                    platform = model_metadata.platform
                    versions = model_metadata.versions
                    inputs = model_metadata.inputs
                    outputs = model_metadata.outputs
                else:
                    platform = model_metadata['platform']
                    versions = model_metadata['versions']
                    inputs = model_metadata['inputs']
                    outputs = model_metadata['outputs']
                
                print(f"   Platform: {platform}")
                print(f"   Version: {versions[-1] if len(versions) > 0 else 'N/A'}")
                
                # Print input information
                print("   Inputs:")
                for input_info in inputs:
                    if is_grpc:
                        input_name = input_info.name
                        input_shape = input_info.shape
                        input_datatype = input_info.datatype
                    else:
                        input_name = input_info['name']
                        input_shape = input_info['shape']
                        input_datatype = input_info['datatype']
                        
                    shape_str = str(input_shape).replace("[", "").replace("]", "")
                    print(f"     • {input_name} (shape: {shape_str}, datatype: {input_datatype})")
                
                # Print output information
                print("   Outputs:")
                for output_info in outputs:
                    if is_grpc:
                        output_name = output_info.name
                        output_shape = output_info.shape
                        output_datatype = output_info.datatype
                    else:
                        output_name = output_info['name']
                        output_shape = output_info['shape']
                        output_datatype = output_info['datatype']
                        
                    shape_str = str(output_shape).replace("[", "").replace("]", "")
                    print(f"     • {output_name} (shape: {shape_str}, datatype: {output_datatype})")
                
                # Get model config
                model_config = triton_client.get_model_config(model_name)
                
                # Extract max batch size - handle different formats based on protocol
                if is_grpc and hasattr(model_config, 'config') and hasattr(model_config.config, 'max_batch_size'):
                    max_batch_size = model_config.config.max_batch_size
                    print(f"   Max Batch Size: {max_batch_size}")
                elif not is_grpc and 'config' in model_config and 'max_batch_size' in model_config['config']:
                    max_batch_size = model_config['config']['max_batch_size']
                    print(f"   Max Batch Size: {max_batch_size}")
                
                # Get model statistics
                try:
                    model_stats = triton_client.get_inference_statistics(model_name)
                    
                    if is_grpc and model_stats.model_stats:
                        stats = model_stats.model_stats[0]
                        print(f"   Inference Count: {stats.inference_stats.success.count}")
                        print(f"   Execution Count: {stats.execution_count}")
                        if hasattr(stats, 'cache_hit'):
                            print(f"   Cache Hit Count: {stats.cache_hit.count}")
                    elif not is_grpc and 'model_stats' in model_stats and model_stats['model_stats']:
                        stats = model_stats['model_stats'][0]
                        print(f"   Inference Count: {stats['inference_stats']['success']['count']}")
                        print(f"   Execution Count: {stats['execution_count']}")
                        if 'cache_hit' in stats:
                            print(f"   Cache Hit Count: {stats['cache_hit']['count']}")
                except Exception as e:
                    print(f"   ⚠️  Could not retrieve statistics: {e}")
                    
            except Exception as e:
                print(f"   ⚠️  Could not retrieve detailed metadata: {e}")
        
        print("-" * 50)
    
    print("\nℹ️  To see detailed information about each model, run with --verbose flag")

if __name__ == "__main__":
    main()
