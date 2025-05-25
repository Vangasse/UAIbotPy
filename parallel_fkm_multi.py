import uaibot as ub
import time
import numpy as np
import torch

if __name__ == "__main__":
    robot = ub.Robot.create_kuka_kr5()

    configs = 3  # Number of random configurations to test
    total_time = 0
    processing_times = []

    np.random.seed(42)  

    random_joint_values_matrix = np.random.uniform(
        low=-np.pi, 
        high=np.pi, 
        size=(configs, len(robot.links))
    )
    
    print("#" * 20)
    print("FKM - Python (Serial CPU)")
    cpu_results = []
    for i in range(configs):
        start_time = time.time()
        result = robot.fkm(q=random_joint_values_matrix[i])
        cpu_results.append(result)
        end_time = time.time()
        processing_times.append(end_time - start_time)
        print(f"Configuration {i+1} HTM:\n{result}")

    total_time = sum(processing_times)
    average_time = total_time / configs
    print(f"\nAverage processing time: {average_time:.6f} seconds")
    print(f"Total processing time: {total_time:.6f} seconds")

    print("\n" + "#" * 20)
    print("FKM - GPU (Parallel)")
    start_time = time.time()
    gpu_results = robot.fkm(q=random_joint_values_matrix, mode='gpu_multi')
    end_time = time.time()

    total_time = end_time - start_time
    average_time = total_time / configs
    print(f"\nBatch HTM results shape: {gpu_results.shape}")  # Expected (3, 4, 4)
    
    # Compare results
    print("\n" + "#" * 20)
    print("Comparison between CPU and GPU results:")
    tolerance = 1e-6  # Numerical tolerance for floating point comparison
    
    for i in range(configs):
        cpu_htm = cpu_results[i]
        gpu_htm = gpu_results[i]
        
        print(f"\nConfiguration {i+1} comparison:")
        print("CPU HTM:\n", cpu_htm)
        print("GPU HTM:\n", gpu_htm)
        
        # Convert to numpy for comparison if needed
        if isinstance(gpu_htm, torch.Tensor):
            gpu_htm = gpu_htm.cpu().numpy()
        
        # Calculate absolute difference
        diff = np.abs(cpu_htm - gpu_htm)
        max_diff = np.max(diff)
        
        print(f"Maximum difference: {max_diff:.8f}")
        if np.allclose(cpu_htm, gpu_htm, atol=tolerance):
            print("✅ Results match within tolerance")
        else:
            print("❌ Results differ beyond tolerance")

    print(f"\nGPU processing time: {total_time:.6f} seconds")
    print(f"GPU average per config: {average_time:.6f} seconds")