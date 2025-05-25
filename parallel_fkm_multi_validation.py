import uaibot as ub
import time
import numpy as np
import torch

if __name__ == "__main__":
    robot = ub.Robot.create_kuka_kr5()
    configs = 1000  # Número de configurações para teste
    tolerance = 1e-6  # Tolerância para comparação numérica
    warmup_runs = 10  # Execuções de aquecimento para a GPU

    np.random.seed(42)
    random_joint_values_matrix = np.random.uniform(
        low=-np.pi, 
        high=np.pi, 
        size=(configs, len(robot.links)))
    
    print("#" * 50)
    print(f"Executando FKM para {configs} configurações")
    print("#" * 50 + "\n")

    # Aquecimento da GPU (opcional, mas recomendado)
    print("Aquecendo a GPU...")
    for _ in range(warmup_runs):
        _ = robot.fkm(q=random_joint_values_matrix[:10], mode='gpu_multi')

    # Modo CPU (serial)
    print("Processando FKM serial (CPU)...")
    cpu_time_start = time.perf_counter()
    cpu_results = [robot.fkm(q=random_joint_values_matrix[i]) for i in range(configs)]
    cpu_time_total = time.perf_counter() - cpu_time_start

    # Modo GPU (paralelo)
    print("Processando FKM paralelo (GPU)...")
    torch.cuda.synchronize()  # Garante que todas operações CUDA estejam completas
    gpu_time_start = time.perf_counter()
    gpu_results = robot.fkm(q=random_joint_values_matrix, mode='gpu_multi')
    torch.cuda.synchronize()  # Garante que todas operações CUDA estejam completas
    gpu_time_total = time.perf_counter() - gpu_time_start

    # Comparação dos resultados
    print("\nVerificando consistência dos resultados...")
    all_match = True
    
    if isinstance(gpu_results, torch.Tensor):
        gpu_results = gpu_results.cpu().numpy()
    
    for i in range(configs):
        if not np.allclose(cpu_results[i], gpu_results[i], atol=tolerance):
            all_match = False
            break

    # Relatório final
    print("\n" + "#" * 50)
    print(" RELATÓRIO FINAL ".center(50, '#'))
    print("#" * 50)
    print(f"Configurações testadas: {configs}")
    print(f"Todos os resultados batem? {'✅ SIM' if all_match else '❌ NÃO'}")
    print(f"\nTempo total CPU: {cpu_time_total:.6f}s ({cpu_time_total/configs:.6f}s por config)")
    print(f"Tempo total GPU: {gpu_time_total:.6f}s ({gpu_time_total/configs:.6f}s por config)")
    
    if gpu_time_total > 0:
        print(f"Speedup GPU: {cpu_time_total/gpu_time_total:.2f}x")
    else:
        print("Speedup GPU: ∞ (tempo GPU insignificante)")
    
    print("#" * 50)