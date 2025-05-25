import uaibot as ub
import time
import numpy as np
import torch
from statistics import mean, stdev

if __name__ == "__main__":
    robot = ub.Robot.create_kuka_kr5()
    samples = 1000  # Número de amostras por execução
    runs = 100     # Número de execuções
    tolerance = 1e-6

    np.random.seed(42)
    random_joint_values_matrix = np.random.uniform(
        low=-np.pi, 
        high=np.pi, 
        size=(samples, len(robot.links)))
    
    print("#" * 50)
    print(f"Benchmark com {runs} execuções de {samples} amostras")
    print("#" * 50 + "\n")

    # Dicionários para armazenar resultados
    results = {
        'cpu_times': [],
        'gpu_times': [],
        'speedups': []
    }

    # Aquecimento da GPU
    print("Aquecendo a GPU...")
    for _ in range(10):
        _ = robot.fkm(q=random_joint_values_matrix[:10], mode='gpu_multi')

    # Benchmark principal
    for run in range(runs):
        print(f"Execução {run+1}/{runs}", end='\r')
        
        # CPU
        cpu_start = time.perf_counter()
        cpu_res = [robot.fkm(q=random_joint_values_matrix[i]) for i in range(samples)]
        results['cpu_times'].append(time.perf_counter() - cpu_start)
        
        # GPU
        torch.cuda.synchronize()
        gpu_start = time.perf_counter()
        gpu_res = robot.fkm(q=random_joint_values_matrix, mode='gpu_multi')
        torch.cuda.synchronize()
        results['gpu_times'].append(time.perf_counter() - gpu_start)
        
        # Speedup
        if results['gpu_times'][-1] > 0:
            results['speedups'].append(results['cpu_times'][-1]/results['gpu_times'][-1])

    # Verificação de consistência (apenas na última execução)
    print("\nVerificando consistência dos resultados...")
    if isinstance(gpu_res, torch.Tensor):
        gpu_res = gpu_res.cpu().numpy()
    
    all_match = all(np.allclose(cpu_res[i], gpu_res[i], atol=tolerance) for i in range(samples))

    # Estatísticas
    stats = {
        'cpu_mean': mean(results['cpu_times']),
        'cpu_std': stdev(results['cpu_times']) if runs > 1 else 0,
        'gpu_mean': mean(results['gpu_times']),
        'gpu_std': stdev(results['gpu_times']) if runs > 1 else 0,
        'speedup_mean': mean(results['speedups']) if results['speedups'] else float('inf'),
        'speedup_std': stdev(results['speedups']) if runs > 1 and len(results['speedups']) > 1 else 0
    }

    # Relatório
    print("\n" + "#" * 50)
    print(" RELATÓRIO ESTATÍSTICO ".center(50, '#'))
    print("#" * 50)
    print(f"Configurações: {runs} execuções de {samples} amostras")
    print(f"Consistência: {'✅ TODAS' if all_match else '❌ INCONSISTÊNCIAS'}")
    
    print("\nCPU (Serial):")
    print(f"  Tempo médio: {stats['cpu_mean']:.6f}s ± {stats['cpu_std']:.6f}s")
    print(f"  Tempo/amostra: {stats['cpu_mean']/samples:.6f}s")
    
    print("\nGPU (Paralelo):")
    print(f"  Tempo médio: {stats['gpu_mean']:.6f}s ± {stats['gpu_std']:.6f}s")
    print(f"  Tempo/amostra: {stats['gpu_mean']/samples:.6f}s")
    
    print("\nSpeedup GPU vs CPU:")
    print(f"  Média: {stats['speedup_mean']:.2f}x ± {stats['speedup_std']:.2f}x")
    print(f"  Melhor caso: {max(results['speedups']):.2f}x" if results['speedups'] else "N/A")
    print(f"  Pior caso: {min(results['speedups']):.2f}x" if results['speedups'] else "N/A")
    print("#" * 50)

    # Salvar resultados para análise (opcional)
    np.savez('benchmark_results.npz',
             cpu_times=results['cpu_times'],
             gpu_times=results['gpu_times'],
             speedups=results['speedups'])