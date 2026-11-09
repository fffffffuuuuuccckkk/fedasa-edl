import numpy as np
import time
import os
import ctypes
import gc

def check_memory_stability():
    print(f"--- 启动非 Root 内存压力测试 (PID: {os.getpid()}) ---")
    
    # 强制清理当前环境
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except:
        pass

    memory_blocks = []
    block_size_gb = 2  # 每次申请 2GB
    total_iterations = 20 # 循环测试次数
    
    try:
        for i in range(1, total_iterations + 1):
            print(f"\n[第 {i} 轮] 正在申请并校验 {block_size_gb}GB 内存...")
            
            # 1. 申请大块内存并写入随机数据
            # 使用 uint8 减少计算开销，专注于内存填充
            block = np.random.randint(0, 255, size=(block_size_gb * 1024 * 1024 * 1024), dtype=np.uint8)
            
            # 2. 执行一次求和校验（强制触发 CPU 与内存的实际交互）
            checksum = block.sum()
            print(f"   校验和计算完成: {checksum} (内存读写正常)")
            
            # 3. 将块存入列表，保持内存占用
            memory_blocks.append(block)
            
            # 打印当前进程占用内存 (Linux 专用)
            with open('/proc/self/status') as f:
                vm_rss = [line for line in f if 'VmRSS' in line][0].strip()
                print(f"   当前进程实耗物理内存: {vm_rss}")
                
            time.sleep(1) # 给系统一点喘息时间

    except MemoryError:
        print("\n[!] 达到系统内存上限 (OOM)，测试停止。")
    except Exception as e:
        print(f"\n[!] 捕获到异常退出: {e}")
    finally:
        print("\n--- 测试结束，正在释放内存 ---")
        memory_blocks.clear()
        gc.collect()

if __name__ == "__main__":
    check_memory_stability()