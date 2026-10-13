""" MA3.py

Student: Victor Johansson Blazevic
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from numba import njit

# Exc1
def approximate_pi(n):

    # n is the number of points
    inside_x, inside_y = [], []
    outside_x, outside_y = [], []
    
    for _ in range(n):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            inside_x.append(x)
            inside_y.append(y)
        else:
            outside_x.append(x)
            outside_y.append(y)
            
    pi_approx = 4 * len(inside_x) / n
    print(f"Approximation off pi for n={n}: {pi_approx}")
    
    # Skapa och spara figur
    plt.figure(figsize=(5, 5))
    plt.scatter(inside_x, inside_y, color='red', s=1)
    plt.scatter(outside_x, outside_y, color='blue', s=1)
    plt.savefig(f'pi_approx_{n}.png')
    plt.close()
    
    return pi_approx
    

# Exc2, approximation
def sphere_volume(n, d): 
    # n is the number of points
    # d is the number of dimensions of the sphere 
    
    # Generera n punkter med d dimensioner mha listbyggare
    points = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
    
    # Använd filter, map och lambda för att räkna punkter inuti
    inside = list(filter(lambda pt: sum(map(lambda x: x**2, pt)) <= 1, points))
    
    vol_approx = (len(inside) / n) * (2**d)
    return vol_approx

#Exc2, real value
def hypersphere_exact(n, d):
    # n is the number of points
    # d is the number of dimensions of the sphere 
    # Analytisk formel för volymen
    return (m.pi**(d/2)) / m.gamma(d/2 + 1)



#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    # n is the number of points
    # d is the number of dimensions of the sphere
    inside = 0
    for i in range(n):
        r_squared = 0.0
        for j in range(d):
            r_squared += random.uniform(-1, 1)**2
        if r_squared <= 1.0:
            inside += 1
    return (inside / n) * (2**d)



# Hjälpfunktion för parallellkörning (måste ligga på modulnivå för ProcessPoolExecutor)
def worker_task(args):
    chunk_size, d = args
    inside = 0
    for _ in range(chunk_size):
        if sum(random.uniform(-1, 1)**2 for _ in range(d)) <= 1:
            inside += 1
    return inside



#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel(n, d, np=10):
    # Dela upp arbetet (n) på antalet processer (np)
    chunks = [n // np] * np
    chunks[0] += n % np # Lägg till resten på första
    
    args = [(chunk, d) for chunk in chunks]
    total_inside = 0
    
    # Använd concurrent.futures för att köra processerna
    with future.ProcessPoolExecutor(max_workers=np) as executor:
        results = executor.map(worker_task, args)
        total_inside = sum(results)
        
    return (total_inside / n) * (2**d)
    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    approx_vol = sphere_volume(n, d)
    print(f"Approximation (d={d}): {approx_vol}")
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    approx_vol = sphere_volume(n, d)
    print(f"Approximation (d={d}): {approx_vol}")
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    
    # 1. Vanlig sekventiell (som du redan gjort)
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc3: Vanlig Python tog: {stop-start} sekunder")

    # 2. Numba (måste köras 3 gånger för att se skillnaden)
    # Förutsätter att du har skapat en funktion med @njit som heter t.ex. sphere_volume_numba
    start = pc()
    sphere_volume_numba(n, d)
    print(f"Numba körning 1: {pc()-start} sekunder")
    
    start = pc()
    sphere_volume_numba(n, d)
    print(f"Numba körning 2: {pc()-start} sekunder")
    
    start = pc()
    sphere_volume_numba(n, d)
    print(f"Numba körning 3: {pc()-start} sekunder")

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume_parallel(n, d)
    stop = pc()
    print(f"Exc4: Parallell time of {d} and {n}: {stop-start}")

    
    

if __name__ == '__main__':
	main()
