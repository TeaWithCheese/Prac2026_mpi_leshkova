import matplotlib.pyplot as plt
import csv

ranks = []
speedup = []

with open('final_scaling.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ranks.append(int(row['ranks']))
        speedup.append(float(row['speedup']))

plt.figure(figsize=(8, 6))
plt.plot(ranks, speedup, 'o-', label='Фактическое ускорение', color='blue')
plt.plot(ranks, ranks, '--', label='Идеальное ускорение', color='gray')

plt.title('График сильной масштабируемости')
plt.xlabel('Число процессов p')
plt.ylabel('Ускорение S(p)')
plt.grid(True)
plt.legend()
plt.savefig('scaling_plot.png')