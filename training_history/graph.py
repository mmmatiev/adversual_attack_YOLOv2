import json
import matplotlib.pyplot as plt

# Конфигурация
files = [
    {'path': 'obj.json', 'label': 'Objectness', 'color': 'red'},
    {'path': 'cls_history.json', 'label': 'Class', 'color': 'green'},
    {'path': 'obj-cls_history.json', 'label': 'Objectness*Class', 'color': 'blue'}
]

plt.figure(figsize=(12, 6), dpi=100)

# Загрузка и построение данных
for file_info in files:
    try:
        with open(file_info['path'], 'r') as f:
            data = json.load(f)

        plt.plot(
            data['epochs'],
            data['losses'],
            label=file_info['label'],
            color=file_info['color'],
            linewidth=2,
            marker='o',
            markersize=1
        )
    except Exception as e:
        print(f"Ошибка при загрузке {file_info['path']}: {str(e)}")

# Оформление графика
plt.xlabel('Номер эпохи', fontsize=12)
plt.ylabel('Значение функции потерь', fontsize=12)
plt.grid(True, alpha=0.4, linestyle='--')
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()

# Показать график
plt.show()