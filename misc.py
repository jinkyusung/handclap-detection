import math
import numpy as np
import matplotlib.pyplot as plt


def highlight_outliers(label, pattern, outliers, fps, answer):
    x = np.array(range(0, len(pattern))) / fps
    plt.figure(figsize=(10, 6))
    plt.plot(x, pattern, label=label, color='royalblue')
    plt.title(f'Pattern of {label}')
    plt.xlabel('Sec')
    plt.ylabel(f'{label}')
    plt.legend()

    if answer:
        for start, end in answer:
            plt.axvspan(start, end, facecolor='green', alpha=0.1)
    if outliers:
        for start, end in outliers:
            plt.axvspan(start/fps, end/fps, facecolor='lightcoral', alpha=0.2)


def color(message, color='red'):
    color_map = {'red':91, 'green':92, 'blue':94}
    color_code = str(color_map[color])
    return f"\033[{color_code}m{message}\033[0m"


def echo(type, distance=None):
    if type == 'both':
        msg = color(f'Distance : {distance}', 'green')
    elif type == 'right-miss':
        msg = color("              Right-miss", 'red')
    elif type == 'left-miss':
        msg = color("Left-miss               ", 'blue')
    else:
        msg = color("Left-miss", 'blue') + color("     Right-miss", 'red') 
    print(msg)
