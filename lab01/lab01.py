import av
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

# Функция для вычисления двумерной автокорреляции
def autocorrelate_2d(frame):
    frame_gray = np.mean(frame, axis=2)  # Переводим кадр в градации серого
    autocorr = correlate2d(frame_gray, frame_gray, mode='full')
    return autocorr

# Открываем видеофайл
input_video_path = 'lr1_1.AVI'
container = av.open(input_video_path)

# Выбираем первый кадр из видео
for frame in container.decode(video=0):
    # Преобразуем кадр в numpy-массив
    frame_array = np.array(frame.to_image())
    
    # Вычисляем автокорреляцию для кадра
    autocorr_result = autocorrelate_2d(frame_array)
    
    # Строим график двумерной автокорреляции
    plt.imshow(autocorr_result, cmap='hot', interpolation='nearest')
    plt.title('2D Autocorrelation of the Frame')
    plt.colorbar()
    plt.show()
    
    # Останавливаем после первого кадра
    break
