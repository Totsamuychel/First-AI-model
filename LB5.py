import numpy
from PIL import Image
import scipy.special
import matplotlib.pyplot as plt
import scipy.ndimage

# определение класса нейронной сети
class neuralNetwork:
  
    # Инициализировать нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
      # задать количество узлов во входном, скрытом и выходном слоях
      self.inodes = inputnodes
      self.hnodes = hiddennodes
      self.onodes = outputnodes


      # Матрицы весовых коэффициентов связей wih и who
      # Весовые коэффицициенты связей между узлом i и узлом j
      # следующего слоя обозначені как w_ij:
      # w11 w21
      # w12 w22 и т.д.
      self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
      self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
      # коэффициент обучения
      self.lr = learningrate
      # Использование сигмоиды в качестве функции активации
      self.activation_function = lambda x: scipy.special.expit(x)

    # тренировка нейронной сети
    def train(self, inputs_list, targets_list):
        # 1. преобразовать список входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 2. Прямое распространение сигнала
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 3. Вычислить ошибки (разницу между целевыми и реальным выходом)
        output_errors = targets - final_outputs

        # 4. Вычисляем ошибки скрытого слоя (обратное распространение)
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 5. Обновляем веса связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # 6. Обновляем веса между входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
   
    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразовать список входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        # вычислить входные сигналы в скрытый слой
        hidden_inputs = numpy.dot(self.wih, inputs)
        # вычислить выходные сигналы из скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # вычислить входные сигналы в выходной слой
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # вычислить выходные сигналы из выходного слоя
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    def backquery(self, targets_list):
        final_outputs = numpy.array(targets_list, ndmin=2).T
         # Ограничиваем значения для избежания ошибок в logit
        final_outputs = numpy.clip(final_outputs, 0.01, 0.99)
        final_inputs = scipy.special.logit(final_outputs)

        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        hidden_outputs = (hidden_outputs - hidden_outputs.min()) / (hidden_outputs.max() - hidden_outputs.min())
        hidden_outputs = hidden_outputs * 0.98 + 0.01
        hidden_inputs = scipy.special.logit(hidden_outputs)

        inputs = numpy.dot(self.wih.T, hidden_inputs)
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        inputs = inputs * 0.98 + 0.01
        return inputs
    
# Функция аугментации с поворотом изображений    
def train_with_augmentation(network, image_array, label):
    """Тренируем сеть на исходном и повёрнутых на ±10° изображениях"""
    inputs = (image_array / 255.0 * 0.99) + 0.01
    inputs = inputs.reshape(784)
    targets = numpy.zeros(10) + 0.01
    targets[label] = 0.99
    network.train(inputs, targets)
    for angle in [-10, 10]:
        rotated = scipy.ndimage.rotate(image_array, angle, reshape=False, order=1, mode='nearest')
        inputs_rot = (rotated / 255.0 * 0.99) + 0.01
        inputs_rot = inputs_rot.reshape(784)
        network.train(inputs_rot, targets)

def safe_recognize_digit(filename):
    """Безопасно распознаём одиночное PNG‑изображение"""
    try:
        img = Image.open(filename)
    except FileNotFoundError:
        print(f"Файл {filename} не найден!")
        return None
    except Exception as e:
        print(f"Ошибка при открытии {filename}: {e}")
        return None

    plt.imshow(img, cmap='Greys'); plt.title(f"Исходник {filename}"); plt.show()
    img = img.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
    img = Image.eval(img, lambda x: 255 - x)
    img_array = numpy.array(img)

    img_norm = (img_array / 255.0 * 0.99) + 0.01
    result = n.query(img_norm.reshape(784))
    plt.imshow(img_array, cmap='Greys'); plt.title("После предобработки"); plt.show()
    print("Ответ сети:", result.ravel())
    print(">>> Предсказано:", numpy.argmax(result))
    return numpy.argmax(result)

# ======= Параметры сети и создание =======
input_nodes, hidden_nodes, output_nodes = 784, 100, 10
learning_rate = 0.1
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# ======= Загрузка тренировочного набора =======
try:
    with open("mnist_train_60K.csv", 'r') as f:
        data_list = f.readlines()
    print(f"Загружено {len(data_list)-1} образцов для обучения")
except FileNotFoundError:
    raise SystemExit("Файл mnist_train_60K.csv не найден!")

"""
# Пример: визуализировать цифру
all_values = data_list[1].split(',')
image_array = numpy.asarray(all_values[1:], dtype=float).reshape((28, 28))
plt.imshow(image_array, cmap='Greys')
plt.show()
"""

# Количество эпох (сколько раз прогоняем все данные через сеть)
epochs = 7

for e in range(epochs):
    total_samples = len(data_list) - 1
    # перебрать все записи в тренировочном наборе данных
    for i, record in enumerate(data_list[1:]):
        # получить список значений из записи, используя символы
        # запятой (',) в качестве разделителя
        all_values = record.split(',')
        label = int(all_values[0])  # правильный ответ (какая цифра на картинке)
        image_array = numpy.asarray(all_values[1:], dtype=float).reshape((28, 28))
        train_with_augmentation(n, image_array, label)
        if i % 5000 == 0:
            progress = (i / total_samples) * 100
            print(f"Эпоха {e+1}, прогресс: {progress:.1f}% ({i}/{total_samples})")
    print(f"Эпоха {e+1} завершена")

# ======= Тестирование на официальном тест‑сете =======
try:
    with open("mnist_test10K.csv", 'r') as f:
        test_list = f.readlines()
except FileNotFoundError:
    print("Test‑файл mnist_test10K.csv отсутствует – пропускаю расчёт accuracy.")
    test_list = []

if test_list:
    scorecard = []
    for record in test_list[1:]:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (numpy.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        scorecard.append(int(label == correct_label))
    scorecard_array = numpy.asarray(scorecard)
    print("Accuracy на тест‑сете:", scorecard_array.mean()*100, "%")

# Дообучение на своих png-изображениях с аугментацией
for i in range(1, 5):
    try:
        img = Image.open(f"{i}.png").convert('L').resize((28, 28))
        train_with_augmentation(n, numpy.array(img), i if i < 10 else 0)
    except Exception as ex:
        print(f"Ошибка с файлом {i}.png: {ex}")

# Проверить свои изображения
for i in range(1, 5):
    safe_recognize_digit(f"{i}.png")

# Визуализация "обратного запроса"
for i in range(10):
    target_outputs = numpy.zeros(10) + 0.01
    target_outputs[i] = 0.99
    img = n.backquery(target_outputs)
    plt.imshow(img.reshape(28, 28), cmap='Greys')
    plt.title(f'Цифра {i} глазами сети')
    plt.show()