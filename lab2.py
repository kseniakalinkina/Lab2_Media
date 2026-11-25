import os
import datetime
import cv2
import numpy as np

# Пути к папкам (относительно файла lab2.py)
BASE_DIR   = "."
INPUT_DIR  = os.path.join(BASE_DIR, "data", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
LOG_FILE   = os.path.join(BASE_DIR, "logs", "history.txt")


def log_result(filename, filter_name, status):
    """Записывает строку в журнал (режим 'a', UTF-8, с новой строки)."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] Файл: {filename} | Фильтр: {filter_name} | Статус: {status}\n"
    try:
        with open(LOG_FILE, "a", encoding="UTF-8") as f:
            f.write(line)
    except Exception as e:
        print(f"Не удалось записать лог: {e}")


def gamma_correction(img, gamma=1.5):
    """Вариант 68, задание №2 = 3 — Гамма-коррекция."""
    img_f = img.astype(np.float32) / 255.0
    out = np.power(img_f, gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def box_filter(img, ksize=5):
    """Вариант 68, задание №1 = 3 — Box-фильтр (усреднение)."""
    return cv2.blur(img, (ksize, ksize))


def process_image(img):
    """Последовательно применяем оба фильтра."""
    img = box_filter(img, ksize=5)         # Сначала Box-фильтр
    img = gamma_correction(img, gamma=1.5) # Затем гамма-коррекция
    return img


def main():
    filename = input("Введите имя файла для обработки (например, image1.jpg или image2.png): ").strip()
    if not filename:
        print("Ошибка: имя файла не задано.")
        return

    filepath = os.path.join(INPUT_DIR, filename)
    filter_desc = "Box-фильтр + Гамма-коррекция"

    # 1) Проверка существования файла
    if not os.path.exists(filepath):
        print("Ошибка: Файл не найден")
        log_result(filename, filter_desc, "Ошибка: файл не найден")
        return

    # 2) Чтение файла с перехватом ошибки прав доступа
    try:
        # np.fromfile вызывает настоящий PermissionError в Windows при запрете ACL
        data = np.fromfile(filepath, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print("Ошибка: не удалось декодировать изображение.")
            log_result(filename, filter_desc, "Ошибка: битый файл")
            return
    except PermissionError:
        print(f"Ошибка: Нет прав на чтение файла {filename}. Обратитесь к администратору")
        log_result(filename, filter_desc, "Ошибка доступа")
        return
    except Exception:
        # На всякий случай ловим другие ошибки ввода-вывода
        print(f"Ошибка: Нет прав на чтение файла {filename}. Обратитесь к администратору")
        log_result(filename, filter_desc, "Ошибка доступа")
        return

    # 3) Обработка
    try:
        result = process_image(img)
    except Exception as e:
        print(f"Ошибка при обработке: {e}")
        log_result(filename, filter_desc, "Ошибка обработки")
        return

    # 4) Сохранение
    os.makedirs(OUTPUT_DIR, exist_ok=True) # На случай, если папки нет
    out_path = os.path.join(OUTPUT_DIR, "result.jpg")
    try:
        ok = cv2.imwrite(out_path, result)
        if not ok:
            raise IOError("cv2.imwrite вернул False")
    except Exception as e:
        print(f"Ошибка при сохранении результата: {e}")
        log_result(filename, filter_desc, "Ошибка сохранения")
        return

    # 5) Успех
    print(f"Готово! Результат сохранён: {out_path}")
    log_result(filename, filter_desc, "Успешно")


if __name__ == "__main__":
    main()