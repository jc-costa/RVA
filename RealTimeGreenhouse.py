import cv2
import time
from ultralytics import YOLO

# Carrega o modelo YOLOv8 com segmentação
segmenter = YOLO('yolov8n-seg.pt')

# Inicializa a captura de vídeo pela webcam padrão
camera = cv2.VideoCapture(0)

def calcular_fps(inicio, fim):
    """Calcula o FPS com base no tempo de início e fim."""
    tempo_execucao = fim - inicio
    return int(1 / tempo_execucao) if tempo_execucao > 0 else 0

try:
    while camera.isOpened():
        sucesso, frame = camera.read()
        if not sucesso:
            print("Nenhuma câmera detectada. Encerrando execução.")
            break

        inicio_tempo = time.perf_counter()
        resultado = segmenter(frame)
        fim_tempo = time.perf_counter()

        imagem_segmentada = resultado[0].plot()
        fps_atual = calcular_fps(inicio_tempo, fim_tempo)

        cv2.putText(imagem_segmentada, f'FPS: {fps_atual}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Segmentação de Imagem", imagem_segmentada)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Encerrando visualização...")
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
