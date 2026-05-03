import cv2
import numpy as np


class FaceDetector:
    """
    Detecta rostos em imagens usando o classificador Haar Cascade do OpenCV.
    """

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: tuple = (30, 30)):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detecta rostos em um frame.

        Args:
            frame: Imagem BGR (numpy array).

        Returns:
            Lista de bounding boxes (x, y, w, h).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces if len(faces) > 0 else []

    def extract_face(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int], target_size: tuple = (160, 160), margin: int = 20
    ) -> np.ndarray:
        """
        Recorta e redimensiona o rosto detectado.

        Args:
            frame: Imagem BGR original.
            bbox: Bounding box (x, y, w, h).
            target_size: Tamanho de saída (largura, altura).
            margin: Margem extra ao redor do rosto.

        Returns:
            Imagem do rosto redimensionada.
        """
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w_frame, x + w + margin)
        y2 = min(h_frame, y + h + margin)

        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, target_size)
        return face

    def draw_faces(self, frame: np.ndarray, faces: list, labels: list = None, color: tuple = (0, 255, 0)) -> np.ndarray:
        """
        Desenha os bounding boxes e labels no frame.

        Args:
            frame: Imagem BGR.
            faces: Lista de bounding boxes.
            labels: Lista de labels (nome + confiança).
            color: Cor do retângulo (BGR).

        Returns:
            Frame anotado.
        """
        output = frame.copy()
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            if labels and i < len(labels):
                label = labels[i]
                cv2.putText(
                    output,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
        return output
