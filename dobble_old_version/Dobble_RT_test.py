import cv2
import numpy as np


# Fonction de prétraitement pour chaque frame
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(32, 32))
    gray_image = clahe.apply(gray)
    gray_image = cv2.medianBlur(gray_image, 21)

    # Seuillage
    thresh_image = cv2.adaptiveThreshold(
        gray_image, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        51, -2
    )
    kernel = np.ones((9, 9), np.uint8)
    thresh_image = cv2.dilate(thresh_image, kernel, iterations=1)

    return thresh_image

# Détection et normalisation des symboles
def detect_and_normalize_symbols(morph_image, original_frame):
    contours, hierarchy = cv2.findContours(morph_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    symbols = []
    positions = []

    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:  
            area = cv2.contourArea(contour)
            if 800 < area < 110000:  # Ajustez les seuils
                x, y, w, h = cv2.boundingRect(contour)
                mask = np.zeros((h, w), dtype=np.uint8)
                shifted_contour = contour - [x, y]
                cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
                mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)

                symbol = original_frame[y:y+h, x:x+w]
                symbol_with_mask = cv2.bitwise_and(symbol, symbol, mask=mask)

                symbols.append(symbol_with_mask)
                positions.append((x, y, w, h))
                cv2.drawContours(original_frame, [contour], -1, (0, 0, 255), 2)
    return symbols, positions

# Comparaison des symboles
def compare_symbols(orb, symbol1, symbols):
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(symbol1, cv2.COLOR_BGR2GRAY), None)
    for idx, symbol2 in enumerate(symbols):
        kp2, des2 = orb.detectAndCompute(cv2.cvtColor(symbol2, cv2.COLOR_BGR2GRAY), None)
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if len(matches) > 50:  # ici c'est pour ajuster le seuil
                return idx
    return -1


# Initialisation de la caméra
cap = cv2.VideoCapture(0)  # index de la caméra... 0-1 ça dépend de la cam

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

orb = cv2.ORB_create(nfeatures=1000)  # Initialisation d'ORB pour la comparaison
# Boucle principale pour le traitement en temps réel
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire une frame.")
        break

    # Prétraitement
    morph_image = preprocess_frame(frame)
    
    # Détection des symboles
    symbols, positions = detect_and_normalize_symbols(morph_image, frame.copy())
    
    # Comparaison des symboles
    for i, symbol1 in enumerate(symbols):
        match_idx = compare_symbols(orb, symbol1, symbols)
        if match_idx != -1 and match_idx != i:
            x1, y1, w1, h1 = positions[i]
            x2, y2, w2, h2 = positions[match_idx]

            # Dessiner des rectangles sur les symboles similaires
            cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (255, 0, 255), 2)
            cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), (255, 0, 255), 2)
            break

    # Afficher le résultat
    cv2.imshow("Dobble Detection", frame)

    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
