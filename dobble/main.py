import cv2
import numpy as np

def preprocess_image(image_path):
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image non trouvée ou chemin incorrect.")
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Appliquer le seuillage adaptatif pour isoler les symboles en blanc sur fond noir
    # thresh_image = cv2.adaptiveThreshold(
    #     gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #     cv2.THRESH_BINARY_INV, 199, 5
    # )
    _, thresh_image = cv2.threshold(gray_image, 251, 255, cv2.THRESH_BINARY_INV) # enlver le max de blanc il faut jouer avec 251 ou bien faire un truc avant pour le fond blanc soit different que la couleur blanche a l'interieur des symbole
    # cv2.imshow("se", thresh_image)
    # cv2.waitKey(0)
    kernel = np.ones((7, 7), np.uint8)
    eroded = cv2.erode(thresh_image, kernel, iterations=1)  #reduire le bors
    # cv2.imshow("ERR", eroded)
    # cv2.waitKey(0)

    # Appliquer la dilatation pour regagner un peu de taille
    kernel2 = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(eroded, kernel2, iterations=1)
    # cv2.imshow("DI", dilated)
    # cv2.waitKey(0)

    # Appliquer un filtre gaussien pour lisser l'image
    smoothed = cv2.GaussianBlur(dilated, (5, 5), 0)
    # cv2.imshow("GAU", smoothed)
    # cv2.waitKey(0) 


    # Opération de fermeture morphologique pour remplir les petits trous
    # Appliquer une fermeture morphologique avec un noyau plus petit
    # kernel_close = np.ones((3,3), np.uint8)
    # morph_image = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel_close)
    

    outils = cv2.subtract(gray_image, smoothed)


    _, seuillageAuto = cv2.threshold(outils, 0, 255, cv2.THRESH_BINARY+  cv2.THRESH_OTSU) # avoir image binaire


    cv2.imshow("se", thresh_image)
    cv2.waitKey(0)
    return image, seuillageAuto

def detect_and_normalize_symbols(morph_image, original_image):
    # Détecter les contours avec hiérarchie
    contours, hierarchy = cv2.findContours(morph_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    symbols = []
    positions = []

    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1: #contour de la carte
             if hierarchy[0][i][3] == -1:  # Ajuste la plage selon le besoin
                cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 4)  # Vert
        else:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                # Filtrer en fonction de la taille et hiérarchie
                if area > 500 :  # Ajustez le seuil d'aire au cas ou on a des des pixels qui n'ont pas été filtré plus tot
                    x, y, w, h = cv2.boundingRect(contour)

                    # Créer un masque de la taille de l'image du symbole
                    mask = np.zeros((h, w), dtype=np.uint8)

                    # Ajuster les contours par rapport à la région du symbole
                    shifted_contour = contour - [x, y]  # Décalage du contour pour le placer dans le masque
                    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

                    # Appliquer le masque pour extraire le symbole de l'image d'origine
                    symbol = original_image[y:y+h, x:x+w]
                    symbol_with_mask = cv2.bitwise_and(symbol, symbol, mask=mask)

                    # Redimensionner le symbole à une taille fixe pour pouvoir les comparer plutard
                    # symbol_normalized = cv2.resize(symbol_with_mask, (400, 400))

                    # Ajouter aux résultats
                    # symbols.append(symbol_normalized)
                    symbols.append(symbol_with_mask)
                    positions.append((x, y, w, h))
                    cv2.drawContours(original_image, [contour], -1, (0, 0, 255), 2)  # Couleur rouge (BGR: (0,0,255)), épaisseur 2


    print(len(symbols))
    cv2.imshow("image", original_image)
    cv2.waitKey(0)
    return symbols, positions


# Exemple d'utilisation
image_path = '.\\dobble\\images\\1.png'
original_image, morph_image = preprocess_image(image_path)
symbols, positions = detect_and_normalize_symbols(morph_image, original_image)

# Affichage des symboles extraits
for idx, symbol in enumerate(symbols):
    cv2.imshow(f"Symbole Extrait {idx + 1}", symbol)

cv2.waitKey(0)
cv2.destroyAllWindows()
