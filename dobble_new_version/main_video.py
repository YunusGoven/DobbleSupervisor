import cv2
import numpy as np
from picamera2 import Picamera2, Preview
from libcamera import Transform
import time


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def preprocess_image(image_frame):
    lab = cv2.cvtColor(image_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # final = cv2.GaussianBlur(final, (11, 11), 0)      
    gray_image = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)
    thresh_image = cv2.threshold(gray_image, 190, 255, cv2.THRESH_BINARY)[1]
    
    ######################################################

    kernel = np.ones((3, 3), np.uint8)
    thresh_image = cv2.erode(thresh_image, kernel, iterations=1)  #reduire le bors
    # cv2.imshow("ERR", eroded)
    # cv2.waitKey(0)

    outils = cv2.subtract(gray_image, thresh_image)

    imageshE = image_resize(outils, width= 876 ,height= 444)

    _, seuillageAuto = cv2.threshold(outils, 0, 255, cv2.THRESH_BINARY+  cv2.THRESH_OTSU) # avoir image binaire
    seuillageAuto = cv2.GaussianBlur(seuillageAuto, (5,5), 0)
    # seuillageAuto = cv2.medianBlur(seuillageAuto, 3)

    seise = image_resize(seuillageAuto, width= 876 ,height= 444)

    # cv2.imshow("seuil", seise)
    
    return image_frame, seuillageAuto


def detect_and_normalize_symbols(morph_image, original_image):
    # DÃ©tecter les contours avec hiÃ©rarchie
    contours, hierarchy = cv2.findContours(morph_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(contours)
    symbols = []
    positions = []

    for i, contour in enumerate(contours):
         if hierarchy[0][i][3] != -1:  
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            #ici 
            if perimeter > 0 and (area > 800 and area < 110000):# Ajustez le seuil d'aire au cas ou on a des des pixels qui n'ont pas Ã©tÃ© filtrÃ© plus tot
                    x, y, w, h = cv2.boundingRect(contour)

                    # CrÃ©er un masque de la taille de l'image du symbole
                    mask = np.zeros((h, w), dtype=np.uint8)

                    # Ajuster les contours par rapport Ã  la rÃ©gion du symbole
                    shifted_contour = contour - [x, y]  # DÃ©calage du contour pour le placer dans le masque
                    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
                    
                    kernel = np.ones((7, 7), np.uint8)  # POUR ENLEVER LE BLANC DES CONTOURES DES SYMBOLES 
                    mask = cv2.erode(mask, kernel, iterations=1)
                    
                    # Appliquer le masque pour extraire le symbole de l'image d'origine
                    symbol = original_image[y:y+h, x:x+w]
                    symbol_with_mask = cv2.bitwise_and(symbol, symbol, mask=mask)

                    # Redimensionner le symbole Ã  une taille fixe pour pouvoir les comparer plutard
                    symbol_normalized = cv2.resize(symbol_with_mask, (400, 400))

                    # Ajouter aux rÃ©sultats
                    symbols.append(symbol_normalized)
                    # symbols.append(symbol_with_mask)
                    positions.append((x, y, w, h))
                    cv2.drawContours(original_image, [contour], -1, (0, 0, 255), 2)  # Couleur rouge (BGR: (0,0,255)), Ã©paisseur 2
                    # cv2.imshow(f"Symbole Extrait", symbol)
                    # cv2.waitKey(0)
                    
    # del symbols[0]
    # del symbols[8]
    ori = image_resize(original_image, width= 876 ,height= 444)

    cv2.imshow("Video Stream", ori)
    return symbols, positions

def preprocess_imageee(image):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # AmÃ©liorer le contraste (en utilisant l'Ã©galisation d'histogramme)
    gray = cv2.equalizeHist(gray)
    
    return gray


def compare_symbols(orb, symbol1, symbols):
    symbol1 = preprocess_imageee(symbol1)
    kp1, des1 = orb.detectAndCompute(symbol1, None)
    img1  = cv2.drawKeypoints(symbol1, kp1, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
    for jdx in range(8, 16):  # Limiter Ã  l'intervalle des indices 8 Ã  15
        symbol2 = symbols[jdx]
        symbol2 = preprocess_imageee(symbol2)
        
        # DÃ©tecter les points d'intÃ©rÃªt et les descripteurs ORB pour chaque image
        kp2, des2 = orb.detectAndCompute(symbol2, None)

        img = cv2.drawKeypoints(symbol2, kp2, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
        # cv2.imshow("Keypoints 1", img1)
        # cv2.imshow("Keypoints 2", img)
        
        # cv2.destroyAllWindows()

        # Si les descripteurs ne sont pas trouvÃ©s, retourner False
        if des1 is None or des2 is None:
            return False

        # CrÃ©er un matcher de descripteurs (Brute Force avec Hamming)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Trouver les correspondances entre les descripteurs
        matches = bf.match(des1, des2)
        print(len(matches))
        
    
        # Si le nombre de correspondances est supÃ©rieur Ã  un seuil, on les considÃ¨re comme similaires
        if len(matches) ==221 :  # Ce seuil peut Ãªtre ajustÃ© 152
             return jdx
    return -1

def take_picture():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1640,1232)})

    picam2.configure(config)
    # DÃ©marrer la camÃ©ra
    picam2.start()
    # Flux vidÃ©o
    try:
        while True:
            frame = picam2.capture_array()
            cv2.imshow("Video Stream", frame)
            original_image, morph_image = preprocess_image(frame)
            symbols, positions = detect_and_normalize_symbols(morph_image, original_image)
            
            # orb = cv2.ORB_create(nfeatures=1000)
            # # Supposons que 'symbols' est une liste d'images extraites
            # for idx in range(8):  # Limiter aux 8 premiers symboles (indices 0 Ã  7)
            #     is_same = compare_symbols(orb, symbols[idx], symbols)
            #     if is_same != -1:
            #         print(f"Les symboles {idx + 1} et {is_same +1} sont  similaires.")
            #         pos1 = positions[idx]
            #         x, y, largeur, hauteur = pos1  
            #         pos2 = positions[is_same+1]
            #         x1, y1, largeur1, hauteur1 = pos2  
            #         # Couleur mauve en BGR

            # # Calculer le coin infÃ©rieur droit du rectangle
            #         coin_oppose1  = (x + largeur, y + hauteur)

            #         coin_oppose2 = (x1 + largeur1, y1 + hauteur1)
            #         # Couleur mauve en BGR
            #         mauve = (255, 0, 255)

            #         # Dessiner le rectangle
            #         cv2.rectangle(original_image, (x,y), coin_oppose1, color=mauve, thickness=2)  # Adjuste thickness selon ton besoin
            #         cv2.rectangle(original_image, (x1,y1), coin_oppose2, color=mauve, thickness=2)  # Adjuste thickness selon ton besoin

                    # Afficher l'image avec le rectangle mauve
                    # cv2.imshow("Image avec rectangle mauve", original_image)

            #        break
            # Quitter avec la touche 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Capture interrompue.")

    # ArrÃªter et nettoyer
    cv2.destroyAllWindows()
    picam2.stop()

# Exemple d'utilisatio
take_picture()
