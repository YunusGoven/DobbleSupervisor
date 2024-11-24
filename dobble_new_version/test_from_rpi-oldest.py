import cv2
import imutils

import numpy as np
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

def preprocess_image(image_path):
    # Charger l'image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image non trouvée ou chemin incorrect.")
    #############################################################
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(32, 32))
    gray_image = clahe.apply(gray)
    gray_image = cv2.medianBlur(gray_image, 21)
    gray_image = cv2.medianBlur(gray_image, 21)


    imageshE = image_resize(gray_image, width= 876 ,height= 444)

    cv2.imshow("imageshE", imageshE)
    cv2.waitKey(0)
    # cv2.imshow("finam", final)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # final = cv2.GaussianBlur(final, (3, 3), 0)      
    # gray_image = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)

    #175
    thresh_image = cv2.adaptiveThreshold(gray_image, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 
                                     51, -2)
    
    imagesh = image_resize(thresh_image, width= 876 ,height= 444)
    thresh_image = cv2.medianBlur(thresh_image, 7)

    cv2.imshow("OT1", imagesh)
    cv2.waitKey(0)
    thresh_image = cv2.threshold(gray_image, 173, 255, cv2.THRESH_BINARY_INV)[1]
    thresh_image = cv2.medianBlur(thresh_image, 7)

    imagesh = image_resize(thresh_image, width= 876 ,height= 444)

    cv2.imshow("OTRESH&1", imagesh)
    # cv2.waitKey(0)
    ######################################################

    # thresh_image = cv2.GaussianBlur(thresh_image, (5,5), 0)
    thresh_image = cv2.medianBlur(thresh_image, 15)  
    trsh0 = image_resize(thresh_image,  width= 876 ,height= 444)


    cv2.imshow("medianBlur&", trsh0)
    cv2.waitKey(0)

    # kernel2 = np.ones((3, 3), np.uint8)
    # thresh_image = cv2.erode(thresh_image, kernel2, iterations=1)


    trsh1 = image_resize(thresh_image,  width= 876 ,height= 444)


    cv2.imshow("1dilate&", trsh1)
    cv2.waitKey(0)
   
    # kernel_close = np.ones((5,5), np.uint8)
    # thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel_close)
    
    # trsh2 = image_resize(thresh_image, width= 876 ,height= 444)

    # cv2.imshow("2open&", trsh2)

    kernel = np.ones((9,9), np.uint8) # A 11 c'est mieux mais detecte 2 pour 1 
    thresh_image = cv2.dilate(thresh_image, kernel, iterations=1)  #reduire le bors

    trsh3 = image_resize(thresh_image,  width= 876 ,height= 444)

    cv2.imshow("3erode&", trsh3)

    # kernel_close = np.ones((7,7), np.uint8)
    # thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel_close)
    # # thresh_image = cv2.GaussianBlur(thresh_image, (5, 5), 0)
    # trsh4 = image_resize(thresh_image, width= 876 ,height= 444)

    # cv2.imshow("4close&fin", trsh4)
    cv2.waitKey(0)

    ######################3
    outils = cv2.subtract(gray_image, thresh_image)
    

    outilsSh = image_resize(outils, width= 876 ,height= 444)

    cv2.imshow("OUTIL&", outilsSh)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ##################3

    # FAIRE SUR THRESH_IMAGE - outils
    _, seuillageAuto = cv2.threshold(outils, 0, 255, cv2.THRESH_BINARY_INV+  cv2.THRESH_OTSU) # avoir image binaire
    # seuillageAuto = cv2.GaussianBlur(seuillageAuto, (3,3), 0)


    seuillageAutoSh = image_resize(seuillageAuto, width= 876 ,height= 444)

    cv2.imshow("OTRESH&2", seuillageAutoSh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    
    return image, seuillageAuto


def detect_and_normalize_symbols(morph_image, original_image):
    # Détecter les contours avec hiérarchie
    contours, hierarchy = cv2.findContours(morph_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(contours)
    symbols = []
    positions = []

    for i, contour in enumerate(contours):
         if hierarchy[0][i][3] != -1:  
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            #ici 
            if perimeter > 0 and (area > 800 and area < 110000):# Ajustez le seuil d'aire au cas ou on a des des pixels qui n'ont pas été filtré plus tot
                    x, y, w, h = cv2.boundingRect(contour)

                    # Créer un masque de la taille de l'image du symbole
                    mask = np.zeros((h, w), dtype=np.uint8)

                    # Ajuster les contours par rapport à la région du symbole
                    shifted_contour = contour - [x, y]  # Décalage du contour pour le placer dans le masque
                    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
                    
                    kernel = np.ones((7, 7), np.uint8)  # POUR ENLEVER LE BLANC DES CONTOURES DES SYMBOLES 
                    mask = cv2.erode(mask, kernel, iterations=1)
                    
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
                    print(f'area {area} permiter {perimeter}')
                    # cv2.imshow(f"Symbole Extrait", symbol)
                    # cv2.waitKey(0)
                    
    # del symbols[0]
    # del symbols[8]
    print(len(symbols))
    original_image_sh = image_resize(original_image, width= 876 ,height= 444)

    cv2.imshow("image", original_image_sh )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return symbols, positions

def preprocess_imageee(image):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Améliorer le contraste (en utilisant l'égalisation d'histogramme)
    gray = cv2.equalizeHist(gray)
    
    return gray


def compare_symbols(orb, symbol1, symbols):
    symbol1 = preprocess_imageee(symbol1)
    kp1, des1 = orb.detectAndCompute(symbol1, None)
    img1  = cv2.drawKeypoints(symbol1, kp1, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
    for jdx in range(8, 16):  # Limiter à l'intervalle des indices 8 à 15
        symbol2 = symbols[jdx]
        symbol2 = preprocess_imageee(symbol2)
        
        # Détecter les points d'intérêt et les descripteurs ORB pour chaque image
        kp2, des2 = orb.detectAndCompute(symbol2, None)

        img = cv2.drawKeypoints(symbol2, kp2, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
        # cv2.imshow("Keypoints 1", img1)
        # cv2.imshow("Keypoints 2", img)
        
        # cv2.destroyAllWindows()

        # Si les descripteurs ne sont pas trouvés, retourner False
        if des1 is None or des2 is None:
            return False

        # Créer un matcher de descripteurs (Brute Force avec Hamming)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Trouver les correspondances entre les descripteurs
        matches = bf.match(des1, des2)
        print(len(matches))
        
    
        # Si le nombre de correspondances est supérieur à un seuil, on les considère comme similaires
        if len(matches) ==221 :  # Ce seuil peut être ajusté 152
             return jdx
    return -1


# Exemple d'utilisation
image_path = '.\\dobble\\images\\image.jpg'
original_image, morph_image = preprocess_image(image_path)
symbols, positions = detect_and_normalize_symbols(morph_image, original_image)

# Affichage des symboles extraits
for idx, symbol in enumerate(symbols):
    cv2.imshow(f"Symbole Extrait {idx + 1}", symbol)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

cv2.waitKey(0)
cv2.destroyAllWindows()   

orb = cv2.ORB_create(nfeatures=1000)
# Supposons que 'symbols' est une liste d'images extraites
for idx in range(8):  # Limiter aux 8 premiers symboles (indices 0 à 7)
    # cv2.imshow(f"Symbole Extrait {idx + 1}", symbols[idx])
    is_same = compare_symbols(orb, symbols[idx], symbols)
    if is_same != -1:
        print(f"Les symboles {idx + 1} et {is_same +1} sont  similaires.")
        pos1 = positions[idx]
        x, y, largeur, hauteur = pos1  
        pos2 = positions[is_same+1]
        x1, y1, largeur1, hauteur1 = pos2  
        # Couleur mauve en BGR

 # Calculer le coin inférieur droit du rectangle
        coin_oppose1  = (x + largeur, y + hauteur)

        coin_oppose2 = (x1 + largeur1, y1 + hauteur1)
        # Couleur mauve en BGR
        mauve = (255, 0, 255)

        # Dessiner le rectangle
        cv2.rectangle(original_image, (x,y), coin_oppose1, color=mauve, thickness=2)  # Adjuste thickness selon ton besoin
        cv2.rectangle(original_image, (x1,y1), coin_oppose2, color=mauve, thickness=2)  # Adjuste thickness selon ton besoin
        original_image_sh = image_resize(original_image, width= 876 ,height= 444)

        # Afficher l'image avec le rectangle mauve
        cv2.imshow("Image avec rectangle mauve", original_image_sh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
    
    


cv2.waitKey(0)
cv2.destroyAllWindows()


