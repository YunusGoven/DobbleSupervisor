import cv2
import numpy as np


def load_test_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur de chargement de l'image.")
    return image


# Capture une image depuis la caméra du Raspberry Pi
def capture_image():
    camera = cv2.VideoCapture(0)  # Assure-toi que le bon index de caméra est utilisé
    ret, frame = camera.read()
    camera.release()
    return frame if ret else None

# Prétraitement de l'image (conversion en niveaux de gris et flou gaussien)
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred



# Détecte les contours des cartes Dobble dans l'image
def detect_cards(image):
    _, seuillageAUTO = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(seuillageAUTO, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []
    circular_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Calcul de la circularité
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            # Plus la circularité est proche de 1, plus la forme est circulaire
            if 0.7 < circularity <= 1.2:  # Ajuste la plage selon le besoin
                circular_contours.append(contour)
                cards.append(contour)  # Ajoute le contour directement

    # Dessin des contours circulaires détectés
    return cards

# Découpe chaque carte détectée et isole les symboles
def extract_symbols(image, cards):
    symbols = []
    width, height = 200, 200  # Dimension normalisée

    for card in cards:
        if len(card) >= 5:  # Vérifie que le contour a au moins quelques points
            # Transformation perspective pour aligner la carte
            epsilon = 0.08 * cv2.arcLength(card, True)
            approx = cv2.approxPolyDP(card, epsilon, True)

            if len(approx) == 4:  # Vérifie que le contour a bien 4 points
                pts = np.float32(approx)
                dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

                # Appliquer la transformation de perspective
                matrix = cv2.getPerspectiveTransform(pts, dst)
                card_img = cv2.warpPerspective(image, matrix, (width, height))

                # Transformation en niveaux de gris
                gray_card = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

                # Seuillage et détection des contours avec Canny
                edges = cv2.Canny(gray_card, 0, 150)
                
                # Opérations morphologiques pour améliorer la segmentation
                kernel = np.ones((5, 5), np.uint8)  # Ajuster la taille du noyau selon le besoin
                closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)  # Fermeture

                # Détection des contours des symboles
                symbol_contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Dessin des symboles détectés sur l'image de la carte
                for symbol_contour in symbol_contours:
                    area = cv2.contourArea(symbol_contour)
                    if area > 50:  # Ajuste ce seuil selon la taille des symboles
                        # Obtenir le rectangle englobant
                        x, y, w, h = cv2.boundingRect(symbol_contour)

                        # Extraire le symbole
                        symbol = card_img[y:y + h, x:x + w]

                        # Optionnel : Redimensionner les symboles pour normaliser la taille
                        if symbol.shape[0] > 0 and symbol.shape[1] > 0:
                            # Optionnel : Redimensionner les symboles pour normaliser la taille
                            symbol_resized = cv2.resize(symbol, (50, 50))  # Ajuste la taille selon tes besoins
                            symbols.append(symbol_resized)

                            # Dessiner le contour autour du symbole détecté sur la carte
                            cv2.drawContours(card_img, [symbol_contour], -1, (255, 0, 0), 2)  # Contour rouge

                # Afficher l'image de la carte avec les symboles détectés
                cv2.imshow("Carte avec Symboles", card_img)
                cv2.waitKey(0)  # Attendre une touche pour continuer

    print(f"Nombre de symboles extraits : {len(symbols)}")
    for idx, sym in enumerate(symbols):
        cv2.imshow(f"Symbole Extrait {idx + 1}", sym)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return symbols

def find_matching_symbols(symbols_card1, symbols_card2):
    orb = cv2.ORB_create()
    matches_found = []

    for sym1 in symbols_card1:
        kp1, des1 = orb.detectAndCompute(sym1, None)
            
        for sym2 in symbols_card2:
            kp2, des2 = orb.detectAndCompute(sym2, None)

            # Comparaison des descripteurs
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Si suffisamment de correspondances sont trouvées
            if len(matches) > 5 and matches[0].distance < 50:
                matches_found.append((sym1, sym2))  # Ajout des symboles correspondants

    if matches_found:
        print(f"Symboles communs trouvés : {len(matches_found)}")
        for idx, (sym1, sym2) in enumerate(matches_found):
            cv2.imshow(f"Symbole Commun {idx + 1} - Carte 1", sym1)
            cv2.imshow(f"Symbole Commun {idx + 1} - Carte 2", sym2)
            cv2.waitKey(0)  # Attendre une touche pour continuer
            cv2.destroyAllWindows()
    else:
        print("Aucun symbole commun détecté.")

    return matches_found

def main():
    # Capture l'image et prétraitement
    # image = capture_image()
    image = load_test_image(".\\dobble\\images\\1.png")

    if image is None:
        print("Erreur de capture de l'image.")
        return
    
    processed_image = preprocess_image(image)
    
    # Détection des cartes
    cards = detect_cards(processed_image)
    
    # Extraction des symboles sur chaque carte
    if len(cards) >= 2:  # Vérifie qu'il y a au moins deux cartes pour Dobble
        symbols_card1 = extract_symbols(image, [cards[0]])
        symbols_card2 = extract_symbols(image, [cards[1]])
        
        if len(symbols_card1) == len(symbols_card2):
          find_matching_symbols(symbols_card1, symbols_card2)
        else :
            print("Nombre de symboles différent")
    else:
        print("Nombre de cartes insuffisant.")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
