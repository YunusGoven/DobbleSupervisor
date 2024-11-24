import subprocess
import time
from picamera2 import Picamera2

# Initialiser la caméra
picam2 = Picamera2()

# Créer une configuration de capture vidéo
camera_config = picam2.create_video_configuration(main={"size": (1920, 1080), "format": "RGB888"})
picam2.configure(camera_config)

# Démarrer la caméra
picam2.start()

# Lancer le processus ffmpeg pour streamer
ffmpeg_command = [
    'ffmpeg',
    '-y',                        # Ecraser le fichier de sortie sans demander
    '-f', 'rawvideo',             # Format brut vidéo
    '-pix_fmt', 'rgb24',        # Format de pixel compatible avec la caméra
    '-r','60',
    '-s', '1920x1080',              # Taille de la vidéo
    '-i', '-',                    # Entrée en format brut via stdin
    '-c:v', 'libx264',            # Codec vidéo H.264
    '-preset', 'ultrafast',       # Préréglage de vitesse d'encodage (plus rapide, moins compressé)
    '-f', 'flv',                  # Format du flux de sortie (FLV pour RTMP)
    'rtmp://172.20.10.13/live/stream'  # L'URL de ton serveur RTMP
]

# Lancer le processus ffmpeg
ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

# Capture et envoi des frames à ffmpeg
try:
    while True:
        # Capturer une frame de la caméra
        frame = picam2.capture_array()

        # Envoyer la frame à ffmpeg via stdin
        ffmpeg_process.stdin.write(frame.tobytes())

        time.sleep(0.033)  # Attendre environ 1/30s entre les images pour 30fps

except KeyboardInterrupt:
    print("Streaming interrompu.")

finally:
    # Terminer le processus ffmpeg et arrêter la caméra
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    picam2.stop()
