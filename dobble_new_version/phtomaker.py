from picamera2 import Picamera2, Preview
from libcamera import Transform
import time
picam2 = Picamera2()
print(f' prop: {picam2.camera_properties}')
#config = picam2.create_still_configuration(main={"size": picam2.sensor_resolution})
#config = picam2.create_preview_configuration(main={"size": (3280x2464)})
#config = picam2.create_preview_configuration(main={"size": (1920,1080)})
config = picam2.create_preview_configuration(main={"size": (1640,1232)})
picam2.configure(config)
picam2.start_preview(Preview.QTGL)
picam2.start()
time.sleep(5)
picam2.capture_file('/home/pi/Desktop/dobble/images/image.jpg') 
picam2.stop_preview()
picam2.stop()
